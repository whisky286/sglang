import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.data_parallel_controller import DataParallelController
from sglang.srt.managers.fault_tolerance import FaultToleranceManager
from sglang.srt.managers.io_struct import (
    AbortReq,
    FaultToleranceApplyReqInput,
    FaultToleranceCommandReqInput,
    FaultToleranceCommandReqOutput,
    FaultToleranceMLPSyncMetadata,
    FaultToleranceMetadataParityProbeReqInput,
    FaultToleranceMetadataProbeReqInput,
    FaultToleranceRecoverableErrorOutput,
    FaultToleranceSchedulerMetadata,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _ack(
    request: FaultToleranceCommandReqInput,
    original_rank: int,
    *,
    success: bool = True,
    engine_paused: bool = False,
    scheduler_metadata: FaultToleranceSchedulerMetadata | None = None,
    mlp_sync_local_metadata: FaultToleranceMLPSyncMetadata | None = None,
    mlp_sync_collective_metadata: list[FaultToleranceMLPSyncMetadata] | None = None,
    mlp_sync_group: str | None = None,
) -> FaultToleranceCommandReqOutput:
    return FaultToleranceCommandReqOutput(
        command_id=request.command_id,
        command=request.command,
        original_rank=original_rank,
        success=success,
        engine_paused=engine_paused,
        scheduler_metadata=scheduler_metadata,
        mlp_sync_local_metadata=mlp_sync_local_metadata,
        mlp_sync_collective_metadata=mlp_sync_collective_metadata,
        mlp_sync_group=mlp_sync_group,
    )


def _mlp_sync_metadata(original_rank: int) -> FaultToleranceMLPSyncMetadata:
    return FaultToleranceMLPSyncMetadata(
        num_tokens=11 * (original_rank + 1),
        num_tokens_for_logprob=7 * (original_rank + 1),
        can_cuda_graph=original_rank % 2 == 0,
        is_extend_in_batch=original_rank % 3 == 1,
        local_can_run_tbo=original_rank % 3 != 2,
        local_forward_mode=(original_rank % 4) + 1,
        can_run_breakable_cuda_graph=original_rank % 2 == 1,
    )


class TestFaultToleranceStatus(CustomTestCase):
    def _new_manager(self, dispatch, timeout: float = 0.02):
        return FaultToleranceManager(
            original_world_size=4,
            command_timeout=timeout,
            dispatch_command=dispatch,
        )

    def test_status_collects_all_original_ranks_and_is_read_only(self):
        requests = []
        manager = None

        def dispatch(request):
            requests.append(request)
            for rank in request.target_original_ranks:
                manager.handle_command_output(_ack(request, rank))

        manager = self._new_manager(dispatch)

        first_code, first = asyncio.run(manager.status())
        second_code, second = asyncio.run(manager.status())

        self.assertEqual(first_code, 200)
        self.assertEqual(second_code, 200)
        self.assertTrue(first["success"])
        self.assertEqual(first["service_state"], "HEALTHY")
        self.assertEqual(first["original_ranks"], [0, 1, 2, 3])
        self.assertEqual(
            [rank["available"] for rank in first["ranks"]],
            [True, True, True, True],
        )
        self.assertNotEqual(first["command_id"], second["command_id"])
        self.assertEqual(requests[0].target_original_ranks, [0, 1, 2, 3])
        self.assertEqual(manager._pending_commands, {})

    def test_duplicate_and_stale_acks_do_not_complete_the_query(self):
        manager = None

        def dispatch(request):
            manager.handle_command_output(_ack(request, 0))
            manager.handle_command_output(_ack(request, 0))
            manager.handle_command_output(_ack(request, 1))
            manager.handle_command_output(_ack(request, 2))
            manager.handle_command_output(
                FaultToleranceCommandReqOutput(
                    command_id="stale-command",
                    command=request.command,
                    original_rank=3,
                    success=True,
                    engine_paused=False,
                )
            )

        manager = self._new_manager(dispatch)
        status_code, body = asyncio.run(manager.status())

        self.assertEqual(status_code, 503)
        self.assertFalse(body["success"])
        self.assertEqual(body["service_state"], "UNKNOWN")
        self.assertEqual(body["missing_ranks"], [3])
        self.assertIn("Timed out", body["last_error"])
        self.assertEqual(manager._pending_commands, {})

    def test_failed_rank_ack_is_reported_without_waiting_for_timeout(self):
        manager = None

        def dispatch(request):
            for rank in request.target_original_ranks:
                manager.handle_command_output(_ack(request, rank, success=(rank != 2)))

        manager = self._new_manager(dispatch, timeout=1.0)
        status_code, body = asyncio.run(manager.status())

        self.assertEqual(status_code, 503)
        self.assertEqual(body["failed_ranks"], [2])
        self.assertEqual(body["missing_ranks"], [])
        self.assertFalse(body["ranks"][2]["available"])

    def test_status_reports_a_consistent_paused_state(self):
        manager = None

        def dispatch(request):
            for rank in request.target_original_ranks:
                manager.handle_command_output(_ack(request, rank, engine_paused=True))

        manager = self._new_manager(dispatch)
        status_code, body = asyncio.run(manager.status())

        self.assertEqual(status_code, 200)
        self.assertEqual(body["service_state"], "PAUSED")


class TestFaultToleranceControlRouting(CustomTestCase):
    def test_command_and_ack_round_trip_over_msgpack(self):
        request = FaultToleranceCommandReqInput(
            command_id="command-msgpack",
            command="status",
            target_original_ranks=[0, 1, 2, 3],
            request_id="request-msgpack",
        )
        output = _ack(request, 2)
        event = FaultToleranceRecoverableErrorOutput(
            event_id="event-msgpack",
            original_rank=2,
            request_id="request-msgpack",
            message="recoverable",
        )

        self.assertEqual(msgpack_decode(msgpack_encode(request)), request)
        self.assertEqual(msgpack_decode(msgpack_encode(output)), output)
        self.assertEqual(msgpack_decode(msgpack_encode(event)), event)
        apply_request = FaultToleranceApplyReqInput(action="retry")
        self.assertEqual(msgpack_decode(msgpack_encode(apply_request)), apply_request)
        probe_request = FaultToleranceMetadataProbeReqInput(active_mask=[1, 0, 1, 1])
        self.assertEqual(msgpack_decode(msgpack_encode(probe_request)), probe_request)

    def test_dpc_routes_only_to_available_target_original_ranks(self):
        controller = DataParallelController.__new__(DataParallelController)
        controller.workers = [
            MagicMock(name="rank_0"),
            MagicMock(name="rank_1"),
            MagicMock(name="rank_2"),
            MagicMock(name="rank_3"),
        ]
        controller.status = [True, False, True, True]
        request = FaultToleranceCommandReqInput(
            command_id="command-1",
            command="status",
            target_original_ranks=[3, 0, 3, 1, 9],
        )

        with patch(
            "sglang.srt.managers.data_parallel_controller.sock_send"
        ) as mock_send:
            controller.send_fault_tolerance_command(request)

        self.assertEqual(
            mock_send.call_args_list,
            [
                call(controller.workers[3], request),
                call(controller.workers[0], request),
            ],
        )

    def test_scheduler_returns_its_original_rank_status(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ps = SimpleNamespace(dp_rank=2)
        scheduler._engine_paused = False
        request = FaultToleranceCommandReqInput(
            command_id="command-2",
            command="status",
            target_original_ranks=[0, 2, 3],
        )

        output = scheduler.handle_fault_tolerance_command(request)

        self.assertEqual(output.command_id, "command-2")
        self.assertEqual(output.original_rank, 2)
        self.assertTrue(output.success)
        self.assertFalse(output.engine_paused)

        untargeted_request = FaultToleranceCommandReqInput(
            command_id="command-3",
            command="status",
            target_original_ranks=[0, 3],
        )
        self.assertIsNone(scheduler.handle_fault_tolerance_command(untargeted_request))


class TestFaultToleranceSurvivorMetadataProbe(CustomTestCase):
    def _new_manager(self, dispatch, timeout: float = 0.02):
        return FaultToleranceManager(
            original_world_size=4,
            command_timeout=timeout,
            dispatch_command=dispatch,
        )

    def test_contacts_only_survivors_and_fills_dead_original_rank_slot(self):
        requests = []
        manager = None

        def dispatch(request):
            requests.append(request)
            for rank in request.target_original_ranks:
                manager.handle_command_output(
                    _ack(
                        request,
                        rank,
                        scheduler_metadata=FaultToleranceSchedulerMetadata(
                            num_running_requests=rank + 1,
                            num_waiting_requests=rank,
                            last_batch_forward_mode="DECODE",
                        ),
                    )
                )

        manager = self._new_manager(dispatch)
        status_code, body = asyncio.run(
            manager.probe_survivor_metadata(active_mask=[1, 0, 1, 1])
        )

        self.assertEqual(status_code, 200)
        self.assertTrue(body["success"])
        self.assertEqual(requests[0].target_original_ranks, [0, 2, 3])
        self.assertEqual(body["target_original_ranks"], [0, 2, 3])
        self.assertEqual(body["acknowledged_ranks"], [0, 2, 3])
        self.assertEqual(
            [slot["original_rank"] for slot in body["slots"]], [0, 1, 2, 3]
        )
        self.assertEqual(body["slots"][1]["source"], "IDLE/fallback")
        self.assertEqual(
            body["slots"][1]["metadata"],
            {
                "num_running_requests": 0,
                "num_waiting_requests": 0,
                "last_batch_forward_mode": "IDLE",
            },
        )
        self.assertEqual(body["slots"][2]["source"], "Scheduler")
        self.assertEqual(body["slots"][2]["metadata"]["num_running_requests"], 3)

    def test_rejects_invalid_active_masks_before_dispatch(self):
        dispatch = MagicMock()
        manager = self._new_manager(dispatch)

        for mask in ([1, 0, 1], [1, 2, 1, 1], [0, 0, 0, 0]):
            with self.subTest(mask=mask):
                status_code, body = asyncio.run(
                    manager.probe_survivor_metadata(active_mask=mask)
                )
                self.assertEqual(status_code, 400)
                self.assertFalse(body["success"])

        dispatch.assert_not_called()

    def test_missing_active_rank_fails_without_filling_it_as_idle(self):
        manager = None

        def dispatch(request):
            for rank in (0, 2):
                manager.handle_command_output(
                    _ack(
                        request,
                        rank,
                        scheduler_metadata=FaultToleranceSchedulerMetadata(
                            num_running_requests=0,
                            num_waiting_requests=0,
                            last_batch_forward_mode="IDLE",
                        ),
                    )
                )

        manager = self._new_manager(dispatch, timeout=0.001)
        status_code, body = asyncio.run(
            manager.probe_survivor_metadata(active_mask=[1, 0, 1, 1])
        )

        self.assertEqual(status_code, 503)
        self.assertFalse(body["success"])
        self.assertEqual(body["missing_ranks"], [3])
        self.assertEqual(body["slots"][3]["source"], "MISSING")
        self.assertIsNone(body["slots"][3]["metadata"])

    def test_scheduler_reports_its_current_local_state(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ps = SimpleNamespace(dp_rank=2)
        scheduler._engine_paused = True
        scheduler.running_batch = SimpleNamespace(reqs=[object(), object()])
        scheduler.waiting_queue = [object()]
        scheduler.last_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(name="DECODE")
        )
        request = FaultToleranceCommandReqInput(
            command_id="metadata-command",
            command="probe_scheduler_metadata",
            target_original_ranks=[0, 2, 3],
        )

        output = scheduler.handle_fault_tolerance_command(request)

        self.assertTrue(output.success)
        self.assertEqual(output.original_rank, 2)
        self.assertTrue(output.engine_paused)
        self.assertEqual(output.scheduler_metadata.num_running_requests, 2)
        self.assertEqual(output.scheduler_metadata.num_waiting_requests, 1)
        self.assertEqual(output.scheduler_metadata.last_batch_forward_mode, "DECODE")


class TestFaultToleranceMetadataParityProbe(CustomTestCase):
    def _new_manager(self, dispatch, timeout: float = 0.02):
        return FaultToleranceManager(
            original_world_size=4,
            command_timeout=timeout,
            dispatch_command=dispatch,
        )

    def test_cpu_control_slots_match_every_collective_view_field_by_field(self):
        requests = []
        manager = None
        collective = [_mlp_sync_metadata(rank) for rank in range(4)]

        def dispatch(request):
            requests.append(request)
            for rank in request.target_original_ranks:
                if request.command == "status":
                    output = _ack(request, rank, engine_paused=True)
                else:
                    output = _ack(
                        request,
                        rank,
                        engine_paused=True,
                        mlp_sync_local_metadata=_mlp_sync_metadata(rank),
                        mlp_sync_collective_metadata=collective,
                        mlp_sync_group="tp-cpu-all-gather",
                    )
                manager.handle_command_output(output)

        manager = self._new_manager(dispatch)
        status_code, body = asyncio.run(manager.probe_metadata_parity())

        self.assertEqual(status_code, 200)
        self.assertTrue(body["success"])
        self.assertEqual(
            [request.command for request in requests],
            ["status", "probe_mlp_sync_metadata_parity"],
        )
        self.assertEqual(body["mlp_sync_group"], "tp-cpu-all-gather")
        self.assertEqual(
            [slot["original_rank"] for slot in body["cpu_control_slots"]],
            [0, 1, 2, 3],
        )
        self.assertEqual(body["cpu_control_slots"][2]["metadata"]["num_tokens"], 33)
        self.assertTrue(all(item["matched"] for item in body["comparisons"]))
        self.assertTrue(all(not item["mismatches"] for item in body["comparisons"]))

    def test_field_mismatch_is_reported_with_rank_and_values(self):
        manager = None

        def dispatch(request):
            for rank in request.target_original_ranks:
                if request.command == "status":
                    output = _ack(request, rank, engine_paused=True)
                else:
                    collective = [_mlp_sync_metadata(i) for i in range(4)]
                    if rank == 2:
                        collective[1] = FaultToleranceMLPSyncMetadata(
                            num_tokens=999,
                            num_tokens_for_logprob=14,
                            can_cuda_graph=False,
                            is_extend_in_batch=True,
                            local_can_run_tbo=True,
                            local_forward_mode=2,
                            can_run_breakable_cuda_graph=True,
                        )
                    output = _ack(
                        request,
                        rank,
                        engine_paused=True,
                        mlp_sync_local_metadata=_mlp_sync_metadata(rank),
                        mlp_sync_collective_metadata=collective,
                        mlp_sync_group="tp-cpu-all-gather",
                    )
                manager.handle_command_output(output)

        manager = self._new_manager(dispatch)
        status_code, body = asyncio.run(manager.probe_metadata_parity())

        self.assertEqual(status_code, 503)
        self.assertFalse(body["success"])
        rank_two = next(
            item for item in body["comparisons"] if item["reporting_rank"] == 2
        )
        self.assertFalse(rank_two["matched"])
        self.assertEqual(
            rank_two["mismatches"],
            [
                {
                    "original_rank": 1,
                    "field": "num_tokens",
                    "cpu_control": 22,
                    "collective": 999,
                }
            ],
        )

    def test_unpaused_rank_blocks_probe_before_any_collective_command(self):
        requests = []
        manager = None

        def dispatch(request):
            requests.append(request)
            for rank in request.target_original_ranks:
                manager.handle_command_output(
                    _ack(request, rank, engine_paused=(rank != 1))
                )

        manager = self._new_manager(dispatch)
        status_code, body = asyncio.run(manager.probe_metadata_parity())

        self.assertEqual(status_code, 409)
        self.assertFalse(body["success"])
        self.assertEqual(body["phase"], "pause-precheck")
        self.assertEqual(body["unpaused_ranks"], [1])
        self.assertEqual([request.command for request in requests], ["status"])

    def test_scheduler_returns_the_collective_probe_payload(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ps = SimpleNamespace(dp_rank=3)
        scheduler._engine_paused = True
        collective = [_mlp_sync_metadata(rank) for rank in range(4)]
        scheduler._probe_mlp_sync_metadata_parity = MagicMock(
            return_value=(
                _mlp_sync_metadata(3),
                collective,
                "tp-cpu-all-gather",
            )
        )
        request = FaultToleranceCommandReqInput(
            command_id="a5-command",
            command="probe_mlp_sync_metadata_parity",
            target_original_ranks=[0, 1, 2, 3],
        )

        output = scheduler.handle_fault_tolerance_command(request)

        self.assertTrue(output.success)
        self.assertEqual(output.original_rank, 3)
        self.assertEqual(output.mlp_sync_local_metadata.num_tokens, 44)
        self.assertEqual(len(output.mlp_sync_collective_metadata), 4)
        self.assertEqual(output.mlp_sync_group, "tp-cpu-all-gather")

    def test_new_probe_structs_round_trip_through_msgpack(self):
        request = FaultToleranceMetadataParityProbeReqInput()
        output = FaultToleranceCommandReqOutput(
            command_id="a5-command",
            command="probe_mlp_sync_metadata_parity",
            original_rank=0,
            success=True,
            engine_paused=True,
            mlp_sync_local_metadata=_mlp_sync_metadata(0),
            mlp_sync_collective_metadata=[
                _mlp_sync_metadata(rank) for rank in range(4)
            ],
            mlp_sync_group="tp-cpu-all-gather",
        )

        decoded_request = msgpack_decode(
            msgpack_encode(request), FaultToleranceMetadataParityProbeReqInput
        )
        decoded_output = msgpack_decode(
            msgpack_encode(output), FaultToleranceCommandReqOutput
        )

        self.assertIsInstance(
            decoded_request, FaultToleranceMetadataParityProbeReqInput
        )
        self.assertEqual(decoded_output.mlp_sync_local_metadata.num_tokens, 11)
        self.assertEqual(decoded_output.mlp_sync_collective_metadata[3].num_tokens, 44)


class TestFaultToleranceCoordinatedPause(CustomTestCase):
    def _new_manager(self, dispatch, timeout: float = 0.02):
        return FaultToleranceManager(
            original_world_size=4,
            command_timeout=timeout,
            dispatch_command=dispatch,
        )

    def test_arm_recoverable_error_targets_one_original_rank(self):
        requests = []
        manager = None

        def dispatch(request):
            requests.append(request)
            manager.handle_command_output(_ack(request, 1))

        manager = self._new_manager(dispatch)
        status_code, body = asyncio.run(
            manager.arm_recoverable_error(
                original_rank=1,
                request_id="a2-request",
            )
        )

        self.assertEqual(status_code, 200)
        self.assertTrue(body["success"])
        self.assertEqual(body["acknowledged_ranks"], [1])
        self.assertEqual(requests[0].command, "arm_recoverable_error")
        self.assertEqual(requests[0].target_original_ranks, [1])
        self.assertEqual(requests[0].request_id, "a2-request")
        self.assertEqual(
            manager.armed_injection,
            {"original_rank": 1, "request_id": "a2-request"},
        )

    def test_ambiguous_arm_failure_closes_admission(self):
        manager = self._new_manager(lambda _request: None, timeout=0.001)

        status_code, body = asyncio.run(
            manager.arm_recoverable_error(
                original_rank=1,
                request_id="a2-request",
            )
        )

        self.assertEqual(status_code, 503)
        self.assertFalse(body["success"])
        self.assertTrue(manager.admission_closed)
        self.assertIsNone(manager.armed_injection)
        self.assertEqual(manager.last_transition["state"], "FAILED")
        _, status = asyncio.run(manager.status())
        self.assertEqual(status["service_state"], "FAIL_STOP")

    def test_recoverable_error_closes_admission_and_pauses_every_rank(self):
        async def scenario():
            manager = None
            paused = {rank: False for rank in range(4)}

            def dispatch(request):
                if request.command == "arm_recoverable_error":
                    manager.handle_command_output(_ack(request, 1))
                elif request.command == "pause":
                    for rank in request.target_original_ranks:
                        paused[rank] = True
                        manager.handle_command_output(
                            _ack(request, rank, engine_paused=True)
                        )
                elif request.command == "status":
                    for rank in request.target_original_ranks:
                        manager.handle_command_output(
                            _ack(request, rank, engine_paused=paused[rank])
                        )

            manager = self._new_manager(dispatch)
            await manager.arm_recoverable_error(
                original_rank=1,
                request_id="a2-request",
            )
            manager.handle_recoverable_error(
                FaultToleranceRecoverableErrorOutput(
                    event_id="event-a2",
                    original_rank=1,
                    request_id="a2-request",
                    message="injected recoverable error",
                )
            )
            await manager._pause_task
            return manager, await manager.status()

        manager, (status_code, body) = asyncio.run(scenario())

        self.assertTrue(manager.admission_closed)
        self.assertEqual(status_code, 200)
        self.assertEqual(body["service_state"], "PAUSED")
        self.assertEqual(body["last_fault"]["original_rank"], 1)
        self.assertEqual(body["last_fault"]["request_id"], "a2-request")
        self.assertEqual(body["last_transition"]["command"], "pause")
        self.assertEqual(body["last_transition"]["state"], "SUCCEEDED")
        self.assertEqual(body["last_transition"]["acknowledged_ranks"], [0, 1, 2, 3])

    def test_armed_injection_admits_only_the_exact_target_request(self):
        manager = self._new_manager(lambda _request: None)
        manager.armed_injection = {
            "original_rank": 1,
            "request_id": "a2-request",
        }

        self.assertIsNone(
            manager.get_admission_error(
                request_id="a2-request",
                routed_dp_rank=1,
            )
        )
        self.assertIsNotNone(
            manager.get_admission_error(
                request_id="different-request",
                routed_dp_rank=1,
            )
        )
        self.assertIsNotNone(
            manager.get_admission_error(
                request_id="a2-request",
                routed_dp_rank=0,
            )
        )

    def test_scheduler_injection_is_request_specific_and_one_shot(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ps = SimpleNamespace(dp_rank=1)
        scheduler._engine_paused = False
        scheduler._fault_tolerance_injection_request_id = None
        send_output = MagicMock()
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=send_output)
        )

        arm = FaultToleranceCommandReqInput(
            command_id="arm-command",
            command="arm_recoverable_error",
            target_original_ranks=[1],
            request_id="a2-request",
        )
        arm_output = scheduler.handle_fault_tolerance_command(arm)
        self.assertTrue(arm_output.success)

        self.assertIsNone(
            scheduler._maybe_inject_recoverable_error(
                SimpleNamespace(rid="different-request")
            )
        )
        abort = scheduler._maybe_inject_recoverable_error(
            SimpleNamespace(rid="a2-request")
        )
        self.assertIsInstance(abort, AbortReq)
        self.assertEqual(abort.finished_reason["status_code"], 503)
        event = send_output.call_args.args[0]
        self.assertIsInstance(event, FaultToleranceRecoverableErrorOutput)
        self.assertEqual(event.original_rank, 1)
        self.assertEqual(event.request_id, "a2-request")
        self.assertIsNone(
            scheduler._maybe_inject_recoverable_error(SimpleNamespace(rid="a2-request"))
        )

    def test_scheduler_pause_ack_happens_after_pause_and_abort(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ps = SimpleNamespace(dp_rank=2)
        scheduler._engine_paused = False

        def pause_generation(_request):
            scheduler._engine_paused = True

        scheduler.pause_generation = MagicMock(side_effect=pause_generation)
        scheduler.abort_request = MagicMock()
        request = FaultToleranceCommandReqInput(
            command_id="pause-command",
            command="pause",
            target_original_ranks=[0, 1, 2, 3],
        )

        output = scheduler.handle_fault_tolerance_command(request)

        self.assertTrue(output.success)
        self.assertTrue(output.engine_paused)
        scheduler.pause_generation.assert_called_once()
        scheduler.abort_request.assert_called_once()
        self.assertTrue(scheduler.abort_request.call_args.args[0].abort_all)


class TestFaultToleranceRetry(CustomTestCase):
    def _new_manager(self, dispatch, timeout: float = 0.02):
        manager = FaultToleranceManager(
            original_world_size=4,
            command_timeout=timeout,
            dispatch_command=dispatch,
        )
        manager.admission_closed = True
        manager.last_transition = {
            "command": "pause",
            "state": "SUCCEEDED",
            "acknowledged_ranks": [0, 1, 2, 3],
        }
        return manager

    def test_retry_opens_admission_only_after_every_rank_acknowledges(self):
        async def scenario():
            manager = None
            requests = []

            def dispatch(request):
                requests.append(request)
                if request.command == "retry":
                    self.assertTrue(manager.admission_closed)
                    for rank in request.target_original_ranks:
                        manager.handle_command_output(
                            _ack(request, rank, engine_paused=False)
                        )
                elif request.command == "status":
                    for rank in request.target_original_ranks:
                        manager.handle_command_output(
                            _ack(request, rank, engine_paused=False)
                        )

            manager = self._new_manager(dispatch)
            retry_result = await manager.apply_retry()
            status_result = await manager.status()
            return manager, requests, retry_result, status_result

        manager, requests, (retry_code, retry), (status_code, status) = asyncio.run(
            scenario()
        )

        self.assertEqual(retry_code, 200)
        self.assertTrue(retry["success"])
        self.assertEqual(retry["acknowledged_ranks"], [0, 1, 2, 3])
        self.assertEqual(requests[0].command, "retry")
        self.assertEqual(requests[0].target_original_ranks, [0, 1, 2, 3])
        self.assertFalse(manager.admission_closed)
        self.assertEqual(manager.last_transition["command"], "retry")
        self.assertEqual(manager.last_transition["state"], "SUCCEEDED")
        self.assertEqual(status_code, 200)
        self.assertEqual(status["service_state"], "HEALTHY")

    def test_failed_retry_keeps_admission_closed(self):
        manager = None

        def dispatch(request):
            for rank in request.target_original_ranks:
                manager.handle_command_output(
                    _ack(
                        request,
                        rank,
                        success=(rank != 2),
                        engine_paused=(rank == 2),
                    )
                )

        manager = self._new_manager(dispatch, timeout=1.0)
        status_code, body = asyncio.run(manager.apply_retry())

        self.assertEqual(status_code, 503)
        self.assertFalse(body["success"])
        self.assertEqual(body["failed_ranks"], [2])
        self.assertTrue(manager.admission_closed)
        self.assertEqual(manager.last_transition["state"], "FAILED")

    def test_retry_requires_a_successful_coordinated_pause(self):
        manager = FaultToleranceManager(
            original_world_size=4,
            command_timeout=0.02,
            dispatch_command=MagicMock(),
        )

        status_code, body = asyncio.run(manager.apply_retry())

        self.assertEqual(status_code, 409)
        self.assertFalse(body["success"])
        self.assertIn("not closed", body["last_error"])
        manager._dispatch_command.assert_not_called()

    def test_scheduler_retry_reuses_existing_runtime_objects(self):
        tp_group = object()
        moe_ep_group = object()
        deep_ep_buffer = object()
        expert_metadata = object()
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ps = SimpleNamespace(
            dp_rank=3,
            tp_group=tp_group,
            moe_ep_group=moe_ep_group,
        )
        scheduler.deep_ep_buffer = deep_ep_buffer
        scheduler.expert_metadata = expert_metadata
        scheduler._engine_paused = True

        def continue_generation(request):
            self.assertFalse(request.torch_empty_cache)
            scheduler._engine_paused = False

        scheduler.continue_generation = MagicMock(side_effect=continue_generation)
        request = FaultToleranceCommandReqInput(
            command_id="retry-command",
            command="retry",
            target_original_ranks=[0, 1, 2, 3],
        )

        output = scheduler.handle_fault_tolerance_command(request)

        self.assertTrue(output.success)
        self.assertFalse(output.engine_paused)
        scheduler.continue_generation.assert_called_once()
        self.assertIs(scheduler.ps.tp_group, tp_group)
        self.assertIs(scheduler.ps.moe_ep_group, moe_ep_group)
        self.assertIs(scheduler.deep_ep_buffer, deep_ep_buffer)
        self.assertIs(scheduler.expert_metadata, expert_metadata)


class TestFaultToleranceFailureSemantics(CustomTestCase):
    def test_pause_ack_loss_wrong_command_id_and_failure_are_fail_stop(self):
        async def scenario(failure_mode):
            manager = None

            def dispatch(request):
                if request.command == "arm_recoverable_error":
                    manager.handle_command_output(_ack(request, 1))
                elif request.command == "pause":
                    for rank in request.target_original_ranks:
                        if failure_mode == "missing" and rank == 3:
                            continue
                        if failure_mode == "wrong_command_id" and rank == 3:
                            manager.handle_command_output(
                                FaultToleranceCommandReqOutput(
                                    command_id="wrong-command-id",
                                    command=request.command,
                                    original_rank=rank,
                                    success=True,
                                    engine_paused=True,
                                )
                            )
                            continue
                        manager.handle_command_output(
                            _ack(
                                request,
                                rank,
                                success=not (failure_mode == "failed" and rank == 2),
                                engine_paused=True,
                            )
                        )
                elif request.command == "status":
                    for rank in request.target_original_ranks:
                        manager.handle_command_output(
                            _ack(request, rank, engine_paused=True)
                        )

            manager = FaultToleranceManager(
                original_world_size=4,
                command_timeout=0.001,
                dispatch_command=dispatch,
            )
            await manager.arm_recoverable_error(
                original_rank=1,
                request_id=f"a4-pause-{failure_mode}",
            )
            manager.handle_recoverable_error(
                FaultToleranceRecoverableErrorOutput(
                    event_id=f"a4-event-{failure_mode}",
                    original_rank=1,
                    request_id=f"a4-pause-{failure_mode}",
                    message="A4 pause failure injection",
                )
            )
            await manager._pause_task
            return manager, await manager.status()

        for failure_mode in ("missing", "wrong_command_id", "failed"):
            with self.subTest(failure_mode=failure_mode):
                manager, (status_code, status) = asyncio.run(scenario(failure_mode))

                self.assertTrue(manager.admission_closed)
                self.assertEqual(manager.last_transition["command"], "pause")
                self.assertEqual(manager.last_transition["state"], "FAILED")
                if failure_mode == "failed":
                    self.assertEqual(manager.last_transition["failed_ranks"], [2])
                    self.assertEqual(manager.last_transition["missing_ranks"], [])
                else:
                    self.assertEqual(manager.last_transition["failed_ranks"], [])
                    self.assertEqual(manager.last_transition["missing_ranks"], [3])
                    self.assertIn("Timed out", manager.last_transition["last_error"])
                self.assertEqual(status_code, 200)
                self.assertEqual(status["service_state"], "FAIL_STOP")
                self.assertTrue(status["admission_closed"])

    def test_retry_ack_loss_and_wrong_command_id_never_reopen_admission(self):
        async def scenario(failure_mode):
            manager = None

            def dispatch(request):
                for rank in request.target_original_ranks:
                    if rank != 3:
                        manager.handle_command_output(
                            _ack(request, rank, engine_paused=False)
                        )
                    elif failure_mode == "wrong_command_id":
                        manager.handle_command_output(
                            FaultToleranceCommandReqOutput(
                                command_id="wrong-command-id",
                                command=request.command,
                                original_rank=rank,
                                success=True,
                                engine_paused=False,
                            )
                        )

            manager = FaultToleranceManager(
                original_world_size=4,
                command_timeout=0.001,
                dispatch_command=dispatch,
            )
            manager.admission_closed = True
            manager.last_transition = {
                "command": "pause",
                "state": "SUCCEEDED",
                "acknowledged_ranks": [0, 1, 2, 3],
            }
            return manager, await manager.apply_retry()

        for failure_mode in ("missing", "wrong_command_id"):
            with self.subTest(failure_mode=failure_mode):
                manager, (status_code, body) = asyncio.run(scenario(failure_mode))

                self.assertEqual(status_code, 503)
                self.assertFalse(body["success"])
                self.assertEqual(body["acknowledged_ranks"], [0, 1, 2])
                self.assertEqual(body["missing_ranks"], [3])
                self.assertIn("Timed out", body["last_error"])
                self.assertTrue(manager.admission_closed)
                self.assertEqual(manager.last_transition["command"], "retry")
                self.assertEqual(manager.last_transition["state"], "FAILED")


if __name__ == "__main__":
    unittest.main()
