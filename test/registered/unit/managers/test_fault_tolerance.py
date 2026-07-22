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
    FaultToleranceCommandReqInput,
    FaultToleranceCommandReqOutput,
    FaultToleranceRecoverableErrorOutput,
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
) -> FaultToleranceCommandReqOutput:
    return FaultToleranceCommandReqOutput(
        command_id=request.command_id,
        command=request.command,
        original_rank=original_rank,
        success=success,
        engine_paused=engine_paused,
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
        self.assertEqual(
            body["last_transition"]["acknowledged_ranks"], [0, 1, 2, 3]
        )

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
            scheduler._maybe_inject_recoverable_error(
                SimpleNamespace(rid="a2-request")
            )
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

if __name__ == "__main__":
    unittest.main()
