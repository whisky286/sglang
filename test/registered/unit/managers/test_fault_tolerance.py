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
    FaultToleranceCommandReqInput,
    FaultToleranceCommandReqOutput,
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
        )
        output = _ack(request, 2)

        self.assertEqual(msgpack_decode(msgpack_encode(request)), request)
        self.assertEqual(msgpack_decode(msgpack_encode(output)), output)

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


if __name__ == "__main__":
    unittest.main()
