import asyncio
import unittest
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sglang.srt.fault_tolerance.manager import FaultToleranceManager
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    FaultToleranceCommandReqOutput,
    FaultToleranceRankFaultOutput,
    ProcessActiveRanksOutput,
    WatchdogHeartbeatOutput,
)


def make_manager(*, dp_size=2, ranks_per_dp=1, strategy="pause"):
    manager = FaultToleranceManager(
        server_args=SimpleNamespace(
            dp_size=dp_size,
            tp_size=dp_size * ranks_per_dp,
            fault_tolerance_on_error_strategy=strategy,
            fault_tolerance_timeout=1,
        ),
        send_to_scheduler=AsyncMock(),
        send_to_dpc=AsyncMock(),
    )
    manager.bind_event_loop(asyncio.get_running_loop())
    return manager


class TestFaultToleranceManager(unittest.IsolatedAsyncioTestCase):
    async def _stop_watchdog(self, manager):
        await asyncio.sleep(0)
        task = manager._watchdog_lease_task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def test_retry_is_maskless_and_uses_expected_topology(self):
        manager = make_manager(dp_size=4)
        manager.state.expected_dp_mask = [True, True, False, True]
        manager.state.process_alive_global_rank_mask[2] = False
        manager.state.unhealthy_dp_ranks.add(0)
        manager._send_command_collect = AsyncMock(return_value={0, 1, 3})
        manager._publish_route_dp_mask = AsyncMock()

        status, response = await manager.apply({"instruction": "retry", "params": {}})

        self.assertEqual(status, 200)
        self.assertEqual(response["ranks"][2]["state"], "dead")
        manager._send_command_collect.assert_awaited_once_with(
            command="retry",
            target_ranks=[0, 1, 3],
            timeout_sec=1,
        )
        manager._publish_route_dp_mask.assert_awaited_once_with(
            [True, True, False, True], 1
        )

    async def test_retry_rejects_process_loss(self):
        manager = make_manager()
        manager.state.unhealthy_dp_ranks.add(0)
        manager.state.observe_process_active_ranks([1], active=False)

        status, response = await manager.apply({"instruction": "retry", "params": {}})

        self.assertEqual(status, 400)
        self.assertEqual(
            response["message"], "retry_requires_all_expected_processes_alive"
        )

    async def test_scale_down_has_one_scheduler_phase_after_shutdown(self):
        manager = make_manager(dp_size=4)
        manager.state.unhealthy_dp_ranks.add(0)
        order = []

        async def shutdown(ranks, timeout):
            order.append("shutdown")
            manager.state.observe_process_active_ranks(ranks, active=False)

        async def command(**kwargs):
            order.append(kwargs["command"])
            return set(kwargs["target_ranks"])

        async def publish(mask, timeout):
            order.append("route")

        manager._shutdown_dp_processes = shutdown
        manager._send_command_collect = command
        manager._publish_route_dp_mask = publish

        status, response = await manager.apply(
            {"instruction": "scale_down", "params": {"ranks": [2]}}
        )

        self.assertEqual(status, 200)
        self.assertEqual(order, ["shutdown", "scale_down", "route"])
        self.assertEqual(manager.state.expected_dp_mask, [True, True, False, True])
        self.assertEqual(response["ranks"][2]["state"], "dead")

    async def test_scale_down_sends_sparse_global_mask(self):
        manager = make_manager(dp_size=4, ranks_per_dp=2)
        manager.state.unhealthy_dp_ranks.add(0)
        manager._shutdown_dp_processes = AsyncMock()
        manager._send_command_collect = AsyncMock(return_value={0, 1, 3})
        manager._publish_route_dp_mask = AsyncMock()

        await manager.apply({"instruction": "scale_down", "params": {"ranks": [2]}})

        manager._send_command_collect.assert_awaited_once_with(
            command="scale_down",
            target_ranks=[0, 1, 3],
            timeout_sec=1,
            membership_epoch=1,
            active_global_rank_mask=[
                True,
                True,
                True,
                True,
                False,
                False,
                True,
                True,
            ],
        )

    async def test_successive_membership_changes_have_distinct_controller_epochs(self):
        manager = make_manager(dp_size=4)
        manager._shutdown_dp_processes = AsyncMock()
        manager._send_command_collect = AsyncMock()
        manager._publish_route_dp_mask = AsyncMock()

        for rank in (2, 3):
            manager.state.unhealthy_dp_ranks.add(0)
            status, _ = await manager.apply(
                {"instruction": "scale_down", "params": {"ranks": [rank]}}
            )
            self.assertEqual(status, 200)

        self.assertEqual(
            [
                call.kwargs["membership_epoch"]
                for call in manager._send_command_collect.await_args_list
            ],
            [1, 2],
        )
        self.assertEqual(manager.state.membership_epoch, 2)

    async def test_pause_native_ready_automatically_reopens_route(self):
        manager = make_manager()
        manager.state.finish_scale_down([1])
        manager._route_dp_mask = [True, False]
        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[1], active=False)
        )
        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[1], active=True)
        )

        self.assertEqual(manager.status()[1]["ranks"][1]["state"], "dead")
        self.assertEqual(manager.state.pending_recovery_global_ranks, {1})

        update = manager.observe_active_ranks(ActiveRanksOutput(status=[True, True]))

        self.assertEqual(update.status, [True, True])
        self.assertEqual(manager.state.expected_dp_mask, [True, True])
        self.assertEqual(manager.state.pending_recovery_global_ranks, set())
        self.assertEqual(manager.status()[1]["ranks"][1]["state"], "healthy")

    async def test_recover_instruction_is_not_an_ft_api(self):
        manager = make_manager()

        status, response = await manager.apply(
            {"instruction": "recover", "params": {"ranks": [1]}}
        )

        self.assertEqual(status, 400)
        self.assertIn("unsupported instruction: recover", response["message"])

    async def test_native_ready_before_process_up_still_auto_recovers(self):
        manager = make_manager()
        manager.state.finish_scale_down([1])
        manager._route_dp_mask = [True, False]
        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[1], active=False)
        )

        self.assertIsNone(
            manager.observe_active_ranks(ActiveRanksOutput(status=[True, True]))
        )
        self.assertEqual(manager.state.expected_dp_mask, [True, False])
        self.assertEqual(manager.status()[1]["ranks"][1]["state"], "dead")

        update = manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[1], active=True)
        )

        self.assertEqual(update.status, [True, True])
        self.assertEqual(manager.state.expected_dp_mask, [True, True])
        self.assertEqual(manager.status()[1]["ranks"][1]["state"], "healthy")

    async def test_process_up_waits_for_native_ready_before_reopening_route(self):
        manager = make_manager()
        manager.state.expected_dp_mask[1] = False
        manager._route_dp_mask = [True, False]
        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[1], active=False)
        )
        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[1], active=True)
        )
        self.assertEqual(manager.status()[1]["ranks"][1]["state"], "dead")

        update = manager.observe_active_ranks(ActiveRanksOutput(status=[True, True]))
        self.assertEqual(manager.status()[1]["ranks"][1]["state"], "healthy")
        self.assertEqual(update.status, [True, True])

    async def test_continue_route_intersects_native_process_and_expected(self):
        manager = make_manager(dp_size=2, ranks_per_dp=2, strategy="continue")
        manager.state.expected_dp_mask[1] = False
        # DP1 (global ranks 2,3) fully down and not yet native-recovered.
        manager.state.observe_process_active_ranks([2, 3], active=False)

        update = manager.observe_active_ranks(ActiveRanksOutput(status=[True, False]))

        # route = expected & process_alive & native; DP1 cannot auto-recover yet.
        self.assertEqual(manager.state.expected_dp_mask, [True, False])
        self.assertEqual(update.status, [True, False])

    async def test_continue_native_recovery_auto_recovers_dead_dp(self):
        manager = make_manager(dp_size=2, ranks_per_dp=2, strategy="continue")
        # Scale-down DP1 then kill it; it stays dead until it rejoins and its
        # data plane recovers, at which point "continue" re-admits it automatically.
        manager.state.finish_scale_down([1])
        manager.state.observe_process_active_ranks([2, 3], active=False)
        manager.state.observe_process_active_ranks([2, 3], active=True)
        self.assertEqual(manager.state.expected_dp_mask, [True, False])
        self.assertEqual(manager.state.pending_recovery_global_ranks, {2, 3})

        update = manager.observe_active_ranks(ActiveRanksOutput(status=[True, True]))

        # The DP-level native callback cleared both global pending ranks and
        # auto-recovered DP1 without an explicit recover.
        self.assertEqual(manager.state.pending_recovery_global_ranks, set())
        self.assertEqual(manager.state.expected_dp_mask, [True, True])
        # The route was already all-true, so no new publish is emitted; recovery
        # shows up purely as the DP returning to HEALTHY.
        self.assertIsNone(update)
        self.assertEqual(manager.status()[1]["ranks"][1]["state"], "healthy")

    async def test_watchdog_lease_expiry_marks_global_processes_down(self):
        manager = make_manager(dp_size=2, ranks_per_dp=2, strategy="continue")
        manager.observe_watchdog_heartbeat(
            WatchdogHeartbeatOutput(node_rank=1, ranks=[2, 3])
        )
        last_seen, ranks = manager._watchdog_leases[1]

        await manager._sweep_expired_watchdog_leases(now=last_seen + 6)

        self.assertEqual(ranks, (2, 3))
        self.assertEqual(
            manager.state.process_alive_global_rank_mask,
            [True, True, False, False],
        )
        manager.send_to_scheduler.assert_awaited_once()
        await self._stop_watchdog(manager)

    async def test_shutdown_completion_comes_from_process_down_events(self):
        manager = make_manager(dp_size=2, ranks_per_dp=2)
        manager._watchdog_leases = {
            0: (0.0, (0, 1)),
            1: (0.0, (2, 3)),
        }

        task = asyncio.create_task(manager._shutdown_dp_processes({1}, 1))
        await asyncio.sleep(0)
        manager.send_to_dpc.assert_awaited_once()
        nodes, request = manager.send_to_dpc.await_args.args
        self.assertEqual(nodes, [1])
        self.assertEqual(request.target_dp_ranks, [1])

        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[2, 3], active=False)
        )
        await task
        self.assertIsNone(manager._shutdown_waiter)

    async def test_command_waits_for_target_dp_acks(self):
        manager = make_manager()
        task = asyncio.create_task(
            manager._send_command_collect(
                command="retry", target_ranks=[0, 1], timeout_sec=1
            )
        )
        await asyncio.sleep(0)
        request = manager.send_to_scheduler.await_args.args[0]
        for rank in (0, 1):
            manager.handle_command_output(
                FaultToleranceCommandReqOutput(
                    request_id=request.request_id,
                    rank=rank,
                    success=True,
                )
            )
        self.assertEqual(await task, {0, 1})

    async def test_rank_fault_only_updates_unhealthy_state(self):
        manager = make_manager()
        manager.handle_rank_fault(FaultToleranceRankFaultOutput(rank=1, message="boom"))

        self.assertEqual(manager.state.unhealthy_dp_ranks, {1})
        self.assertFalse(hasattr(manager, "_paused_failstop_handle"))


if __name__ == "__main__":
    unittest.main()
