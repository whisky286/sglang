from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.fault_tolerance.controller import FaultToleranceState, ft_failure
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    ActiveRanksUpdateReqOutput,
    FaultToleranceCommandReqInput,
    FaultToleranceCommandReqOutput,
    FaultToleranceDPCRecoverReqInput,
    FaultToleranceDPCShutdownReqInput,
    FaultToleranceRankFaultOutput,
    ProcessActiveRanksOutput,
    WatchdogHeartbeatOutput,
)
from sglang.srt.utils import kill_process_tree
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)

WATCHDOG_LEASE_SWEEP_INTERVAL_SEC = 1
WATCHDOG_LEASE_TIMEOUT_SEC = 5


@dataclasses.dataclass
class PendingFTCommand:
    target_ranks: set[int]
    future: asyncio.Future
    acked: set[int] = dataclasses.field(default_factory=set)
    failed: Dict[int, str] = dataclasses.field(default_factory=dict)

    def finish_if_ready(self) -> None:
        if (
            not self.future.done()
            and self.acked.union(self.failed) >= self.target_ranks
        ):
            self.future.set_result(None)


class FaultToleranceManager:
    def __init__(self, *, server_args, send_to_scheduler, send_to_dpc):
        self.server_args = server_args
        self.send_to_scheduler = send_to_scheduler
        self.send_to_dpc = send_to_dpc
        self.state = FaultToleranceState(
            dp_size=server_args.dp_size,
            strategy=server_args.fault_tolerance_on_error_strategy,
            global_rank_count=server_args.tp_size,
        )
        self.event_loop = None
        self.asyncio_tasks = None
        self._pending_commands: Dict[str, PendingFTCommand] = {}
        self._pending_active_rank_updates: Dict[str, asyncio.Future] = {}
        self._shutdown_waiter: Optional[Tuple[set[int], asyncio.Future]] = None
        self._watchdog_leases: Dict[int, Tuple[float, Tuple[int, ...]]] = {}
        self._watchdog_lease_task: Optional[asyncio.Task] = None
        self._route_dp_mask = [True] * server_args.dp_size
        self._process_event_floor_epoch = 0
        self._recovering_epoch = None
        self._recovery_exited_ranks: set[int] = set()

    def bind_event_loop(self, loop) -> None:
        if self.event_loop is loop:
            return
        if self.event_loop is not None:
            raise RuntimeError(
                "fault tolerance manager is already bound to an event loop"
            )
        self.event_loop = loop
        self.asyncio_tasks = set()

    def status(self) -> tuple[int, dict]:
        return 200, self.state.status_response()

    def _parse_apply_args(self, obj: Dict[str, Any]) -> Tuple[str, List[int], int]:
        instruction = obj["instruction"]
        if instruction not in ("retry", "scale_down", "recover"):
            raise ValueError(f"unsupported instruction: {instruction}")
        params = obj["params"]
        timeout = params.get("timeout", self.server_args.fault_tolerance_timeout)
        ranks = params.get("ranks", [])
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be positive")
        if not isinstance(ranks, list) or any(
            type(rank) is not int or rank < 0 or rank >= self.state.dp_size
            for rank in ranks
        ):
            raise ValueError("rank out of range")
        if len(set(ranks)) != len(ranks):
            raise ValueError("ranks must be unique")
        return instruction, ranks, timeout

    async def apply(self, obj: Dict[str, Any]) -> tuple[int, dict]:
        try:
            instruction, ranks, timeout = self._parse_apply_args(obj)
        except Exception as exc:
            return 400, ft_failure(f"invalid params: {exc}")
        if self.state.ft_operation_in_progress:
            return 409, ft_failure("ft_operation_in_progress")

        try:
            if instruction == "retry":
                return await self._apply_retry(timeout)
            if instruction == "recover":
                return await self._apply_recover(ranks, timeout)
            return await self._apply_scale_down(ranks, timeout)
        except Exception as exc:
            self._failstop(f"fault tolerance apply {instruction} failed: {exc}")

    async def _apply_retry(self, timeout: int) -> tuple[int, dict]:
        st = self.state
        if not st.unhealthy_dp_ranks:
            return 400, ft_failure("retry_requires_unhealthy_rank")
        process_alive_dp_mask = st.process_alive_dp_mask()
        if any(
            expected and not process_alive_dp_mask[rank]
            for rank, expected in enumerate(st.expected_dp_mask)
        ):
            return 400, ft_failure("retry_requires_all_expected_processes_alive")

        st.ft_operation_in_progress = True
        await self._send_command_collect(
            command="retry", target_ranks=st.expected_dp_ranks(), timeout_sec=timeout
        )
        await self._publish_route_dp_mask(st.expected_dp_mask, timeout)
        return 200, st.finish_retry()

    async def _apply_scale_down(
        self, ranks: List[int], timeout: int
    ) -> tuple[int, dict]:
        st = self.state
        requested = set(ranks)
        if not st.has_incident():
            return 400, ft_failure("scale_down_requires_incident")
        if not requested:
            return 400, ft_failure("scale_down_requires_ranks")
        if any(not st.expected_dp_mask[rank] for rank in requested):
            return 400, ft_failure("scale_down_requires_expected_ranks")

        candidate_dp_mask = [
            expected and rank not in requested
            for rank, expected in enumerate(st.expected_dp_mask)
        ]
        if not any(candidate_dp_mask):
            return 400, ft_failure("cannot_scale_down_all_expected_ranks")

        st.ft_operation_in_progress = True
        st.membership_epoch += 1
        await self._shutdown_dp_processes(requested, timeout)
        await self._send_command_collect(
            command="scale_down",
            target_ranks=[
                rank for rank, active in enumerate(candidate_dp_mask) if active
            ],
            timeout_sec=timeout,
            active_global_rank_mask=st.expand_dp_mask_to_global_rank_mask(
                candidate_dp_mask
            ),
            membership_epoch=st.membership_epoch,
        )
        await self._publish_route_dp_mask(candidate_dp_mask, timeout)
        return 200, st.finish_scale_down(requested)

    def validate_routed_rank(self, rank: int) -> None:
        if not 0 <= rank < len(self._route_dp_mask) or not self._route_dp_mask[rank]:
            raise ValueError(f"routed_dp_rank={rank} is not active")

    async def _apply_recover(self, ranks: List[int], timeout: int) -> tuple[int, dict]:
        st = self.state
        if not getattr(self.server_args, "enable_npu_ft_recovery", False):
            return 400, ft_failure("recover_requires_enable_npu_ft_recovery")
        removed = {rank for rank, active in enumerate(st.expected_dp_mask) if not active}
        if not ranks or set(ranks) != removed:
            return 400, ft_failure("recover_requires_all_removed_original_ranks")
        if st.has_incident():
            return 400, ft_failure("recover_requires_healthy_survivors")
        if not any(0 in owned for _, owned in self._watchdog_leases.values()):
            return 409, ft_failure("recover_requires_live_local_watchdog")
        st.ft_operation_in_progress = True
        st.membership_epoch += 1
        self._recovering_epoch = st.membership_epoch
        self._recovery_exited_ranks.clear()
        previous = st.expand_dp_mask_to_global_rank_mask(st.expected_dp_mask)
        target = [True] * st.dp_size
        survivors = st.expected_dp_ranks()
        await self._send_command_collect(
            command="recover_pause",
            target_ranks=survivors,
            timeout_sec=timeout,
            membership_epoch=st.membership_epoch,
        )
        # Start PG construction on survivors before the DPC starts replacements.
        prepare = asyncio.create_task(
            self._send_command_collect(
                command="recover_prepare",
                target_ranks=survivors,
                timeout_sec=timeout,
                active_global_rank_mask=[True] * st.global_rank_count,
                previous_active_mask=previous,
                membership_epoch=st.membership_epoch,
            )
        )
        try:
            await self.send_to_dpc(
                [0],
                FaultToleranceDPCRecoverReqInput(
                    target_dp_ranks=sorted(removed),
                    membership_epoch=st.membership_epoch,
                    previous_active_mask=previous,
                ),
            )
            await prepare
        finally:
            if not prepare.done():
                prepare.cancel()
                await asyncio.gather(prepare, return_exceptions=True)
        await self._send_command_collect(
            command="recover_validate",
            target_ranks=list(range(st.dp_size)),
            timeout_sec=timeout,
            active_global_rank_mask=[True] * st.global_rank_count,
            membership_epoch=st.membership_epoch,
        )
        if self._recovery_exited_ranks:
            raise RuntimeError("scheduler exited during NPU recovery validation")
        await self._publish_route_dp_mask(target, timeout)
        await self._send_command_collect(
            command="recover_commit",
            target_ranks=list(range(st.dp_size)),
            timeout_sec=timeout,
            membership_epoch=st.membership_epoch,
        )
        if self._recovery_exited_ranks:
            raise RuntimeError("scheduler exited during NPU recovery commit")
        self._process_event_floor_epoch = st.membership_epoch
        self._recovering_epoch = None
        return 200, st.finish_recover(ranks)

    def should_reject_admission(self) -> bool:
        return self.state.should_reject_admission(self._route_dp_mask)

    def _route_dp_update(
        self, route_dp_mask: List[bool]
    ) -> Optional[ActiveRanksOutput]:
        if route_dp_mask == self._route_dp_mask:
            return None
        self._route_dp_mask = list(route_dp_mask)
        return ActiveRanksOutput(status=route_dp_mask)

    def _auto_recover_ready_dps(self) -> List[int]:
        st = self.state
        if st.ft_operation_in_progress or (
            getattr(self.server_args, "device", None) == "npu"
            and getattr(self.server_args, "elastic_ep_backend", None) == "mc2"
        ):
            return []

        process_alive_dp_mask = st.process_alive_dp_mask()
        recovered_dp_ranks = []
        for dp_rank in range(st.dp_size):
            if st.expected_dp_mask[dp_rank]:
                continue
            if (
                process_alive_dp_mask[dp_rank]
                and st.native_active_dp_mask[dp_rank]
                and not any(
                    member in st.pending_recovery_global_ranks
                    for member in st.global_ranks_for_dp(dp_rank)
                )
            ):
                st.expected_dp_mask[dp_rank] = True
                recovered_dp_ranks.append(dp_rank)
        return recovered_dp_ranks

    def _route_after_observation(
        self, recovered_dp_ranks: List[int]
    ) -> Optional[ActiveRanksOutput]:
        st = self.state
        if st.strategy == "continue":
            process_alive_dp_mask = st.process_alive_dp_mask()
            return self._route_dp_update(
                [
                    expected
                    and process_alive_dp_mask[dp_rank]
                    and st.native_active_dp_mask[dp_rank]
                    for dp_rank, expected in enumerate(st.expected_dp_mask)
                ]
            )

        if not recovered_dp_ranks:
            return None
        route_dp_mask = list(self._route_dp_mask)
        for dp_rank in recovered_dp_ranks:
            route_dp_mask[dp_rank] = True
        return self._route_dp_update(route_dp_mask)

    def observe_active_ranks(
        self, ranks: ActiveRanksOutput
    ) -> Optional[ActiveRanksOutput]:
        # Scheduler-originated ActiveRanksOutput.status is a DP-rank mask.
        self.state.observe_native_active_dp_mask(list(ranks.status))
        return self._route_after_observation(self._auto_recover_ready_dps())

    def observe_process_active_ranks(
        self, ranks: ProcessActiveRanksOutput
    ) -> Optional[ActiveRanksOutput]:
        epoch = ranks.membership_epoch
        if epoch is not None and epoch < self._process_event_floor_epoch:
            return None
        if not ranks.active and epoch is not None and epoch == self._recovering_epoch:
            self._recovery_exited_ranks.update(ranks.ranks)
        self.state.observe_process_active_ranks(ranks.ranks, active=ranks.active)
        self._finish_shutdown_if_ready()
        return self._route_after_observation(self._auto_recover_ready_dps())

    def observe_watchdog_heartbeat(self, heartbeat: WatchdogHeartbeatOutput) -> None:
        existing = self._watchdog_leases.get(heartbeat.node_rank)
        ranks = tuple(sorted(set(heartbeat.ranks))) if existing is None else existing[1]
        self._watchdog_leases[heartbeat.node_rank] = (time.monotonic(), ranks)
        if self._watchdog_lease_task is None or self._watchdog_lease_task.done():
            self._watchdog_lease_task = self._create_task(
                self._watchdog_lease_sweep_loop()
            )

    async def _watchdog_lease_sweep_loop(self) -> None:
        try:
            while self._watchdog_leases:
                await asyncio.sleep(WATCHDOG_LEASE_SWEEP_INTERVAL_SEC)
                await self._sweep_expired_watchdog_leases()
        finally:
            self._watchdog_lease_task = None

    async def _sweep_expired_watchdog_leases(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        expired = [
            node
            for node, (last_seen, _) in self._watchdog_leases.items()
            if now - last_seen >= WATCHDOG_LEASE_TIMEOUT_SEC
        ]
        inactive = set()
        for node in expired:
            _, ranks = self._watchdog_leases.pop(node)
            inactive.update(ranks)
        if not inactive:
            return
        logger.warning(
            "FT watchdog lease expired: nodes=%s global_ranks=%s",
            sorted(expired),
            sorted(inactive),
        )
        update = self.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=sorted(inactive), active=False)
        )
        if update is not None:
            await self.send_to_scheduler(update)

    def handle_command_output(self, output: FaultToleranceCommandReqOutput) -> None:
        pending = self._pending_commands.get(output.request_id)
        if pending is None:
            logger.warning("Unknown fault tolerance command ack: rank=%s", output.rank)
            return
        if (
            output.rank not in pending.target_ranks
            or output.rank in pending.acked.union(pending.failed)
        ):
            return
        if output.success:
            pending.acked.add(output.rank)
        else:
            pending.failed[output.rank] = output.message
        pending.finish_if_ready()

    def handle_active_ranks_update_output(
        self, output: ActiveRanksUpdateReqOutput
    ) -> None:
        future = self._pending_active_rank_updates.get(output.request_id)
        if future is None or future.done():
            return
        if output.success:
            future.set_result(None)
        else:
            future.set_exception(RuntimeError(output.message))

    def handle_rank_fault(self, event: FaultToleranceRankFaultOutput) -> None:
        self.state.observe_rank_fault(event.rank)
        logger.warning(
            "FT observed scheduler exception on rank %s: %s", event.rank, event.message
        )

    def _create_task(self, coro):
        if self.event_loop is None:
            coro.close()
            raise RuntimeError("fault tolerance manager has no bound event loop")
        task = self.event_loop.create_task(self._fatal_task_wrapper(coro))
        self.asyncio_tasks.add(task)
        task.add_done_callback(self.asyncio_tasks.discard)
        return task

    async def _fatal_task_wrapper(self, coro):
        try:
            await coro
        except Exception:
            self._failstop(
                f"FaultToleranceManager hit an exception: {get_exception_traceback()}"
            )

    async def _publish_route_dp_mask(
        self, route_dp_mask: List[bool], timeout_sec: int
    ) -> None:
        request_id = uuid.uuid4().hex
        future = self.event_loop.create_future()
        self._pending_active_rank_updates[request_id] = future
        try:
            await self.send_to_scheduler(
                ActiveRanksOutput(status=list(route_dp_mask), request_id=request_id)
            )
            await asyncio.wait_for(future, timeout=timeout_sec)
            self._route_dp_mask = list(route_dp_mask)
        finally:
            self._pending_active_rank_updates.pop(request_id, None)

    async def _send_command_collect(
        self,
        *,
        command: str,
        target_ranks: List[int],
        timeout_sec: int,
        active_global_rank_mask: Optional[List[bool]] = None,
        membership_epoch: Optional[int] = None,
        previous_active_mask: Optional[List[bool]] = None,
    ) -> set[int]:
        targets = set(target_ranks)
        if not targets:
            return set()
        request_id = uuid.uuid4().hex
        pending = PendingFTCommand(
            target_ranks=targets, future=self.event_loop.create_future()
        )
        self._pending_commands[request_id] = pending
        req = FaultToleranceCommandReqInput(
            request_id=request_id,
            command=command,
            target_ranks=sorted(targets),
            # This command is consumed by per-scheduler Elastic EP control, so
            # its active_mask remains a physical scheduler/global-rank mask.
            active_mask=active_global_rank_mask,
            membership_epoch=membership_epoch,
            previous_active_mask=previous_active_mask,
        )
        await self.send_to_scheduler(req)
        try:
            await asyncio.wait_for(pending.future, timeout=timeout_sec)
        finally:
            self._pending_commands.pop(request_id, None)
        if pending.failed:
            raise RuntimeError(
                f"fault tolerance command {command} failed: {pending.failed}"
            )
        return pending.acked

    def _finish_shutdown_if_ready(self) -> None:
        if self._shutdown_waiter is None:
            return
        targets, future = self._shutdown_waiter
        if not future.done() and not any(
            self.state.process_alive_global_rank_mask[rank] for rank in targets
        ):
            future.set_result(None)

    async def _shutdown_dp_processes(
        self, target_dp_ranks: set[int], timeout_sec: int
    ) -> None:
        targets = set(self.state.global_ranks_for_dps(target_dp_ranks))
        live_targets = {
            rank for rank in targets if self.state.process_alive_global_rank_mask[rank]
        }
        if not live_targets:
            return
        nodes = [
            node
            for node, (_, owned) in self._watchdog_leases.items()
            if live_targets.intersection(owned)
        ]
        future = self.event_loop.create_future()
        self._shutdown_waiter = (targets, future)
        try:
            await self.send_to_dpc(
                nodes,
                FaultToleranceDPCShutdownReqInput(
                    target_dp_ranks=sorted(target_dp_ranks)
                ),
            )
            await asyncio.wait_for(future, timeout=timeout_sec)
        finally:
            self._shutdown_waiter = None

    @staticmethod
    def _failstop(message: str) -> None:
        logger.error(message)
        kill_process_tree(os.getpid(), include_parent=True)
        raise RuntimeError(message)
