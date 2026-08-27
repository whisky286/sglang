"""Rebuilt graph-external process groups for Ascend fault tolerance.

MC2 dispatch/combine keeps its graph-captured original-rank communication
window and consumes a mutable ``elastic_info`` tensor.  Every communication
path outside the captured graph uses the survivor-only groups managed here.

The TCPStore is hosted by the DataParallelController, which survives a
Scheduler/rank exit.  It is only a rendezvous store: tensors are exchanged by
fresh Gloo/HCCL process groups, never serialized through the store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch.distributed import PrefixStore, TCPStore

logger = logging.getLogger(__name__)


def _parse_tcp_endpoint(endpoint: str) -> tuple[str, int]:
    if not endpoint.startswith("tcp://"):
        raise ValueError(f"expected a tcp:// endpoint, got {endpoint!r}")
    address = endpoint[len("tcp://") :]
    if address.startswith("["):
        host, port = address.rsplit("]:", 1)
        host = host[1:]
    else:
        host, port = address.rsplit(":", 1)
    return host, int(port)


def create_npu_ft_metadata_store_server(endpoint: str) -> TCPStore:
    """Create the persistent rendezvous store in the controller process."""

    host, port = _parse_tcp_endpoint(endpoint)
    return TCPStore(
        host_name=host,
        port=port,
        is_master=True,
        wait_for_workers=False,
    )


def prewarm_npu_ft_original_mlp_sync_group(
    group: dist.ProcessGroup,
    *,
    original_rank: int,
) -> None:
    """Eagerly connect the original Gloo group while every rank is healthy.

    TorchNPU/PyTorch may defer Gloo's full-mesh connection until the first CPU
    collective.  If that first use races with a rank failure, a survivor can
    block in rendezvous before it has a chance to consume the scale-down
    command.  A startup barrier removes that post-failure lazy-init path.
    """

    dist.barrier(group=group)
    logger.info(
        "[NPU FT] prewarmed original graph-external MLP-sync Gloo group: "
        "original_rank=%d",
        original_rank,
    )


@dataclass
class NpuFTSurvivorProcessGroups:
    """Own the latest survivor-only CPU and NPU process groups.

    Ranks in these groups are compact.  ``active_original_ranks`` and
    ``group_rank`` provide the explicit mapping back to the immutable original
    rank namespace used by MC2 and expert-location metadata.

    Old groups are retained instead of destroyed.  A later failure can poison
    the previous survivor group, and destroying that group may itself require
    participation from the newly failed rank.  At most ``original_world_size``
    generations can be created by scale-down, so this retention is bounded.
    """

    store: TCPStore
    original_rank: int
    original_world_size: int
    timeout_sec: float
    generation: int = 0
    active_original_ranks: tuple[int, ...] = ()
    cpu_group: Any = None
    scheduler_device_group: Any = None
    eplb_device_group: Any = None
    device: torch.device | None = None
    _retired_groups: list[Any] = field(default_factory=list)

    @classmethod
    def connect(
        cls,
        endpoint: str,
        *,
        original_rank: int,
        original_world_size: int,
        timeout_sec: float,
    ) -> "NpuFTSurvivorProcessGroups":
        host, port = _parse_tcp_endpoint(endpoint)
        store = TCPStore(
            host_name=host,
            port=port,
            is_master=False,
            wait_for_workers=False,
        )
        return cls(
            store=store,
            original_rank=original_rank,
            original_world_size=original_world_size,
            timeout_sec=timeout_sec,
        )

    @property
    def is_rebuilt(self) -> bool:
        return (
            self.cpu_group is not None
            and self.scheduler_device_group is not None
            and self.eplb_device_group is not None
        )

    @property
    def compact_rank(self) -> int:
        self._require_rebuilt()
        return self.active_original_ranks.index(self.original_rank)

    @property
    def world_size(self) -> int:
        self._require_rebuilt()
        return len(self.active_original_ranks)

    def _resolve_active_original_ranks(
        self, active_ranks: Sequence[bool]
    ) -> tuple[int, ...]:
        mask = [bool(value) for value in active_ranks]
        if len(mask) != self.original_world_size:
            raise ValueError(
                "active_ranks length must match original_world_size "
                f"({self.original_world_size}), got {len(mask)}"
            )
        active = tuple(rank for rank, enabled in enumerate(mask) if enabled)
        if not active:
            raise ValueError("process-group rebuild requires at least one active rank")
        if self.original_rank not in active:
            raise RuntimeError(
                f"inactive original rank {self.original_rank} entered process-group rebuild"
            )
        return active

    def rebuild(
        self,
        *,
        active_ranks: Sequence[bool],
        device: torch.device | str,
    ) -> None:
        """Build fresh compact-rank Gloo and HCCL groups on all survivors."""

        device = torch.device(device)
        active = self._resolve_active_original_ranks(active_ranks)
        if self.is_rebuilt and active == self.active_original_ranks:
            return

        next_generation = self.generation + 1
        compact_rank = active.index(self.original_rank)
        membership = "".join(
            "1" if rank in active else "0"
            for rank in range(self.original_world_size)
        )
        prefix = f"npu-ft/process-groups/{membership}/{next_generation}"
        timeout = timedelta(seconds=self.timeout_sec)

        # init_custom_process_group directly constructs an independent group.
        # dist.new_group() is intentionally not used: it requires participation
        # by the failed member of the old default group on supported versions.
        from sglang.srt.distributed.parallel_state import (
            get_torch_distributed_pg_options,
        )
        from sglang.srt.utils import init_custom_process_group

        cpu_group = init_custom_process_group(
            backend="gloo",
            store=PrefixStore(f"{prefix}/gloo", self.store),
            timeout=timeout,
            world_size=len(active),
            rank=compact_rank,
            group_name=f"npu_ft_gloo_{membership}_{next_generation}",
        )
        scheduler_device_group = init_custom_process_group(
            backend="hccl",
            store=PrefixStore(f"{prefix}/scheduler-hccl", self.store),
            timeout=timeout,
            world_size=len(active),
            rank=compact_rank,
            group_name=f"npu_ft_scheduler_hccl_{membership}_{next_generation}",
            pg_options=get_torch_distributed_pg_options(
                "moe_npu_ft_scheduler_survivors"
            ),
            device_id=device if device.index is not None else None,
        )
        eplb_device_group = init_custom_process_group(
            backend="hccl",
            store=PrefixStore(f"{prefix}/eplb-hccl", self.store),
            timeout=timeout,
            world_size=len(active),
            rank=compact_rank,
            group_name=f"npu_ft_eplb_hccl_{membership}_{next_generation}",
            pg_options=get_torch_distributed_pg_options(
                "moe_npu_ft_eplb_survivors"
            ),
            device_id=device if device.index is not None else None,
        )

        # Force lazy communicator initialization before publishing the groups.
        # Subsequent callers can therefore treat a successful rebuild as a
        # completed graph-external communication-domain transition.
        dist.barrier(group=cpu_group)
        warmup = torch.zeros(1, dtype=torch.int32, device=device)
        dist.all_reduce(
            warmup, op=dist.ReduceOp.SUM, group=scheduler_device_group
        )
        dist.all_reduce(warmup, op=dist.ReduceOp.SUM, group=eplb_device_group)

        if self.cpu_group is not None:
            self._retired_groups.append(self.cpu_group)
        if self.scheduler_device_group is not None:
            self._retired_groups.append(self.scheduler_device_group)
        if self.eplb_device_group is not None:
            self._retired_groups.append(self.eplb_device_group)
        self.cpu_group = cpu_group
        self.scheduler_device_group = scheduler_device_group
        self.eplb_device_group = eplb_device_group
        self.device = device
        self.active_original_ranks = active
        self.generation = next_generation
        logger.info(
            "[NPU FT] rebuilt graph-external process groups: "
            "generation=%d original_rank=%d compact_rank=%d "
            "active_original_ranks=%s "
            "domains=[gloo,scheduler_hccl,eplb_hccl]",
            self.generation,
            self.original_rank,
            compact_rank,
            list(active),
        )

    def group_rank(self, original_rank: int) -> int:
        self._require_rebuilt()
        try:
            return self.active_original_ranks.index(original_rank)
        except ValueError as exc:
            raise ValueError(
                f"original rank {original_rank} is not in survivor group "
                f"{list(self.active_original_ranks)}"
            ) from exc

    def all_gather_tensor(self, local_tensor: torch.Tensor) -> dict[int, torch.Tensor]:
        """Gather device tensors and return them in original-rank namespace."""

        self._require_rebuilt()
        local_tensor = local_tensor.to(device=self.device).contiguous()
        if self.world_size == 1:
            tensors = [local_tensor.clone()]
        else:
            tensors = [torch.empty_like(local_tensor) for _ in range(self.world_size)]
            dist.all_gather(
                tensors, local_tensor, group=self.scheduler_device_group
            )
        return dict(zip(self.active_original_ranks, tensors, strict=True))

    def all_gather_cpu_tensor(
        self,
        local_tensor: torch.Tensor,
        *,
        timeout_sec: float | None = None,
    ) -> dict[int, torch.Tensor]:
        """Gather CPU tensors over the rebuilt survivor Gloo group.

        The Scheduler passes a short timeout for steady-state MLP-sync.  Gloo
        waits on the host and can therefore return control after a later
        survivor failure.  Do not replace this path with an async HCCL wait:
        TorchNPU WorkHCCL.wait() may only make the current NPU stream wait and
        report success before the collective has completed.
        """

        self._require_rebuilt()
        if timeout_sec is not None and timeout_sec <= 0:
            raise ValueError("all-gather timeout must be positive")
        local_tensor = local_tensor.detach().to(device="cpu").contiguous()
        if self.world_size == 1:
            tensors = [local_tensor.clone()]
        else:
            tensors = [torch.empty_like(local_tensor) for _ in range(self.world_size)]
            if timeout_sec is None:
                dist.all_gather(tensors, local_tensor, group=self.cpu_group)
            else:
                work = dist.all_gather(
                    tensors,
                    local_tensor,
                    group=self.cpu_group,
                    async_op=True,
                )
                try:
                    completed = work.wait(
                        timeout=timedelta(seconds=timeout_sec)
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "NPU MC2 survivor MLP-sync Gloo collective failed; "
                        "returning to the FT control loop"
                    ) from exc
                if completed is False:
                    raise TimeoutError(
                        "NPU MC2 survivor MLP-sync Gloo collective timed out after "
                        f"{timeout_sec:g}s; returning to the FT control loop"
                    )
        return dict(zip(self.active_original_ranks, tensors, strict=True))

    def all_reduce_sum_tensor(self, local_tensor: torch.Tensor) -> torch.Tensor:
        """Sum a device tensor over the rebuilt survivor HCCL group."""

        self._require_rebuilt()
        result = local_tensor.to(device=self.device).contiguous().clone()
        if self.world_size > 1:
            dist.all_reduce(
                result, op=dist.ReduceOp.SUM, group=self.eplb_device_group
            )
        return result

    def broadcast_control(self, control_reqs: list) -> list:
        """Broadcast control objects over the rebuilt survivor Gloo group."""

        self._require_rebuilt()
        payload = [control_reqs if self.compact_rank == 0 else None]
        if self.world_size > 1:
            dist.broadcast_object_list(payload, src=0, group=self.cpu_group)
        return payload[0]

    def _require_rebuilt(self) -> None:
        if not self.is_rebuilt:
            raise RuntimeError(
                "NPU FT graph-external process groups have not been rebuilt"
            )

    def barrier(self, *, timeout_sec: float = 60.0) -> None:
        """Wait until every rebuilt survivor reaches the same FT phase."""

        self._require_rebuilt()

        if self.world_size == 1:
            return

        work = dist.barrier(
            group=self.cpu_group,
            async_op=True,
        )

        try:
            completed = work.wait(
                timeout=timedelta(seconds=timeout_sec)
            )
        except Exception as exc:
            raise RuntimeError(
                "NPU MC2 survivor recovery barrier failed"
            ) from exc

        if completed is False:
            raise TimeoutError(
                "NPU MC2 survivor recovery barrier timed out "
                f"after {timeout_sec:g}s"
            )


def init_npu_ft_metadata_group(
    endpoint: str,
    *,
    original_rank: int,
    original_world_size: int,
    timeout_sec: float,
) -> NpuFTSurvivorProcessGroups:
    """Connect to the persistent store; groups are built after scale-down."""

    from sglang.srt.runtime_context import get_resources

    resources = get_resources()
    existing = resources.buffers.get("npu_ft_survivor_process_groups")
    if existing is not None:
        return existing
    groups = NpuFTSurvivorProcessGroups.connect(
        endpoint,
        original_rank=original_rank,
        original_world_size=original_world_size,
        timeout_sec=timeout_sec,
    )
    resources.buffers["npu_ft_survivor_process_groups"] = groups
    return groups


def get_npu_ft_metadata_group() -> NpuFTSurvivorProcessGroups:
    from sglang.srt.runtime_context import get_resources

    groups = get_resources().buffers.get("npu_ft_survivor_process_groups")
    if groups is None:
        raise RuntimeError("NPU FT survivor process-group manager is not initialized")
    return groups
