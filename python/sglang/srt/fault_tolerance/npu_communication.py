from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch.distributed import PrefixStore, TCPStore

from sglang.srt.eplb.process_group_context import (
    EPLBProcessGroupContext,
    set_eplb_process_group_context,
)

logger = logging.getLogger(__name__)


def _parse_tcp_endpoint(endpoint: str) -> tuple[str, int]:
    address = endpoint.removeprefix("tcp://")
    host, port = address.rsplit(":", 1)
    return host, int(port)


def create_npu_ft_store(endpoint: str) -> TCPStore:
    host, port = _parse_tcp_endpoint(endpoint)
    return TCPStore(host_name=host, port=port, is_master=True, wait_for_workers=False)


@dataclass
class NpuFTCommunication:
    store: TCPStore
    original_rank: int
    timeout_sec: float
    mlp_sync_group: Any
    active_original_ranks: tuple[int, ...]
    generation: int = 0

    def rebuild_survivor_groups(
        self,
        active_mask: Sequence[bool],
        *,
        membership_epoch: int | None = None,
    ) -> None:
        active_ranks = tuple(rank for rank, active in enumerate(active_mask) if active)
        if self.original_rank not in active_ranks:
            raise ValueError("local rank must belong to the candidate membership")
        compact_rank = active_ranks.index(self.original_rank)
        generation = (
            self.generation + 1 if membership_epoch is None else membership_epoch
        )
        if type(generation) is not int or generation <= self.generation:
            raise ValueError("membership_epoch must advance the local generation")
        membership = "".join("1" if active else "0" for active in active_mask)
        prefix = f"npu-ft/{membership}/{generation}"
        timeout = timedelta(seconds=self.timeout_sec)

        from sglang.srt.distributed.parallel_state import get_moe_ep_group
        from sglang.srt.utils import init_custom_process_group

        mlp_sync_group = init_custom_process_group(
            backend="gloo",
            store=PrefixStore(f"{prefix}/mlp-sync", self.store),
            timeout=timeout,
            world_size=len(active_ranks),
            rank=compact_rank,
            group_name=f"npu_ft_mlp_sync_{membership}_{generation}",
        )
        original_eplb_device_group = get_moe_ep_group().device_group
        dist.barrier(group=mlp_sync_group)
        self.mlp_sync_group = mlp_sync_group
        self.active_original_ranks = active_ranks
        self.generation = generation
        set_eplb_process_group_context(
            EPLBProcessGroupContext(
                control_group=mlp_sync_group,
                device_group=original_eplb_device_group,
                active_original_ranks=active_ranks,
                control_group_uses_cpu=True,
            )
        )
        logger.info(
            "Rebuilt NPU FT survivor groups: original_rank=%s "
            "active_original_ranks=%s",
            self.original_rank,
            active_ranks,
        )


_communication: NpuFTCommunication | None = None


def all_gather_into_tensor_with_timeout(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    *,
    group,
    timeout_sec: float,
) -> None:
    try:
        work = dist.all_gather_into_tensor(
            output_tensor,
            input_tensor,
            group=group,
            async_op=True,
        )
        completed = work.wait(timeout=timedelta(seconds=timeout_sec))
    except Exception as exc:
        raise RuntimeError(
            "NPU MC2 MLP-sync collective failed; entering the FT control loop"
        ) from exc
    if completed is False:
        raise RuntimeError(
            "NPU MC2 MLP-sync collective timed out after "
            f"{timeout_sec:g}s; entering the FT control loop"
        )


def init_npu_ft_communication(
    endpoint: str,
    *,
    original_rank: int,
    original_world_size: int,
    timeout_sec: float,
    mlp_sync_group,
) -> NpuFTCommunication:
    global _communication
    host, port = _parse_tcp_endpoint(endpoint)
    _communication = NpuFTCommunication(
        store=TCPStore(
            host_name=host,
            port=port,
            is_master=False,
            wait_for_workers=False,
        ),
        original_rank=original_rank,
        timeout_sec=timeout_sec,
        mlp_sync_group=mlp_sync_group,
        active_original_ranks=tuple(range(original_world_size)),
    )
    return _communication


def get_npu_ft_communication() -> NpuFTCommunication | None:
    return _communication
