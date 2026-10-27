from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterable, List, Tuple


class RankState(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEAD = "dead"


def ft_failure(message: str) -> Dict[str, Any]:
    return {"success": False, "message": message}


def _dp_attention_gate(server_args) -> bool:
    if not getattr(server_args, "enable_dp_attention", False):
        return True
    attention_block_count = getattr(server_args, "dp_size", 1) * getattr(
        server_args, "attn_cp_size", 1
    )
    return getattr(server_args, "tp_size", 1) % attention_block_count == 0


_FT_SUPPORT_GATES = [
    (lambda a: getattr(a, "dp_size", 1) > 1, "ft_requires_dp_gt1"),
    (lambda a: getattr(a, "enable_dp_attention", False), "ft_requires_dp_attention"),
    (lambda a: getattr(a, "enable_eplb", False), "ft_requires_eplb"),
    (
        lambda a: getattr(a, "max_ep_size", None) in (None, getattr(a, "tp_size", 1)),
        "ft_unsupported_with_runtime_ep_scale",
    ),
    (
        lambda a: getattr(a, "ep_join_mode", None) != "scale",
        "ft_unsupported_with_runtime_ep_scale",
    ),
    (lambda a: getattr(a, "pp_size", 1) == 1, "ft_requires_pp1"),
    (
        lambda a: (
            getattr(a, "elastic_ep_backend", None) == "mooncake"
            or (
                getattr(a, "device", None) == "npu"
                and getattr(a, "elastic_ep_backend", None) == "mc2"
            )
        ),
        "ft_requires_mooncake_active_rank_backend",
    ),
    (
        lambda a: getattr(a, "disaggregation_mode", "null") == "null",
        "ft_unsupported_with_pd",
    ),
    (
        lambda a: getattr(a, "tokenizer_worker_num", 1) <= 1,
        "ft_unsupported_with_multi_tokenizer",
    ),
    (lambda a: not getattr(a, "use_ray", False), "ft_unsupported_with_ray_engine"),
    (_dp_attention_gate, "ft_requires_tp_divisible_by_dp_and_attn_cp"),
]


def is_ft_supported_config(server_args) -> Tuple[bool, str]:
    for supported, error in _FT_SUPPORT_GATES:
        if not supported(server_args):
            return False, error
    return True, ""


class FaultToleranceState:
    def __init__(self, *, dp_size: int, strategy: str, global_rank_count: int):
        self.dp_size = dp_size
        self.strategy = strategy
        self.global_rank_count = global_rank_count
        self.global_ranks_per_dp = global_rank_count // dp_size
        self.expected_dp_mask = [True] * dp_size
        self.native_active_dp_mask = [True] * dp_size
        self.process_alive_global_rank_mask = [True] * global_rank_count
        # Process events use physical scheduler/global ranks, so recovery pending
        # stays at that same granularity until a DP-level native-ready event arrives.
        self.pending_recovery_global_ranks: set[int] = set()
        self.unhealthy_dp_ranks: set[int] = set()
        self.ft_operation_in_progress = False
        self.cluster_paused = False
        # Assigned by the FT manager, never independently by joining workers.
        self.membership_epoch = 0

    def global_ranks_for_dp(self, dp_rank: int) -> range:
        # FT keeps this mapping local until the upstream topology helper is stable.
        # With PP=1, scheduler/global ranks for one DP form a contiguous block;
        # the block can contain multiple attention TP/CP ranks.
        start = dp_rank * self.global_ranks_per_dp
        return range(start, start + self.global_ranks_per_dp)

    def global_ranks_for_dps(self, dp_ranks: Iterable[int]) -> List[int]:
        return [
            rank
            for dp_rank in sorted(set(dp_ranks))
            for rank in self.global_ranks_for_dp(dp_rank)
        ]

    def expand_dp_mask_to_global_rank_mask(self, dp_mask: List[bool]) -> List[bool]:
        return [
            dp_mask[rank // self.global_ranks_per_dp]
            for rank in range(self.global_rank_count)
        ]

    def project_global_rank_mask_to_dp_mask(
        self, global_rank_mask: List[bool]
    ) -> List[bool]:
        return [
            all(
                rank < len(global_rank_mask) and global_rank_mask[rank]
                for rank in self.global_ranks_for_dp(dp)
            )
            for dp in range(self.dp_size)
        ]

    def process_alive_dp_mask(self) -> List[bool]:
        return self.project_global_rank_mask_to_dp_mask(
            self.process_alive_global_rank_mask
        )

    def expected_dp_ranks(self) -> List[int]:
        return [rank for rank, expected in enumerate(self.expected_dp_mask) if expected]

    def _rank_state(self, rank: int) -> RankState:
        if not self.expected_dp_mask[rank] or not self.process_alive_dp_mask()[rank]:
            return RankState.DEAD
        if rank in self.unhealthy_dp_ranks:
            return RankState.UNHEALTHY
        return RankState.HEALTHY

    def status_response(self) -> Dict[str, Any]:
        return {
            "ranks": [
                {"rank": rank, "state": self._rank_state(rank).value}
                for rank in range(self.dp_size)
            ]
        }

    def has_incident(self) -> bool:
        process_alive_dp_mask = self.process_alive_dp_mask()
        return bool(self.unhealthy_dp_ranks) or any(
            expected and not process_alive_dp_mask[rank]
            for rank, expected in enumerate(self.expected_dp_mask)
        )

    def should_reject_admission(self, route_dp_mask: List[bool]) -> bool:
        return (
            not any(route_dp_mask)
            or self.ft_operation_in_progress
            or (self.strategy == "pause" and self.cluster_paused)
        )

    def observe_process_active_ranks(
        self, ranks: Iterable[int], *, active: bool
    ) -> None:
        for rank in ranks:
            if 0 <= rank < self.global_rank_count:
                self.process_alive_global_rank_mask[rank] = active
        if not active:
            self.pending_recovery_global_ranks.update(
                rank for rank in ranks if 0 <= rank < self.global_rank_count
            )
            self.cluster_paused = True

    def observe_native_active_dp_mask(self, active_dp_mask: List[bool]) -> None:
        # ActiveRanksOutput is a DP-rank mask at the FT boundary. A true DP bit
        # means all physical scheduler ranks in that DP have rejoined the native
        # data plane, so their global-rank pending markers can be retired together.
        self.native_active_dp_mask = list(active_dp_mask)
        for dp_rank, active in enumerate(self.native_active_dp_mask):
            if active and dp_rank < self.dp_size:
                self.pending_recovery_global_ranks.difference_update(
                    self.global_ranks_for_dp(dp_rank)
                )

    def observe_rank_fault(self, rank: int) -> None:
        if self.strategy == "pause" and 0 <= rank < self.dp_size:
            self.unhealthy_dp_ranks.add(rank)
            self.cluster_paused = True

    def finish_retry(self) -> Dict[str, Any]:
        self.unhealthy_dp_ranks.clear()
        self.ft_operation_in_progress = False
        self.cluster_paused = False
        return self._success("fault_tolerance_retry_committed")

    def finish_scale_down(self, ranks: Iterable[int]) -> Dict[str, Any]:
        removed = sorted(set(ranks))
        for rank in removed:
            self.expected_dp_mask[rank] = False
        self.unhealthy_dp_ranks.clear()
        self.ft_operation_in_progress = False
        self.cluster_paused = False
        body = self._success("fault_tolerance_scale_down_committed")
        body["removed_ranks"] = removed
        return body

    def _success(self, message: str) -> Dict[str, Any]:
        body = self.status_response()
        body.update({"success": True, "message": message})
        return body
