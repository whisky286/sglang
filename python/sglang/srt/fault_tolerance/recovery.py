"""Immutable membership plan for restoring removed ranks within launch capacity.

The plan describes membership, not readiness. A joining process must load its
target expert slots and pass device/data-plane validation before this membership
can be published to request routing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RankRecoveryPlan:
    membership_epoch: int
    previous_active_mask: tuple[bool, ...]
    recovering_ranks: tuple[int, ...]
    ranks_per_dp: int = 1

    def __post_init__(self) -> None:
        # Normalize wire-format lists so a frozen plan cannot change underneath
        # participants after validation.
        object.__setattr__(
            self, "previous_active_mask", tuple(self.previous_active_mask)
        )
        object.__setattr__(self, "recovering_ranks", tuple(self.recovering_ranks))
        if type(self.membership_epoch) is not int or self.membership_epoch <= 0:
            raise ValueError("membership_epoch must be a positive integer")
        if type(self.ranks_per_dp) is not int or self.ranks_per_dp <= 0:
            raise ValueError("ranks_per_dp must be a positive integer")
        mask = self.previous_active_mask
        if not mask or any(type(active) is not bool for active in mask):
            raise ValueError("previous_active_mask must contain booleans")
        if not any(mask):
            raise ValueError("rank recovery requires at least one surviving rank")
        if len(mask) % self.ranks_per_dp:
            raise ValueError("launch capacity must contain complete DP groups")
        for start in range(0, len(mask), self.ranks_per_dp):
            block = mask[start : start + self.ranks_per_dp]
            if any(block) != all(block):
                raise ValueError("previous membership must contain complete DP groups")

        ranks = self.recovering_ranks
        if not ranks or any(type(rank) is not int for rank in ranks):
            raise ValueError("recovering_ranks must contain integer ranks")
        if len(set(ranks)) != len(ranks):
            raise ValueError("recovering_ranks must be unique")
        if any(rank < 0 or rank >= len(mask) for rank in ranks):
            raise ValueError("rank recovery cannot exceed launch capacity")
        if any(mask[rank] for rank in ranks):
            raise ValueError("recovering ranks must be inactive")
        recovering = set(ranks)
        for rank in ranks:
            start = rank // self.ranks_per_dp * self.ranks_per_dp
            if not set(range(start, start + self.ranks_per_dp)) <= recovering:
                raise ValueError("rank recovery requires complete DP groups")
        object.__setattr__(self, "recovering_ranks", tuple(sorted(ranks)))

    @property
    def target_active_mask(self) -> tuple[bool, ...]:
        recovering = set(self.recovering_ranks)
        return tuple(
            active or rank in recovering
            for rank, active in enumerate(self.previous_active_mask)
        )

    @property
    def active_original_ranks(self) -> tuple[int, ...]:
        return tuple(
            rank for rank, active in enumerate(self.target_active_mask) if active
        )

    @property
    def store_prefix(self) -> str:
        membership = "".join(
            "1" if active else "0" for active in self.target_active_mask
        )
        return f"npu-ft/{membership}/{self.membership_epoch}"
