# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence

import torch
import torch.distributed
import torch.nn.functional as F

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _prefer_same_node_experts(server_args: ServerArgs) -> bool:
    from sglang.srt.elastic_ep.elastic_ep import elastic_expanded_world_enabled

    return server_args.ep_join_mode != "scale" and not elastic_expanded_world_enabled()


def _compute_elastic_expert_layout(
    base_num_physical_experts: int,
    initial_ep_size: int,
    effective_ep_size: int,
) -> tuple[int, int]:
    assert base_num_physical_experts % initial_ep_size == 0
    num_local_physical_experts = base_num_physical_experts // initial_ep_size
    return (
        num_local_physical_experts * effective_ep_size,
        num_local_physical_experts,
    )


@dataclass
class ExpertLocationMetadata:
    physical_to_logical_map: torch.Tensor  # (layers, num_physical_experts)
    physical_to_logical_map_cpu: torch.Tensor
    logical_to_all_physical_map: torch.Tensor  # (layers, num_logical_experts, X)
    logical_to_all_physical_map_cpu: torch.Tensor  # CPU copy for performance
    logical_to_all_physical_map_num_valid: torch.Tensor  # (layers, num_logical_experts)
    ep_size: int
    # (layers, num_logical_experts)
    logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]

    # -------------------------------- properties ------------------------------------

    @property
    def num_layers(self) -> int:
        return self.physical_to_logical_map.shape[0]

    @property
    def num_physical_experts(self) -> int:
        return self.physical_to_logical_map.shape[1]

    @property
    def num_local_physical_experts(self) -> int:
        ans, remainder = divmod(self.num_physical_experts, self.ep_size)
        assert remainder == 0
        return ans

    @property
    def num_logical_experts(self) -> int:
        return self.logical_to_all_physical_map.shape[1]

    def __post_init__(self):
        num_layers_0, num_physical_experts_0 = self.physical_to_logical_map.shape
        num_layers_1, num_logical_experts_0, num_physical_experts_1 = (
            self.logical_to_all_physical_map.shape
        )
        num_layers_2, num_logical_experts_1 = (
            self.logical_to_all_physical_map_num_valid.shape
        )
        assert num_layers_0 == num_layers_1 == num_layers_2
        assert num_logical_experts_0 == num_logical_experts_1
        assert num_physical_experts_0 == num_physical_experts_1

    # -------------------------------- construction ------------------------------------

    @staticmethod
    def init_trivial(
        server_args: ServerArgs, model_config: ModelConfig, moe_ep_rank: int
    ):
        """Trivial location - logical expert i corresponds to physical expert i"""
        common = ExpertLocationMetadata._init_common(server_args, model_config)

        if common is None:
            return None

        num_physical_experts = common["num_physical_experts"]
        model_config_for_expert_location = common["model_config_for_expert_location"]
        num_layers = model_config_for_expert_location.num_layers
        num_logical_experts = model_config_for_expert_location.num_logical_experts

        base_num_physical_experts = common["base_num_physical_experts"]
        physical_to_logical_map = (
            torch.arange(0, base_num_physical_experts).repeat(num_layers, 1)
            % num_logical_experts
        )
        physical_to_logical_map = append_trivial_expert_slots(
            physical_to_logical_map,
            num_physical_experts - base_num_physical_experts,
            num_logical_experts,
        )

        return ExpertLocationMetadata.init_by_mapping(
            server_args,
            model_config,
            physical_to_logical_map=physical_to_logical_map,
            moe_ep_rank=moe_ep_rank,
        )

    @staticmethod
    def init_by_mapping(
        server_args: ServerArgs,
        model_config: ModelConfig,
        physical_to_logical_map,
        moe_ep_rank: int = None,
    ):
        if not isinstance(physical_to_logical_map, torch.Tensor):
            physical_to_logical_map = torch.tensor(physical_to_logical_map)
        physical_to_logical_map = physical_to_logical_map.to(server_args.device)

        common = ExpertLocationMetadata._init_common(server_args, model_config)

        if common is None:
            return None

        model_config_for_expert_location = common["model_config_for_expert_location"]
        if common["num_physical_experts"] > common["base_num_physical_experts"]:
            if physical_to_logical_map.shape[-1] == common["base_num_physical_experts"]:
                physical_to_logical_map = append_trivial_expert_slots(
                    physical_to_logical_map,
                    common["num_physical_experts"]
                    - common["base_num_physical_experts"],
                    model_config_for_expert_location.num_logical_experts,
                )
            assert physical_to_logical_map.shape[-1] == common["num_physical_experts"]
        logical_to_all_physical_map = _compute_logical_to_all_physical_map(
            server_args=server_args,
            physical_to_logical_map=physical_to_logical_map,
            num_logical_experts=model_config_for_expert_location.num_logical_experts,
            ep_size=common["ep_size"],
            moe_ep_rank=moe_ep_rank,
        )

        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
            moe_ep_rank=moe_ep_rank,
        )

    @staticmethod
    def init_by_eplb(
        server_args: ServerArgs, model_config: ModelConfig, logical_count: torch.Tensor
    ):
        if not isinstance(logical_count, torch.Tensor):
            logical_count = torch.tensor(logical_count)
        if len(logical_count.shape) == 2:
            logical_count = logical_count.unsqueeze(0)
        logical_count = logical_count.to(server_args.device)

        common = ExpertLocationMetadata._init_common(server_args, model_config)

        if common is None:
            return None

        model_config_for_expert_location = common["model_config_for_expert_location"]
        num_physical_experts = common["num_physical_experts"]
        num_groups = model_config_for_expert_location.num_groups
        num_nodes = server_args.nnodes

        from sglang.srt.eplb import eplb_algorithms

        physical_to_logical_map, logical_to_all_physical_map, expert_count = (
            eplb_algorithms.rebalance_experts(
                tokens_per_expert=logical_count,
                num_physical_experts=num_physical_experts,
                num_local_physical_experts=num_physical_experts // common["ep_size"],
                num_groups=num_groups,
                num_nodes=num_nodes,
                algorithm=eplb_algorithms.compute_algorithm(
                    raw_algorithm=server_args.eplb_algorithm,
                    num_groups=num_groups,
                    num_nodes=num_nodes,
                ),
            )
        )

        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map.to(server_args.device),
            logical_to_all_physical_map=logical_to_all_physical_map.to(
                server_args.device
            ),
        )

    @staticmethod
    def init_for_fault_recovery(
        server_args: ServerArgs,
        old_metadata: "ExpertLocationMetadata",
        active_ranks: Sequence[bool] | torch.Tensor,
        moe_ep_rank: Optional[int] = None,
        strategy: str = "minimal",
    ) -> "ExpertLocationMetadata":
        """Build an expert layout for rank-fault recovery.

        strategy:
        - minimal:
            Production recovery strategy. Preserve survivor slots whenever
            possible and reload only logical experts that disappeared with
            failed ranks.

        - shuffle:
            Debug/stress strategy. Start from a valid minimal survivor layout,
            then deliberately reshuffle expert slots across survivor ranks in
            order to exercise EPLB P2P/internal-format weight movement.
        """

        strategy = strategy.strip().lower()

        if strategy == "minimal":
            return ExpertLocationMetadata.init_for_fault_recovery_minimal(
                server_args=server_args,
                old_metadata=old_metadata,
                active_ranks=active_ranks,
                moe_ep_rank=moe_ep_rank,
            )

        if strategy == "shuffle":
            return ExpertLocationMetadata.init_for_fault_recovery_shuffle(
                server_args=server_args,
                old_metadata=old_metadata,
                active_ranks=active_ranks,
                moe_ep_rank=moe_ep_rank,
            )

        raise ValueError(
            "Unsupported FT expert recovery strategy: "
            f"{strategy!r}. Expected one of: minimal, shuffle"
        )

    @staticmethod
    def init_for_fault_recovery_minimal(
        server_args: ServerArgs,
        old_metadata: "ExpertLocationMetadata",
        active_ranks: Sequence[bool] | torch.Tensor,
        moe_ep_rank: Optional[int] = None,
    ) -> "ExpertLocationMetadata":
        """Build a survivor-only layout with the minimum required movement.

        Normal EPLB is free to reshuffle every physical slot for load balance.
        During rank-fault recovery that needlessly sends internal-format expert
        weights over P2P even when the survivors already cover every logical
        expert.  Preserve every survivor slot first.  Only logical experts
        with no surviving copy replace a redundant survivor slot; those slots
        then follow the updater's DRAM/checkpoint recovery path.

        The physical map remains in the immutable original-rank namespace, but
        the dispatch map contains only active physical IDs.
        """

        if isinstance(active_ranks, torch.Tensor):
            active_rank_values = active_ranks.detach().to(device="cpu").tolist()
        else:
            active_rank_values = list(active_ranks)
        if len(active_rank_values) != old_metadata.ep_size:
            raise ValueError(
                "active_ranks length must match expert metadata EP size "
                f"({old_metadata.ep_size}), got {len(active_rank_values)}"
            )
        if not any(bool(value) for value in active_rank_values):
            raise ValueError("fault recovery requires at least one active rank")

        num_local_physical_experts = old_metadata.num_local_physical_experts
        active_original_ranks = [
            rank
            for rank, is_active in enumerate(active_rank_values)
            if bool(is_active)
        ]
        active_physical_ids = [
            rank * num_local_physical_experts + local_slot
            for rank in active_original_ranks
            for local_slot in range(num_local_physical_experts)
        ]
        active_physical_ids_by_rank = {
            rank: list(
                range(
                    rank * num_local_physical_experts,
                    (rank + 1) * num_local_physical_experts,
                )
            )
            for rank in active_original_ranks
        }
        num_logical_experts = old_metadata.num_logical_experts
        if len(active_physical_ids) < num_logical_experts:
            raise RuntimeError(
                "insufficient survivor expert slots for fault recovery: "
                f"active_slots={len(active_physical_ids)}, "
                f"logical_experts={num_logical_experts}"
            )

        physical_to_logical_map_cpu = (
            old_metadata.physical_to_logical_map_cpu.clone()
        )
        logical_to_physical_by_layer = []
        max_replica_count = 0
        for layer_id in range(old_metadata.num_layers):
            layer_map = physical_to_logical_map_cpu[layer_id]
            active_logical_ids = layer_map[active_physical_ids].to(torch.int64)
            if bool(
                ((active_logical_ids < 0) | (active_logical_ids >= num_logical_experts))
                .any()
                .item()
            ):
                raise RuntimeError(
                    "fault recovery found an invalid logical expert ID in "
                    f"survivor slots for layer {layer_id}"
                )

            replica_counts = torch.bincount(
                active_logical_ids,
                minlength=num_logical_experts,
            )
            missing_logical_ids = torch.nonzero(
                replica_counts == 0,
                as_tuple=False,
            ).flatten()
            # Spread checkpoint/DRAM reloads across survivor ranks.  A simple
            # scan over active_physical_ids would consume every redundant slot
            # on the lowest original rank first.  Sequential scale-down from
            # four ranks to two can then make one survivor reload half of the
            # model while the other waits for its FT command acknowledgement.
            replacement_rank_cursor = 0
            for missing_logical_id_tensor in missing_logical_ids:
                missing_logical_id = int(missing_logical_id_tensor.item())
                replacement_physical_id = None
                for rank_offset in range(len(active_original_ranks)):
                    rank_index = (
                        replacement_rank_cursor + rank_offset
                    ) % len(active_original_ranks)
                    candidate_rank = active_original_ranks[rank_index]
                    replacement_physical_id = next(
                        (
                            physical_id
                            for physical_id in active_physical_ids_by_rank[
                                candidate_rank
                            ]
                            if replica_counts[
                                int(layer_map[physical_id].item())
                            ]
                            > 1
                        ),
                        None,
                    )
                    if replacement_physical_id is not None:
                        replacement_rank_cursor = (
                            rank_index + 1
                        ) % len(active_original_ranks)
                        break
                if replacement_physical_id is None:
                    raise RuntimeError(
                        "unable to reserve a survivor slot for missing logical "
                        f"expert {missing_logical_id} in layer {layer_id}"
                    )
                replaced_logical_id = int(
                    layer_map[replacement_physical_id].item()
                )
                layer_map[replacement_physical_id] = missing_logical_id
                replica_counts[replaced_logical_id] -= 1
                replica_counts[missing_logical_id] += 1

            layer_logical_to_physical = [
                [] for _ in range(num_logical_experts)
            ]
            for physical_id in active_physical_ids:
                logical_id = int(layer_map[physical_id].item())
                layer_logical_to_physical[logical_id].append(physical_id)
            if any(not locations for locations in layer_logical_to_physical):
                raise RuntimeError(
                    "fault recovery layout does not cover every logical expert "
                    f"in layer {layer_id}"
                )
            max_replica_count = max(
                max_replica_count,
                max(len(locations) for locations in layer_logical_to_physical),
            )
            logical_to_physical_by_layer.append(layer_logical_to_physical)

        logical_to_all_physical_map_cpu = torch.full(
            (
                old_metadata.num_layers,
                num_logical_experts,
                max_replica_count,
            ),
            -1,
            dtype=physical_to_logical_map_cpu.dtype,
            device="cpu",
        )
        for layer_id, layer_mapping in enumerate(logical_to_physical_by_layer):
            for logical_id, physical_ids in enumerate(layer_mapping):
                logical_to_all_physical_map_cpu[
                    layer_id, logical_id, : len(physical_ids)
                ] = torch.tensor(
                    physical_ids,
                    dtype=logical_to_all_physical_map_cpu.dtype,
                )

        device = old_metadata.physical_to_logical_map.device
        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            ep_size=old_metadata.ep_size,
            physical_to_logical_map=physical_to_logical_map_cpu.to(device=device),
            logical_to_all_physical_map=logical_to_all_physical_map_cpu.to(
                device=device
            ),
            moe_ep_rank=moe_ep_rank,
        )
    @staticmethod
    def init_for_fault_recovery_shuffle(
        server_args: ServerArgs,
        old_metadata: "ExpertLocationMetadata",
        active_ranks: Sequence[bool] | torch.Tensor,
        moe_ep_rank: Optional[int] = None,
    ) -> "ExpertLocationMetadata":
        """Build a deliberately shuffled survivor-only FT layout.

        This is a DEBUG/STRESS strategy, not the production recovery policy.

        Procedure:
        1. Build the normal minimal recovery layout first. This guarantees that
            every logical expert is represented by at least one survivor slot.
        2. Rotate whole expert blocks among survivor ranks.
        3. Also rotate local slots by one position to maximize changed slots.

        The global multiset of logical experts on survivor slots is unchanged, so
        logical-expert coverage remains identical to the valid minimal layout.

        Failed-rank physical slots remain untouched. Only survivor slots are
        reshuffled.
        """

        if isinstance(active_ranks, torch.Tensor):
            active_rank_values = (
                active_ranks.detach().to(device="cpu").tolist()
            )
        else:
            active_rank_values = list(active_ranks)

        if len(active_rank_values) != old_metadata.ep_size:
            raise ValueError(
                "active_ranks length must match expert metadata EP size "
                f"({old_metadata.ep_size}), got {len(active_rank_values)}"
            )

        active_original_ranks = [
            rank
            for rank, is_active in enumerate(active_rank_values)
            if bool(is_active)
        ]

        if len(active_original_ranks) < 2:
            raise RuntimeError(
                "FT shuffle recovery requires at least two survivor ranks, "
                f"got {active_original_ranks}"
            )

        #
        # First create the known-correct minimal layout.
        #
        minimal_metadata = (
            ExpertLocationMetadata.init_for_fault_recovery_minimal(
                server_args=server_args,
                old_metadata=old_metadata,
                active_ranks=active_rank_values,
                moe_ep_rank=moe_ep_rank,
            )
        )

        num_local_physical_experts = (
            old_metadata.num_local_physical_experts
        )
        num_logical_experts = old_metadata.num_logical_experts

        #
        # Keep dead-rank slots unchanged. Only modify active-rank slots.
        #
        physical_to_logical_map_cpu = (
            minimal_metadata.physical_to_logical_map_cpu.clone()
        )

        #
        # Read from an immutable snapshot so that the shuffle itself cannot
        # overwrite a source block before another destination consumes it.
        #
        minimal_map_cpu = (
            minimal_metadata.physical_to_logical_map_cpu.clone()
        )

        for layer_id in range(old_metadata.num_layers):
            source_blocks = {}

            for rank in active_original_ranks:
                begin = rank * num_local_physical_experts
                end = begin + num_local_physical_experts

                source_blocks[rank] = (
                    minimal_map_cpu[layer_id, begin:end].clone()
                )

            #
            # Rotate rank blocks:
            #
            #   survivor[0] <- survivor[-1]
            #   survivor[1] <- survivor[0]
            #   survivor[2] <- survivor[1]
            #
            # Therefore an expert that existed only on one survivor is forced to
            # cross ranks instead of merely changing a local slot.
            #
            for dst_rank_index, dst_rank in enumerate(
                active_original_ranks
            ):
                src_rank = active_original_ranks[
                    (dst_rank_index - 1)
                    % len(active_original_ranks)
                ]

                shuffled_block = source_blocks[src_rank]

                #
                # Also rotate local slots. This makes the debug layout much less
                # likely to accidentally match the old layout when replicas happen
                # to exist on multiple survivor ranks.
                #
                if num_local_physical_experts > 1:
                    shuffled_block = torch.roll(
                        shuffled_block,
                        shifts=1,
                        dims=0,
                    )

                dst_begin = (
                    dst_rank * num_local_physical_experts
                )
                dst_end = (
                    dst_begin + num_local_physical_experts
                )

                physical_to_logical_map_cpu[
                    layer_id,
                    dst_begin:dst_end,
                ] = shuffled_block

        #
        # Verify that the debug shuffle actually changed survivor slots.
        #
        active_physical_ids = [
            rank * num_local_physical_experts + local_slot
            for rank in active_original_ranks
            for local_slot in range(num_local_physical_experts)
        ]

        changed_active_slots = int(
            (
                physical_to_logical_map_cpu[
                    :, active_physical_ids
                ]
                != minimal_map_cpu[
                    :, active_physical_ids
                ]
            )
            .sum()
            .item()
        )

        if changed_active_slots == 0:
            raise RuntimeError(
                "FT shuffle recovery produced no changed survivor slots; "
                "the debug strategy did not exercise expert movement"
            )

        #
        # Rebuild logical -> active physical mapping.
        #
        # IMPORTANT:
        # Only ACTIVE physical IDs participate here. Dead-rank slots stay in the
        # physical map solely because FT preserves the immutable original-rank
        # namespace; they must not become dispatch destinations.
        #
        logical_to_physical_by_layer = []
        max_replica_count = 0

        for layer_id in range(old_metadata.num_layers):
            layer_map = physical_to_logical_map_cpu[layer_id]

            layer_logical_to_physical = [
                [] for _ in range(num_logical_experts)
            ]

            for physical_id in active_physical_ids:
                logical_id = int(
                    layer_map[physical_id].item()
                )

                if not (
                    0 <= logical_id < num_logical_experts
                ):
                    raise RuntimeError(
                        "FT shuffle produced invalid logical expert ID: "
                        f"layer={layer_id} "
                        f"physical_id={physical_id} "
                        f"logical_id={logical_id}"
                    )

                layer_logical_to_physical[
                    logical_id
                ].append(physical_id)

            missing_logical_ids = [
                logical_id
                for logical_id, physical_ids
                in enumerate(layer_logical_to_physical)
                if not physical_ids
            ]

            if missing_logical_ids:
                raise RuntimeError(
                    "FT shuffle lost logical-expert coverage: "
                    f"layer={layer_id} "
                    f"missing_logical_ids={missing_logical_ids}"
                )

            max_replica_count = max(
                max_replica_count,
                max(
                    len(physical_ids)
                    for physical_ids
                    in layer_logical_to_physical
                ),
            )

            logical_to_physical_by_layer.append(
                layer_logical_to_physical
            )

        logical_to_all_physical_map_cpu = torch.full(
            (
                old_metadata.num_layers,
                num_logical_experts,
                max_replica_count,
            ),
            -1,
            dtype=physical_to_logical_map_cpu.dtype,
            device="cpu",
        )

        for layer_id, layer_mapping in enumerate(
            logical_to_physical_by_layer
        ):
            for logical_id, physical_ids in enumerate(
                layer_mapping
            ):
                logical_to_all_physical_map_cpu[
                    layer_id,
                    logical_id,
                    : len(physical_ids),
                ] = torch.tensor(
                    physical_ids,
                    dtype=(
                        logical_to_all_physical_map_cpu.dtype
                    ),
                )

        logger.warning(
            "[NPU FT DEBUG] shuffled expert recovery layout: "
            "active_original_ranks=%s "
            "changed_active_slots=%d",
            active_original_ranks,
            changed_active_slots,
        )

        device = old_metadata.physical_to_logical_map.device

        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            ep_size=old_metadata.ep_size,
            physical_to_logical_map=(
                physical_to_logical_map_cpu.to(
                    device=device
                )
            ),
            logical_to_all_physical_map=(
                logical_to_all_physical_map_cpu.to(
                    device=device
                )
            ),
            moe_ep_rank=moe_ep_rank,
        )

    @staticmethod
    def _init_common(server_args: ServerArgs, model_config: ModelConfig):
        model_config_for_expert_location = (
            ModelConfigForExpertLocation.from_model_config(model_config)
        )

        if model_config_for_expert_location is None:
            return None

        base_num_physical_experts = (
            model_config_for_expert_location.num_logical_experts
            + server_args.ep_num_redundant_experts
        )
        ep_size = server_args.ep_size
        num_physical_experts = base_num_physical_experts
        initial_ep_size = server_args.elastic_ep_initial_size
        if initial_ep_size is not None:
            if server_args.ep_join_mode == "scale":
                ep_size = max(
                    ep_size,
                    server_args.ep_join_rank_offset + server_args.tp_size,
                )
            num_physical_experts, num_local_physical_experts = (
                _compute_elastic_expert_layout(
                    base_num_physical_experts,
                    initial_ep_size,
                    ep_size,
                )
            )
        else:
            assert num_physical_experts % ep_size == 0
            num_local_physical_experts = num_physical_experts // ep_size

        return dict(
            model_config_for_expert_location=model_config_for_expert_location,
            base_num_physical_experts=base_num_physical_experts,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            ep_size=ep_size,
        )

    @staticmethod
    def _init_raw(
        server_args: ServerArgs,
        ep_size: int,
        physical_to_logical_map: torch.Tensor,
        logical_to_all_physical_map: torch.Tensor,
        moe_ep_rank: Optional[int] = None,
    ):
        _, num_physical_experts = physical_to_logical_map.shape

        logical_to_all_physical_map_padded = F.pad(
            logical_to_all_physical_map,
            (0, num_physical_experts - logical_to_all_physical_map.shape[-1]),
            value=-1,
        )

        logical_to_all_physical_map_num_valid = torch.count_nonzero(
            logical_to_all_physical_map != -1, dim=-1
        )

        return ExpertLocationMetadata(
            physical_to_logical_map=physical_to_logical_map,
            physical_to_logical_map_cpu=physical_to_logical_map.cpu(),
            logical_to_all_physical_map=logical_to_all_physical_map_padded,
            logical_to_all_physical_map_cpu=logical_to_all_physical_map_padded.cpu(),
            logical_to_all_physical_map_num_valid=logical_to_all_physical_map_num_valid,
            ep_size=ep_size,
            logical_to_rank_dispatch_physical_map=(
                compute_logical_to_rank_dispatch_physical_map(
                    server_args=server_args,
                    logical_to_all_physical_map=logical_to_all_physical_map,
                    ep_size=ep_size,
                    num_physical_experts=num_physical_experts,
                    ep_rank=(
                        moe_ep_rank
                        if moe_ep_rank is not None
                        else torch.distributed.get_rank() % ep_size
                    ),
                )
                if server_args.ep_dispatch_algorithm == "static"
                else None
            ),
        )

    # -------------------------------- mutation ------------------------------------

    def update(
        self,
        other: ExpertLocationMetadata,
        update_layer_ids: List[int],
    ):
        for field in [
            "ep_size",
        ]:
            assert getattr(self, field) == getattr(other, field)

        for field in [
            "physical_to_logical_map",
            "physical_to_logical_map_cpu",
            "logical_to_all_physical_map",
            "logical_to_all_physical_map_cpu",
            "logical_to_all_physical_map_num_valid",
            "logical_to_rank_dispatch_physical_map",
        ]:
            other_field = getattr(other, field)
            self_field = getattr(self, field)
            assert (other_field is not None) == (self_field is not None)
            if self_field is not None:
                mask_update = torch.tensor(
                    [i in update_layer_ids for i in range(self.num_layers)]
                )
                mask_update = mask_update.view(*([-1] + [1] * (self_field.dim() - 1)))
                mask_update = mask_update.to(self_field.device, non_blocking=True)
                self_field[...] = torch.where(mask_update, other_field, self_field)

    # -------------------------------- usage ------------------------------------

    def logical_to_all_physical(
        self,
        layer_id: int,
        logical_expert_id: int,
        require_global_experts: bool = False,
    ) -> List[int]:
        # Use CPU copy to avoid GPU→CPU sync on every call, which is expensive in update weights scenario
        cpu_map = self.logical_to_all_physical_map_cpu
        # Draft workers can query MoE layers whose layer_id lies beyond the
        # target-sized expert map; fall back to the identity mapping (no EPLB
        # rebalancing for those layers) instead of indexing out of range.
        if layer_id >= cpu_map.shape[0]:
            if require_global_experts:
                num_physical_experts = cpu_map.shape[-1]
                return list(
                    range(
                        logical_expert_id,
                        num_physical_experts,
                        self.num_logical_experts,
                    )
                )
            return [logical_expert_id]
        if require_global_experts:
            num_physical_experts = cpu_map[layer_id].shape[-1]
            return list(
                range(logical_expert_id, num_physical_experts, self.num_logical_experts)
            )
        return [
            physical_expert_id
            for physical_expert_id in cpu_map[layer_id, logical_expert_id].tolist()
            if physical_expert_id != -1
        ]


def format_expert_location_layout(
    metadata: Optional[ExpertLocationMetadata],
    layer_ids: Optional[Iterable[int]] = None,
) -> str:
    if metadata is None:
        return "<none>"

    return format_physical_to_logical_map(
        metadata.physical_to_logical_map_cpu,
        ep_size=metadata.ep_size,
        layer_ids=layer_ids,
    )


def format_expert_location_layout_diff(
    old_metadata: Optional[ExpertLocationMetadata],
    new_metadata: Optional[ExpertLocationMetadata],
    layer_ids: Optional[Iterable[int]] = None,
) -> str:
    if old_metadata is None or new_metadata is None:
        return "<none>"

    old_map = old_metadata.physical_to_logical_map_cpu
    new_map = new_metadata.physical_to_logical_map_cpu
    if old_map.shape != new_map.shape:
        return f"shape_changed old_shape={tuple(old_map.shape)} new_shape={tuple(new_map.shape)}"

    layer_ids = _normalize_layer_ids(layer_ids, num_layers=old_map.shape[0])
    num_physical_experts = old_map.shape[1]

    changed_by_layer = []
    for layer_id in layer_ids:
        num_changed = torch.count_nonzero(old_map[layer_id] != new_map[layer_id]).item()
        if num_changed > 0:
            changed_by_layer.append((layer_id, num_changed))

    total_changed = sum(num_changed for _, num_changed in changed_by_layer)
    total_slots = len(layer_ids) * num_physical_experts
    lines = [f"changed_physical_slots={total_changed}/{total_slots}"]
    if not changed_by_layer:
        lines.append("changed_layers=[]")
        return "\n".join(lines)

    for layer_id, num_changed in changed_by_layer:
        lines.append(f"layer={layer_id}: changed={num_changed}/{num_physical_experts}")
    return "\n".join(lines)


def format_physical_to_logical_map(
    physical_to_logical_map: torch.Tensor,
    ep_size: int,
    layer_ids: Optional[Iterable[int]] = None,
) -> str:
    physical_to_logical_map = physical_to_logical_map.cpu()
    if physical_to_logical_map.numel() == 0:
        return "<empty>"

    layer_ids = _normalize_layer_ids(
        layer_ids, num_layers=physical_to_logical_map.shape[0]
    )
    num_physical_experts = physical_to_logical_map.shape[1]
    num_local_physical_experts, remainder = divmod(num_physical_experts, ep_size)

    lines = [
        "physical_to_logical_map "
        f"num_layers={physical_to_logical_map.shape[0]} "
        f"num_physical_experts={num_physical_experts} "
        f"ep_size={ep_size}"
    ]
    for layer_id in layer_ids:
        row = physical_to_logical_map[layer_id].tolist()
        if remainder != 0:
            lines.append(
                f"layer={layer_id}: "
                f"physical={json.dumps(row, separators=(',', ':'))}"
            )
            continue

        rank_chunks = []
        for ep_rank in range(ep_size):
            start = ep_rank * num_local_physical_experts
            end = start + num_local_physical_experts
            rank_chunks.append(
                f"ep{ep_rank}={json.dumps(row[start:end], separators=(',', ':'))}"
            )
        lines.append(f"layer={layer_id}: " + " ".join(rank_chunks))

    return "\n".join(lines)


def _normalize_layer_ids(
    layer_ids: Optional[Iterable[int]],
    num_layers: int,
) -> List[int]:
    if layer_ids is None:
        return list(range(num_layers))

    normalized_layer_ids = [int(layer_id) for layer_id in layer_ids]
    for layer_id in normalized_layer_ids:
        assert 0 <= layer_id < num_layers, f"{layer_id=} {num_layers=}"
    return normalized_layer_ids


def get_global_expert_location_metadata():
    from sglang.srt.runtime_context import get_resources

    return get_resources().expert_location_metadata


def set_global_expert_location_metadata(value, allow_overwrite=False):
    from sglang.srt.runtime_context import get_resources

    resources = get_resources()
    if not allow_overwrite:
        assert resources.expert_location_metadata is None
    resources.expert_location_metadata = value


def append_trivial_expert_slots(
    physical_to_logical_map: torch.Tensor,
    count: int,
    num_logical_experts: int,
    start: int = 0,
) -> torch.Tensor:
    if count <= 0:
        return physical_to_logical_map
    new_slots = torch.arange(
        start,
        start + count,
        dtype=physical_to_logical_map.dtype,
        device=physical_to_logical_map.device,
    ).unsqueeze(0)
    new_slots = new_slots.expand(physical_to_logical_map.shape[0], -1)
    return torch.cat([physical_to_logical_map, new_slots % num_logical_experts], dim=1)


def broadcast_global_expert_location_metadata(
    model_config: ModelConfig,
    moe_ep_rank: int,
    src_rank: int = 0,
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> ExpertLocationMetadata:
    from sglang.srt.runtime_context import get_server_args

    server_args = get_server_args()
    metadata = get_global_expert_location_metadata()
    assert metadata is not None

    if group is None and os.environ.get("MOONCAKE_EP_FORCE_FALLBACK") == "1":
        _broadcast_global_expert_location_metadata_via_cpu_group(
            metadata=metadata,
            src_rank=src_rank,
        )
        return

    # Ensure device tensors are contiguous before broadcasting in-place
    metadata.physical_to_logical_map = metadata.physical_to_logical_map.contiguous()
    torch.distributed.broadcast(
        metadata.physical_to_logical_map, src=src_rank, group=group
    )
    metadata = ExpertLocationMetadata.init_by_mapping(
        server_args,
        model_config,
        metadata.physical_to_logical_map,
        moe_ep_rank=moe_ep_rank,
    )
    set_global_expert_location_metadata(metadata, allow_overwrite=True)
    return metadata


def _broadcast_global_expert_location_metadata_via_cpu_group(
    metadata: ExpertLocationMetadata,
    src_rank: int,
):
    from sglang.srt.distributed.parallel_state import get_world_group

    logger.info(
        "Broadcast expert location metadata over CPU group in Mooncake forced "
        "fallback path."
    )

    physical_to_logical_map_cpu = metadata.physical_to_logical_map_cpu.contiguous()
    logical_to_all_physical_map_cpu = (
        metadata.logical_to_all_physical_map_cpu.contiguous()
    )

    torch.distributed.broadcast(
        physical_to_logical_map_cpu,
        src=src_rank,
        group=get_world_group().cpu_group,
    )
    torch.distributed.broadcast(
        logical_to_all_physical_map_cpu,
        src=src_rank,
        group=get_world_group().cpu_group,
    )

    logical_to_all_physical_map_num_valid_cpu = torch.count_nonzero(
        logical_to_all_physical_map_cpu != -1,
        dim=-1,
    )

    logical_to_rank_dispatch_physical_map_cpu = None
    if metadata.logical_to_rank_dispatch_physical_map is not None:
        logical_to_rank_dispatch_physical_map_cpu = (
            metadata.logical_to_rank_dispatch_physical_map.detach().cpu().contiguous()
        )
        torch.distributed.broadcast(
            logical_to_rank_dispatch_physical_map_cpu,
            src=src_rank,
            group=get_world_group().cpu_group,
        )

    metadata.physical_to_logical_map_cpu = physical_to_logical_map_cpu
    metadata.logical_to_all_physical_map_cpu = logical_to_all_physical_map_cpu
    metadata.physical_to_logical_map = physical_to_logical_map_cpu.to(
        device=metadata.physical_to_logical_map.device,
        non_blocking=True,
    )
    metadata.logical_to_all_physical_map = logical_to_all_physical_map_cpu.to(
        device=metadata.logical_to_all_physical_map.device,
        non_blocking=True,
    )
    metadata.logical_to_all_physical_map_num_valid = (
        logical_to_all_physical_map_num_valid_cpu.to(
            device=metadata.logical_to_all_physical_map_num_valid.device,
            non_blocking=True,
        )
    )
    if logical_to_rank_dispatch_physical_map_cpu is not None:
        metadata.logical_to_rank_dispatch_physical_map = (
            logical_to_rank_dispatch_physical_map_cpu.to(
                device=metadata.logical_to_rank_dispatch_physical_map.device,
                non_blocking=True,
            )
        )


def _compute_logical_to_all_physical_map(
    server_args: ServerArgs,
    physical_to_logical_map: torch.Tensor,
    num_logical_experts: int,
    ep_size: int,
    moe_ep_rank: int,
):
    # This is rarely called, so we use for loops for maximum clarity

    num_layers, num_physical_experts = physical_to_logical_map.shape

    logical_to_all_physical_map = [
        [[] for _ in range(num_logical_experts)] for _ in range(num_layers)
    ]

    # Find out the candidate physical experts for each logical expert on each layer
    for layer_id in range(num_layers):
        for physical_expert_id in range(num_physical_experts):
            logical_expert_id = physical_to_logical_map[
                layer_id, physical_expert_id
            ].item()
            logical_to_all_physical_map[layer_id][logical_expert_id].append(
                physical_expert_id
            )

    # Replace by the physical expert on local GPU or node if possible
    if moe_ep_rank is not None:
        num_local_gpu_physical_experts = num_physical_experts // ep_size
        prefer_same_node = _prefer_same_node_experts(server_args)
        num_gpus_per_node = (
            server_args.ep_size // server_args.nnodes if prefer_same_node else None
        )
        num_local_node_physical_experts = (
            num_local_gpu_physical_experts * num_gpus_per_node
            if num_gpus_per_node is not None
            else None
        )
        for layer_id in range(num_layers):
            for logical_expert_id in range(num_logical_experts):
                # Try to find the nearest physical expert
                nearest_expert = _find_nearest_expert(
                    candidate_physical_expert_ids=logical_to_all_physical_map[layer_id][
                        logical_expert_id
                    ],
                    num_local_gpu_physical_experts=num_local_gpu_physical_experts,
                    moe_ep_rank=moe_ep_rank,
                    num_gpus_per_node=num_gpus_per_node,
                    num_local_node_physical_experts=num_local_node_physical_experts,
                )

                # Replace by the nearest physical expert
                if nearest_expert != -1:
                    logical_to_all_physical_map[layer_id][logical_expert_id] = [
                        nearest_expert
                    ]

    logical_to_all_physical_map = _pad_nested_array(
        logical_to_all_physical_map, pad_value=-1
    )

    return torch.tensor(
        logical_to_all_physical_map, device=physical_to_logical_map.device
    )


def _pad_nested_array(arr, pad_value):
    max_len = max(len(inner) for outer in arr for inner in outer)
    padded = [
        [inner + [pad_value] * (max_len - len(inner)) for inner in outer]
        for outer in arr
    ]
    return padded


# TODO optimize performance (rewrite and/or run in separate process with overlap)
def compute_logical_to_rank_dispatch_physical_map(
    server_args: ServerArgs,
    logical_to_all_physical_map: torch.Tensor,
    ep_size: int,
    num_physical_experts: int,
    ep_rank: int,
    seed: int = 42,
):
    r = random.Random(seed)

    device = logical_to_all_physical_map.device
    logical_to_all_physical_map = logical_to_all_physical_map.cpu()

    num_local_gpu_physical_experts = num_physical_experts // ep_size
    prefer_same_node = _prefer_same_node_experts(server_args)
    num_gpus_per_node = (
        server_args.ep_size // server_args.nnodes if prefer_same_node else None
    )
    num_local_node_physical_experts = (
        num_local_gpu_physical_experts * num_gpus_per_node
        if num_gpus_per_node is not None
        else None
    )
    num_layers, num_logical_experts, _ = logical_to_all_physical_map.shape
    dtype = logical_to_all_physical_map.dtype

    result_list = [
        [[-1] * num_logical_experts for _ in range(num_layers)] for _ in range(ep_size)
    ]

    for layer_id in range(num_layers):
        for logical_expert_id in range(num_logical_experts):
            candidate_physical_expert_ids = _logical_to_all_physical_raw(
                logical_to_all_physical_map, layer_id, logical_expert_id
            )

            remaining_ranks = []
            for moe_ep_rank in range(ep_size):
                val = _find_nearest_expert(
                    candidate_physical_expert_ids=candidate_physical_expert_ids,
                    num_local_gpu_physical_experts=num_local_gpu_physical_experts,
                    moe_ep_rank=moe_ep_rank,
                    num_gpus_per_node=num_gpus_per_node,
                    num_local_node_physical_experts=num_local_node_physical_experts,
                )

                result_list[moe_ep_rank][layer_id][logical_expert_id] = val
                if val == -1:
                    remaining_ranks.append(moe_ep_rank)

            if remaining_ranks:
                choices = _fair_choices(
                    candidate_physical_expert_ids, k=len(remaining_ranks), r=r
                )
                for moe_ep_rank, choice in zip(remaining_ranks, choices, strict=True):
                    result_list[moe_ep_rank][layer_id][logical_expert_id] = choice

    logical_to_rank_dispatch_physical_map = torch.tensor(result_list, dtype=dtype)
    assert torch.all(logical_to_rank_dispatch_physical_map != -1)

    return logical_to_rank_dispatch_physical_map[ep_rank, :, :].to(device)


def _logical_to_all_physical_raw(
    logical_to_all_physical_map, layer_id: int, logical_expert_id: int
) -> List[int]:
    return [
        physical_expert_id
        for physical_expert_id in logical_to_all_physical_map[
            layer_id, logical_expert_id
        ].tolist()
        if physical_expert_id != -1
    ]


def _compute_gpu_id_of_physical_expert(
    physical_expert_id: int, num_local_gpu_physical_experts: int
) -> int:
    return physical_expert_id // num_local_gpu_physical_experts


def _compute_node_id_of_physical_expert(
    physical_expert_id: int, num_local_host_physical_experts: int
) -> int:
    return physical_expert_id // num_local_host_physical_experts


def _find_nearest_expert(
    candidate_physical_expert_ids: List[int],
    num_local_gpu_physical_experts: int,
    moe_ep_rank: int,
    num_gpus_per_node: Optional[int],
    num_local_node_physical_experts: Optional[int],
) -> int:
    # 1. If only one candidate, return it directly
    if len(candidate_physical_expert_ids) == 1:
        return candidate_physical_expert_ids[0]

    # 2. Prefer same-GPU experts
    same_gpu_physical_expert_ids = [
        physical_expert_id
        for physical_expert_id in candidate_physical_expert_ids
        if _compute_gpu_id_of_physical_expert(
            physical_expert_id, num_local_gpu_physical_experts
        )
        == moe_ep_rank
    ]
    if len(same_gpu_physical_expert_ids) > 0:
        return same_gpu_physical_expert_ids[0]

    # Prefer same-node experts only when it narrows the candidate set.
    if num_gpus_per_node is not None and num_local_node_physical_experts is not None:
        node_rank = moe_ep_rank // num_gpus_per_node
        same_node_physical_expert_ids = [
            physical_expert_id
            for physical_expert_id in candidate_physical_expert_ids
            if _compute_node_id_of_physical_expert(
                physical_expert_id, num_local_node_physical_experts
            )
            == node_rank
        ]
        if 0 < len(same_node_physical_expert_ids) < len(candidate_physical_expert_ids):
            return same_node_physical_expert_ids[0]

    # 4. At last, leave it as -1 to indicate not found.
    return -1


def _fair_choices(arr: List, k: int, r: random.Random) -> List:
    quotient, remainder = divmod(k, len(arr))
    ans = arr * quotient + r.sample(arr, k=remainder)
    r.shuffle(ans)
    return ans


@dataclass
class ModelConfigForExpertLocation:
    num_layers: int
    num_logical_experts: int
    num_groups: Optional[int] = None

    @staticmethod
    def from_model_config(model_config: ModelConfig):
        from sglang.srt.model_loader import get_model_architecture

        model_class, _ = get_model_architecture(model_config)
        if hasattr(model_class, "get_model_config_for_expert_location"):
            return model_class.get_model_config_for_expert_location(
                model_config.hf_config
            )
        else:
            return None


def compute_initial_expert_location_metadata(
    server_args: ServerArgs,
    model_config: ModelConfig,
    moe_ep_rank: int,
) -> Optional[ExpertLocationMetadata]:
    data = server_args.init_expert_location
    if data == "trivial":
        return ExpertLocationMetadata.init_trivial(
            server_args, model_config, moe_ep_rank
        )

    # TODO unify with the utils function
    if data.endswith(".pt"):
        data_dict = torch.load(data, weights_only=True, map_location="cpu")
    elif data.endswith(".json"):
        data_dict = json.loads(Path(data).read_text())
    else:
        data_dict = json.loads(data)

    if "physical_to_logical_map" in data_dict:
        logger.info(
            "init_expert_location from init_by_mapping using ServerArgs.init_expert_location"
        )
        return ExpertLocationMetadata.init_by_mapping(
            server_args,
            model_config,
            **data_dict,
            moe_ep_rank=moe_ep_rank,
        )
    elif "logical_count" in data_dict:
        logger.info(
            "init_expert_location from init_by_eplb using ServerArgs.init_expert_location"
        )
        return ExpertLocationMetadata.init_by_eplb(
            server_args, model_config, logical_count=data_dict["logical_count"]
        )
    else:
        raise NotImplementedError(
            f"Unknown init_expert_location format ({list(data_dict.keys())=})"
        )
