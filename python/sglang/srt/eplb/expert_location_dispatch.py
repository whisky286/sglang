# Copyright 2023-2025 SGLang Team
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

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.runtime_context import get_server_args

import logging

from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

_VALIDATE_NPU_FT_RUNTIME_ROUTING = get_bool_env_var(
    "SGLANG_VALIDATE_NPU_FT_RUNTIME_ROUTING"
)
_NPU_FT_RUNTIME_ROUTING_PROBED = False

import os

@dataclass
class ExpertLocationDispatchInfo:
    ep_dispatch_algorithm: Literal["static", "dynamic", "fake", "lp"]
    # (num_logical_experts,)
    partial_logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]
    # (num_logical_experts, X)
    partial_logical_to_all_physical_map: torch.Tensor
    # (num_logical_experts,)
    partial_logical_to_all_physical_map_num_valid: torch.Tensor
    num_physical_experts: int
    # EPLB storage IDs use the immutable original-rank namespace, while MC2's
    # effective expert space is compact across survivor ranks. These fields
    # are populated before graph capture and updated in-place after scale-down.
    npu_mc2_elastic_info: Optional[torch.Tensor] = None
    npu_mc2_original_ep_size: int = 0
    npu_mc2_num_local_physical_experts: int = 0

    @classmethod
    def init_new(cls, layer_id: int):
        server_args = get_server_args()
        ep_dispatch_algorithm = server_args.ep_dispatch_algorithm
        expert_location_metadata = get_global_expert_location_metadata()
        assert expert_location_metadata is not None

        if ep_dispatch_algorithm is None:
            return None

        npu_mc2_elastic_info = None
        npu_mc2_original_ep_size = 0
        npu_mc2_num_local_physical_experts = 0
        if (
            server_args.device == "npu"
            and server_args.enable_fault_tolerance
            and server_args.elastic_ep_backend == "mc2"
        ):
            from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

            state = ElasticEPStateManager.instance()
            if state is None or state.npu_mc2_elastic_info is None:
                raise RuntimeError(
                    "NPU MC2 fault tolerance requires fixed elastic_info "
                    "before expert dispatch"
                )
            npu_mc2_elastic_info = state.npu_mc2_elastic_info.tensor
            npu_mc2_original_ep_size = state.original_ep_size
            (
                npu_mc2_num_local_physical_experts,
                remainder,
            ) = divmod(
                expert_location_metadata.num_physical_experts,
                npu_mc2_original_ep_size,
            )
            if remainder != 0:
                raise RuntimeError(
                    "NPU MC2 expert storage must be divisible by the original "
                    "EP size"
                )

        return cls(
            ep_dispatch_algorithm=ep_dispatch_algorithm,
            partial_logical_to_rank_dispatch_physical_map=(
                expert_location_metadata.logical_to_rank_dispatch_physical_map[
                    layer_id, :
                ]
                if expert_location_metadata.logical_to_rank_dispatch_physical_map
                is not None
                else None
            ),
            partial_logical_to_all_physical_map=expert_location_metadata.logical_to_all_physical_map[
                layer_id, :
            ],
            partial_logical_to_all_physical_map_num_valid=expert_location_metadata.logical_to_all_physical_map_num_valid[
                layer_id, :
            ],
            num_physical_experts=expert_location_metadata.num_physical_experts,
            npu_mc2_elastic_info=npu_mc2_elastic_info,
            npu_mc2_original_ep_size=npu_mc2_original_ep_size,
            npu_mc2_num_local_physical_experts=npu_mc2_num_local_physical_experts,
        )


def transform_select_experts_inputs(
    router_logits: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    info: Optional[ExpertLocationDispatchInfo],
):
    if (info is not None) and (info.ep_dispatch_algorithm == "fake"):
        router_logits.uniform_(5, 10)
        if correction_bias is not None:
            correction_bias = torch.zeros_like(correction_bias)
    return router_logits, correction_bias

def _validate_npu_ft_runtime_routing_once(
    logical_topk_ids: torch.Tensor,
    original_physical_ids: torch.Tensor,
    compact_physical_ids: torch.Tensor,
    info: ExpertLocationDispatchInfo,
) -> bool:
    """Validate logical -> original physical -> compact -> original round trip.

    Returns True only when the post-scale-down probe actually ran.
    """

    from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
    from sglang.srt.elastic_ep.npu_mc2 import MC2_ELASTIC_INFO_HEADER_SIZE

    state = ElasticEPStateManager.instance()
    if (
        state is None
        or state.active_ranks_cpu is None
        or bool(state.active_ranks_cpu.all().item())
    ):
        # Still before scale-down.
        return False

    elastic_info = info.npu_mc2_elastic_info
    if elastic_info is None:
        return False

    original_ep_size = info.npu_mc2_original_ep_size
    num_local = info.npu_mc2_num_local_physical_experts

    logical = logical_topk_ids.reshape(-1)
    original = original_physical_ids.reshape(-1)
    compact = compact_physical_ids.reshape(-1)

    valid = logical >= 0
    if not bool(valid.any().item()):
        return False

    # ------------------------------------------------------------------
    # 1. original physical ID 必须属于当前 logical expert 的候选位置
    # ------------------------------------------------------------------
    safe_logical = logical.masked_fill(~valid, 0).long()

    candidates = info.partial_logical_to_all_physical_map[
        safe_logical
    ]
    # candidates: [num_routes, max_replica_count]

    belongs_to_logical = (
        candidates == original.unsqueeze(-1)
    ).any(dim=-1)

    logical_mapping_bad = valid & ~belongs_to_logical

    # ------------------------------------------------------------------
    # 2. original physical ID 范围必须合法
    # ------------------------------------------------------------------
    max_original_physical = original_ep_size * num_local

    original_range_bad = valid & (
        (original < 0)
        | (original >= max_original_physical)
    )

    safe_original = original.masked_fill(
        original_range_bad | ~valid,
        0,
    )

    original_rank = torch.div(
        safe_original,
        num_local,
        rounding_mode="floor",
    )
    local_slot = safe_original % num_local

    # elastic_info layout:
    # [header(4),
    #  original_rank -> effective_rank,
    #  effective_rank -> original_rank]
    original_to_effective = elastic_info[
        MC2_ELASTIC_INFO_HEADER_SIZE :
        MC2_ELASTIC_INFO_HEADER_SIZE + original_ep_size
    ]

    effective_to_original = elastic_info[
        MC2_ELASTIC_INFO_HEADER_SIZE + original_ep_size :
        MC2_ELASTIC_INFO_HEADER_SIZE + 2 * original_ep_size
    ]

    effective_rank = original_to_effective[
        original_rank.long()
    ].to(original.dtype)

    # ------------------------------------------------------------------
    # 3. runtime 不允许路由到 dead original rank
    # ------------------------------------------------------------------
    inactive_rank_bad = valid & (
        ~original_range_bad
    ) & (effective_rank < 0)

    # ------------------------------------------------------------------
    # 4. compact 结果必须严格等于我们根据 elastic_info 算出的结果
    # ------------------------------------------------------------------
    expected_compact = (
        effective_rank * num_local + local_slot
    )

    compact_bad = (
        valid
        & ~original_range_bad
        & ~inactive_rank_bad
        & (compact != expected_compact)
    )

    # ------------------------------------------------------------------
    # 5. compact ID 必须在 survivor effective physical 范围内
    # ------------------------------------------------------------------
    effective_ep_size = int(
        elastic_info[1].item()
    )
    effective_num_physical = int(
        elastic_info[3].item()
    )

    compact_range_bad = valid & (
        (compact < 0)
        | (compact >= effective_num_physical)
    )

    # ------------------------------------------------------------------
    # 6. compact -> original 反解，必须得到完全相同的 original physical ID
    # ------------------------------------------------------------------
    safe_compact = compact.masked_fill(
        compact_range_bad | ~valid,
        0,
    )

    compact_effective_rank = torch.div(
        safe_compact,
        num_local,
        rounding_mode="floor",
    )
    compact_local_slot = safe_compact % num_local

    safe_effective_rank = compact_effective_rank.clamp(
        min=0,
        max=max(effective_ep_size - 1, 0),
    )

    roundtrip_original_rank = effective_to_original[
        safe_effective_rank.long()
    ].to(original.dtype)

    roundtrip_original = (
        roundtrip_original_rank * num_local
        + compact_local_slot
    )

    roundtrip_bad = (
        valid
        & ~compact_range_bad
        & (roundtrip_original != original)
    )

    bad = (
        logical_mapping_bad
        | original_range_bad
        | inactive_rank_bad
        | compact_bad
        | compact_range_bad
        | roundtrip_bad
    )

    bad_indices = torch.nonzero(
        bad,
        as_tuple=False,
    ).flatten()

    active_original_ranks = (
        torch.nonzero(
            state.active_ranks_cpu,
            as_tuple=False,
        )
        .flatten()
        .tolist()
    )

    logger.warning(
        "[NPU FT RUNTIME ROUTING] "
        "active_original_ranks=%s "
        "effective_ep_size=%d "
        "effective_num_physical=%d "
        "num_routes=%d "
        "logical_mapping_bad=%d "
        "original_range_bad=%d "
        "inactive_rank_bad=%d "
        "compact_bad=%d "
        "compact_range_bad=%d "
        "roundtrip_bad=%d "
        "total_bad=%d",
        active_original_ranks,
        effective_ep_size,
        effective_num_physical,
        int(valid.sum().item()),
        int(logical_mapping_bad.sum().item()),
        int(original_range_bad.sum().item()),
        int(inactive_rank_bad.sum().item()),
        int(compact_bad.sum().item()),
        int(compact_range_bad.sum().item()),
        int(roundtrip_bad.sum().item()),
        int(bad_indices.numel()),
    )

    if bad_indices.numel() > 0:
        sample = bad_indices[:16]

        logger.error(
            "[NPU FT RUNTIME ROUTING] mismatch sample: "
            "logical=%s "
            "original=%s "
            "original_rank=%s "
            "local_slot=%s "
            "effective_rank=%s "
            "compact=%s "
            "expected_compact=%s "
            "roundtrip_original=%s",
            logical[sample].detach().cpu().tolist(),
            original[sample].detach().cpu().tolist(),
            original_rank[sample].detach().cpu().tolist(),
            local_slot[sample].detach().cpu().tolist(),
            effective_rank[sample].detach().cpu().tolist(),
            compact[sample].detach().cpu().tolist(),
            expected_compact[sample].detach().cpu().tolist(),
            roundtrip_original[sample].detach().cpu().tolist(),
        )

        raise AssertionError(
            "NPU FT runtime expert routing is inconsistent "
            "with recovered EPLB/MC2 namespaces"
        )

    return True

def topk_ids_logical_to_physical(
    topk_ids: torch.Tensor,
    info: Optional[ExpertLocationDispatchInfo],
    log2phy_prob: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if info is None:
        return topk_ids

    if info.ep_dispatch_algorithm == "static":
        physical_topk_ids = _topk_ids_logical_to_physical_static(topk_ids, info)
    elif info.ep_dispatch_algorithm in ["dynamic", "fake"]:
        physical_topk_ids = _topk_ids_logical_to_physical_dynamic(topk_ids, info)
    elif info.ep_dispatch_algorithm == "lp":
        if log2phy_prob is None:
            raise RuntimeError(
                "ep_dispatch_algorithm='lp' but log2phy_prob is None at dispatch "
                f"time (topk_ids.shape={tuple(topk_ids.shape)})."
            )
        physical_topk_ids = _topk_ids_logical_to_physical_probability(
            topk_ids, info, log2phy_prob
        )
    else:
        raise NotImplementedError(f"Unknown algorithm {info.ep_dispatch_algorithm}")

    if info.npu_mc2_elastic_info is not None:
        from sglang.srt.elastic_ep.npu_mc2 import compact_mc2_physical_expert_ids

        # EPLB routing table produces physical IDs in the immutable
        # original-rank namespace.
        original_physical_topk_ids = physical_topk_ids

        dump_mc2_routing = (
            os.environ.get("SGLANG_NPU_FT_DUMP_MC2_ROUTING", "0") == "1"
            and torch.distributed.get_rank() == 3
        )
        dump_dir = "/home/hww/projects/sglang/artifacts/npu_ft/tensor_dump"

        if dump_mc2_routing:
            torch.ops.npu.save_npugraph_tensor(
                original_physical_topk_ids,
                save_path=f"{dump_dir}/mc2_routing_raw.pt",
            )
            torch.ops.npu.save_npugraph_tensor(
                info.npu_mc2_elastic_info,
                save_path=f"{dump_dir}/mc2_routing_elastic_info.pt",
            )

        # Translate original physical expert IDs into MC2's compact
        # survivor namespace.
        compact_physical_topk_ids = compact_mc2_physical_expert_ids(
            original_physical_topk_ids,
            elastic_info=info.npu_mc2_elastic_info,
            original_ep_size=info.npu_mc2_original_ep_size,
            num_local_physical_experts=info.npu_mc2_num_local_physical_experts,
        )

        if dump_mc2_routing:
            torch.ops.npu.save_npugraph_tensor(
                compact_physical_topk_ids,
                save_path=f"{dump_dir}/mc2_routing_compact.pt",
            )

        global _NPU_FT_RUNTIME_ROUTING_PROBED

        if (
            _VALIDATE_NPU_FT_RUNTIME_ROUTING
            and not _NPU_FT_RUNTIME_ROUTING_PROBED
        ):
            probed = _validate_npu_ft_runtime_routing_once(
                logical_topk_ids=topk_ids,
                original_physical_ids=original_physical_topk_ids,
                compact_physical_ids=compact_physical_topk_ids,
                info=info,
            )
            if probed:
                _NPU_FT_RUNTIME_ROUTING_PROBED = True

        physical_topk_ids = compact_physical_topk_ids

    return physical_topk_ids


def _topk_ids_logical_to_physical_static(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    physical_topk_ids = info.partial_logical_to_rank_dispatch_physical_map[topk_ids]
    if physical_topk_ids.dtype != topk_ids.dtype:
        physical_topk_ids = physical_topk_ids.to(topk_ids.dtype)
    return physical_topk_ids


def _topk_ids_logical_to_physical_dynamic(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    topk_ids_original_shape = topk_ids.shape
    original_dtype = topk_ids.dtype
    device = topk_ids.device
    topk_ids = topk_ids.flatten()

    chosen_dispatch_index = (
        torch.randint(0, 65536, topk_ids.shape, dtype=torch.int32, device=device)
        % info.partial_logical_to_all_physical_map_num_valid[topk_ids]
    )
    topk_ids = info.partial_logical_to_all_physical_map[topk_ids, chosen_dispatch_index]
    if topk_ids.dtype != original_dtype:
        topk_ids = topk_ids.to(original_dtype)

    topk_ids = topk_ids.view(topk_ids_original_shape)
    return topk_ids


def _topk_ids_logical_to_physical_probability(
    topk_ids: torch.Tensor,
    info: ExpertLocationDispatchInfo,
    log2phy_prob: torch.Tensor,
) -> torch.Tensor:
    """Select physical experts via the JIT-compiled CUDA dispatch kernel.

    Raises if ``topk_ids`` isn't on CUDA — the LP path requires the fused
    kernel and there is no torch reference fallback at runtime.
    """
    if not topk_ids.is_cuda:
        raise RuntimeError(
            "LP dispatch requires CUDA tensors; got topk_ids on " f"{topk_ids.device}."
        )
    from sglang.kernels.ops.lplb import cuda_solver

    return cuda_solver.dispatch_probability(
        topk_ids, log2phy_prob, info.partial_logical_to_all_physical_map
    )
