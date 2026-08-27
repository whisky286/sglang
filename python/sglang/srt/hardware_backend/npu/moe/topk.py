from typing import TYPE_CHECKING, Optional

import torch
from sgl_kernel_npu.norm.l1_norm import l1_norm

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import topk_ids_logical_to_physical
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    capture_routed_experts_if_allowed,
    select_experts,
)

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.moe.topk import TopKConfig, TopKOutput

import logging
from sglang.srt.utils import get_bool_env_var

_VALIDATE_NPU_FT_RUNTIME_ROUTING = get_bool_env_var(
    "SGLANG_VALIDATE_NPU_FT_RUNTIME_ROUTING"
)

_NPU_FT_RUNTIME_ROUTING_PROBED = False

logger = logging.getLogger(__name__)

def _apply_routed_scaling_after_renorm(
    topk_weights: torch.Tensor,
    topk_config: "TopKConfig",
) -> torch.Tensor:
    """Mirror GPU post-renorm scaling when apply_routed_scaling_factor_on_output is set."""
    if (
        topk_config.renormalize
        and topk_config.apply_routed_scaling_factor_on_output
        and topk_config.routed_scaling_factor is not None
    ):
        return topk_weights * topk_config.routed_scaling_factor
    return topk_weights


def fused_topk_npu(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: "TopKConfig",
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional["ExpertLocationDispatchInfo"] = None,
    layer_id: Optional[int] = None,
) -> "TopKOutput":

    use_grouped_topk = topk_config.use_grouped_topk
    renormalize = topk_config.renormalize
    correction_bias = topk_config.correction_bias

    # Fast path: simple top-k without grouped routing and bias
    if not use_grouped_topk and correction_bias is None:
        topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k_softmax(
            router_logits,
            k=topk_config.top_k,
        )

        if renormalize:
            topk_weights = l1_norm(
                topk_weights
                if topk_config.num_fused_shared_experts == 0
                else topk_weights[:, :-1]
            )
        topk_weights = topk_weights.to(torch.float32)

    # sqrtsoftplus (DSV4 noaux_tc): the NPU op only scores sigmoid/softmax, so use
    # a torch path. top-k over (scores + bias); weights from un-biased scores.
    elif topk_config.scoring_func == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(router_logits.float()).sqrt()
        scores_for_choice = (
            scores + correction_bias.unsqueeze(0).float()
            if correction_bias is not None
            else scores
        )
        _, topk_ids = torch.topk(
            scores_for_choice, k=topk_config.top_k, dim=-1, sorted=False
        )
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = scores.gather(1, topk_ids)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        else:
            topk_weights = topk_weights * topk_config.routed_scaling_factor
        topk_weights = topk_weights.to(torch.float32)

    # Support grouped top-k or correction bias or sigmoid or routed_scaling_factor
    elif (
        correction_bias is not None
        or topk_config.scoring_func == "sigmoid"
        or num_token_non_padded is not None
    ):
        topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k(
            router_logits.to(torch.float32),
            k=topk_config.top_k,
            bias=(
                correction_bias.to(torch.float32)
                if correction_bias is not None
                else None
            ),
            # num_expert_group and topk_group in some topk_config without group is None, (not supported by this ops)
            k_group=topk_config.topk_group if use_grouped_topk else 1,
            group_count=topk_config.num_expert_group if use_grouped_topk else 1,
            group_select_mode=(1 if use_grouped_topk else 0),
            renorm=0,
            # 1 for sigmoid, 0 for softmax
            norm_type=1,
            routed_scaling_factor=(
                topk_config.routed_scaling_factor
                if topk_config.apply_routed_scaling_factor_on_output
                else 1
            ),
            eps=float(1e-20),
        )
        topk_weights = topk_weights.to(torch.float32)

    # torch native is not yet supported num_token_non_padded
    # Fallback to torch native implementation
    else:
        topk_config.torch_native = True
        return select_experts(
            hidden_states=hidden_states,
            layer_id=layer_id,
            router_logits=router_logits,
            topk_config=topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    global _NPU_FT_RUNTIME_ROUTING_PROBED

    probe_this_forward = (
        _VALIDATE_NPU_FT_RUNTIME_ROUTING
        and not _NPU_FT_RUNTIME_ROUTING_PROBED
        and layer_id == 0
        and expert_location_dispatch_info is not None
        and topk_ids.numel() > 0
    )
    logical_topk_ids_for_probe = topk_ids.clone() if probe_this_forward else None
    if expert_location_dispatch_info is not None:
        topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    if probe_this_forward:
        _validate_npu_ft_runtime_routing(
            logical_topk_ids_for_probe,
            topk_ids,
            expert_location_dispatch_info,
            layer_id,
        )
        _NPU_FT_RUNTIME_ROUTING_PROBED = True
    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)
    capture_routed_experts_if_allowed(topk_config, layer_id, topk_ids)

    return StandardTopKOutput(topk_weights, topk_ids, router_logits)

def _validate_npu_ft_runtime_routing(
    logical_topk_ids: torch.Tensor,
    compact_topk_ids: torch.Tensor,
    expert_location_dispatch_info,
    layer_id: int,
):
    from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
    from sglang.srt.elastic_ep.npu_mc2 import (
        MC2_ELASTIC_INFO_HEADER_SIZE,
    )
    from sglang.srt.eplb.expert_location import (
        get_global_expert_location_metadata,
    )

    state = ElasticEPStateManager.instance()
    if (
        state is None
        or state.active_ranks_cpu is None
        or bool(state.active_ranks_cpu.all().item())
    ):
        return

    info = expert_location_dispatch_info
    elastic_info = info.npu_mc2_elastic_info

    if elastic_info is None:
        raise RuntimeError(
            "runtime routing probe requires NPU MC2 elastic_info"
        )

    original_ep_size = info.npu_mc2_original_ep_size
    num_local = info.npu_mc2_num_local_physical_experts

    effective_ep_size = int(elastic_info[1].item())
    effective_num_physical = int(elastic_info[3].item())

    # First verify that MC2's compact-rank order is the same as
    # the rebuilt survivor rank order.
    active_original_ranks = (
        torch.nonzero(
            state.active_ranks_cpu,
            as_tuple=False,
        )
        .flatten()
        .tolist()
    )

    effective_to_original = elastic_info[
        MC2_ELASTIC_INFO_HEADER_SIZE + original_ep_size :
        MC2_ELASTIC_INFO_HEADER_SIZE + 2 * original_ep_size
    ]

    actual_effective_to_original = (
        effective_to_original[:effective_ep_size]
        .detach()
        .cpu()
        .tolist()
    )

    if actual_effective_to_original != active_original_ranks:
        raise AssertionError(
            "NPU FT effective-rank namespace mismatch: "
            f"active_original_ranks={active_original_ranks} "
            f"effective_to_original={actual_effective_to_original}"
        )

    logical = logical_topk_ids.reshape(-1)
    compact = compact_topk_ids.reshape(-1)

    valid = logical >= 0

    # Check compact IDs themselves are legal for the shrunken EP.
    compact_range_bad = valid & (
        (compact < 0)
        | (compact >= effective_num_physical)
    )

    safe_compact = compact.masked_fill(~valid, 0)

    effective_rank = torch.div(
        safe_compact,
        num_local,
        rounding_mode="floor",
    )

    # Avoid illegal indexing while collecting diagnostics.
    safe_effective_rank = effective_rank.clamp(
        min=0,
        max=max(effective_ep_size - 1, 0),
    )

    original_rank = effective_to_original[
        safe_effective_rank.long()
    ].to(compact.dtype)

    local_expert_id = safe_compact % num_local

    original_physical_id = (
        original_rank * num_local + local_expert_id
    )

    metadata = get_global_expert_location_metadata()

    routed_logical = metadata.physical_to_logical_map[
        layer_id,
        original_physical_id.long(),
    ].to(logical.dtype)

    mapping_bad = valid & (routed_logical != logical)

    bad = compact_range_bad | mapping_bad
    bad_indices = torch.nonzero(
        bad,
        as_tuple=False,
    ).flatten()

    logger.warning(
        "[NPU FT RUNTIME ROUTING] "
        "layer=%d "
        "active_original_ranks=%s "
        "effective_to_original=%s "
        "effective_ep_size=%d "
        "effective_num_physical=%d "
        "logical_min=%d logical_max=%d "
        "compact_min=%d compact_max=%d "
        "routing_mismatch_count=%d "
        "num_routes=%d",
        layer_id,
        active_original_ranks,
        actual_effective_to_original,
        effective_ep_size,
        effective_num_physical,
        int(logical[valid].min().item()) if bool(valid.any()) else -1,
        int(logical[valid].max().item()) if bool(valid.any()) else -1,
        int(compact[valid].min().item()) if bool(valid.any()) else -1,
        int(compact[valid].max().item()) if bool(valid.any()) else -1,
        int(bad_indices.numel()),
        int(valid.sum().item()),
    )

    if bad_indices.numel() > 0:
        sample = bad_indices[:16]

        logger.error(
            "[NPU FT RUNTIME ROUTING] mismatch sample: "
            "logical=%s compact=%s "
            "original_rank=%s "
            "local_slot=%s "
            "original_physical=%s "
            "physical_map_logical=%s",
            logical[sample].detach().cpu().tolist(),
            compact[sample].detach().cpu().tolist(),
            original_rank[sample].detach().cpu().tolist(),
            local_expert_id[sample].detach().cpu().tolist(),
            original_physical_id[sample].detach().cpu().tolist(),
            routed_logical[sample].detach().cpu().tolist(),
        )

        raise AssertionError(
            "NPU FT runtime expert routing does not match "
            "the recovered expert layout"
        )