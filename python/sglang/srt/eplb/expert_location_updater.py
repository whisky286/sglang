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
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import einops
import torch
import torch.distributed
from torch.distributed import P2POp

from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    get_global_expert_location_metadata,
)
from sglang.srt.runtime_context import get_server_args
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)


_LOG_INPUT = get_bool_env_var("SGLANG_EXPERT_LOCATION_UPDATER_LOG_INPUT")
_LOG_P2P_SCHEDULE = get_bool_env_var(
    "SGLANG_EXPERT_LOCATION_UPDATER_LOG_P2P_SCHEDULE"
)

_VALIDATE_NPU_NZ_EXPERT_COPY = get_bool_env_var(
    "SGLANG_VALIDATE_NPU_NZ_EXPERT_COPY"
)

_NPU_NZ_COPY_VALIDATE_COUNT = 0

_VALIDATE_NPU_P2P_NAN = get_bool_env_var(
    "SGLANG_VALIDATE_NPU_P2P_NAN"
)

_NPU_P2P_NAN_LAYER = int(
    os.environ.get("SGLANG_NPU_P2P_NAN_LAYER", "0")
)


class ExpertLocationUpdater:
    def __init__(self):
        self._first_execution = True

    def update(
        self,
        routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: List[int],
        nnodes: int,
        rank: int,
        survivor_process_groups=None,
    ):
        """
        Update experts' physical location after EPLB.

        Returns a map of layer_id to expert_ids that are missing due to rank
        failures during fault conditions when elastic EP is enabled.
        """
        if self._first_execution:
            self._first_execution = False
            torch.get_device_module().empty_cache()

        old_expert_location_metadata = get_global_expert_location_metadata()
        assert old_expert_location_metadata is not None

        missing_logical_experts_by_layers = _update_expert_weights(
            routed_experts_weights_of_layer=routed_experts_weights_of_layer,
            old_expert_location_metadata=old_expert_location_metadata,
            new_expert_location_metadata=new_expert_location_metadata,
            update_layer_ids=update_layer_ids,
            nnodes=nnodes,
            rank=rank,
            survivor_process_groups=survivor_process_groups,
        )
        old_expert_location_metadata.update(
            new_expert_location_metadata,
            update_layer_ids=update_layer_ids,
        )

        return missing_logical_experts_by_layers


def _update_expert_weights(**kwargs):
    if get_bool_env_var("SGLANG_EXPERT_LOCATION_UPDATER_CANARY"):
        return _update_expert_weights_with_canary(**kwargs)
    else:
        return _update_expert_weights_raw(**kwargs)


# can add watchdog as well
def _update_expert_weights_with_canary(
    routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
    old_expert_location_metadata: ExpertLocationMetadata,
    new_expert_location_metadata: ExpertLocationMetadata,
    update_layer_ids: List[int],
    nnodes: int,
    rank: int,
    survivor_process_groups=None,
):
    num_local_physical_experts = old_expert_location_metadata.num_local_physical_experts

    def _get_canary_value(meta: ExpertLocationMetadata, layer_id: int):
        return meta.physical_to_logical_map_cpu[
            layer_id,
            num_local_physical_experts * rank : num_local_physical_experts * (rank + 1),
        ]

    routed_experts_weights_of_layer = {
        k: [x for x in v] for k, v in routed_experts_weights_of_layer.items()
    }
    for layer_id in update_layer_ids:
        canary_tensor = (
            _get_canary_value(old_expert_location_metadata, layer_id)
            .clone()
            .to(device=get_server_args().device, non_blocking=True)
        )
        routed_experts_weights_of_layer[layer_id].append(canary_tensor)

    missing_logical_experts_by_layers = _update_expert_weights_raw(
        routed_experts_weights_of_layer=routed_experts_weights_of_layer,
        old_expert_location_metadata=old_expert_location_metadata,
        new_expert_location_metadata=new_expert_location_metadata,
        update_layer_ids=update_layer_ids,
        nnodes=nnodes,
        rank=rank,
        survivor_process_groups=survivor_process_groups,
    )

    for layer_id in update_layer_ids:
        expect_value = _get_canary_value(new_expert_location_metadata, layer_id)
        actual_value = routed_experts_weights_of_layer[layer_id][-1].cpu()
        _validate_survivor_movement_canary(
            expect_value=expect_value,
            actual_value=actual_value,
            missing_logical_experts=missing_logical_experts_by_layers.get(
                layer_id, []
            ),
            layer_id=layer_id,
        )

    return missing_logical_experts_by_layers


def _validate_survivor_movement_canary(
    *,
    expect_value: torch.Tensor,
    actual_value: torch.Tensor,
    missing_logical_experts: List[int],
    layer_id: int,
) -> None:
    """Validate movement completed before the fallback weight reload.

    Experts absent from every survivor are only recorded by the raw updater;
    their DRAM/checkpoint load happens after this function returns.  Exclude
    exactly those destination slots from the movement canary instead of
    treating the intentionally deferred load as a P2P-copy failure.
    """

    expect_value = expect_value.detach().to(device="cpu")
    actual_value = actual_value.detach().to(device="cpu")
    if expect_value.shape != actual_value.shape:
        raise AssertionError(
            "expert recovery canary shape mismatch: "
            f"layer={layer_id} expected={tuple(expect_value.shape)} "
            f"actual={tuple(actual_value.shape)}"
        )

    pending_reload = set(missing_logical_experts)
    validate_mask = torch.tensor(
        [
            int(logical_id) not in pending_reload
            for logical_id in expect_value.tolist()
        ],
        dtype=torch.bool,
    )
    mismatch_slots = torch.nonzero(
        validate_mask & (expect_value != actual_value),
        as_tuple=False,
    ).flatten()
    if mismatch_slots.numel() == 0:
        return

    mismatch_slot_values = mismatch_slots.tolist()
    raise AssertionError(
        "expert recovery survivor-movement canary mismatch: "
        f"layer={layer_id} local_slots={mismatch_slot_values} "
        f"expected={expect_value[mismatch_slots].tolist()} "
        f"actual={actual_value[mismatch_slots].tolist()} "
        f"pending_reload_experts={sorted(pending_reload)}"
    )


def _update_expert_weights_raw(
    routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
    old_expert_location_metadata: ExpertLocationMetadata,
    new_expert_location_metadata: ExpertLocationMetadata,
    update_layer_ids: List[int],
    nnodes: int,
    rank: int,
    survivor_process_groups=None,
):
    log_metrics = get_bool_env_var("SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS")

    temp_buffers = create_temp_buffers(
        routed_experts_weights_of_layer[update_layer_ids[0]]
    )

    world_size = torch.distributed.get_world_size()
    num_local_physical_experts = old_expert_location_metadata.num_local_physical_experts
    num_gpu_per_node = world_size // nnodes

    missing_logical_experts_by_layers: Dict[int, List[int]] = {}

    for layer_id in update_layer_ids:
        missing_logical_experts_info: List[int] = []
        update_expert_weights_single_layer(
            routed_experts_weights=routed_experts_weights_of_layer[layer_id],
            temp_buffers=temp_buffers,
            old_physical_to_logical_map=old_expert_location_metadata.physical_to_logical_map_cpu[
                layer_id
            ].tolist(),
            new_physical_to_logical_map=new_expert_location_metadata.physical_to_logical_map_cpu[
                layer_id
            ].tolist(),
            num_local_physical_experts=num_local_physical_experts,
            num_gpu_per_node=num_gpu_per_node,
            rank=rank,
            world_size=world_size,
            missing_logical_experts_info=missing_logical_experts_info,
            survivor_process_groups=survivor_process_groups,
            log_metrics=log_metrics,
            layer_id=layer_id,
        )
        if len(missing_logical_experts_info) > 0:
            missing_logical_experts_by_layers[layer_id] = list(
                dict.fromkeys(missing_logical_experts_info)
            )
    return missing_logical_experts_by_layers


def create_temp_buffers(sample_tensors):
    return [torch.empty_like(tensor) for tensor in sample_tensors]


def _copy_expert_tensor_(
    destination_tensor: torch.Tensor, source_tensor: torch.Tensor, *, debug_context: str = ""
) -> None:
    """Copy an expert while preserving an NPU internal-format slot layout."""

    global _NPU_NZ_COPY_VALIDATE_COUNT

    if destination_tensor.device.type == "npu":
        import torch_npu
        from sglang.srt.hardware_backend.npu.utils import (
            copy_npu_formatted_tensor_,
            is_npu_internal_format_tensor,
        )

        supports_offset_zero_alias = destination_tensor.dtype in (
            torch.float16,
            torch.float32,
            torch.bfloat16,
        )

        is_nz_copy = (
            supports_offset_zero_alias
            and is_npu_internal_format_tensor(destination_tensor)
        )

        if is_nz_copy:
            src_format = torch_npu.get_npu_format(source_tensor)
            dst_format = torch_npu.get_npu_format(destination_tensor)

            source_snapshot = None

            should_validate = (
                _VALIDATE_NPU_NZ_EXPERT_COPY
                and debug_context.startswith("p2p_recv_stage")
                and _NPU_NZ_COPY_VALIDATE_COUNT < 20
            )

            if should_validate:
                # Snapshot BEFORE the copy. This is important: if the copy
                # accidentally overwrites the source region as well, comparing
                # source and destination only after the copy could hide it.
                try:
                    source_snapshot = source_tensor.detach().cpu()
                except Exception:
                    logger.exception(
                        "[NPU FT NZ COPY] source read failed BEFORE copy: "
                        "context=%s "
                        "shape=%s dtype=%s "
                        "src_format=%s src_ptr=%d src_offset=%d "
                        "dst_format=%s dst_ptr=%d dst_offset=%d",
                        debug_context,
                        tuple(source_tensor.shape),
                        source_tensor.dtype,
                        src_format,
                        source_tensor.data_ptr(),
                        source_tensor.storage_offset(),
                        dst_format,
                        destination_tensor.data_ptr(),
                        destination_tensor.storage_offset(),
                    )
                    raise

            try:
                copy_npu_formatted_tensor_(
                    destination_tensor,
                    source_tensor,
                )
            except Exception:
                logger.exception(
                    "[NPU FT NZ COPY] formatted copy itself failed: "
                    "context=%s "
                    "shape=%s dtype=%s "
                    "src_format=%s src_ptr=%d src_offset=%d "
                    "dst_format=%s dst_ptr=%d dst_offset=%d",
                    debug_context,
                    tuple(source_tensor.shape),
                    source_tensor.dtype,
                    src_format,
                    source_tensor.data_ptr(),
                    source_tensor.storage_offset(),
                    dst_format,
                    destination_tensor.data_ptr(),
                    destination_tensor.storage_offset(),
                )
                raise

            if should_validate:
                try:
                    destination_snapshot = destination_tensor.detach().cpu()
                except Exception:
                    logger.exception(
                        "[NPU FT NZ COPY] destination read failed AFTER copy: "
                        "context=%s "
                        "shape=%s dtype=%s "
                        "src_format=%s src_ptr=%d src_offset=%d "
                        "dst_format=%s dst_ptr=%d dst_offset=%d",
                        debug_context,
                        tuple(destination_tensor.shape),
                        destination_tensor.dtype,
                        src_format,
                        source_tensor.data_ptr(),
                        source_tensor.storage_offset(),
                        dst_format,
                        destination_tensor.data_ptr(),
                        destination_tensor.storage_offset(),
                    )
                    raise

                src_flat = source_snapshot.reshape(-1)
                dst_flat = destination_snapshot.reshape(-1)

                is_float = source_snapshot.is_floating_point()

                # 1. Compute mismatch mask.
                # Treat NaN at the same position as equal for copy validation.
                if is_float:
                    src_nan = torch.isnan(src_flat)
                    dst_nan = torch.isnan(dst_flat)

                    both_nan = src_nan & dst_nan
                    value_equal = src_flat == dst_flat

                    mismatch_mask = ~(value_equal | both_nan)

                    src_nan_count = int(src_nan.sum().item())
                    dst_nan_count = int(dst_nan.sum().item())
                else:
                    mismatch_mask = src_flat != dst_flat

                    # These must still be defined for the success log below.
                    src_nan_count = 0
                    dst_nan_count = 0

                mismatch_indices = torch.nonzero(
                    mismatch_mask,
                    as_tuple=False,
                ).flatten()

                # 2. Compute max_abs_diff for logging.
                # Ignore NaN / Inf pairs here.
                if is_float:
                    finite_pair_mask = (
                        torch.isfinite(src_flat)
                        & torch.isfinite(dst_flat)
                    )
                else:
                    finite_pair_mask = torch.ones_like(
                        src_flat,
                        dtype=torch.bool,
                    )

                if bool(finite_pair_mask.any().item()):
                    max_abs_diff = float(
                        (
                            src_flat[finite_pair_mask].float()
                            - dst_flat[finite_pair_mask].float()
                        )
                        .abs()
                        .max()
                        .item()
                    )
                else:
                    max_abs_diff = float("nan")

                # 3. Fail only on a real mismatch.
                if mismatch_indices.numel() > 0:
                    mismatch_count = int(mismatch_indices.numel())
                    first_mismatch = int(mismatch_indices[0].item())

                    src_value = src_flat[first_mismatch].item()
                    dst_value = dst_flat[first_mismatch].item()

                    logger.error(
                        "[NPU FT NZ COPY] VALIDATION FAILED: "
                        "context=%s "
                        "shape=%s dtype=%s "
                        "src_format=%s src_ptr=%d src_offset=%d "
                        "dst_format=%s dst_ptr=%d dst_offset=%d "
                        "src_nan_count=%d dst_nan_count=%d "
                        "mismatch_count=%d first_mismatch=%d "
                        "src_value=%s dst_value=%s "
                        "max_abs_diff=%s",
                        debug_context,
                        tuple(source_tensor.shape),
                        source_tensor.dtype,
                        src_format,
                        source_tensor.data_ptr(),
                        source_tensor.storage_offset(),
                        dst_format,
                        destination_tensor.data_ptr(),
                        destination_tensor.storage_offset(),
                        src_nan_count,
                        dst_nan_count,
                        mismatch_count,
                        first_mismatch,
                        src_value,
                        dst_value,
                        max_abs_diff,
                    )

                    raise AssertionError(
                        "NPU FT NZ expert copy validation failed: "
                        f"context={debug_context} "
                        f"src_offset={source_tensor.storage_offset()} "
                        f"dst_offset={destination_tensor.storage_offset()} "
                        f"mismatch_count={mismatch_count} "
                        f"first_mismatch={first_mismatch}"
                    )

                # 4. Success path.
                _NPU_NZ_COPY_VALIDATE_COUNT += 1

                if (
                    _NPU_NZ_COPY_VALIDATE_COUNT <= 10
                    or _NPU_NZ_COPY_VALIDATE_COUNT % 100 == 0
                ):
                    logger.warning(
                        "[NPU FT NZ COPY] validation success: "
                        "count=%d context=%s "
                        "shape=%s dtype=%s "
                        "src_offset=%d dst_offset=%d "
                        "src_nan_count=%d dst_nan_count=%d "
                        "max_abs_diff=%s",
                        _NPU_NZ_COPY_VALIDATE_COUNT,
                        debug_context,
                        tuple(source_tensor.shape),
                        source_tensor.dtype,
                        source_tensor.storage_offset(),
                        destination_tensor.storage_offset(),
                        src_nan_count,
                        dst_nan_count,
                        max_abs_diff,
                    )

            return

    destination_tensor.copy_(source_tensor)


def _needs_npu_p2p_staging(tensor: torch.Tensor) -> bool:
    if tensor.device.type != "npu":
        return False

    from sglang.srt.hardware_backend.npu.utils import is_npu_internal_format_tensor

    return (
        tensor.storage_offset() != 0
        or is_npu_internal_format_tensor(tensor)
    )

def _debug_npu_p2p_nan_count(tensor: torch.Tensor) -> int:
    if not tensor.is_floating_point():
        return 0

    return int(torch.isnan(tensor).sum().item())

def _stage_npu_p2p_ops(
    p2p_ops: List[P2POp],
) -> Tuple[List[P2POp], List[Tuple[torch.Tensor, torch.Tensor, str]]]:
    """Give HCCL offset-zero buffers for internal-format tensor views.

    Expert weights are stored with the local-expert dimension first, so
    selecting any expert after index 0 usually returns a view with a non-zero
    storage offset. HCCL rejects such views when their underlying NPU tensor
    uses an internal format. Stage only those NPU views; CUDA/ROCm retain the
    existing zero-copy behavior.
    """

    staged_ops = []
    recv_copy_infos = []
    staged_send_tensors = {}
    for op in p2p_ops:
        tensor = op.tensor
        if not _needs_npu_p2p_staging(tensor):
            staged_ops.append(op)
            continue

        if op.op == torch.distributed.irecv:
            staged_tensor = _new_npu_nd_staging_like(tensor)
            recv_copy_infos.append((staged_tensor, tensor))
        elif op.op == torch.distributed.isend:
            # A single expert tensor can be sent to multiple peers. Reuse one
            # offset-zero clone instead of multiplying the memory peak by the
            # destination fanout.
            send_key = (
                tensor.device,
                tensor.data_ptr(),
                tensor.storage_offset(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.dtype,
            )
            staged_tensor = staged_send_tensors.get(send_key)
            if staged_tensor is None:
                staged_tensor = _new_npu_nd_staging_like(tensor)
                # ND destination <- logical NZ source.
                # Here we intentionally want format conversion, not raw formatted-byte copy.
                staged_tensor.copy_(tensor)
                staged_send_tensors[send_key] = staged_tensor
        else:
            raise ValueError(f"Unsupported P2P operation: {op.op}")

        if staged_tensor.storage_offset() != 0:
            raise RuntimeError(
                "Failed to create an offset-zero NPU staging tensor for EPLB P2P."
            )

        staged_ops.append(
            P2POp(
                op=op.op,
                tensor=staged_tensor,
                peer=op.peer,
                group=op.group,
                tag=op.tag,
            )
        )

    return staged_ops, recv_copy_infos

def _new_npu_nd_staging_like(tensor: torch.Tensor) -> torch.Tensor:
    import torch_npu

    from sglang.srt.hardware_backend.npu.utils import NPUACLFormat

    staged = torch_npu.empty_with_format(
        tuple(tensor.shape),
        dtype=tensor.dtype,
        device=tensor.device,
        acl_format=int(NPUACLFormat.ACL_FORMAT_ND),
    )

    if staged.storage_offset() != 0:
        raise RuntimeError(
            "NPU EPLB ND staging tensor must have storage_offset=0"
        )

    actual_format = torch_npu.get_npu_format(staged)
    if actual_format != int(NPUACLFormat.ACL_FORMAT_ND):
        raise RuntimeError(
            f"NPU EPLB staging tensor is not ND: format={actual_format}"
        )

    return staged


def _copy_staged_p2p_recvs(
    recv_copy_infos: List[Tuple[torch.Tensor, torch.Tensor]],
):
    for staged_tensor, destination_tensor in recv_copy_infos:
        if destination_tensor.device.type == "npu":
            import torch_npu

            from sglang.srt.hardware_backend.npu.utils import (
                NPUACLFormat,
                is_npu_internal_format_tensor,
            )

            if is_npu_internal_format_tensor(destination_tensor):
                dst_format = torch_npu.get_npu_format(
                    destination_tensor
                )

                # HCCL received into ND.
                # Convert the complete offset-zero expert tensor back to
                # destination's physical format first.
                formatted_staged_tensor = torch.ops.npu.npu_format_cast(
                    staged_tensor,
                    dst_format,
                )

                # Now both tensors have the same physical format.
                # Existing formatted-copy implementation safely updates
                # the non-zero-offset destination block.
                _copy_expert_tensor_(
                    destination_tensor,
                    formatted_staged_tensor,
                )
                continue

        _copy_expert_tensor_(
            destination_tensor,
            staged_tensor,
        )


def update_expert_weights_single_layer(
    routed_experts_weights: List[torch.Tensor],
    temp_buffers: List[torch.Tensor],
    old_physical_to_logical_map: List[int],  # (num_physical_Experts,)
    new_physical_to_logical_map: List[int],  # (num_physical_Experts,)
    num_local_physical_experts: int,
    num_gpu_per_node: int,
    rank: int,
    world_size: Optional[int] = None,
    missing_logical_experts_info: Optional[List[int]] = None,
    survivor_process_groups=None,
    debug: bool = False,
    log_metrics: bool = False,
    layer_id: Optional[int] = None,
):
    assert all(
        tensor.shape[0] == num_local_physical_experts
        for tensor in routed_experts_weights
    ), f"{num_local_physical_experts=} {[x.shape for x in routed_experts_weights]=}"
    assert isinstance(old_physical_to_logical_map, list)
    assert isinstance(new_physical_to_logical_map, list)

    if _LOG_INPUT:
        logger.info(
            "update_expert_weights_single_layer "
            f"{[x.shape for x in routed_experts_weights]=} "
            f"{[x.shape for x in temp_buffers]=} "
            f"{old_physical_to_logical_map=} "
            f"{new_physical_to_logical_map=} "
            f"{num_local_physical_experts=} "
            f"{num_gpu_per_node=} "
            f"{rank=} "
            f"{world_size=} "
        )

    output_logs = [] if debug else None

    num_physical_experts = len(old_physical_to_logical_map)
    num_tensors = len(routed_experts_weights)
    if world_size is None:
        world_size = num_physical_experts // num_local_physical_experts

    if survivor_process_groups is None:
        active_rank_mask = [True] * world_size
        p2p_group = None

        def _to_peer_rank(original_rank: int) -> int:
            return original_rank

        def _from_peer_rank(peer_rank: int) -> int:
            return peer_rank

    else:
        active_rank_mask = [
            original_rank in survivor_process_groups.active_original_ranks
            for original_rank in range(world_size)
        ]
        p2p_group = survivor_process_groups.eplb_device_group

        def _to_peer_rank(original_rank: int) -> int:
            return survivor_process_groups.group_rank(original_rank)

        def _from_peer_rank(peer_rank: int) -> int:
            return int(
                survivor_process_groups.active_original_ranks[
                    peer_rank
                ]
            )

    def _rank_is_active(original_rank: int) -> bool:
        return active_rank_mask[original_rank]

    self_node_id = rank // num_gpu_per_node

    local_expert_location_range = (
        rank * num_local_physical_experts,
        (rank + 1) * num_local_physical_experts,
    )

    def _entrypoint():
        # List[Tuple[logical_expert_id, List[P2POp]]]
        p2p_op_infos: List[Tuple[int, List[P2POp]]] = []
        # List[Tuple[temp_buffers_expert_location, routed_experts_weights_expert_location]]
        buffer2weight_copy_infos: List[Tuple[int, int]] = []

        _handle_recv(buffer2weight_copy_infos, p2p_op_infos)
        _create_isend_ops(p2p_op_infos)
        _filter_p2p_ops(p2p_op_infos)
        _execute_p2p_ops(p2p_op_infos)
        _execute_buffer2weight_copies(buffer2weight_copy_infos)

        if log_metrics:
            _log_p2p_op_metrics(
                p2p_op_infos,
                world_size=world_size,
                num_gpu_per_node=num_gpu_per_node,
                self_node_id=self_node_id,
            )

        if debug:
            output_logs.append(f"{p2p_op_infos=}")
            output_logs.append(f"{buffer2weight_copy_infos=}")

    def _handle_recv(buffer2weight_copy_infos, p2p_op_infos):
        for dst_expert_location in range(*local_expert_location_range):
            _handle_recv_of_dst_expert_location(
                dst_expert_location, buffer2weight_copy_infos, p2p_op_infos
            )

    def _handle_recv_of_dst_expert_location(
        dst_expert_location: int, buffer2weight_copy_infos, p2p_op_infos
    ):
        logical_expert_id = new_physical_to_logical_map[dst_expert_location]

        # case 1: unchanged
        if old_physical_to_logical_map[dst_expert_location] == logical_expert_id:
            if debug:
                output_logs.append(
                    f"handle_recv_of_dst_expert_location {dst_expert_location=} case=unchanged"
                )
            return

        # case 2: same-gpu
        for src_expert_location in range(*local_expert_location_range):
            if old_physical_to_logical_map[src_expert_location] == logical_expert_id:
                for i in range(num_tensors):
                    _copy_expert_tensor_(
                        _get_tensor(temp_buffers, i, dst_expert_location),
                        _get_tensor(routed_experts_weights, i, src_expert_location),
                        debug_context=(
                            f"same_gpu_to_temp "
                            f"layer={layer_id} "
                            f"logical_expert={logical_expert_id} "
                            f"tensor_idx={i} "
                            f"src_global_slot={src_expert_location} "
                            f"dst_global_slot={dst_expert_location}"
                        ),
                    )
                buffer2weight_copy_infos.append(
                    (dst_expert_location, dst_expert_location)
                )
                if debug:
                    output_logs.append(
                        f"handle_recv_of_dst_expert_location {dst_expert_location=} case=same-gpu {src_expert_location=}"
                    )
                return

        # case 3: free-rider
        for src_expert_location in range(
            rank * num_local_physical_experts, dst_expert_location
        ):
            if new_physical_to_logical_map[src_expert_location] == logical_expert_id:
                buffer2weight_copy_infos.append(
                    (src_expert_location, dst_expert_location)
                )
                if debug:
                    output_logs.append(
                        f"handle_recv_of_dst_expert_location {dst_expert_location=} case=free-rider {src_expert_location=}"
                    )
                return

        same_node_mapping, cross_node_mapping, need_comm_self_node_dst_ranks = (
            _compute_comm_info(logical_expert_id=logical_expert_id)
        )

        # No surviving rank has this logical expert.  Only this case reaches
        # the DRAM/disk recovery loader; local reuse and survivor P2P were both
        # exhausted first.
        if not same_node_mapping.chunk_values and not cross_node_mapping.chunk_values:
            if missing_logical_experts_info is None:
                raise RuntimeError("missing expert recovery requires tracking")
            missing_logical_experts_info.append(logical_expert_id)
            if debug:
                output_logs.append(
                    "handle_recv_of_dst_expert_location "
                    f"{dst_expert_location=} case=reload-no-survivor-copy"
                )
            return

        # case 4: same-node
        if rank in need_comm_self_node_dst_ranks:
            chosen_src_rank = same_node_mapping.chunk_value_from_element_value(
                element_value=rank
            )
            _create_p2p_recv_and_buffer2weight_copy(
                buffer2weight_copy_infos,
                p2p_op_infos,
                src_rank=chosen_src_rank,
                logical_expert_id=logical_expert_id,
                dst_expert_location=dst_expert_location,
            )
            if debug:
                output_logs.append(
                    f"handle_recv_of_dst_expert_location {dst_expert_location=} case=same-node {chosen_src_rank=}"
                )
            return

        # case 5: cross-node
        # Future work: can optimize when there are multiple ranks in the same dst node that uses the same logical expert
        chosen_src_rank = cross_node_mapping.chunk_value_from_element_value(
            element_value=rank
        )
        _create_p2p_recv_and_buffer2weight_copy(
            buffer2weight_copy_infos,
            p2p_op_infos,
            src_rank=chosen_src_rank,
            logical_expert_id=logical_expert_id,
            dst_expert_location=dst_expert_location,
        )
        if debug:
            output_logs.append(
                f"handle_recv_of_dst_expert_location {dst_expert_location=} case=cross-node {chosen_src_rank=}"
            )
        return

    def _create_p2p_recv_and_buffer2weight_copy(
        buffer2weight_copy_infos,
        p2p_op_infos,
        *,
        logical_expert_id: int,
        src_rank: int,
        dst_expert_location: int,
    ):
        p2p_op_infos.append(
            (
                logical_expert_id,
                [
                    P2POp(
                        op=torch.distributed.irecv,
                        tensor=_get_tensor(temp_buffers, i, dst_expert_location),
                        peer=_to_peer_rank(src_rank),
                        group=p2p_group,
                    )
                    for i in range(num_tensors)
                ],
            )
        )
        buffer2weight_copy_infos.append((dst_expert_location, dst_expert_location))

    def _create_isend_ops(p2p_op_infos):
        handled_logical_expert_ids = set()
        for src_expert_location in range(*local_expert_location_range):
            logical_expert_id = old_physical_to_logical_map[src_expert_location]

            if logical_expert_id in handled_logical_expert_ids:
                continue
            handled_logical_expert_ids.add(logical_expert_id)

            _create_isend_ops_of_logical_expert_id(
                logical_expert_id, src_expert_location, p2p_op_infos
            )

    def _create_isend_ops_of_logical_expert_id(
        logical_expert_id, src_expert_location, p2p_op_infos
    ):
        same_node_mapping, cross_node_mapping, need_comm_self_node_dst_ranks = (
            _compute_comm_info(logical_expert_id=logical_expert_id)
        )

        same_node_dst_ranks = same_node_mapping.element_values_from_chunk_value(
            chunk_value=rank
        )
        cross_node_dst_ranks = cross_node_mapping.element_values_from_chunk_value(
            chunk_value=rank
        )
        all_dst_ranks = same_node_dst_ranks + cross_node_dst_ranks

        if debug:
            output_logs.append(
                f"create_isend_ops_of_logical_expert_id {logical_expert_id=} {src_expert_location=} {same_node_dst_ranks=} {cross_node_dst_ranks=}"
            )

        p2p_op_infos.append(
            (
                logical_expert_id,
                [
                    P2POp(
                        op=torch.distributed.isend,
                        tensor=_get_tensor(
                            routed_experts_weights, i, src_expert_location
                        ),
                        peer=_to_peer_rank(dst_rank),
                        group=p2p_group,
                    )
                    for dst_rank in all_dst_ranks
                    for i in range(num_tensors)
                ],
            )
        )

    def _compute_comm_info(logical_expert_id: int):
        all_src_ranks = _deduplicate_ordered(
            [
                x // num_local_physical_experts
                for x in range(num_physical_experts)
                if old_physical_to_logical_map[x] == logical_expert_id
                and _rank_is_active(x // num_local_physical_experts)
            ]
        )
        all_src_nodes = [x // num_gpu_per_node for x in all_src_ranks]
        self_node_src_ranks = [
            x for x in all_src_ranks if x // num_gpu_per_node == self_node_id
        ]

        need_comm_dst_ranks = _deduplicate_ordered(
            [
                x // num_local_physical_experts
                for x in range(num_physical_experts)
                if new_physical_to_logical_map[x] == logical_expert_id
                and _rank_is_active(x // num_local_physical_experts)
                and x // num_local_physical_experts not in all_src_ranks
            ]
        )
        need_comm_self_node_dst_ranks = (
            [x for x in need_comm_dst_ranks if x // num_gpu_per_node == self_node_id]
            if len(self_node_src_ranks) > 0
            else []
        )
        need_comm_cross_node_dst_ranks = [
            x
            for x in need_comm_dst_ranks
            if (x // num_gpu_per_node) not in all_src_nodes
        ]

        same_node_mapping = _ChunkUtils(
            chunk_values=self_node_src_ranks,
            element_values=need_comm_self_node_dst_ranks,
        )

        cross_node_mapping = _ChunkUtils(
            chunk_values=all_src_ranks,
            element_values=need_comm_cross_node_dst_ranks,
        )

        return same_node_mapping, cross_node_mapping, need_comm_self_node_dst_ranks

    def _filter_p2p_ops(p2p_op_infos):
        if survivor_process_groups is not None:
            # Source/destination discovery already uses survivor membership and
            # every peer is expressed in the rebuilt group's compact namespace.
            return
        elastic_ep_state = ElasticEPStateManager.instance()
        if elastic_ep_state is not None and missing_logical_experts_info is not None:
            # Filter out inactive P2P ops and record missing expert IDs in missing_logical_experts_info
            is_active = elastic_ep_state.active_ranks_cpu
            for i, (logical_expert_id, ops) in enumerate(p2p_op_infos):
                has_isend = any(op.op == torch.distributed.isend for op in ops)
                has_irecv = any(op.op == torch.distributed.irecv for op in ops)
                assert not (has_isend and has_irecv), (
                    "Each p2p_op_infos entry is expected to contain only send "
                    "or only recv ops."
                )

                if has_isend:
                    p2p_op_infos[i] = (
                        logical_expert_id,
                        [op for op in ops if is_active[op.peer]],
                    )
                elif has_irecv:
                    if any(not is_active[op.peer] for op in ops):
                        missing_logical_experts_info.append(logical_expert_id)
                        p2p_op_infos[i] = (logical_expert_id, [])

    def _execute_p2p_ops(p2p_op_infos):
        sorted_infos = sorted(p2p_op_infos, key=lambda info: info[0])
        p2p_ops = [op for _, ops in sorted_infos for op in ops]
        if len(p2p_ops) == 0:
            return

        if _LOG_P2P_SCHEDULE:
            schedules = defaultdict(list)
            for logical_expert_id, ops in sorted_infos:
                for op in ops:
                    direction = (
                        "send"
                        if op.op == torch.distributed.isend
                        else "recv"
                    )
                    schedules[f"{direction}:{op.peer}"].append(
                        (
                            logical_expert_id,
                            op.tensor.numel(),
                            str(op.tensor.dtype),
                        )
                    )
            elastic_ep_state = ElasticEPStateManager.instance()
            active_ranks = (
                elastic_ep_state.active_ranks_cpu.tolist()
                if elastic_ep_state is not None
                else None
            )
            logger.info(
                "[ExpertLocationUpdaterP2PSchedule] "
                f"{rank=} {active_ranks=} {dict(schedules)=}"
            )

        # Submit P2P ops in batches to prevent NCCL/RCCL GPU-side accumulation
        # hangs on large rebalances. All ranks use the same expert_id ranges
        # (based on num_physical_experts) so matching send/recv pairs land in
        # the same batch. Set batch_chunk_size >= num_physical_experts to disable.
        batch_chunk_size = envs.SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE.get()
        ops_by_expert = {eid: ops for eid, ops in sorted_infos}
        for start in range(0, num_physical_experts, batch_chunk_size):
            batch_ops = []
            batch_debug_infos = []

            for eid in range(
                start,
                min(
                    start + batch_chunk_size,
                    num_physical_experts,
                ),
            ):
                if eid not in ops_by_expert:
                    continue

                expert_ops = ops_by_expert[eid]

                for op_index, op in enumerate(expert_ops):
                    #
                    # _create_isend_ops() / recv creation both arrange
                    # tensors in tensor-index order. For send, each peer
                    # owns one contiguous num_tensors-sized block.
                    #
                    tensor_index = op_index % num_tensors

                    if op.op == torch.distributed.isend:
                        src_original_rank = rank
                        dst_original_rank = _from_peer_rank(
                            op.peer
                        )
                        direction = "send"

                    elif op.op == torch.distributed.irecv:
                        src_original_rank = _from_peer_rank(
                            op.peer
                        )
                        dst_original_rank = rank
                        direction = "recv"

                    else:
                        raise ValueError(
                            f"Unsupported P2P operation: {op.op}"
                        )

                    batch_ops.append(op)

                    batch_debug_infos.append(
                        {
                            "logical_expert_id": eid,
                            "tensor_index": tensor_index,
                            "src_rank": src_original_rank,
                            "dst_rank": dst_original_rank,
                            "direction": direction,
                            "tag": op.tag,
                            #
                            # Remember whether _stage_npu_p2p_ops()
                            # is expected to replace this tensor.
                            #
                            "was_staged": _needs_npu_p2p_staging(
                                op.tensor
                            ),
                            "original_offset": (
                                op.tensor.storage_offset()
                            ),
                        }
                    )

            if not batch_ops:
                continue

            #
            # Existing behavior:
            #
            # non-zero-offset internal-format tensor
            #     -> offset-zero staging tensor
            #
            batch_ops, recv_copy_infos = _stage_npu_p2p_ops(
                batch_ops
            )


            def _is_target_nan_case(info):
                return (
                    layer_id == 0
                    and info["logical_expert_id"] in (0, 32)
                    and info["tensor_index"] in (0, 1)
                    and info["src_rank"] == 0
                    and info["dst_rank"] == 3
                )

            #
            # ============================================================
            # CHECKPOINT A:
            # after send staging has been populated,
            # BEFORE batch_isend_irecv().
            # ============================================================
            #
            if _VALIDATE_NPU_P2P_NAN:
                for op, info in zip(batch_ops, batch_debug_infos):
                    if op.op != torch.distributed.isend:
                        continue

                    if not _is_target_nan_case(info):
                        continue

                    nan_count = _debug_npu_p2p_nan_count(op.tensor)

                    logger.warning(
                        "[NPU FT P2P TARGET] PRE_SEND "
                        "layer=%d logical_expert=%d tensor_idx=%d "
                        "src_rank=%d dst_rank=%d "
                        "was_staged=%s "
                        "original_offset=%d staging_offset=%d "
                        "nan_count=%d numel=%d",
                        layer_id,
                        info["logical_expert_id"],
                        info["tensor_index"],
                        info["src_rank"],
                        info["dst_rank"],
                        info["was_staged"],
                        info["original_offset"],
                        op.tensor.storage_offset(),
                        nan_count,
                        op.tensor.numel(),
                    )

            #
            # Actual HCCL P2P. Unchanged.
            #
            reqs = torch.distributed.batch_isend_irecv(batch_ops)

            for req in reqs:
                req.wait()

            if reqs:
                del req


            #
            # ============================================================
            # CHECKPOINT B:
            # HCCL recv has completed,
            # BEFORE recv staging -> expert weight copy.
            # ============================================================
            #
            if _VALIDATE_NPU_P2P_NAN:
                for op, info in zip(batch_ops, batch_debug_infos):
                    if op.op != torch.distributed.irecv:
                        continue

                    if not _is_target_nan_case(info):
                        continue

                    nan_count = _debug_npu_p2p_nan_count(op.tensor)

                    logger.warning(
                        "[NPU FT P2P TARGET] POST_RECV "
                        "layer=%d logical_expert=%d tensor_idx=%d "
                        "src_rank=%d dst_rank=%d "
                        "was_staged=%s "
                        "original_offset=%d staging_offset=%d "
                        "nan_count=%d numel=%d",
                        layer_id,
                        info["logical_expert_id"],
                        info["tensor_index"],
                        info["src_rank"],
                        info["dst_rank"],
                        info["was_staged"],
                        info["original_offset"],
                        op.tensor.storage_offset(),
                        nan_count,
                        op.tensor.numel(),
                    )

            recv_dst_by_ptr = {
                staged_tensor.data_ptr(): destination_tensor
                for staged_tensor, destination_tensor in recv_copy_infos
            }
            #
            # Existing behavior. Do not change.
            #
            _copy_staged_p2p_recvs(
                recv_copy_infos
            )
            if _VALIDATE_NPU_P2P_NAN:
                for op, info in zip(batch_ops, batch_debug_infos):
                    if op.op != torch.distributed.irecv:
                        continue

                    target = (
                        layer_id == 0
                        and info["logical_expert_id"] in (0, 32)
                        and info["tensor_index"] in (0, 1)
                        and info["src_rank"] == 0
                        and info["dst_rank"] == 3
                    )
                    if not target:
                        continue

                    destination_tensor = recv_dst_by_ptr.get(
                        op.tensor.data_ptr()
                    )
                    if destination_tensor is None:
                        raise RuntimeError(
                            "target recv staging has no destination"
                        )

                    # op.tensor is the received ND staging.
                    src = op.tensor.detach().cpu()
                    dst = destination_tensor.detach().cpu()

                    src_flat = src.reshape(-1)
                    dst_flat = dst.reshape(-1)

                    if src.is_floating_point():
                        both_nan = (
                            torch.isnan(src_flat)
                            & torch.isnan(dst_flat)
                        )
                        mismatch = ~(
                            (src_flat == dst_flat)
                            | both_nan
                        )
                    else:
                        mismatch = src_flat != dst_flat

                    mismatch_count = int(mismatch.sum().item())

                    logger.warning(
                        "[NPU FT FINAL WEIGHT] "
                        "layer=%d expert=%d tensor=%d "
                        "src_rank=%d dst_rank=%d "
                        "src_offset=%d dst_offset=%d "
                        "mismatch_count=%d numel=%d",
                        layer_id,
                        info["logical_expert_id"],
                        info["tensor_index"],
                        info["src_rank"],
                        info["dst_rank"],
                        op.tensor.storage_offset(),
                        destination_tensor.storage_offset(),
                        mismatch_count,
                        src.numel(),
                    )
            del (
                reqs,
                recv_copy_infos,
                batch_ops,
                batch_debug_infos,
            )

    def _execute_buffer2weight_copies(buffer2weight_copy_infos):
        for (
            temp_buffers_expert_location,
            routed_experts_weights_expert_location,
        ) in buffer2weight_copy_infos:
            logical_expert_id = new_physical_to_logical_map[
                routed_experts_weights_expert_location
            ]
            for i in range(num_tensors):
                _copy_expert_tensor_(
                    _get_tensor(
                        routed_experts_weights,
                        i,
                        routed_experts_weights_expert_location,
                    ),
                    _get_tensor(temp_buffers, i, temp_buffers_expert_location),
                     debug_context=(
                        f"temp_to_weight "
                        f"layer={layer_id} "
                        f"logical_expert={logical_expert_id} "
                        f"tensor_idx={i} "
                        f"temp_global_slot={temp_buffers_expert_location} "
                        f"dst_global_slot="
                        f"{routed_experts_weights_expert_location}"
                    ),
                )

    def _get_tensor(tensors, tensor_index: int, expert_location: int) -> torch.Tensor:
        return tensors[tensor_index][_get_local_expert_location(expert_location)]

    def _get_local_expert_location(expert_location: int) -> int:
        assert (
            local_expert_location_range[0]
            <= expert_location
            < local_expert_location_range[1]
        )
        return expert_location % num_local_physical_experts

    _entrypoint()

    return output_logs


class _ChunkUtils:
    def __init__(self, *, chunk_values: List, element_values: List):
        self.chunk_values = chunk_values
        self.element_values = element_values

    def chunk_value_from_element_value(self, element_value):
        chunk_index = self._chunk_index_from_element_index(
            num_elements=len(self.element_values),
            num_chunks=len(self.chunk_values),
            element_index=self.element_values.index(element_value),
        )
        return self.chunk_values[chunk_index]

    def element_values_from_chunk_value(self, chunk_value) -> List:
        if len(self.element_values) == 0:
            return []
        element_slice = self._element_slice_from_chunk_index(
            num_elements=len(self.element_values),
            num_chunks=len(self.chunk_values),
            chunk_index=self.chunk_values.index(chunk_value),
        )
        return self.element_values[element_slice]

    @staticmethod
    def _chunk_index_from_element_index(
        num_elements: int, num_chunks: int, element_index: int
    ) -> int:
        short_chunk_size, num_long_chunks = divmod(num_elements, num_chunks)
        num_elements_for_long_chunks = num_long_chunks * (short_chunk_size + 1)
        if element_index < num_elements_for_long_chunks:
            return element_index // (short_chunk_size + 1)
        else:
            return (
                num_long_chunks
                + (element_index - num_elements_for_long_chunks) // short_chunk_size
            )

    @staticmethod
    def _element_slice_from_chunk_index(
        num_elements: int, num_chunks: int, chunk_index: int
    ) -> slice:
        short_chunk_size, num_long_chunks = divmod(num_elements, num_chunks)
        start = chunk_index * short_chunk_size + min(chunk_index, num_long_chunks)
        end = start + short_chunk_size + int(chunk_index < num_long_chunks)
        return slice(start, end)


def _deduplicate_ordered(arr: List[int]):
    output = []
    for item in arr:
        if len(output) == 0 or item != output[-1]:
            output.append(item)
    return output


def _log_p2p_op_metrics(
    p2p_op_infos: List[Tuple[int, List[P2POp]]],
    num_gpu_per_node: int,
    world_size: int,
    self_node_id: int,
):
    text = ""
    all_ops = [op for _, ops in p2p_op_infos for op in ops]

    for direction, ops in _group_by(all_ops, _get_direction_from_op).items():
        nbytes_of_gpu = [0] * world_size
        for op in ops:
            nbytes_of_gpu[op.peer] += op.tensor.nbytes
        nbytes_of_gpu = torch.tensor(nbytes_of_gpu, dtype=torch.int64)

        nbytes_of_node = einops.reduce(
            nbytes_of_gpu,
            "(num_nodes num_gpu_per_node) -> num_nodes",
            num_gpu_per_node=num_gpu_per_node,
            reduction="sum",
        )

        nbytes_curr_node = nbytes_of_node[self_node_id]
        nbytes_cross_node = torch.sum(nbytes_of_node) - nbytes_curr_node

        text += (
            f"{direction}_nbytes_of_gpu={nbytes_of_gpu.tolist()} "
            f"{direction}_nbytes_of_node={nbytes_of_node.tolist()} "
            f"{direction}_nbytes_curr_node={nbytes_curr_node.item()} "
            f"{direction}_nbytes_cross_node={nbytes_cross_node.item()} "
        )

    logger.info(f"[ExpertLocationUpdater] {text}")


def _get_direction_from_op(op: P2POp):
    if op.op == torch.distributed.isend:
        return "isend"
    if op.op == torch.distributed.irecv:
        return "irecv"
    raise NotImplementedError


def _group_by(items, keyfunc):
    ans = defaultdict(list)
    for item in items:
        ans[keyfunc(item)].append(item)
    return dict(ans)
