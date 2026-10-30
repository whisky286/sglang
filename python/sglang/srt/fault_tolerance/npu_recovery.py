"""Explicit, fail-closed restoration of the original single-node NPU cohort.

All members use the controller's epoch and an independent TCPStore namespace.
Old groups/buffers remain owned until process exit: destroying a group with a
dead member is not part of the recovery contract. No request route is opened
by this module; the FT manager commits it after every scheduler's health ACK.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from sglang.srt.fault_tolerance.recovery import RankRecoveryPlan


def recovery_config_error(args) -> str | None:
    """The first serving implementation deliberately restores the full cohort."""
    requirements = {
        "requires_npu_mc2_ft": getattr(args, "device", None) == "npu"
        and getattr(args, "enable_fault_tolerance", False)
        and getattr(args, "elastic_ep_backend", None) == "mc2",
        "requires_single_node": getattr(args, "nnodes", 1) == 1,
        "requires_primary_launcher": getattr(args, "ep_join_mode", None) is None,
        "requires_attention_tp1_ep_equals_dp": args.tp_size
        == args.dp_size
        == args.ep_size,
        "requires_cp1_moe_dp1": getattr(args, "attn_cp_size", 1) == 1
        and getattr(args, "moe_dp_size", 1) == 1
        and getattr(args, "dcp_size", 1) == 1,
        "requires_eager": getattr(args, "disable_cuda_graph", False)
        and not getattr(args, "enable_piecewise_cuda_graph", False)
        and not getattr(args, "enable_prefill_cuda_graph", False),
        "requires_no_overlap": getattr(args, "disable_overlap_schedule", False),
        "requires_deepep_low_latency": getattr(args, "moe_a2a_backend", None)
        == "deepep"
        and getattr(args, "deepep_mode", None) == "low_latency",
        "requires_plain_model": not any(
            getattr(args, key, None)
            for key in (
                "quantization",
                "speculative_algorithm",
                "enable_lora",
                "enable_hierarchical_cache",
                "enable_pdmux",
                "enable_memory_saver",
                "enable_two_batch_overlap",
                "pre_warm_nccl",
                "enable_symm_mem",
                "enable_elastic_expert_backup",
                "enable_waterfill",
                "is_embedding",
                "use_ray",
            )
        ),
        "requires_single_tokenizer": getattr(args, "tokenizer_worker_num", 1) == 1,
        "requires_static_dispatch": getattr(args, "ep_dispatch_algorithm", None)
        in (None, "static"),
        "requires_standard_process_groups": os.getenv(
            "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS", "0"
        ).lower()
        not in ("1", "true", "yes", "on")
        and os.getenv("SGLANG_USE_MESSAGE_QUEUE_BROADCASTER", "1").lower()
        not in ("1", "true", "yes", "on"),
        "requires_default_deepep_strategy": os.getenv(
            "DEEP_USE_MODE", "default"
        ).lower()
        == "default",
    }
    return next((name for name, passed in requirements.items() if not passed), None)


@dataclass
class RecoveryGroups:
    store: Any
    plan: RankRecoveryPlan
    rank: int
    timeout_sec: float
    groups: dict = field(default_factory=dict)
    retired: list = field(default_factory=list)

    def group(self, role: str, ranks, backend: str):
        from torch.distributed.distributed_c10d import GroupMember, _world

        from sglang.srt.utils import init_custom_process_group

        ranks = tuple(ranks)
        if self.rank not in ranks:
            return GroupMember.NON_GROUP_MEMBER
        key = (role, ranks, backend)
        if key not in self.groups:
            suffix = f"{role}/{'-'.join(map(str, ranks))}/{backend}"
            prefix = f"{self.plan.store_prefix}/restore/{suffix}"
            pg = init_custom_process_group(
                backend=backend,
                store=dist.PrefixStore(prefix, self.store),
                timeout=timedelta(seconds=self.timeout_sec),
                world_size=len(ranks),
                rank=ranks.index(self.rank),
                group_name=prefix.replace("/", "_"),
            )
            # c10d object collectives interpret src/dst as original global ranks.
            _world.pg_group_ranks[pg] = {
                rank: index for index, rank in enumerate(ranks)
            }
            self.groups[key] = pg
        return self.groups[key]

    def install_default(self):
        from torch.distributed.distributed_c10d import _update_default_pg, _world

        if _world.default_pg is not None:
            self.retired.append(_world.default_pg)
        _update_default_pg(
            self.group("default", self.plan.active_original_ranks, "hccl")
        )

    def snapshot(self):
        key = f"{self.plan.store_prefix}/snapshot"
        self.store.wait([key], timedelta(seconds=self.timeout_sec))
        snapshot = json.loads(self.store.get(key))
        if snapshot["membership_epoch"] != self.plan.membership_epoch:
            raise RuntimeError("NPU recovery snapshot epoch mismatch")
        return snapshot


_groups: RecoveryGroups | None = None


def current_recovery_groups():
    return _groups


def begin_joiner_bootstrap(args, *, gpu_id, rank):
    """Install a fresh default PG before normal model-parallel construction."""
    global _groups
    from sglang.srt.fault_tolerance.npu_communication import _parse_tcp_endpoint

    config = getattr(args, "_npu_ft_recovery", None)
    if config is None:
        return
    plan = RankRecoveryPlan(**config["plan"])
    if rank not in plan.recovering_ranks or not all(plan.target_active_mask):
        raise ValueError("NPU joiner must restore a removed rank in the full cohort")
    host, port = _parse_tcp_endpoint(config["endpoint"])
    store = dist.TCPStore(
        host,
        port,
        is_master=False,
        wait_for_workers=False,
        timeout=timedelta(seconds=args.fault_tolerance_timeout),
    )
    _groups = RecoveryGroups(store, plan, rank, args.fault_tolerance_timeout)
    snapshot = _groups.snapshot()
    torch.npu.set_device(gpu_id)
    _groups.install_default()
    # Match the serving pool upper bound without running startup all-reduces
    # against survivors which already own their model and KV pools.
    args.override("npu_ft.recovery_pool", max_total_tokens=snapshot["max_total_tokens"])


def joiner_metadata(args, model_config, rank):
    if getattr(args, "_npu_ft_recovery", None) is None:
        return None
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata

    return ExpertLocationMetadata.init_by_mapping(
        args,
        model_config,
        physical_to_logical_map=_groups.snapshot()["physical_to_logical_map"],
        moe_ep_rank=rank,
    )


def prepare_survivor(scheduler, request):
    global _groups
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.distributed.device_communicators.npu_communicator import (
        NpuCommunicator,
    )
    from sglang.srt.eplb.expert_location import (
        ExpertLocationMetadata,
        get_global_expert_location_metadata,
    )
    from sglang.srt.fault_tolerance.npu_communication import get_npu_ft_communication

    runner = scheduler.tp_worker.model_runner
    comm = get_npu_ft_communication()
    plan = RankRecoveryPlan(
        request.membership_epoch,
        tuple(request.previous_active_mask),
        tuple(
            rank
            for rank, active in enumerate(request.previous_active_mask)
            if not active
        ),
    )
    if not all(plan.target_active_mask) or request.active_mask != list(
        plan.target_active_mask
    ):
        raise ValueError("NPU recovery must restore the complete original cohort")
    if plan.membership_epoch <= comm.generation:
        raise ValueError("stale NPU recovery epoch")
    current = tuple(
        rank in comm.active_original_ranks
        for rank in range(len(plan.previous_active_mask))
    )
    if current != plan.previous_active_mask:
        raise ValueError("NPU recovery previous membership mismatch")
    old = get_global_expert_location_metadata()
    candidate = ExpertLocationMetadata.init_for_rank_recovery(
        runner.server_args,
        old,
        plan,
        moe_ep_rank=runner.ps.moe_ep_rank,
    )
    context = RecoveryGroups(comm.store, plan, comm.original_rank, comm.timeout_sec)
    if _groups is not None:
        context.retired.append(_groups)
    _groups = context
    if comm.original_rank == min(comm.active_original_ranks):
        comm.store.set(
            f"{plan.store_prefix}/snapshot",
            json.dumps(
                {
                    "membership_epoch": plan.membership_epoch,
                    "physical_to_logical_map": candidate.physical_to_logical_map_cpu.tolist(),
                    "max_total_tokens": runner.max_total_num_tokens,
                }
            ),
        )
    # A fresh process constructs these same role/rank keys through the normal
    # GroupCoordinator initializer. Singleton attention/PP groups never changed.
    context.install_default()
    for role, coordinator in (
        ("world", ps.get_world_group()),
        ("tp", ps.get_tp_group()),
        ("moe_ep", ps.get_moe_ep_group()),
    ):
        context.retired.extend((coordinator.device_group, coordinator.cpu_group))
        coordinator.device_group = context.group(role, coordinator.ranks, "hccl")
        coordinator.cpu_group = context.group(role, coordinator.ranks, "gloo")
        if coordinator.npu_communicator is not None:
            coordinator.npu_communicator = NpuCommunicator(coordinator.device_group)
        coordinator.active_ranks.fill_(1)
        coordinator.active_ranks_cpu.fill_(1)
    scheduler.tp_cpu_group = scheduler.tp_group.cpu_group
    # Preserve tensors already referenced by MoE routing layers.
    old.update(candidate, update_layer_ids=list(range(old.num_layers)))
    install_serving_membership(runner, plan)
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer

    DeepEPBuffer.retire_for_npu_recovery(runner.model, context.retired)
    for rank in plan.recovering_ranks:
        comm.store.wait(
            [f"{plan.store_prefix}/loaded/{rank}"], timedelta(seconds=comm.timeout_sec)
        )


def install_serving_membership(runner, plan):
    from sglang.srt.distributed.parallel_state import get_moe_ep_group, get_tp_group
    from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
    from sglang.srt.eplb.process_group_context import (
        EPLBProcessGroupContext,
        set_eplb_process_group_context,
    )
    from sglang.srt.fault_tolerance.npu_communication import get_npu_ft_communication

    state = ElasticEPStateManager.instance()
    state.active_ranks.fill_(1)
    state.active_ranks_cpu.fill_(1)
    state.update_npu_mc2_elastic_info()
    state.snapshot_active_to_last()
    comm = get_npu_ft_communication()
    comm.mlp_sync_group = get_tp_group().cpu_group
    comm.active_original_ranks = plan.active_original_ranks
    comm.generation = plan.membership_epoch
    set_eplb_process_group_context(
        EPLBProcessGroupContext(
            control_group=comm.mlp_sync_group,
            device_group=get_moe_ep_group().device_group,
            active_original_ranks=comm.active_original_ranks,
            control_group_uses_cpu=True,
        )
    )


def mark_joiner_loaded(scheduler):
    if getattr(scheduler.server_args, "_npu_ft_recovery", None) is None:
        return
    install_serving_membership(scheduler.tp_worker.model_runner, _groups.plan)
    if scheduler.max_total_num_tokens != _groups.snapshot()["max_total_tokens"]:
        raise RuntimeError("replacement KV capacity differs from the serving cohort")
    scheduler._engine_paused = True
    scheduler._ft_recovery_epoch = _groups.plan.membership_epoch
    scheduler._ft_recovery_validated = False
    _groups.store.set(f"{_groups.plan.store_prefix}/loaded/{_groups.rank}", "1")


def validate_cohort(scheduler, request):
    from sglang.srt.distributed.parallel_state import (
        get_moe_ep_group,
        get_tp_group,
        get_world_group,
    )
    from sglang.srt.fault_tolerance.npu_communication import get_npu_ft_communication

    comm = get_npu_ft_communication()
    if request.membership_epoch != comm.generation:
        raise ValueError("NPU health check must match the installed recovery epoch")
    runner = scheduler.tp_worker.model_runner
    dist.barrier(group=comm.mlp_sync_group)
    # HCCL communicators initialize lazily. Materialize every replaced device
    # domain before DeepEP asks its backend for the communicator name.
    for coordinator in (get_world_group(), get_tp_group(), get_moe_ep_group()):
        value = torch.tensor([comm.original_rank + 1.0], device=runner.device)
        dist.all_reduce(value, group=coordinator.device_group)
        if value.item() != sum(rank + 1 for rank in comm.active_original_ranks):
            raise RuntimeError("restored HCCL membership check failed")
    with scheduler.forward_stream_ctx:
        output = runner.run_npu_fault_tolerance_dummy_batch(request.active_mask)
    runner.synchronize_npu_fault_tolerance_health_gate()
    validate_health_output(output)
    dist.barrier(group=comm.mlp_sync_group)


def validate_health_output(output):
    logits = getattr(getattr(output, "logits_output", None), "next_token_logits", None)
    if not isinstance(logits, torch.Tensor) or logits.numel() == 0:
        raise RuntimeError("NPU recovery health check returned no model logits")
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("NPU recovery health check returned non-finite model logits")
