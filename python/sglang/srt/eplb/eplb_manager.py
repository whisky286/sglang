from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import torch.cuda
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    format_expert_location_layout,
    format_expert_location_layout_diff,
    get_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.runtime_context import get_server_args

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class EPLBManager:
    def __init__(
        self,
        *,
        server_args: ServerArgs,
        model_config: ModelConfig,
        ps: Any,
        get_model: Callable[[], nn.Module],
        get_expert_location_updater: Callable[[], ExpertLocationUpdater],
        get_expert_backup_client: Callable[[], Any],
        get_weight_updater: Callable[[], Any],
    ):
        super().__init__()
        # These collaborators are set on ModelRunner AFTER EPLBManager is
        # constructed (model load, expert_backup_client, weight_updater), so
        # they are read through getters at rebalance time, not captured here.
        self._server_args = server_args
        self._model_config = model_config
        self._ps = ps
        self._get_model = get_model
        self._get_expert_location_updater = get_expert_location_updater
        self._get_expert_backup_client = get_expert_backup_client
        self._get_weight_updater = get_weight_updater
        self._rebalance_layers_per_chunk = (
            self._server_args.eplb_rebalance_layers_per_chunk
        )
        self._rebalance_num_iterations = self._server_args.eplb_rebalance_num_iterations
        self._rebalance_disabled_reason = None
        self._rebalance_disabled_logged = False

        # Otherwise, the circular buffer will contain stale data. If the case is needed, it can be implemented.
        assert (
            self._server_args.eplb_rebalance_num_iterations
            >= self._server_args.expert_distribution_recorder_buffer_size
        ), "eplb_rebalance_num_iterations must be greater than expert_distribution_recorder_buffer_size"

        if not get_global_expert_distribution_recorder().recording:
            get_global_expert_distribution_recorder().start_record()

        logger.info(
            f"[EPLBManager] system started, will rebalance per {self._rebalance_num_iterations} iterations."
        )

        self._main_generator = self._entrypoint()

    def on_forward_pass_end(self):
        next(self._main_generator)

    def reset_generator(self):
        self._main_generator = self._entrypoint()

    def disable_rebalance(self, reason: str):
        self._rebalance_disabled_reason = reason
        self._rebalance_disabled_logged = False
        self.reset_generator()

    # can be more complex if needed
    def _entrypoint(self):
        while True:
            for _ in range(self._rebalance_num_iterations):
                yield

            yield from self.rebalance()

    def rebalance(
        self,
        *,
        force: bool = False,
        logical_count_override: Optional[torch.Tensor] = None,
    ):
        if self._rebalance_disabled_reason is not None:
            if not self._rebalance_disabled_logged:
                logger.debug(
                    "[EPLBManager] rebalance disabled: %s",
                    self._rebalance_disabled_reason,
                )
                self._rebalance_disabled_logged = True
            return

        logger.info("[EPLBManager] rebalance start")

        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

        elastic_ep_state = ElasticEPStateManager.instance()
        recovering_from_rank_fault = (
            elastic_ep_state is not None
            and elastic_ep_state.active_ranks_cpu is not None
            and not bool(elastic_ep_state.active_ranks_cpu.all().item())
        )
        enable_timing = (
            self._rebalance_layers_per_chunk is None
            and not recovering_from_rank_fault
        )
        if recovering_from_rank_fault:
            logger.info(
                "[EPLBManager] skip device-wide timing synchronization during rank-fault recovery"
            )

        if enable_timing:
            torch.get_device_module().synchronize()
            time_start = time.time()

        if recovering_from_rank_fault:
            old_expert_location_metadata = get_global_expert_location_metadata()
            expert_location_metadata = ExpertLocationMetadata.init_for_fault_recovery(
                self._server_args,
                old_expert_location_metadata,
                elastic_ep_state.active_ranks_cpu,
                moe_ep_rank=self._ps.tp_rank,
            )
            changed_slots = int(
                (
                    expert_location_metadata.physical_to_logical_map_cpu
                    != old_expert_location_metadata.physical_to_logical_map_cpu
                )
                .sum()
                .item()
            )
            logger.info(
                "[NPU FT] minimal expert recovery layout: rank=%d "
                "active_original_ranks=%s changed_physical_slots=%d",
                self._ps.tp_rank,
                torch.nonzero(
                    elastic_ep_state.active_ranks_cpu,
                    as_tuple=False,
                )
                .flatten()
                .tolist(),
                changed_slots,
            )
        else:
            if logical_count_override is None:
                dump_record_output = (
                    get_global_expert_distribution_recorder().dump_record(
                        output_mode="object"
                    )
                )
                logical_count = dump_record_output["logical_count"]
                average_utilization_rate_over_window = dump_record_output[
                    "average_utilization_rate_over_window"
                ]
            else:
                logical_count = logical_count_override
                average_utilization_rate_over_window = None

            # Check whether rebalancing is needed
            if not force and not self._check_rebalance_needed(
                average_utilization_rate_over_window
            ):
                return

            expert_location_metadata = ExpertLocationMetadata.init_by_eplb(
                self._server_args, self._model_config, logical_count
            )

        from sglang.srt.model_executor.model_runner_components.moe_ep_setup import (
            init_lplb_solvers,
        )

        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        all_update_layer_ids = [
            layer_id for chunk in update_layer_ids_chunks for layer_id in chunk
        ]
        self._log_rebalance_layout_before_update(
            expert_location_metadata,
            update_layer_ids=all_update_layer_ids,
        )
        for chunk_layer_ids in update_layer_ids_chunks:
            if len(update_layer_ids_chunks) > 1:
                yield
            survivor_process_groups = None
            if (
                recovering_from_rank_fault
                and self._server_args.elastic_ep_backend == "mc2"
            ):
                from sglang.srt.fault_tolerance.npu_metadata import (
                    get_npu_ft_metadata_group,
                )

                survivor_process_groups = get_npu_ft_metadata_group()
            update_expert_location_with_recovery(
                expert_location_updater=self._get_expert_location_updater(),
                model=self._get_model(),
                new_expert_location_metadata=expert_location_metadata,
                update_layer_ids=chunk_layer_ids,
                nnodes=self._server_args.nnodes,
                tp_rank=self._ps.tp_rank,
                expert_backup_client=self._get_expert_backup_client(),
                update_weights_from_disk_callable=self._get_weight_updater().update_weights_from_disk,
                ep_dispatch_algorithm=self._server_args.ep_dispatch_algorithm,
                init_lplb_solvers_callable=lambda: init_lplb_solvers(
                    model_config=self._model_config
                ),
                survivor_process_groups=survivor_process_groups,
            )

        self._log_rebalance_layout_after_update(update_layer_ids=all_update_layer_ids)

        if (
            recovering_from_rank_fault
            and self._server_args.device == "npu"
        ):
            logger.info(
                "[NPU FT] synchronizing expert recovery before scale-down commit"
            )
            torch.get_device_module().synchronize()
            logger.info(
                "[NPU FT] expert recovery synchronized"
            )

        msg = f"[EPLBManager] rebalance end"
        if enable_timing:
            torch.get_device_module().synchronize()
            time_end = time.time()
            msg += f" time={time_end - time_start:.3f}s"
        logger.info(msg)

    def _check_rebalance_needed(self, average_utilization_rate_over_window):
        if average_utilization_rate_over_window is None:
            return True

        if (
            average_utilization_rate_over_window
            > self._server_args.eplb_min_rebalancing_utilization_threshold
        ):
            logger.info(
                f"[EPLBManager] Skipped ep rebalancing: current GPU utilization {average_utilization_rate_over_window:.2f} > minimum rebalance threshold {self._server_args.eplb_min_rebalancing_utilization_threshold:.2f}"
            )
            return False

        return True

    def _compute_update_layer_ids_chunks(self) -> List[List[int]]:
        all_layer_ids = sorted(
            list(self._get_model().routed_experts_weights_of_layer.keys())
        )
        chunk_size = self._rebalance_layers_per_chunk or 1000000
        return list(_chunk_list(all_layer_ids, chunk_size=chunk_size))

    def _should_log_expert_location_metadata(self) -> bool:
        return self._ps.tp_rank == 0 and envs.SGLANG_LOG_EXPERT_LOCATION_METADATA.get()

    def _log_rebalance_layout_before_update(
        self,
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: List[int],
    ):
        if not self._should_log_expert_location_metadata():
            return

        old_expert_location_metadata = get_global_expert_location_metadata()
        logger.info(
            "[EPLBManager] rebalance layout before:\n%s",
            format_expert_location_layout(
                old_expert_location_metadata,
                layer_ids=update_layer_ids,
            ),
        )
        logger.info(
            "[EPLBManager] rebalance layout target:\n%s",
            format_expert_location_layout(
                new_expert_location_metadata,
                layer_ids=update_layer_ids,
            ),
        )
        logger.info(
            "[EPLBManager] rebalance layout diff:\n%s",
            format_expert_location_layout_diff(
                old_expert_location_metadata,
                new_expert_location_metadata,
                layer_ids=update_layer_ids,
            ),
        )

    def _log_rebalance_layout_after_update(self, update_layer_ids: List[int]):
        if not self._should_log_expert_location_metadata():
            return

        logger.info(
            "[EPLBManager] rebalance layout after:\n%s",
            format_expert_location_layout(
                get_global_expert_location_metadata(),
                layer_ids=update_layer_ids,
            ),
        )


def update_expert_location_with_recovery(
    *,
    expert_location_updater: ExpertLocationUpdater,
    model: nn.Module,
    new_expert_location_metadata: ExpertLocationMetadata,
    update_layer_ids: List[int],
    nnodes: int,
    tp_rank: int,
    expert_backup_client,
    update_weights_from_disk_callable,
    ep_dispatch_algorithm: str,
    init_lplb_solvers_callable,
    survivor_process_groups=None,
):
    p2p_missing_logical_experts = expert_location_updater.update(
        model.routed_experts_weights_of_layer,
        new_expert_location_metadata,
        update_layer_ids=update_layer_ids,
        nnodes=nnodes,
        rank=tp_rank,
        survivor_process_groups=survivor_process_groups,
    )

    if len(p2p_missing_logical_experts) > 0:
        missing_expert_layer_pairs = sum(
            len(expert_ids)
            for expert_ids in p2p_missing_logical_experts.values()
        )
        unique_missing_experts = sorted(
            {
                expert_id
                for expert_ids in p2p_missing_logical_experts.values()
                for expert_id in expert_ids
            }
        )
        # Load the missing expert weights from disk
        if callable(getattr(model, "generate_weight_name_filter", None)):
            # Filter and load only missing expert weights
            weight_name_filter = model.generate_weight_name_filter(
                p2p_missing_logical_experts
            )
        else:
            # Do a full reload from disk/DRAM
            logger.info(
                "[Elastic EP] Model does not implement generate_weight_name_filter. "
                "Performing full weight reload."
            )
            weight_name_filter = None

        reload_source = (
            "dram"
            if expert_backup_client is not None and expert_backup_client.use_backup
            else "disk"
        )
        reload_start = time.monotonic()
        logger.info(
            "[NPU FT] missing-expert reload begin: rank=%d source=%s "
            "layers=%d expert_layer_pairs=%d unique_experts=%d",
            tp_rank,
            reload_source,
            len(p2p_missing_logical_experts),
            missing_expert_layer_pairs,
            len(unique_missing_experts),
        )
        if expert_backup_client is not None and expert_backup_client.use_backup:
            # Load the missing weights from the DRAM backup
            expert_backup_client.update_weights(weight_name_filter)
        else:
            # Load the missing weights from disk
            update_result = update_weights_from_disk_callable(
                get_server_args().model_path,
                get_server_args().load_format,
                weight_name_filter=weight_name_filter,
            )
            if (
                survivor_process_groups is not None
                and isinstance(update_result, tuple)
                and not update_result[0]
            ):
                raise RuntimeError(
                    "NPU FT failed to reload missing experts from disk: "
                    f"{update_result[1]}"
                )
            reload_stats = getattr(
                weight_name_filter, "_sglang_ft_reload_stats", None
            )
            if reload_stats is not None:
                expected_pairs = reload_stats["expected_pairs"]
                selected_pairs = reload_stats["selected_pairs"]
                unmatched_pairs = sorted(expected_pairs - selected_pairs)
                logger.info(
                    "[NPU FT] missing-expert checkpoint coverage: rank=%d "
                    "expected_pairs=%d selected_pairs=%d selected_tensors=%d",
                    tp_rank,
                    len(expected_pairs),
                    len(selected_pairs),
                    reload_stats["selected_weight_names"],
                )
                if unmatched_pairs:
                    raise RuntimeError(
                        "NPU FT checkpoint filter did not find every requested "
                        "expert: "
                        f"rank={tp_rank} unmatched_pairs={unmatched_pairs[:32]} "
                        f"unmatched_count={len(unmatched_pairs)}"
                    )
                if envs.SGLANG_NPU_FT_VERIFY_EXPERT_RELOAD.get():
                    validate_probe = getattr(
                        model, "validate_ft_reloaded_expert_probe", None
                    )
                    if not callable(validate_probe):
                        raise RuntimeError(
                            "SGLANG_NPU_FT_VERIFY_EXPERT_RELOAD requires a "
                            "model-specific expert content validator"
                        )
                    probe_result = validate_probe(reload_stats, tp_rank)
                    logger.info(
                        "[NPU FT] missing-expert content probe: rank=%d %s",
                        tp_rank,
                        probe_result,
                    )
        logger.info(
            "[NPU FT] missing-expert reload end: rank=%d source=%s "
            "layers=%d expert_layer_pairs=%d elapsed=%.3fs",
            tp_rank,
            reload_source,
            len(p2p_missing_logical_experts),
            missing_expert_layer_pairs,
            time.monotonic() - reload_start,
        )

    # Re-init LPLB solvers after expert location update
    if ep_dispatch_algorithm == "lp":
        init_lplb_solvers_callable()


def _chunk_list(items: List, chunk_size):
    for start_index in range(0, len(items), chunk_size):
        yield items[start_index : start_index + chunk_size]
