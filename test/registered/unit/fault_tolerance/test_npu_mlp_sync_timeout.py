import unittest
from datetime import timedelta
from unittest.mock import Mock, patch

import torch

from sglang.srt.fault_tolerance.npu_communication import (
    NpuFTCommunication,
    all_gather_into_tensor_with_timeout,
)
from sglang.srt.managers.scheduler_components.dp_attn import MLPSyncBatchInfo
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNpuMLPSyncTimeout(CustomTestCase):
    @staticmethod
    def _make_mlp_sync_info(*, dp_size=2, attn_tp_size=2, attn_cp_size=1):
        return MLPSyncBatchInfo(
            dp_size=dp_size,
            tp_size=attn_tp_size,
            cp_size=attn_cp_size,
            num_tokens=1,
            num_tokens_for_logprob=1,
            can_cuda_graph=True,
            is_extend_in_batch=False,
            local_can_run_tbo=True,
            local_forward_mode=ForwardMode.DECODE.value,
            can_run_breakable_cuda_graph=False,
        )

    @staticmethod
    def _run_physical_rank_gather(info, communication, gathered_rows, device="cpu"):
        gathered_rows = torch.tensor(gathered_rows, dtype=torch.int64)

        def fake_all_gather(output_tensor, input_tensor, *, group, timeout_sec):
            output_tensor.copy_(gathered_rows.flatten())

        num_physical_slots = info.dp_size * info.tp_size * info.cp_size
        with (
            patch(
                "sglang.srt.managers.scheduler_components.dp_attn.get_npu_ft_communication",
                return_value=communication,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.dp_attn.all_gather_into_tensor_with_timeout",
                side_effect=fake_all_gather,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.dp_attn.get_tp_group",
                return_value=Mock(
                    active_ranks_cpu=torch.ones(num_physical_slots, dtype=torch.int64)
                ),
            ),
        ):
            info.all_gather(device=device, group=communication.mlp_sync_group)

    def test_npu_metadata_stays_on_cpu_for_prefill_and_idle_survivors(self):
        info = self._make_mlp_sync_info(dp_size=4, attn_tp_size=1)
        communication = NpuFTCommunication(
            store=Mock(),
            original_rank=0,
            timeout_sec=5,
            mlp_sync_group=Mock(),
            active_original_ranks=(0, 1, 3),
            generation=1,
        )
        self._run_physical_rank_gather(
            info,
            communication,
            [
                [6, 1, 0, 1, 1, ForwardMode.EXTEND.value, 0],
                [0, 0, 1, 0, 1, ForwardMode.IDLE.value, 0],
                [0, 0, 1, 0, 1, ForwardMode.IDLE.value, 0],
            ],
            # A real NPU caller passes its device even though the control PG is
            # Gloo. This test must run on CPU without allocating any NPU tensor.
            device="npu:0",
        )
        self.assertEqual(info.tp0_info.device.type, "cpu")
        self.assertEqual(info.global_num_tokens, [6, 0, 0, 0])
        self.assertEqual(info.global_num_tokens_for_logprob, [1, 0, 0, 0])
        self.assertTrue(info.is_extend_in_batch)
        self.assertFalse(info.can_cuda_graph)

    def test_mlp_sync_scatter_uses_physical_rank_layout_with_attention_tp(self):
        info = self._make_mlp_sync_info(dp_size=2, attn_tp_size=2)
        communication = NpuFTCommunication(
            store=Mock(),
            original_rank=0,
            timeout_sec=5,
            mlp_sync_group=Mock(),
            active_original_ranks=(0, 1, 2, 3),
        )

        self._run_physical_rank_gather(
            info,
            communication,
            [
                [10, 1, 1, 0, 1, ForwardMode.DECODE.value, 0],
                [11, 1, 1, 0, 1, ForwardMode.DECODE.value, 0],
                [20, 2, 1, 0, 1, ForwardMode.DECODE.value, 0],
                [21, 2, 1, 0, 1, ForwardMode.DECODE.value, 0],
            ],
        )

        # Physical ranks 0 and 2 are the attention-TP0 members of DP0 and DP1.
        self.assertEqual(info.global_num_tokens, [10, 20])
        self.assertEqual(info.global_num_tokens_for_logprob, [1, 2])

    def test_mlp_sync_scatter_preserves_dead_dp_fallback_after_scale_down(self):
        info = self._make_mlp_sync_info(dp_size=2, attn_tp_size=2)
        communication = NpuFTCommunication(
            store=Mock(),
            original_rank=2,
            timeout_sec=5,
            mlp_sync_group=Mock(),
            active_original_ranks=(2, 3),
        )

        self._run_physical_rank_gather(
            info,
            communication,
            [
                [20, 2, 1, 0, 1, ForwardMode.DECODE.value, 0],
                [21, 2, 1, 0, 1, ForwardMode.DECODE.value, 0],
            ],
        )

        self.assertEqual(info.global_num_tokens, [0, 20])
        self.assertEqual(info.global_num_tokens_for_logprob, [0, 2])

    def test_rebuild_survivor_groups_separates_control_and_device_planes(self):
        communication = NpuFTCommunication(
            store=Mock(),
            original_rank=2,
            timeout_sec=5,
            mlp_sync_group=Mock(),
            active_original_ranks=(0, 1, 2, 3),
        )
        mlp_sync_group = object()
        original_device_group = object()

        with (
            patch(
                "sglang.srt.utils.init_custom_process_group",
                return_value=mlp_sync_group,
            ) as init_group,
            patch(
                "sglang.srt.distributed.parallel_state.get_moe_ep_group",
                return_value=Mock(device_group=original_device_group),
            ),
            patch(
                "sglang.srt.fault_tolerance.npu_communication.PrefixStore",
                side_effect=lambda prefix, store: (prefix, store),
            ),
            patch(
                "sglang.srt.fault_tolerance.npu_communication.dist.barrier"
            ) as barrier,
            patch(
                "sglang.srt.fault_tolerance.npu_communication.dist.all_reduce"
            ) as all_reduce,
            patch(
                "sglang.srt.fault_tolerance.npu_communication.set_eplb_process_group_context"
            ) as set_context,
        ):
            communication.rebuild_survivor_groups([False, True, True, True])

        init_group.assert_called_once()
        self.assertEqual(init_group.call_args.kwargs["world_size"], 3)
        self.assertEqual(init_group.call_args.kwargs["rank"], 1)
        self.assertEqual(init_group.call_args.kwargs["backend"], "gloo")
        barrier.assert_called_once_with(group=mlp_sync_group)
        all_reduce.assert_not_called()
        context = set_context.call_args.args[0]
        self.assertIs(context.control_group, mlp_sync_group)
        self.assertIs(context.device_group, original_device_group)
        self.assertEqual(context.active_original_ranks, (1, 2, 3))
        self.assertTrue(context.control_group_uses_cpu)
        self.assertEqual(communication.active_original_ranks, (1, 2, 3))

    def test_returning_process_uses_controller_epoch_instead_of_local_counter(self):
        prefixes = []
        with (
            patch("sglang.srt.utils.init_custom_process_group", return_value=object()),
            patch(
                "sglang.srt.distributed.parallel_state.get_moe_ep_group",
                return_value=Mock(device_group=object()),
            ),
            patch(
                "sglang.srt.fault_tolerance.npu_communication.PrefixStore",
                side_effect=lambda prefix, store: prefixes.append(prefix) or store,
            ),
            patch("sglang.srt.fault_tolerance.npu_communication.dist.barrier"),
            patch(
                "sglang.srt.fault_tolerance.npu_communication.set_eplb_process_group_context"
            ),
        ):
            for rank, generation in ((0, 1), (2, 0)):
                communication = NpuFTCommunication(
                    store=Mock(),
                    original_rank=rank,
                    timeout_sec=5,
                    mlp_sync_group=object(),
                    active_original_ranks=(0, 1, 3) if rank == 0 else (2,),
                    generation=generation,
                )
                communication.rebuild_survivor_groups(
                    [True, True, True, True], membership_epoch=2
                )
                self.assertEqual(communication.generation, 2)
                self.assertEqual(communication.active_original_ranks, (0, 1, 2, 3))

        self.assertEqual(prefixes, ["npu-ft/1111/2/mlp-sync"] * 2)

    def test_rejects_stale_epoch_and_excluded_local_rank_before_building_group(self):
        communication = NpuFTCommunication(
            store=Mock(),
            original_rank=0,
            timeout_sec=5,
            mlp_sync_group=object(),
            active_original_ranks=(0, 1, 3),
            generation=2,
        )
        with patch("sglang.srt.utils.init_custom_process_group") as init_group:
            for epoch in (1, 2, True):
                with self.subTest(epoch=epoch):
                    with self.assertRaisesRegex(ValueError, "membership_epoch"):
                        communication.rebuild_survivor_groups(
                            [True, True, True, True], membership_epoch=epoch
                        )
            with self.assertRaisesRegex(ValueError, "local rank"):
                communication.rebuild_survivor_groups(
                    [False, True, True, True], membership_epoch=3
                )
            init_group.assert_not_called()
        self.assertEqual(communication.generation, 2)
        self.assertEqual(communication.active_original_ranks, (0, 1, 3))

    def test_failed_candidate_barrier_does_not_install_new_control_group(self):
        old_group = object()
        communication = NpuFTCommunication(
            store=Mock(),
            original_rank=0,
            timeout_sec=5,
            mlp_sync_group=old_group,
            active_original_ranks=(0, 1, 3),
            generation=1,
        )
        with (
            patch("sglang.srt.utils.init_custom_process_group", return_value=object()),
            patch(
                "sglang.srt.distributed.parallel_state.get_moe_ep_group",
                return_value=Mock(device_group=object()),
            ),
            patch(
                "sglang.srt.fault_tolerance.npu_communication.PrefixStore",
                return_value=Mock(),
            ),
            patch(
                "sglang.srt.fault_tolerance.npu_communication.dist.barrier",
                side_effect=RuntimeError("timeout"),
            ),
            patch(
                "sglang.srt.fault_tolerance.npu_communication.set_eplb_process_group_context"
            ) as install,
        ):
            with self.assertRaisesRegex(RuntimeError, "timeout"):
                communication.rebuild_survivor_groups(
                    [True, True, True, True], membership_epoch=2
                )
        install.assert_not_called()
        self.assertIs(communication.mlp_sync_group, old_group)
        self.assertEqual(communication.generation, 1)

    def test_mlp_sync_gather_has_bounded_wait(self):
        work = Mock()
        work.wait.return_value = True
        output = torch.empty(3)
        local = torch.tensor([2])

        with patch(
            "sglang.srt.fault_tolerance.npu_communication.dist.all_gather_into_tensor",
            return_value=work,
        ) as all_gather:
            all_gather_into_tensor_with_timeout(
                output,
                local,
                group="mlp-sync-gloo",
                timeout_sec=5,
            )

        all_gather.assert_called_once_with(
            output,
            local,
            group="mlp-sync-gloo",
            async_op=True,
        )
        work.wait.assert_called_once_with(timeout=timedelta(seconds=5))

    def test_mlp_sync_timeout_enters_ft_control_loop(self):
        work = Mock()
        work.wait.return_value = False

        with (
            patch(
                "sglang.srt.fault_tolerance.npu_communication.dist.all_gather_into_tensor",
                return_value=work,
            ),
            self.assertRaisesRegex(RuntimeError, "entering the FT control loop"),
        ):
            all_gather_into_tensor_with_timeout(
                torch.empty(3),
                torch.tensor([2]),
                group="mlp-sync-gloo",
                timeout_sec=5,
            )

    def test_mlp_sync_wait_failure_is_tagged(self):
        work = Mock()
        work.wait.side_effect = RuntimeError("peer disconnected")

        with (
            patch(
                "sglang.srt.fault_tolerance.npu_communication.dist.all_gather_into_tensor",
                return_value=work,
            ),
            self.assertRaisesRegex(RuntimeError, "entering the FT control loop"),
        ):
            all_gather_into_tensor_with_timeout(
                torch.empty(3),
                torch.tensor([2]),
                group="mlp-sync-gloo",
                timeout_sec=5,
            )


if __name__ == "__main__":
    unittest.main()
