"""Unit tests for DeepEP auto-mode expert distribution gathering."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.eplb import expert_distribution
from sglang.test.test_utils import CustomTestCase


class TestDeepEPAutoSinglePassGatherer(CustomTestCase):
    def _make_gatherer(self):
        metadata = MagicMock()
        normal_gatherer = MagicMock(spec=expert_distribution._SinglePassGatherer)
        low_latency_gatherer = MagicMock(spec=expert_distribution._SinglePassGatherer)

        with patch.object(
            expert_distribution,
            "_DeepepLowLatencySinglePassGatherer",
            return_value=low_latency_gatherer,
        ):
            gatherer = expert_distribution._DeepEPAutoSinglePassGatherer(
                metadata,
                rank=0,
                normal_gatherer=normal_gatherer,
            )

        return gatherer, normal_gatherer, low_latency_gatherer

    def test_reported_server_configuration_is_supported(self):
        server_args = SimpleNamespace(
            expert_distribution_recorder_mode="stat",
            moe_a2a_backend="deepep",
            deepep_mode="auto",
            elastic_ep_backend=None,
        )
        metadata = MagicMock()
        normal_gatherer = MagicMock(spec=expert_distribution._SinglePassGatherer)
        low_latency_gatherer = MagicMock(spec=expert_distribution._SinglePassGatherer)

        with (
            patch.object(
                expert_distribution,
                "_SelectExpertsSinglePassGatherer",
                return_value=normal_gatherer,
            ),
            patch.object(
                expert_distribution,
                "_DeepepLowLatencySinglePassGatherer",
                return_value=low_latency_gatherer,
            ),
        ):
            gatherer = expert_distribution._SinglePassGatherer.init_new(
                server_args, metadata, rank=0
            )

        self.assertIsInstance(
            gatherer, expert_distribution._DeepEPAutoSinglePassGatherer
        )

    def test_forward_batch_restores_mode_for_graph_replay(self):
        gatherer, normal_gatherer, low_latency_gatherer = self._make_gatherer()
        normal_gatherer.collect.return_value = {"path": "normal"}
        low_latency_gatherer.collect.return_value = {"path": "low_latency"}

        gatherer.reset()
        gatherer.on_forward_pass_start(SimpleNamespace(is_extend_in_batch=False))

        self.assertEqual(gatherer.collect(), {"path": "low_latency"})
        low_latency_gatherer.on_forward_pass_start.assert_called_once()
        normal_gatherer.on_forward_pass_start.assert_not_called()

    def test_select_experts_follows_runtime_auto_mode(self):
        gatherer, normal_gatherer, low_latency_gatherer = self._make_gatherer()
        topk_ids = torch.tensor([[1, 2]], dtype=torch.int32)

        with patch.object(
            expert_distribution, "get_is_extend_in_batch", return_value=True
        ):
            gatherer.on_select_experts(layer_idx=3, topk_ids=topk_ids)
        normal_gatherer.on_select_experts.assert_called_once_with(3, topk_ids)
        low_latency_gatherer.on_select_experts.assert_not_called()

        with patch.object(
            expert_distribution, "get_is_extend_in_batch", return_value=False
        ):
            gatherer.on_select_experts(layer_idx=4, topk_ids=topk_ids)
        low_latency_gatherer.on_select_experts.assert_called_once_with(4, topk_ids)

    def test_dispatch_hook_selects_matching_collector(self):
        gatherer, normal_gatherer, low_latency_gatherer = self._make_gatherer()
        normal_gatherer.collect.return_value = {"path": "normal"}
        low_latency_gatherer.collect.return_value = {"path": "low_latency"}
        local_count = torch.tensor([2, 1], dtype=torch.int32)

        gatherer.on_deepep_dispatch_low_latency(
            layer_idx=2,
            local_physical_count_of_layer=local_count,
        )

        low_latency_gatherer.on_deepep_dispatch_low_latency.assert_called_once_with(
            2, local_count
        )
        self.assertEqual(gatherer.collect(), {"path": "low_latency"})
        normal_gatherer.collect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
