import unittest

import torch

from sglang.srt.elastic_ep.npu_mc2 import (
    NpuMC2ElasticInfo,
    build_mc2_elastic_info,
    compact_mc2_physical_expert_ids,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNpuMC2ElasticInfo(CustomTestCase):
    def test_build_elastic_info_for_sparse_survivors(self):
        info = build_mc2_elastic_info(
            [True, False, True, True],
            original_ep_size=4,
            num_local_physical_experts=3,
        )

        self.assertEqual(
            info.tolist(),
            [1, 3, 0, 9, 0, -1, 1, 2, 0, 2, 3, -1],
        )

    def test_compact_physical_expert_ids_uses_effective_rank_namespace(self):
        info = build_mc2_elastic_info(
            [True, False, True, True],
            original_ep_size=4,
            num_local_physical_experts=3,
        )
        physical_ids = torch.tensor([[0, 7, 11, -1]])

        compact = compact_mc2_physical_expert_ids(
            physical_ids,
            elastic_info=info,
            original_ep_size=4,
            num_local_physical_experts=3,
        )

        self.assertEqual(compact.tolist(), [[0, 4, 8, -1]])

    def test_update_preserves_graph_captured_tensor_address(self):
        elastic_info = NpuMC2ElasticInfo.create(
            [True, True, True, True],
            original_ep_size=4,
            num_physical_experts=12,
            device="cpu",
        )
        address = elastic_info.tensor.data_ptr()

        elastic_info.update([True, False, True, True])

        self.assertEqual(elastic_info.tensor.data_ptr(), address)
        self.assertEqual(
            elastic_info.tensor.tolist(),
            [1, 3, 0, 9, 0, -1, 1, 2, 0, 2, 3, -1],
        )

    def test_recovery_restores_capacity_and_mapping_in_existing_storage(self):
        elastic_info = NpuMC2ElasticInfo.create(
            [True] * 4, original_ep_size=4, num_physical_experts=12, device="cpu"
        )
        address = elastic_info.tensor.data_ptr()
        physical_ids = torch.tensor([[0, 4, 8, 11]])
        for mask, expected_ids in (
            ([True, False, True, True], [[0, -1, 5, 8]]),
            ([True, True, True, True], [[0, 4, 8, 11]]),
            ([True, False, True, False], [[0, -1, 5, -1]]),
            ([True, True, True, False], [[0, 4, 8, -1]]),
        ):
            with self.subTest(mask=mask):
                elastic_info.update(mask)
                mapped = compact_mc2_physical_expert_ids(
                    physical_ids,
                    elastic_info=elastic_info.tensor,
                    original_ep_size=4,
                    num_local_physical_experts=3,
                )
                self.assertEqual(mapped.tolist(), expected_ids)
                self.assertEqual(elastic_info.tensor.data_ptr(), address)
                self.assertEqual(elastic_info.tensor[1].item(), sum(mask))


if __name__ == "__main__":
    unittest.main()
