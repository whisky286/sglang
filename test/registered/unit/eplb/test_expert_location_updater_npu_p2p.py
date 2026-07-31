"""Unit tests for offset-zero NPU staging in EPLB P2P weight updates."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import torch
from torch.distributed import P2POp

from sglang.srt.eplb import expert_location_updater
from sglang.test.test_utils import CustomTestCase


class TestExpertLocationUpdaterNPUP2P(CustomTestCase):
    def test_non_npu_views_keep_existing_zero_copy_path(self):
        tensor = torch.arange(12).reshape(3, 4)[1]
        op = P2POp(torch.distributed.isend, tensor, peer=1)

        staged_ops, recv_copy_infos = expert_location_updater._stage_npu_p2p_ops([op])

        self.assertIs(staged_ops[0], op)
        self.assertEqual(recv_copy_infos, [])

    def test_staged_send_and_recv_have_zero_offset_and_copy_back(self):
        send_tensor = torch.arange(12).reshape(3, 4)[1]
        recv_tensor = torch.zeros(12).reshape(3, 4)[2]
        ops = [
            P2POp(torch.distributed.isend, send_tensor, peer=1),
            P2POp(torch.distributed.irecv, recv_tensor, peer=1),
        ]

        with patch.object(
            expert_location_updater,
            "_needs_npu_p2p_staging",
            return_value=True,
        ):
            staged_ops, recv_copy_infos = expert_location_updater._stage_npu_p2p_ops(
                ops
            )

        staged_send, staged_recv = (op.tensor for op in staged_ops)
        self.assertEqual(staged_send.storage_offset(), 0)
        self.assertEqual(staged_recv.storage_offset(), 0)
        self.assertTrue(torch.equal(staged_send, send_tensor))
        self.assertNotEqual(staged_send.data_ptr(), send_tensor.data_ptr())

        staged_recv.fill_(7)
        expert_location_updater._copy_staged_p2p_recvs(recv_copy_infos)
        self.assertTrue(torch.equal(recv_tensor, torch.full_like(recv_tensor, 7)))

    def test_multicast_reuses_one_staged_send_tensor(self):
        send_tensor = torch.arange(12).reshape(3, 4)[1]
        ops = [
            P2POp(torch.distributed.isend, send_tensor, peer=1),
            P2POp(torch.distributed.isend, send_tensor, peer=2),
        ]

        with patch.object(
            expert_location_updater,
            "_needs_npu_p2p_staging",
            return_value=True,
        ):
            staged_ops, recv_copy_infos = expert_location_updater._stage_npu_p2p_ops(
                ops
            )

        self.assertIs(staged_ops[0].tensor, staged_ops[1].tensor)
        self.assertEqual(staged_ops[0].tensor.storage_offset(), 0)
        self.assertNotEqual(staged_ops[0].tensor.data_ptr(), send_tensor.data_ptr())
        self.assertEqual(recv_copy_infos, [])

    def test_weight_update_uses_staged_buffers_and_preserves_recv_result(self):
        routed_expert_weights = [
            torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            )
        ]
        temp_buffers = [torch.empty_like(routed_expert_weights[0])]
        observed_ops = []

        class FakeRequest:
            def __init__(self, op):
                self.op = op

            def wait(self):
                if self.op.op == torch.distributed.irecv:
                    self.op.tensor.copy_(torch.tensor([20.0, 21.0]))

        def fake_batch_isend_irecv(ops):
            observed_ops.extend(ops)
            for op in ops:
                self.assertEqual(op.tensor.storage_offset(), 0)
            return [FakeRequest(op) for op in ops]

        with (
            patch.object(
                expert_location_updater,
                "_needs_npu_p2p_staging",
                return_value=True,
            ),
            patch.object(
                expert_location_updater.torch.distributed,
                "batch_isend_irecv",
                side_effect=fake_batch_isend_irecv,
            ),
            patch.object(
                expert_location_updater,
                "trace_collective",
                return_value=nullcontext(),
            ),
            patch.object(
                expert_location_updater.ElasticEPStateManager,
                "instance",
                return_value=None,
            ),
            patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=MagicMock(),
            ),
            patch.object(
                expert_location_updater.envs.SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE,
                "get",
                return_value=4,
            ),
        ):
            expert_location_updater.update_expert_weights_single_layer(
                routed_experts_weights=routed_expert_weights,
                temp_buffers=temp_buffers,
                old_physical_to_logical_map=[0, 1, 2, 3],
                new_physical_to_logical_map=[0, 2, 1, 3],
                num_local_physical_experts=2,
                num_gpu_per_node=2,
                rank=0,
                world_size=2,
            )

        self.assertEqual(len(observed_ops), 2)
        self.assertTrue(
            torch.equal(
                routed_expert_weights[0],
                torch.tensor(
                    [
                        [1.0, 2.0],
                        [20.0, 21.0],
                    ]
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
