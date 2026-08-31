"""Unit tests for NPU EPLB P2P staging policy."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb import expert_location_updater
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeTensor:
    def __init__(self, device_type: str, storage_offset: int = 0):
        self.device = SimpleNamespace(type=device_type)
        self._storage_offset = storage_offset

    def storage_offset(self):
        return self._storage_offset


class _FakeP2POp:
    def __init__(self, op, tensor, peer, group=None, tag=0):
        self.op = op
        self.tensor = tensor
        self.peer = peer
        self.group = group
        self.tag = tag


class TestExpertLocationUpdaterNPUP2P(CustomTestCase):
    def test_offset_zero_npu_tensor_always_uses_staging(self):
        original = _FakeTensor("npu", storage_offset=0)
        staged = _FakeTensor("npu", storage_offset=0)
        group = object()
        op = _FakeP2POp(
            torch.distributed.irecv,
            original,
            peer=1,
            group=group,
            tag=7,
        )

        with (
            envs.SGLANG_NPU_EPLB_P2P_USE_ND_STAGING.override(True),
            patch.object(expert_location_updater, "P2POp", _FakeP2POp),
            patch.object(
                expert_location_updater,
                "_new_npu_nd_staging_like",
                return_value=staged,
            ) as new_staging,
        ):
            staged_ops, recv_copy_infos = (
                expert_location_updater._stage_npu_p2p_ops([op])
            )

        new_staging.assert_called_once_with(original)
        self.assertIs(staged_ops[0].tensor, staged)
        self.assertIs(staged_ops[0].group, group)
        self.assertEqual(staged_ops[0].tag, 7)
        self.assertEqual(recv_copy_infos, [(staged, original)])

    def test_ablation_switch_keeps_original_npu_p2p_tensor(self):
        original = _FakeTensor("npu", storage_offset=9)
        op = _FakeP2POp(torch.distributed.irecv, original, peer=1)

        with (
            envs.SGLANG_NPU_EPLB_P2P_USE_ND_STAGING.override(False),
            patch.object(
                expert_location_updater,
                "_new_npu_nd_staging_like",
            ) as new_staging,
        ):
            staged_ops, recv_copy_infos = (
                expert_location_updater._stage_npu_p2p_ops([op])
            )

        new_staging.assert_not_called()
        self.assertIs(staged_ops[0], op)
        self.assertIs(staged_ops[0].tensor, original)
        self.assertEqual(recv_copy_infos, [])

    def test_non_npu_tensor_keeps_zero_copy_path(self):
        original = _FakeTensor("cuda", storage_offset=9)
        op = _FakeP2POp(torch.distributed.isend, original, peer=1)

        with envs.SGLANG_NPU_EPLB_P2P_USE_ND_STAGING.override(True):
            staged_ops, recv_copy_infos = (
                expert_location_updater._stage_npu_p2p_ops([op])
            )

        self.assertIs(staged_ops[0], op)
        self.assertEqual(recv_copy_infos, [])


if __name__ == "__main__":
    unittest.main()
