"""Unit tests for the three-level NPU EPLB P2P staging policy."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.environ import envs
from sglang.srt.eplb import expert_location_updater
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeDevice:
    def __init__(self, device_type: str):
        self.type = device_type

    def __hash__(self):
        return hash(self.type)


class _FakeTensor:
    def __init__(self, device_type: str, storage_offset: int = 0, npu_format: int = 2):
        self.device = _FakeDevice(device_type)
        self._storage_offset = storage_offset
        self.npu_format = npu_format
        self.shape = (4, 4)
        self.dtype = torch.float16

    def storage_offset(self):
        return self._storage_offset

    def data_ptr(self):
        return id(self)

    def stride(self):
        return (4, 1)


class _FakeP2POp:
    def __init__(self, op, tensor, peer, group=None, tag=0):
        self.op = op
        self.tensor = tensor
        self.peer = peer
        self.group = group
        self.tag = tag


class TestExpertLocationUpdaterNPUP2P(CustomTestCase):
    def test_nd_mode_stages_offset_zero_npu_tensor(self):
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
            envs.SGLANG_NPU_EPLB_P2P_STAGING_MODE.override("nd"),
            envs.SGLANG_NPU_EPLB_P2P_USE_ND_STAGING.override(False),
            patch.object(expert_location_updater, "P2POp", _FakeP2POp),
            patch.object(
                expert_location_updater,
                "_new_npu_nd_staging_like",
                return_value=staged,
            ) as new_staging,
            patch.object(
                expert_location_updater,
                "_new_npu_offset_zero_staging_like",
            ) as new_offset_staging,
        ):
            staged_ops, recv_copy_infos = expert_location_updater._stage_npu_p2p_ops(
                [op]
            )

        new_staging.assert_called_once_with(original)
        new_offset_staging.assert_not_called()
        self.assertIs(staged_ops[0].tensor, staged)
        self.assertIs(staged_ops[0].group, group)
        self.assertEqual(staged_ops[0].tag, 7)
        self.assertEqual(recv_copy_infos, [(staged, original)])

    def test_offset_mode_only_stages_nonzero_offset_tensor(self):
        offset_zero = _FakeTensor("npu", storage_offset=0, npu_format=29)
        offset_nonzero = _FakeTensor("npu", storage_offset=9, npu_format=29)
        staged = _FakeTensor("npu", storage_offset=0, npu_format=29)
        direct_op = _FakeP2POp(torch.distributed.irecv, offset_zero, peer=1)
        staged_op = _FakeP2POp(torch.distributed.irecv, offset_nonzero, peer=2)

        with (
            envs.SGLANG_NPU_EPLB_P2P_STAGING_MODE.override("offset"),
            patch.object(expert_location_updater, "P2POp", _FakeP2POp),
            patch.object(
                expert_location_updater,
                "_new_npu_nd_staging_like",
            ) as new_nd_staging,
            patch.object(
                expert_location_updater,
                "_new_npu_offset_zero_staging_like",
                return_value=staged,
            ) as new_offset_staging,
        ):
            staged_ops, recv_copy_infos = expert_location_updater._stage_npu_p2p_ops(
                [direct_op, staged_op]
            )

        new_nd_staging.assert_not_called()
        new_offset_staging.assert_called_once_with(offset_nonzero)
        self.assertIs(staged_ops[0], direct_op)
        self.assertIs(staged_ops[0].tensor, offset_zero)
        self.assertIs(staged_ops[1].tensor, staged)
        self.assertEqual(recv_copy_infos, [(staged, offset_nonzero)])

    def test_direct_mode_keeps_nonzero_offset_npu_tensor(self):
        original = _FakeTensor("npu", storage_offset=9)
        op = _FakeP2POp(torch.distributed.irecv, original, peer=1)

        with (
            envs.SGLANG_NPU_EPLB_P2P_STAGING_MODE.override("direct"),
            patch.object(
                expert_location_updater,
                "_new_npu_nd_staging_like",
            ) as new_nd_staging,
            patch.object(
                expert_location_updater,
                "_new_npu_offset_zero_staging_like",
            ) as new_offset_staging,
        ):
            staged_ops, recv_copy_infos = expert_location_updater._stage_npu_p2p_ops(
                [op]
            )

        new_nd_staging.assert_not_called()
        new_offset_staging.assert_not_called()
        self.assertIs(staged_ops[0], op)
        self.assertIs(staged_ops[0].tensor, original)
        self.assertEqual(recv_copy_infos, [])

    def test_offset_mode_send_uses_same_format_copy(self):
        original = _FakeTensor("npu", storage_offset=9, npu_format=29)
        staged = _FakeTensor("npu", storage_offset=0, npu_format=29)
        op = _FakeP2POp(torch.distributed.isend, original, peer=1)

        with (
            envs.SGLANG_NPU_EPLB_P2P_STAGING_MODE.override("offset"),
            patch.object(expert_location_updater, "P2POp", _FakeP2POp),
            patch.object(
                expert_location_updater,
                "_new_npu_offset_zero_staging_like",
                return_value=staged,
            ),
            patch.object(
                expert_location_updater,
                "_copy_expert_tensor_",
            ) as copy_expert_tensor,
        ):
            staged_ops, recv_copy_infos = expert_location_updater._stage_npu_p2p_ops(
                [op]
            )

        copy_expert_tensor.assert_called_once_with(staged, original)
        self.assertIs(staged_ops[0].tensor, staged)
        self.assertEqual(recv_copy_infos, [])

    def test_non_npu_tensor_keeps_zero_copy_path(self):
        original = _FakeTensor("cuda", storage_offset=9)
        op = _FakeP2POp(torch.distributed.isend, original, peer=1)

        with envs.SGLANG_NPU_EPLB_P2P_STAGING_MODE.override("nd"):
            staged_ops, recv_copy_infos = expert_location_updater._stage_npu_p2p_ops(
                [op]
            )

        self.assertIs(staged_ops[0], op)
        self.assertEqual(recv_copy_infos, [])

    def test_legacy_boolean_is_used_when_new_mode_is_unset(self):
        with (
            envs.SGLANG_NPU_EPLB_P2P_STAGING_MODE.override(None),
            envs.SGLANG_NPU_EPLB_P2P_USE_ND_STAGING.override(True),
        ):
            self.assertEqual(
                expert_location_updater.get_npu_eplb_p2p_staging_mode(),
                "nd",
            )

        with (
            envs.SGLANG_NPU_EPLB_P2P_STAGING_MODE.override(None),
            envs.SGLANG_NPU_EPLB_P2P_USE_ND_STAGING.override(False),
        ):
            self.assertEqual(
                expert_location_updater.get_npu_eplb_p2p_staging_mode(),
                "direct",
            )

    def test_invalid_staging_mode_is_rejected(self):
        with (
            envs.SGLANG_NPU_EPLB_P2P_STAGING_MODE.override("unknown"),
            self.assertRaisesRegex(ValueError, "must be one of"),
        ):
            expert_location_updater.get_npu_eplb_p2p_staging_mode()

    def test_same_format_offset_recv_does_not_cast_again(self):
        staged = _FakeTensor("npu", storage_offset=0, npu_format=29)
        destination = _FakeTensor("npu", storage_offset=7, npu_format=29)
        fake_torch_npu = SimpleNamespace(
            get_npu_format=lambda tensor: tensor.npu_format
        )

        with (
            patch.dict(sys.modules, {"torch_npu": fake_torch_npu}),
            patch(
                "sglang.srt.hardware_backend.npu.utils.is_npu_internal_format_tensor",
                return_value=True,
            ),
            patch.object(
                expert_location_updater,
                "_copy_expert_tensor_",
            ) as copy_expert_tensor,
        ):
            expert_location_updater._copy_staged_p2p_recvs([(staged, destination)])

        copy_expert_tensor.assert_called_once_with(destination, staged)


if __name__ == "__main__":
    unittest.main()
