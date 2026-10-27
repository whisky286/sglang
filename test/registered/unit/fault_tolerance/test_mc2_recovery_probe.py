"""CPU checks for the numerical gate in the manual MC2 recovery probe."""

import importlib.util
import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMC2RecoveryProbe(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        path = (
            Path(__file__).resolve().parents[4]
            / "test/manual/ep/probe_npu_rank_recovery.py"
        )
        spec = importlib.util.spec_from_file_location("mc2_recovery_probe", path)
        cls.probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.probe)

    def setUp(self):
        self.buffers = []
        self.config = {"hidden": 16, "tokens": 4, "local_experts": 8, "topk": 4}
        self.output_mode = "correct"
        self.dispatch_mode = "bf16"
        self.comm_name = "candidate_hccl"
        test = self

        class Buffer:
            get_low_latency_rdma_size_hint = Mock(return_value=4096)

            def __init__(self, group, *args, **kwargs):
                self.group = group
                self.kwargs = kwargs
                self.moe_all_to_all_group_name = test.comm_name
                test.buffers.append(self)

            def low_latency_dispatch(self, x, ids, max_tokens, experts, **kwargs):
                self.input = x
                self.ids = ids
                self.dispatch_kwargs = kwargs
                self.max_tokens = max_tokens
                self.experts = experts
                packed = x.clone()
                if test.dispatch_mode == "quantized":
                    packed = (packed, torch.ones(1))
                return (
                    packed,
                    torch.ones(test.config["local_experts"], dtype=torch.int64),
                    "handle",
                    None,
                    None,
                )

            def low_latency_combine(
                self, *, x, topk_idx, topk_weights, handle, **kwargs
            ):
                output = (x * topk_weights.sum(dim=1, keepdim=True)).to(x.dtype)
                if test.output_mode == "incorrect":
                    output += 1
                elif test.output_mode == "nan":
                    output.fill_(float("nan"))
                elif test.output_mode == "shape":
                    output = output[:, :-1]
                elif test.output_mode == "dtype":
                    output = output.float()
                return output, None, None

        self.deep_ep = ModuleType("deep_ep")
        self.deep_ep.__file__ = "/test/deep_ep/__init__.py"
        self.deep_ep.Buffer = Buffer

    def run_roundtrip(self, group):
        synchronize = Mock()
        with patch.dict("sys.modules", {"deep_ep": self.deep_ep}):
            result = self.probe._mc2_roundtrip(
                group,
                original_rank=2,
                config=self.config,
                device="cpu",
                synchronize=synchronize,
            )
        return result, synchronize

    def test_uses_fresh_buffer_and_candidate_group_for_each_membership(self):
        for size in (4, 3, 4):
            group = Mock()
            group.size.return_value = size
            group.rank.return_value = 1
            (buffer, report), synchronize = self.run_roundtrip(group)
            self.assertIs(buffer.group, group)
            self.assertEqual(buffer.experts, size * self.config["local_experts"])
            self.assertEqual(
                buffer.kwargs["num_qps_per_rank"], self.config["local_experts"]
            )
            self.assertFalse(buffer.dispatch_kwargs["use_fp8"])
            self.assertFalse(buffer.dispatch_kwargs["async_finish"])
            self.assertEqual(report["max_abs_error"], 0)
            self.assertEqual(report["group_size"], size)
            self.assertEqual(synchronize.call_count, 2)
            self.assertTrue(bool((buffer.ids >= 0).all()))
            self.assertTrue(bool((buffer.ids < buffer.experts).all()))
            for row in buffer.ids:
                self.assertEqual(row.unique().numel(), self.config["topk"])
        self.assertEqual(len({id(buffer) for buffer in self.buffers}), 3)

    def test_rejects_wrong_output_nan_and_wrong_shape(self):
        group = Mock()
        group.size.return_value, group.rank.return_value = 3, 1
        for mode in ("incorrect", "nan", "shape", "dtype"):
            with self.subTest(mode=mode):
                self.output_mode = mode
                with self.assertRaisesRegex(RuntimeError, "mismatch"):
                    self.run_roundtrip(group)

    def test_rejects_quantized_dispatch_in_bf16_identity_test(self):
        group = Mock()
        group.size.return_value, group.rank.return_value = 3, 1
        self.dispatch_mode = "quantized"
        with self.assertRaisesRegex(RuntimeError, "BF16 dispatch"):
            self.run_roundtrip(group)

    def test_rejects_empty_communicator_name(self):
        group = Mock()
        group.size.return_value, group.rank.return_value = 3, 1
        self.comm_name = ""
        with self.assertRaisesRegex(RuntimeError, "empty HCCL"):
            self.run_roundtrip(group)

    def test_mc2_mode_cannot_silently_run_as_cpu_probe(self):
        error = io.StringIO()
        with (
            patch("sys.argv", ["probe", "--backend", "gloo", "--mc2"]),
            patch.object(self.probe.dist, "TCPStore") as store,
            redirect_stderr(error),
        ):
            with self.assertRaises(SystemExit) as raised:
                self.probe.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--mc2 requires --backend hccl", error.getvalue())
        store.assert_not_called()


if __name__ == "__main__":
    unittest.main()
