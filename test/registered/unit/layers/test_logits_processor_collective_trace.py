"""Unit tests for LM-head vocab-gather collective tracing."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers import logits_processor

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestLogitsProcessorCollectiveTrace(CustomTestCase):
    def test_attn_tp_vocab_gather_has_dedicated_trace_scope(self):
        processor = logits_processor.LogitsProcessor.__new__(
            logits_processor.LogitsProcessor
        )
        processor.vocab_size = 8
        processor.attn_tp_size = 2
        local_logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        group = MagicMock()
        group.device_group = object()

        def fake_all_gather(output, local):
            output[0].copy_(local)
            output[1].copy_(local + 10)

        with (
            patch.object(
                logits_processor,
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_group=group),
            ),
            patch.object(
                logits_processor,
                "attn_tp_all_gather_into_tensor",
                side_effect=fake_all_gather,
            ),
            patch.object(
                logits_processor,
                "trace_collective",
                return_value=nullcontext(),
            ) as trace,
            patch.object(
                logits_processor.torch.profiler,
                "record_function",
                return_value=nullcontext(),
            ),
        ):
            gathered = processor._gather_attn_tp_logits(local_logits)

        self.assertTrue(
            torch.equal(
                gathered,
                torch.tensor([[1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0]]),
            )
        )
        trace.assert_called_once()
        trace_kwargs = trace.call_args.kwargs
        self.assertEqual(trace.call_args.args, ("all_gather_into_tensor",))
        self.assertEqual(trace_kwargs["scope"], "lm_head_vocab")
        self.assertEqual(
            trace_kwargs["source"], "LogitsProcessor._gather_attn_tp_logits"
        )
        self.assertEqual(trace_kwargs["coordinator"], group)
        self.assertEqual(trace_kwargs["process_group"], group.device_group)
        self.assertEqual(trace_kwargs["extra"]["attn_tp_size"], 2)


if __name__ == "__main__":
    unittest.main()
