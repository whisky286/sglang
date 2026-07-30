"""CPU-only tests for opt-in large-TP communication tracing."""

import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch

import torch

import sglang.srt.distributed.collective_trace as collective_trace
from sglang.srt.distributed.collective_trace import (
    TRACE_PREFIX,
    trace_big_tp_collective,
    trace_collective,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeGroup:
    def __init__(self, unique_name="tp:0"):
        self.unique_name = unique_name
        self.ranks = [0, 1, 2, 3]
        self.world_size = 4
        self.rank_in_group = 1
        self.active_ranks_cpu = torch.tensor([1, 0, 1, 1], dtype=torch.int32)
        self.device_group = None


class TestCollectiveTrace(CustomTestCase):
    def _records(self, output):
        return [
            json.loads(line.removeprefix(TRACE_PREFIX))
            for line in output.splitlines()
            if line.startswith(TRACE_PREFIX)
        ]

    def _traced_group(self, unique_name="tp:0"):
        class TracedGroup(_FakeGroup):
            @trace_big_tp_collective("all_reduce")
            def all_reduce(self, tensor):
                return tensor

            @trace_big_tp_collective("outer")
            def outer(self, tensor):
                return self.all_reduce(tensor)

            @trace_big_tp_collective("failing")
            def failing(self, tensor):
                raise RuntimeError("expected failure")

        return TracedGroup(unique_name)

    def test_disabled_is_silent(self):
        buffer = io.StringIO()
        with patch.object(collective_trace, "_TRACE_ENABLED", False):
            with redirect_stdout(buffer):
                group = self._traced_group()
                group.all_reduce(torch.empty((2, 3)))
        self.assertEqual(buffer.getvalue(), "")

    def test_big_tp_trace_contains_only_tensor_metadata_and_active_mask(self):
        buffer = io.StringIO()
        tensor = torch.empty((2, 3), dtype=torch.bfloat16)
        with patch.object(collective_trace, "_TRACE_ENABLED", True):
            with redirect_stdout(buffer):
                result = self._traced_group().all_reduce(tensor)

        self.assertIs(result, tensor)
        records = self._records(buffer.getvalue())
        self.assertEqual([record["event"] for record in records], ["BEGIN", "RETURN"])
        self.assertEqual(records[0]["scope"], "big_tp")
        self.assertEqual(records[0]["group"], "tp:0")
        self.assertEqual(records[0]["ranks"], [0, 1, 2, 3])
        self.assertEqual(records[0]["active_mask_cpu"], [1, 0, 1, 1])
        self.assertEqual(records[0]["tensors"][0]["shape"], [2, 3])
        self.assertEqual(records[0]["tensors"][0]["dtype"], "torch.bfloat16")
        self.assertIn("device_completion_not_implied", records[1]["completion"])

    def test_non_big_tp_group_is_silent(self):
        buffer = io.StringIO()
        with patch.object(collective_trace, "_TRACE_ENABLED", True):
            with redirect_stdout(buffer):
                self._traced_group("attention_tp:0").all_reduce(torch.empty(1))
        self.assertEqual(buffer.getvalue(), "")

    def test_nested_group_calls_emit_one_operation(self):
        buffer = io.StringIO()
        with patch.object(collective_trace, "_TRACE_ENABLED", True):
            with redirect_stdout(buffer):
                self._traced_group().outer(torch.empty(1))
        records = self._records(buffer.getvalue())
        self.assertEqual(len(records), 2)
        self.assertEqual({record["op"] for record in records}, {"outer"})

    def test_error_is_recorded_and_reraised(self):
        buffer = io.StringIO()
        with patch.object(collective_trace, "_TRACE_ENABLED", True):
            with redirect_stdout(buffer):
                with self.assertRaisesRegex(RuntimeError, "expected failure"):
                    self._traced_group().failing(torch.empty(1))
        records = self._records(buffer.getvalue())
        self.assertEqual([record["event"] for record in records], ["BEGIN", "ERROR"])
        self.assertEqual(records[-1]["error_type"], "RuntimeError")

    def test_explicit_world_scope_is_traceable(self):
        buffer = io.StringIO()
        with patch.object(collective_trace, "_TRACE_ENABLED", True):
            with redirect_stdout(buffer):
                with trace_collective(
                    "broadcast",
                    coordinator=_FakeGroup(),
                    scope="eplb_world",
                    source="test",
                    tensors=torch.empty(4),
                ):
                    pass
        records = self._records(buffer.getvalue())
        self.assertEqual(records[0]["scope"], "eplb_world")
        self.assertEqual(records[0]["source"], "test")

    def test_tensor_metadata_is_bounded(self):
        buffer = io.StringIO()
        tensors = [torch.empty(1) for _ in range(20)]
        with patch.object(collective_trace, "_TRACE_ENABLED", True):
            with redirect_stdout(buffer):
                with trace_collective(
                    "batch_isend_irecv",
                    coordinator=_FakeGroup(),
                    scope="eplb_world_p2p",
                    tensors=tensors,
                ):
                    pass
        begin = self._records(buffer.getvalue())[0]
        self.assertEqual(begin["tensor_count"], 20)
        self.assertEqual(len(begin["tensors"]), 16)
        self.assertEqual(begin["tensor_metadata_truncated"], 4)


if __name__ == "__main__":
    import unittest

    unittest.main()
