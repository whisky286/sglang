"""CPU-only tests for NPU graph profiler phase ranges."""

import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
    NPUCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNPUGraphProfilerRanges(CustomTestCase):
    def setUp(self):
        self.backend = NPUCudaGraphBackend.__new__(NPUCudaGraphBackend)
        self.backend._graphs = {}
        self.backend._outputs = {}
        self.backend._pool = object()
        self.backend._capture_stream = object()
        self.backend._device_module = MagicMock()
        self.backend._tp_group = MagicMock()
        self.backend._memory_saver_adapter = None
        self.backend._enable_torch_compile = False

    @staticmethod
    def _record_ranges(output):
        @contextmanager
        def record_function(name):
            output.append(name)
            yield

        return record_function

    def test_capture_ranges_separate_warmup_from_recording(self):
        phases = []
        graph = MagicMock()
        forward = MagicMock(return_value="captured-output")

        @contextmanager
        def graph_context(*args, **kwargs):
            yield

        fake_npu = SimpleNamespace(
            NPUGraph=MagicMock(return_value=graph),
            graph=graph_context,
        )
        with (
            patch.dict(sys.modules, {"torch_npu": MagicMock()}),
            patch.object(torch, "npu", fake_npu, create=True),
            patch.object(
                torch.profiler,
                "record_function",
                side_effect=self._record_ranges(phases),
            ),
        ):
            self.backend.capture_one("shape", forward)

        self.assertEqual(
            phases,
            ["GRAPH_WARMUP", "GRAPH_WARMUP", "GRAPH_CAPTURE"],
        )
        self.assertEqual(forward.call_count, 3)
        self.assertEqual(self.backend._device_module.synchronize.call_count, 2)
        self.assertEqual(self.backend._tp_group.barrier.call_count, 2)
        self.assertIs(self.backend._graphs["shape"], graph)
        self.assertEqual(self.backend._outputs["shape"], "captured-output")

    def test_both_replay_paths_emit_replay_range(self):
        phases = []
        graph = MagicMock()
        self.backend._graphs["shape"] = graph
        self.backend._outputs["shape"] = "captured-output"

        with patch.object(
            torch.profiler,
            "record_function",
            side_effect=self._record_ranges(phases),
        ):
            direct_output = self.backend.replay("shape", None)
            updated_output = self.backend.replay_with_input_update(
                "shape",
                seq_lens=None,
                cpu_update_input=[{"context_lens": [1, 2]}],
            )

        self.assertEqual(phases, ["GRAPH_REPLAY", "GRAPH_REPLAY"])
        self.assertEqual(graph.replay.call_count, 2)
        graph.update.assert_called_once_with(
            cpu_update_input=[{"context_lens": [1, 2]}]
        )
        self.assertEqual(direct_output, "captured-output")
        self.assertEqual(updated_output, "captured-output")


if __name__ == "__main__":
    unittest.main()
