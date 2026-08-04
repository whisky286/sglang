"""Unit tests for the complete NPU communication-matrix analyzer."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _load_analyzer_module():
    repo_root = Path(__file__).resolve().parents[4]
    analyzer_path = (
        repo_root / "test" / "manual" / "ascend" / "analyze_npu_communication_matrix.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_npu_communication_matrix", analyzer_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_trace(root: Path, rank: int, events):
    trace_root = root / f"rank-{rank}"
    output_root = trace_root / "ASCEND_PROFILER_OUTPUT"
    output_root.mkdir(parents=True)
    (trace_root / f"profiler_info_{rank}.json").write_text("{}", encoding="utf-8")
    (output_root / "trace_view.json").write_text(
        json.dumps({"traceEvents": events}), encoding="utf-8"
    )


class TestNPUCommunicationMatrixAnalyzer(CustomTestCase):
    def test_complete_matrix_distinguishes_graph_internal_and_external(self):
        analyzer = _load_analyzer_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            manifest_dir = run_dir / "stage-0-manifest"
            capture_dir = run_dir / "stage-1a-capture"
            decode_dir = run_dir / "stage-1b-healthy-replay"
            manifest_dir.mkdir()
            (manifest_dir / "resolved-server-info.json").write_text(
                json.dumps(
                    {
                        "tp_size": 4,
                        "dp_size": 4,
                        "ep_size": 4,
                        "attn_cp_size": 1,
                        "enable_dp_attention": True,
                        "enable_dp_lm_head": True,
                    }
                ),
                encoding="utf-8",
            )

            external_specs = (
                analyzer.EXPECTED_SOURCES[0],
                analyzer.EXPECTED_SOURCES[1],
                analyzer.EXPECTED_SOURCES[4],
                analyzer.EXPECTED_SOURCES[5],
            )
            internal_specs = (
                analyzer.EXPECTED_SOURCES[2],
                analyzer.EXPECTED_SOURCES[3],
            )
            lines = []
            sequence = 0
            for rank in range(4):
                for spec in analyzer.EXPECTED_SOURCES:
                    sequence += 1
                    backend = (
                        "gloo"
                        if spec["id"] == "scheduler_control_broadcast"
                        else "hccl"
                    )
                    begin = {
                        "event": "BEGIN",
                        "pid": 1000 + rank,
                        "seq": sequence,
                        "global_rank": rank,
                        "scope": spec["scope"],
                        "source": spec["source"],
                        "op": spec["op"],
                        "group_size": 4,
                        "ranks": [0, 1, 2, 3],
                        "backend": backend,
                        "tensors": (
                            [] if backend == "gloo" else [{"device": f"npu:{rank}"}]
                        ),
                        "extra": {
                            "profile_device_event_patterns": list(
                                spec["device_event_patterns"]
                            )
                        },
                    }
                    lines.extend(
                        [
                            analyzer.TRACE_PREFIX + json.dumps(begin),
                            analyzer.TRACE_PREFIX
                            + json.dumps({**begin, "event": "RETURN"}),
                        ]
                    )
                capture_events = [
                    {"name": analyzer.GRAPH_CAPTURE, "pid": 1000 + rank},
                    *(
                        {
                            "name": analyzer._profile_marker(spec),
                            "pid": 1000 + rank,
                        }
                        for spec in internal_specs
                    ),
                ]
                decode_events = [
                    {"name": analyzer.GRAPH_REPLAY, "pid": 1000 + rank},
                    *(
                        {
                            "name": analyzer._profile_marker(spec),
                            "pid": 1000 + rank,
                        }
                        for spec in external_specs
                    ),
                    {"name": "gloo:broadcast", "pid": 1000 + rank},
                    {"name": "hccl:all_gather", "pid": 1000 + rank},
                    {"name": "MoeLowLatencyDispatchV2", "pid": 1000 + rank},
                    {"name": "MoeLowLatencyCombineV2", "pid": 1000 + rank},
                    {"name": "hccl:all_reduce", "pid": 1000 + rank},
                    {"name": "hccl:send", "pid": 1000 + rank},
                ]
                _write_trace(capture_dir, rank, capture_events)
                _write_trace(decode_dir, rank, decode_events)
            (run_dir / "server.log").write_text("\n".join(lines), encoding="utf-8")

            summary = analyzer.analyze(run_dir)

        rows = {row["id"]: row for row in summary["matrix"]}
        self.assertTrue(summary["coverage"]["inventory_complete"])
        self.assertEqual(
            rows["deepep_low_latency_dispatch"]["graph_membership"]["verdict"],
            "graph_internal",
        )
        self.assertEqual(
            rows["deepep_low_latency_combine"]["graph_membership"]["verdict"],
            "graph_internal",
        )
        self.assertEqual(
            rows["mlp_sync_all_gather"]["graph_membership"]["verdict"],
            "graph_external",
        )
        self.assertEqual(
            rows["eplb_weight_p2p"]["graph_membership"]["verdict"],
            "graph_external",
        )
        self.assertEqual(summary["coverage"]["decode_profiled_ranks"], [0, 1, 2, 3])

    def test_unknown_profile_communication_keeps_inventory_incomplete(self):
        analyzer = _load_analyzer_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "stage-0-manifest").mkdir()
            (run_dir / "server.log").write_text("", encoding="utf-8")
            _write_trace(
                run_dir / "stage-1b-healthy-replay",
                0,
                [
                    {"name": analyzer.GRAPH_REPLAY},
                    {"name": "gloo:all_gather"},
                ],
            )
            summary = analyzer.analyze(run_dir)

        self.assertFalse(summary["coverage"]["inventory_complete"])
        self.assertEqual(
            summary["profile_communication_event_inventory"][
                "unknown_event_name_counts"
            ],
            {"gloo:all_gather": 1},
        )


if __name__ == "__main__":
    unittest.main()
