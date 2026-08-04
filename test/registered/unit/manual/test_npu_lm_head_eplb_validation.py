"""Unit tests for the node-local NPU focused-profile analyzer."""

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
        repo_root
        / "test"
        / "manual"
        / "ascend"
        / "analyze_npu_lm_head_eplb_validation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_npu_lm_head_eplb_validation", analyzer_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestNPULMHeadEPLBValidationAnalyzer(CustomTestCase):
    def test_graph_internal_evidence_remains_conservative(self):
        analyzer = _load_analyzer_module()
        target = analyzer.TARGET_COLLECTIVES["mlp_sync_all_gather"]
        marker = analyzer._target_profile_marker(target)
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            capture_dir = run_dir / "stage-1a-capture"
            replay_dir = run_dir / "stage-1b-healthy-replay"
            capture_dir.mkdir()
            replay_dir.mkdir()
            (capture_dir / "trace_view.json").write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {
                                "name": "GRAPH_CAPTURE",
                                "pid": 1,
                                "ts": 0,
                                "dur": 10,
                            },
                            {"name": marker, "pid": 1, "ts": 2, "dur": 2},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            replay_trace = replay_dir / "trace_view.json"
            replay_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {
                                "name": "GRAPH_REPLAY",
                                "pid": 1,
                                "ts": 20,
                                "dur": 10,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            capture_only = analyzer._collective_graph_membership(run_dir, target)
            replay_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {
                                "name": "GRAPH_REPLAY",
                                "pid": 1,
                                "ts": 20,
                                "dur": 10,
                            },
                            {"name": marker, "pid": 1, "ts": 22, "dur": 2},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            replay_nested = analyzer._collective_graph_membership(run_dir, target)

        self.assertEqual(
            capture_only["verdict"],
            "capture_only_candidate_graph_internal_not_replay_confirmed",
        )
        self.assertEqual(
            replay_nested["verdict"],
            "observed_inside_graph_replay_host_scope_not_device_confirmed",
        )

    def test_fixed_target_and_weight_p2p_are_validated(self):
        analyzer = _load_analyzer_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            manifest_dir = run_dir / "stage-0-manifest"
            trace_dir = run_dir / "stage-1b-healthy-replay" / "rank-0"
            manifest_dir.mkdir()
            trace_dir.mkdir(parents=True)

            (manifest_dir / "resolved-server-info.json").write_text(
                json.dumps(
                    {
                        "tp_size": 4,
                        "dp_size": 4,
                        "attn_cp_size": 1,
                        "enable_dp_attention": True,
                        "enable_dp_lm_head": True,
                    }
                ),
                encoding="utf-8",
            )

            distribution_record = {
                "event": "BEGIN",
                "pid": 2000,
                "seq": 1,
                "global_rank": 0,
                "scope": "eplb_world",
                "source": "_ExpertDistributionRecorderReal.dump",
                "op": "all_reduce",
                "group_size": 4,
                "ranks": [0, 1, 2, 3],
                "backend": "hccl",
                "tensors": [{"device": "npu:0"}],
            }
            p2p_record = {
                "event": "BEGIN",
                "pid": 2000,
                "seq": 2,
                "global_rank": 0,
                "scope": "eplb_world_p2p",
                "source": "_update_expert_weights._execute_p2p_ops",
                "op": "batch_isend_irecv",
                "group_size": 4,
                "ranks": [0, 1, 2, 3],
                "backend": "hccl",
                "tensors": [{"device": "npu:0"}],
                "extra": {"peers": [1]},
            }
            mlp_sync_records = []
            for rank in range(4):
                begin_record = {
                    "event": "BEGIN",
                    "pid": 1000 + rank,
                    "seq": 7,
                    "global_rank": rank,
                    "scope": "scheduler_big_tp",
                    "source": "MLPSyncBatchInfo.all_gather",
                    "op": "all_gather_into_tensor",
                    "group_size": 4,
                    "ranks": [0, 1, 2, 3],
                    "backend": "hccl",
                    "tensors": [
                        {"device": f"npu:{rank}"},
                        {"device": f"npu:{rank}"},
                    ],
                }
                mlp_sync_records.extend(
                    [begin_record, {**begin_record, "event": "RETURN"}]
                )
            (run_dir / "server.log").write_text(
                "\n".join(
                    [
                        "[EPLBManager] rebalance start",
                        analyzer.TRACE_PREFIX + json.dumps(distribution_record),
                        analyzer.TRACE_PREFIX
                        + json.dumps({**distribution_record, "event": "RETURN"}),
                        analyzer.TRACE_PREFIX + json.dumps(p2p_record),
                        analyzer.TRACE_PREFIX
                        + json.dumps({**p2p_record, "event": "RETURN"}),
                        *(
                            analyzer.TRACE_PREFIX + json.dumps(record)
                            for record in mlp_sync_records
                        ),
                        "[EPLBManager] rebalance end",
                    ]
                ),
                encoding="utf-8",
            )
            mlp_marker = analyzer._target_profile_marker(
                analyzer.TARGET_COLLECTIVES["mlp_sync_all_gather"]
            )
            distribution_marker = analyzer._target_profile_marker(
                analyzer.TARGET_COLLECTIVES["eplb_distribution_all_reduce"]
            )
            p2p_marker = analyzer._target_profile_marker(
                analyzer.TARGET_COLLECTIVES["eplb_weight_p2p"]
            )
            (trace_dir / "trace_view.json").write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            *(
                                {
                                    "name": "GRAPH_REPLAY",
                                    "pid": pid,
                                    "ts": 0,
                                    "dur": 10,
                                }
                                for pid in (1000, 1001, 1002, 1003, 2000)
                            ),
                            *(
                                {
                                    "name": mlp_marker,
                                    "pid": pid,
                                    "ts": 20,
                                    "dur": 2,
                                }
                                for pid in (1000, 1001, 1002, 1003)
                            ),
                            {
                                "name": distribution_marker,
                                "pid": 2000,
                                "ts": 30,
                                "dur": 2,
                            },
                            {
                                "name": p2p_marker,
                                "pid": 2000,
                                "ts": 40,
                                "dur": 2,
                            },
                            {
                                "name": "EPLB_DISTRIBUTION_DUMP",
                                "pid": 2000,
                                "ts": 29,
                                "dur": 4,
                            },
                            {
                                "name": "EPLB_WEIGHT_UPDATE",
                                "pid": 2000,
                                "ts": 39,
                                "dur": 4,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            summary = analyzer.analyze(run_dir)

            server_log = run_dir / "server.log"
            server_log.write_text(
                server_log.read_text(encoding="utf-8").replace(
                    '"backend": "hccl"', '"backend": "gloo"'
                ),
                encoding="utf-8",
            )
            non_device_summary = analyzer.analyze(run_dir)

        self.assertEqual(
            summary["lm_head_vocab_gather"]["verdict"],
            "validated_absent_by_design",
        )
        self.assertEqual(
            summary["eplb_weight_movement"]["operation_verdict"],
            "validated_distribution_and_weight_p2p",
        )
        self.assertEqual(
            summary["eplb_weight_movement"]["distribution_graph_membership"]["verdict"],
            "validated_replay_adjacent_graph_external",
        )
        self.assertEqual(
            summary["eplb_weight_movement"]["weight_p2p_graph_membership"]["verdict"],
            "validated_replay_adjacent_graph_external",
        )
        self.assertEqual(
            summary["lm_head_vocab_gather"]["config"]["effective_attn_tp_size"],
            1,
        )
        self.assertEqual(
            summary["mlp_sync_device_all_gather"]["operation_verdict"],
            "validated_device_all_gather",
        )
        self.assertEqual(
            summary["mlp_sync_device_all_gather"]["graph_membership"]["verdict"],
            "validated_replay_adjacent_graph_external",
        )
        self.assertTrue(summary["validation_goal"]["complete"])
        self.assertEqual(
            summary["mlp_sync_device_all_gather"]["validated_global_ranks_observed"],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            non_device_summary["mlp_sync_device_all_gather"]["operation_verdict"],
            "observed_but_incomplete_or_non_device",
        )


if __name__ == "__main__":
    unittest.main()
