#!/usr/bin/env python3
"""Summarize LM-head, EPLB, and MLP-sync evidence on the NPU node."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any


TRACE_PREFIX = "[SGLANG_BIG_TP_COLL] "
PROFILE_MARKER_PREFIX = "SGLANG_COLLECTIVE::"
PROFILE_MARKERS = (
    "GRAPH_CAPTURE",
    "GRAPH_REPLAY",
    "LM_HEAD_VOCAB_GATHER",
    "EPLB_DISTRIBUTION_DUMP",
    "EPLB_WEIGHT_UPDATE",
)
TARGET_COLLECTIVES = {
    "mlp_sync_all_gather": {
        "scope": "scheduler_big_tp",
        "source": "MLPSyncBatchInfo.all_gather",
        "op": "all_gather_into_tensor",
    },
    "eplb_distribution_all_reduce": {
        "scope": "eplb_world",
        "source": "_ExpertDistributionRecorderReal.dump",
        "op": "all_reduce",
    },
    "eplb_weight_p2p": {
        "scope": "eplb_world_p2p",
        "source": "_update_expert_weights._execute_p2p_ops",
        "op": "batch_isend_irecv",
    },
}


def _profile_marker_name(scope: str, source: str, op: str) -> str:
    return f"{PROFILE_MARKER_PREFIX}{scope}::{source}::{op}"


def _target_profile_marker(target: dict[str, str]) -> str:
    return _profile_marker_name(target["scope"], target["source"], target["op"])


def _read_collective_records(server_log: Path) -> tuple[list[dict[str, Any]], int]:
    records = []
    malformed = 0
    decoder = json.JSONDecoder()
    with server_log.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            prefix_index = line.find(TRACE_PREFIX)
            if prefix_index < 0:
                continue
            payload = line[prefix_index + len(TRACE_PREFIX) :].lstrip()
            try:
                record, _ = decoder.raw_decode(payload)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(record, dict):
                records.append(record)
    return records, malformed


def _count_log_lines(server_log: Path, needle: str) -> int:
    count = 0
    with server_log.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            count += line.count(needle)
    return count


def _count_profile_marker_strings(run_dir: Path) -> dict[str, int]:
    counts = Counter()
    max_marker_len = max(map(len, PROFILE_MARKERS))
    for path in run_dir.glob("stage-*/**/trace_view.json"):
        carry = ""
        with path.open(encoding="utf-8", errors="ignore") as file:
            while chunk := file.read(1024 * 1024):
                for marker in PROFILE_MARKERS:
                    counts[marker] += chunk.count(marker)
                    counts[marker] += sum(
                        carry.endswith(marker[:split])
                        and chunk.startswith(marker[split:])
                        for split in range(1, len(marker))
                    )
                carry = (carry + chunk)[-(max_marker_len - 1) :]
    return {marker: counts[marker] for marker in PROFILE_MARKERS}


def _trace_view_paths(stage_dir: Path) -> list[Path]:
    return sorted(
        {
            *stage_dir.rglob("trace_view.json"),
            *stage_dir.rglob("trace_view.json.gz"),
        }
    )


def _read_trace_events(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    if path.suffix == ".gz":
        file_context = opener(path, "rt", encoding="utf-8", errors="replace")
    else:
        file_context = opener(path, "r", encoding="utf-8", errors="replace")
    with file_context as file:
        payload = json.load(file)
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise ValueError("traceEvents must be a list")
    return [event for event in events if isinstance(event, dict)]


def _event_interval(event: dict[str, Any]) -> tuple[float, float] | None:
    try:
        timestamp = float(event["ts"])
        duration = float(event["dur"])
    except (KeyError, TypeError, ValueError):
        return None
    return timestamp, timestamp + max(duration, 0)


def _events_overlap(event: dict[str, Any], graph_event: dict[str, Any]) -> bool:
    interval = _event_interval(event)
    graph_interval = _event_interval(graph_event)
    if interval is None or graph_interval is None:
        return False
    event_pid = event.get("pid")
    graph_pid = graph_event.get("pid")
    if event_pid is not None and graph_pid is not None and event_pid != graph_pid:
        return False
    return interval[0] < graph_interval[1] and graph_interval[0] < interval[1]


def _profile_stage_evidence(
    run_dir: Path,
    stage_name: str,
    graph_marker: str,
    target_marker: str,
) -> dict[str, Any]:
    stage_dir = run_dir / stage_name
    occurrence_count = 0
    overlap_count = 0
    outside_graph_count = 0
    without_graph_reference_count = 0
    invalid_interval_count = 0
    graph_marker_count = 0
    parsed_file_count = 0
    parse_errors = []
    examples = []

    for path in _trace_view_paths(stage_dir):
        try:
            events = _read_trace_events(path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            parse_errors.append(
                {
                    "trace": str(path.relative_to(run_dir)),
                    "error_type": type(error).__name__,
                }
            )
            continue
        parsed_file_count += 1
        graph_events = [event for event in events if event.get("name") == graph_marker]
        target_events = [
            event for event in events if event.get("name") == target_marker
        ]
        graph_marker_count += len(graph_events)
        occurrence_count += len(target_events)

        for event in target_events:
            interval = _event_interval(event)
            same_pid_graph_events = [
                graph_event
                for graph_event in graph_events
                if event.get("pid") is None
                or graph_event.get("pid") is None
                or event.get("pid") == graph_event.get("pid")
            ]
            overlaps_graph = any(
                _events_overlap(event, graph_event)
                for graph_event in same_pid_graph_events
            )
            if interval is None:
                invalid_interval_count += 1
                relation = "invalid_interval"
            elif not same_pid_graph_events:
                without_graph_reference_count += 1
                relation = "no_same_process_graph_marker"
            elif overlaps_graph:
                overlap_count += 1
                relation = "overlaps_graph_host_scope"
            else:
                outside_graph_count += 1
                relation = "outside_graph_host_scope"
            if len(examples) < 8:
                examples.append(
                    {
                        "trace": str(path.relative_to(run_dir)),
                        "pid": event.get("pid"),
                        "tid": event.get("tid"),
                        "ts_us": event.get("ts"),
                        "duration_us": event.get("dur"),
                        "relation": relation,
                    }
                )

    return {
        "stage": stage_name,
        "graph_marker": graph_marker,
        "collective_profile_marker": target_marker,
        "parsed_trace_file_count": parsed_file_count,
        "parse_errors": parse_errors,
        "graph_marker_count": graph_marker_count,
        "collective_marker_count": occurrence_count,
        "overlap_with_graph_host_scope_count": overlap_count,
        "outside_graph_host_scope_count": outside_graph_count,
        "without_same_process_graph_marker_count": without_graph_reference_count,
        "invalid_interval_count": invalid_interval_count,
        "examples": examples,
    }


def _collective_graph_membership(
    run_dir: Path, target: dict[str, str]
) -> dict[str, Any]:
    target_marker = _target_profile_marker(target)
    capture = _profile_stage_evidence(
        run_dir,
        "stage-1a-capture",
        "GRAPH_CAPTURE",
        target_marker,
    )
    replay = _profile_stage_evidence(
        run_dir,
        "stage-1b-healthy-replay",
        "GRAPH_REPLAY",
        target_marker,
    )

    replay_inside = replay["overlap_with_graph_host_scope_count"]
    replay_outside = replay["outside_graph_host_scope_count"]
    replay_observed = replay["collective_marker_count"]
    capture_inside = capture["overlap_with_graph_host_scope_count"]
    capture_observed = capture["collective_marker_count"]
    if replay_outside > 0 and replay_inside == 0:
        verdict = "validated_replay_adjacent_graph_external"
    elif replay_outside > 0 and replay_inside > 0:
        verdict = "mixed_inside_and_outside_graph_host_scope"
    elif replay_inside > 0:
        # Host-scope nesting alone is not enough to prove that HCCL device work
        # was captured into the NPU graph.
        verdict = "observed_inside_graph_replay_host_scope_not_device_confirmed"
    elif replay_observed > 0:
        verdict = "replay_profile_observed_without_graph_timing_reference"
    elif capture_inside > 0:
        verdict = "capture_only_candidate_graph_internal_not_replay_confirmed"
    elif capture_observed > 0:
        verdict = "capture_profile_observed_outside_graph_host_scope"
    else:
        verdict = "not_observed_in_profile"

    return {
        "verdict": verdict,
        "interpretation": (
            "A Python collective marker re-executed outside GRAPH_REPLAY proves "
            "that invocation is graph-external. A marker seen only during "
            "GRAPH_CAPTURE is merely a graph-internal candidate until replay-side "
            "device activity is correlated."
        ),
        "capture_profile": capture,
        "healthy_replay_profile": replay,
    }


def _project_collective(record: dict[str, Any]) -> dict[str, Any]:
    tensor_devices = sorted(
        {
            str(tensor.get("device"))
            for tensor in record.get("tensors", [])
            if isinstance(tensor, dict) and tensor.get("device") is not None
        }
    )
    return {
        "global_rank": record.get("global_rank"),
        "scope": record.get("scope"),
        "source": record.get("source"),
        "op": record.get("op"),
        "group": record.get("group"),
        "group_size": record.get("group_size"),
        "ranks": record.get("ranks"),
        "backend": record.get("backend"),
        "active_mask_cpu": record.get("active_mask_cpu"),
        "tensor_count": record.get("tensor_count"),
        "tensor_devices": tensor_devices,
        "extra": record.get("extra"),
    }


def _unique_examples(records: list[dict[str, Any]], limit: int = 8):
    examples = []
    seen = set()
    for record in records:
        projected = _project_collective(record)
        key = json.dumps(projected, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        examples.append(projected)
        if len(examples) >= limit:
            break
    return examples


def _validated_device_records(
    records: list[dict[str, Any]],
    return_record_keys: set[tuple[Any, Any]],
    expected_group_size: int,
    expected_ranks: list[int],
) -> list[dict[str, Any]]:
    validated_records = []
    for record in records:
        tensor_devices = {
            str(tensor.get("device"))
            for tensor in record.get("tensors", [])
            if isinstance(tensor, dict) and tensor.get("device") is not None
        }
        if (
            record.get("group_size") == expected_group_size
            and record.get("ranks") == expected_ranks
            and "hccl" in str(record.get("backend", "")).lower()
            and tensor_devices
            and all(device.startswith("npu") for device in tensor_devices)
            and (record.get("pid"), record.get("seq")) in return_record_keys
        ):
            validated_records.append(record)
    return validated_records


def analyze(run_dir: Path) -> dict[str, Any]:
    manifest_dir = run_dir / "stage-0-manifest"
    server_info_path = manifest_dir / "resolved-server-info.json"
    server_log = run_dir / "server.log"
    if not server_info_path.is_file():
        raise FileNotFoundError(server_info_path)
    if not server_log.is_file():
        raise FileNotFoundError(server_log)

    server_info = json.loads(server_info_path.read_text(encoding="utf-8"))
    tp_size = int(server_info["tp_size"])
    dp_size = int(server_info["dp_size"])
    attn_cp_size = int(server_info.get("attn_cp_size", 1))
    enable_dp_attention = bool(server_info["enable_dp_attention"])
    enable_dp_lm_head = bool(server_info["enable_dp_lm_head"])
    attn_dp_size = dp_size if enable_dp_attention else 1
    divisor = attn_cp_size * attn_dp_size
    effective_attn_tp_size = tp_size // divisor if tp_size % divisor == 0 else None

    records, malformed_records = _read_collective_records(server_log)
    expected_ranks = list(range(tp_size))
    begin_records = [record for record in records if record.get("event") == "BEGIN"]
    return_record_keys = {
        (record.get("pid"), record.get("seq"))
        for record in records
        if record.get("event") == "RETURN"
    }
    source_counts = Counter(str(record.get("source")) for record in begin_records)
    scope_counts = Counter(str(record.get("scope")) for record in begin_records)

    lm_head_records = [
        record for record in begin_records if record.get("scope") == "lm_head_vocab"
    ]
    eplb_records = [
        record
        for record in begin_records
        if str(record.get("scope", "")).startswith("eplb_")
    ]
    distribution_records = [
        record
        for record in eplb_records
        if record.get("source") == "_ExpertDistributionRecorderReal.dump"
    ]
    weight_p2p_records = [
        record
        for record in eplb_records
        if record.get("source") == "_update_expert_weights._execute_p2p_ops"
    ]
    mlp_sync_records = [
        record
        for record in begin_records
        if record.get("scope") == "scheduler_big_tp"
        and record.get("source") == "MLPSyncBatchInfo.all_gather"
        and record.get("op") == "all_gather_into_tensor"
    ]
    validated_distribution_records = _validated_device_records(
        distribution_records, return_record_keys, tp_size, expected_ranks
    )
    validated_weight_p2p_records = _validated_device_records(
        weight_p2p_records, return_record_keys, tp_size, expected_ranks
    )

    graph_membership = {
        name: _collective_graph_membership(run_dir, target)
        for name, target in TARGET_COLLECTIVES.items()
    }

    marker_hits = _count_profile_marker_strings(run_dir)
    rebalance_starts = _count_log_lines(server_log, "[EPLBManager] rebalance start")
    rebalance_ends = _count_log_lines(server_log, "[EPLBManager] rebalance end")

    lm_head_expected = bool(
        enable_dp_lm_head
        and effective_attn_tp_size is not None
        and effective_attn_tp_size > 1
    )
    if not lm_head_expected and effective_attn_tp_size == 1 and not lm_head_records:
        lm_head_verdict = "validated_absent_by_design"
    elif lm_head_expected and lm_head_records:
        lm_head_verdict = "validated_present"
    else:
        lm_head_verdict = "inconclusive_or_mismatch"

    if validated_distribution_records and validated_weight_p2p_records:
        eplb_operation_verdict = "validated_distribution_and_weight_p2p"
    elif validated_distribution_records:
        eplb_operation_verdict = "validated_distribution_without_weight_p2p"
    elif distribution_records or weight_p2p_records:
        eplb_operation_verdict = "observed_without_successful_python_return"
    else:
        eplb_operation_verdict = "not_observed"

    validated_mlp_sync_records = _validated_device_records(
        mlp_sync_records, return_record_keys, tp_size, expected_ranks
    )
    mlp_sync_ranks_observed = sorted(
        {
            int(record["global_rank"])
            for record in validated_mlp_sync_records
            if isinstance(record.get("global_rank"), int)
        }
    )
    if validated_mlp_sync_records and mlp_sync_ranks_observed == expected_ranks:
        mlp_sync_operation_verdict = "validated_device_all_gather"
    elif mlp_sync_records:
        mlp_sync_operation_verdict = "observed_but_incomplete_or_non_device"
    else:
        mlp_sync_operation_verdict = "not_observed"

    required_external_verdict = "validated_replay_adjacent_graph_external"
    validation_complete = bool(
        mlp_sync_operation_verdict == "validated_device_all_gather"
        and graph_membership["mlp_sync_all_gather"]["verdict"]
        == required_external_verdict
        and eplb_operation_verdict == "validated_distribution_and_weight_p2p"
        and graph_membership["eplb_distribution_all_reduce"]["verdict"]
        == required_external_verdict
        and graph_membership["eplb_weight_p2p"]["verdict"] == required_external_verdict
    )

    return {
        "run_dir": str(run_dir),
        "raw_artifacts_leave_node": False,
        "validation_goal": {
            "mlp_sync_question": (
                "Is the HCCL MLP-sync all_gather graph-internal or "
                "replay-adjacent graph-external?"
            ),
            "eplb_question": (
                "Do EPLB distribution or weight-movement collectives execute "
                "inside the graph?"
            ),
            "complete": validation_complete,
        },
        "profile_marker_string_hits": marker_hits,
        "lm_head_vocab_gather": {
            "config": {
                "tp_size": tp_size,
                "dp_size": dp_size,
                "attn_cp_size": attn_cp_size,
                "effective_attn_tp_size": effective_attn_tp_size,
                "enable_dp_attention": enable_dp_attention,
                "enable_dp_lm_head": enable_dp_lm_head,
            },
            "collective_expected": lm_head_expected,
            "collective_trace_begin_count": len(lm_head_records),
            "examples": _unique_examples(lm_head_records),
            "verdict": lm_head_verdict,
        },
        "eplb_weight_movement": {
            "rebalance_start_count": rebalance_starts,
            "rebalance_end_count": rebalance_ends,
            "distribution_all_reduce_begin_count": len(distribution_records),
            "distribution_all_reduce_validated_device_call_count": len(
                validated_distribution_records
            ),
            "weight_p2p_begin_count": len(weight_p2p_records),
            "weight_p2p_validated_device_call_count": len(validated_weight_p2p_records),
            "source_counts": dict(sorted(source_counts.items())),
            "scope_counts": dict(sorted(scope_counts.items())),
            "distribution_examples": _unique_examples(distribution_records),
            "weight_p2p_examples": _unique_examples(weight_p2p_records),
            "operation_verdict": eplb_operation_verdict,
            "distribution_graph_membership": graph_membership[
                "eplb_distribution_all_reduce"
            ],
            "weight_p2p_graph_membership": graph_membership["eplb_weight_p2p"],
        },
        "mlp_sync_device_all_gather": {
            "expected_backend": "hccl",
            "expected_group_size": tp_size,
            "expected_ranks": expected_ranks,
            "expected_tensor_device_prefix": "npu",
            "collective_trace_begin_count": len(mlp_sync_records),
            "validated_successful_call_count": len(validated_mlp_sync_records),
            "validated_global_ranks_observed": mlp_sync_ranks_observed,
            "examples": _unique_examples(mlp_sync_records),
            "operation_verdict": mlp_sync_operation_verdict,
            "graph_membership": graph_membership["mlp_sync_all_gather"],
        },
        "trace_parse": {
            "parsed_record_count": len(records),
            "malformed_record_count": malformed_records,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    summary = analyze(args.run_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    if args.require_complete and not summary["validation_goal"]["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
