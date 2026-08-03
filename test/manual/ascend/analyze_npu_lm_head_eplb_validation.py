#!/usr/bin/env python3
"""Summarize LM-head, EPLB, and MLP-sync evidence on the NPU node."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


TRACE_PREFIX = "[SGLANG_BIG_TP_COLL] "
PROFILE_MARKERS = (
    "GRAPH_REPLAY",
    "LM_HEAD_VOCAB_GATHER",
    "EPLB_DISTRIBUTION_DUMP",
    "EPLB_WEIGHT_UPDATE",
)


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

    if (
        rebalance_starts > 0
        and rebalance_ends > 0
        and distribution_records
        and weight_p2p_records
        and marker_hits["EPLB_DISTRIBUTION_DUMP"] > 0
        and marker_hits["EPLB_WEIGHT_UPDATE"] > 0
    ):
        eplb_verdict = "validated_graph_external_weight_movement"
    elif rebalance_starts > 0 and rebalance_ends > 0 and distribution_records:
        eplb_verdict = "rebalance_observed_without_p2p_weight_movement"
    else:
        eplb_verdict = "not_observed_or_incomplete"

    expected_ranks = list(range(tp_size))
    validated_mlp_sync_records = []
    for record in mlp_sync_records:
        tensor_devices = {
            str(tensor.get("device"))
            for tensor in record.get("tensors", [])
            if isinstance(tensor, dict) and tensor.get("device") is not None
        }
        if (
            record.get("group_size") == tp_size
            and record.get("ranks") == expected_ranks
            and "hccl" in str(record.get("backend", "")).lower()
            and tensor_devices
            and all(device.startswith("npu") for device in tensor_devices)
            and (record.get("pid"), record.get("seq")) in return_record_keys
        ):
            validated_mlp_sync_records.append(record)
    mlp_sync_ranks_observed = sorted(
        {
            int(record["global_rank"])
            for record in validated_mlp_sync_records
            if isinstance(record.get("global_rank"), int)
        }
    )
    if validated_mlp_sync_records and mlp_sync_ranks_observed == expected_ranks:
        mlp_sync_verdict = "validated_device_all_gather"
    elif mlp_sync_records:
        mlp_sync_verdict = "observed_but_incomplete_or_non_device"
    else:
        mlp_sync_verdict = "not_observed"

    return {
        "run_dir": str(run_dir),
        "raw_artifacts_leave_node": False,
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
            "weight_p2p_begin_count": len(weight_p2p_records),
            "source_counts": dict(sorted(source_counts.items())),
            "scope_counts": dict(sorted(scope_counts.items())),
            "distribution_examples": _unique_examples(distribution_records),
            "weight_p2p_examples": _unique_examples(weight_p2p_records),
            "verdict": eplb_verdict,
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
            "verdict": mlp_sync_verdict,
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

    if args.require_complete:
        if (
            summary["lm_head_vocab_gather"]["verdict"]
            not in {"validated_absent_by_design", "validated_present"}
            or summary["eplb_weight_movement"]["verdict"]
            != "validated_graph_external_weight_movement"
            or summary["mlp_sync_device_all_gather"]["verdict"]
            != "validated_device_all_gather"
        ):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
