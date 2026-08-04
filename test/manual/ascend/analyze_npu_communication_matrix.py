#!/usr/bin/env python3
"""Build a node-local, source-attributed NPU graph communication matrix.

The output contains metadata and relative paths only. It never copies raw
profiler artifacts away from the NPU host.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TRACE_PREFIX = "[SGLANG_BIG_TP_COLL] "
PROFILE_MARKER_PREFIX = "SGLANG_COLLECTIVE::"
GRAPH_CAPTURE = "GRAPH_CAPTURE"
GRAPH_REPLAY = "GRAPH_REPLAY"


EXPECTED_SOURCES = (
    {
        "id": "scheduler_control_broadcast",
        "name_cn": "调度请求/控制消息广播",
        "scope": "scheduler_control",
        "source": (
            "SchedulerRequestReceiver._broadcast_reqs_across_ranks.control_full_tp"
        ),
        "op": "broadcast_pyobj",
        "plane_cn": "CPU 控制面",
        "required_backend": "gloo",
        "device_event_patterns": ("c10d::broadcast", "gloo:broadcast"),
    },
    {
        "id": "mlp_sync_all_gather",
        "name_cn": "MLP-sync 元数据 all-gather",
        "scope": "scheduler_big_tp",
        "source": "MLPSyncBatchInfo.all_gather",
        "op": "all_gather_into_tensor",
        "plane_cn": "NPU 数据面",
        "required_backend": "hccl",
        "device_event_patterns": (
            "c10d::all_gather",
            "c10d::allgather",
            "hccl:all_gather",
            "hccl:allgather",
            "hcomallgather",
        ),
    },
    {
        "id": "deepep_low_latency_dispatch",
        "name_cn": "DeepEP low-latency dispatch",
        "scope": "deepep_ep",
        "source": "_DeepEPDispatcherImplLowLatency._dispatch_core",
        "op": "low_latency_dispatch",
        "plane_cn": "NPU 数据面",
        "required_backend": "hccl",
        "device_event_patterns": (
            "MoeLowLatencyDispatchV2",
            "aclnnMoeLowLatencyDispatchV2",
        ),
    },
    {
        "id": "deepep_low_latency_combine",
        "name_cn": "DeepEP low-latency combine",
        "scope": "deepep_ep",
        "source": "_DeepEPDispatcherImplLowLatency._combine_core",
        "op": "low_latency_combine",
        "plane_cn": "NPU 数据面",
        "required_backend": "hccl",
        "device_event_patterns": (
            "MoeLowLatencyCombineV2",
            "aclnnMoeLowLatencyCombineV2",
        ),
    },
    {
        "id": "eplb_distribution_all_reduce",
        "name_cn": "EPLB 专家负载汇总 all-reduce",
        "scope": "eplb_world",
        "source": "_ExpertDistributionRecorderReal.dump",
        "op": "all_reduce",
        "plane_cn": "NPU 数据面",
        "required_backend": "hccl",
        "device_event_patterns": (
            "c10d::all_reduce",
            "c10d::allreduce",
            "hccl:all_reduce",
            "hccl:allreduce",
            "hcomallreduce",
        ),
    },
    {
        "id": "eplb_weight_p2p",
        "name_cn": "EPLB 专家权重 P2P 迁移",
        "scope": "eplb_world_p2p",
        "source": "_update_expert_weights._execute_p2p_ops",
        "op": "batch_isend_irecv",
        "plane_cn": "NPU 数据面",
        "required_backend": "hccl",
        "device_event_patterns": (
            "c10d::send",
            "c10d::recv",
            "hccl:send",
            "hccl:recv",
            "hcomsend",
            "hcomreceive",
        ),
    },
)


OPTIONAL_DEEPEP_SOURCES = (
    {
        "id": "deepep_normal_dispatch_layout",
        "name_cn": "DeepEP normal dispatch layout",
        "scope": "deepep_ep",
        "source": "_DeepEPDispatcherImplNormal._dispatch_core.layout",
        "op": "normal_dispatch_layout",
        "plane_cn": "NPU 数据面",
        "required_backend": "hccl",
        "device_event_patterns": ("DispatchLayout", "NotifyDispatch"),
    },
    {
        "id": "deepep_normal_dispatch",
        "name_cn": "DeepEP normal dispatch",
        "scope": "deepep_ep",
        "source": "_DeepEPDispatcherImplNormal._dispatch_core.dispatch",
        "op": "normal_dispatch",
        "plane_cn": "NPU 数据面",
        "required_backend": "hccl",
        "device_event_patterns": ("CamMoeDispatchNormal", "MoeDispatchNormal"),
    },
    {
        "id": "deepep_normal_combine",
        "name_cn": "DeepEP normal combine",
        "scope": "deepep_ep",
        "source": "_DeepEPDispatcherImplNormal._combine_core",
        "op": "normal_combine",
        "plane_cn": "NPU 数据面",
        "required_backend": "hccl",
        "device_event_patterns": ("CamMoeCombineNormal", "MoeCombineNormal"),
    },
)


HARNESS_ONLY_EVENT_PATTERNS = (
    "c10d::barrier",
    "gloo:barrier",
)


COMMUNICATION_EVENT_PATTERNS = (
    "c10d::all_",
    "c10d::allgather",
    "c10d::allreduce",
    "c10d::broadcast",
    "c10d::reduce_",
    "c10d::send",
    "c10d::recv",
    "gloo:",
    "hccl:",
    "hcom",
    "moelowlatencydispatch",
    "moelowlatencycombine",
    "cammoedispatchnormal",
    "cammoecombinenormal",
    "dispatchlayout",
    "notifydispatch",
)


OP_DEVICE_EVENT_PATTERNS = {
    "all_reduce": (
        "c10d::all_reduce",
        "c10d::allreduce",
        "hccl:all_reduce",
        "hccl:allreduce",
        "hcomallreduce",
    ),
    "quant_all_reduce": ("all_reduce", "allreduce", "hcomallreduce"),
    "fused_allreduce_rmsnorm": ("all_reduce", "allreduce", "hcomallreduce"),
    "all_gather": (
        "c10d::all_gather",
        "c10d::allgather",
        "hccl:all_gather",
        "hccl:allgather",
        "hcomallgather",
    ),
    "all_gather_into_tensor": (
        "c10d::all_gather",
        "c10d::allgather",
        "hccl:all_gather",
        "hccl:allgather",
        "hcomallgather",
    ),
    "all_gatherv": ("all_gather", "allgather", "hcomallgather"),
    "reduce_scatter": ("reduce_scatter", "reducescatter", "hcomreducescatter"),
    "reduce_scatter_tensor": (
        "reduce_scatter",
        "reducescatter",
        "hcomreducescatter",
    ),
    "reduce_scatterv": ("reduce_scatter", "reducescatter", "hcomreducescatter"),
    "all_to_all_single": ("all_to_all", "alltoall", "hcomalltoall"),
    "broadcast": ("c10d::broadcast", "gloo:broadcast", "hccl:broadcast"),
    "broadcast_pyobj": ("c10d::broadcast", "gloo:broadcast"),
    "batch_isend_irecv": (
        "c10d::send",
        "c10d::recv",
        "hccl:send",
        "hccl:recv",
        "hcomsend",
        "hcomreceive",
    ),
}


def _profile_marker(spec: dict[str, Any]) -> str:
    return f"{PROFILE_MARKER_PREFIX}{spec['scope']}::{spec['source']}::{spec['op']}"


def _source_key(scope: Any, source: Any, op: Any) -> tuple[str, str, str]:
    return str(scope), str(source), str(op)


def _read_collective_records(server_log: Path) -> tuple[list[dict[str, Any]], int]:
    decoder = json.JSONDecoder()
    records = []
    malformed = 0
    with server_log.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            index = line.find(TRACE_PREFIX)
            if index < 0:
                continue
            try:
                record, _ = decoder.raw_decode(
                    line[index + len(TRACE_PREFIX) :].lstrip()
                )
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(record, dict):
                records.append(record)
    return records, malformed


def _trace_view_paths(run_dir: Path) -> list[Path]:
    return sorted(
        {
            *run_dir.glob("stage-*/**/trace_view.json"),
            *run_dir.glob("stage-*/**/trace_view.json.gz"),
        }
    )


def _read_trace_events(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".gz":
        context = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        context = path.open("r", encoding="utf-8", errors="replace")
    with context as file:
        payload = json.load(file)
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise ValueError("traceEvents must be a list")
    return [event for event in events if isinstance(event, dict)]


def _trace_rank(path: Path) -> int | None:
    profiler_root = path.parent.parent
    candidates = sorted(profiler_root.glob("profiler_info_*.json"))
    for candidate in candidates:
        match = re.search(r"profiler_info_(\d+)\.json$", candidate.name)
        if match:
            return int(match.group(1))
    return None


def _is_communication_event(name: str) -> bool:
    if name.startswith(PROFILE_MARKER_PREFIX):
        return False
    lowered = name.lower()
    return any(pattern in lowered for pattern in COMMUNICATION_EVENT_PATTERNS)


def _phase_for_trace(path: Path, graph_capture_count: int, graph_replay_count: int):
    if graph_capture_count:
        return "capture"
    if "stage-1b-healthy-replay" in path.parts:
        return "decode" if graph_replay_count else "prefill"
    return "other"


def _scan_profiles(run_dir: Path) -> dict[str, Any]:
    phases = {
        name: {
            "trace_file_count": 0,
            "profiled_ranks": set(),
            "graph_capture_count": 0,
            "graph_replay_count": 0,
            "marker_counts": Counter(),
            "marker_paths": defaultdict(list),
            "communication_event_counts": Counter(),
            "communication_event_paths": defaultdict(list),
        }
        for name in ("capture", "prefill", "decode", "other")
    }
    parse_errors = []
    for path in _trace_view_paths(run_dir):
        relative_path = str(path.relative_to(run_dir))
        try:
            events = _read_trace_events(path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            parse_errors.append(
                {"trace": relative_path, "error_type": type(error).__name__}
            )
            continue
        names = [event.get("name") for event in events]
        graph_capture_count = sum(name == GRAPH_CAPTURE for name in names)
        graph_replay_count = sum(name == GRAPH_REPLAY for name in names)
        phase_name = _phase_for_trace(path, graph_capture_count, graph_replay_count)
        phase = phases[phase_name]
        phase["trace_file_count"] += 1
        rank = _trace_rank(path)
        if rank is not None:
            phase["profiled_ranks"].add(rank)
        phase["graph_capture_count"] += graph_capture_count
        phase["graph_replay_count"] += graph_replay_count

        for name in names:
            if not isinstance(name, str):
                continue
            if name.startswith(PROFILE_MARKER_PREFIX):
                phase["marker_counts"][name] += 1
                paths = phase["marker_paths"][name]
                if relative_path not in paths and len(paths) < 8:
                    paths.append(relative_path)
            if _is_communication_event(name):
                phase["communication_event_counts"][name] += 1
                paths = phase["communication_event_paths"][name]
                if relative_path not in paths and len(paths) < 8:
                    paths.append(relative_path)

    serializable_phases = {}
    for name, phase in phases.items():
        serializable_phases[name] = {
            **phase,
            "profiled_ranks": sorted(phase["profiled_ranks"]),
            "marker_counts": dict(phase["marker_counts"]),
            "marker_paths": dict(phase["marker_paths"]),
            "communication_event_counts": dict(phase["communication_event_counts"]),
            "communication_event_paths": dict(phase["communication_event_paths"]),
        }
    return {"phases": serializable_phases, "parse_errors": parse_errors}


def _patterns_from_records(records: list[dict[str, Any]]) -> list[str]:
    patterns = set()
    for record in records:
        extra = record.get("extra")
        if not isinstance(extra, dict):
            continue
        values = extra.get("profile_device_event_patterns", [])
        if isinstance(values, list):
            patterns.update(str(value) for value in values)
    return sorted(patterns)


def _patterns_for_op(op: str) -> list[str]:
    return list(OP_DEVICE_EVENT_PATTERNS.get(op, ()))


def _matching_device_evidence(
    phase: dict[str, Any], patterns: list[str]
) -> tuple[int, list[dict[str, Any]]]:
    lowered_patterns = [pattern.lower() for pattern in patterns]
    count = 0
    examples = []
    for name, event_count in phase["communication_event_counts"].items():
        if not any(pattern in name.lower() for pattern in lowered_patterns):
            continue
        count += int(event_count)
        if len(examples) < 8:
            examples.append(
                {
                    "event": name,
                    "count": event_count,
                    "traces": phase["communication_event_paths"].get(name, []),
                }
            )
    return count, examples


def _tensor_devices(records: list[dict[str, Any]]) -> list[str]:
    devices = set()
    for record in records:
        tensors = record.get("tensors", [])
        if not isinstance(tensors, list):
            continue
        for tensor in tensors:
            if isinstance(tensor, dict) and tensor.get("device") is not None:
                devices.add(str(tensor["device"]))
    return sorted(devices)


def _classify_graph_membership(
    capture_marker_count: int,
    prefill_marker_count: int,
    decode_marker_count: int,
    decode_device_event_count: int,
    decode_graph_replay_count: int,
) -> str:
    if decode_marker_count > 0 and decode_graph_replay_count > 0:
        return "graph_external"
    if (
        capture_marker_count > 0
        and decode_marker_count == 0
        and decode_device_event_count > 0
        and decode_graph_replay_count > 0
    ):
        return "graph_internal"
    if prefill_marker_count > 0 and decode_marker_count == 0:
        return "graph_external_prefill_only"
    return "inconclusive"


def _meaning_cn(
    verdict: str,
    *,
    capture_marker_count: int,
    prefill_marker_count: int,
    decode_marker_count: int,
    decode_device_event_count: int,
    decode_graph_replay_count: int,
) -> str:
    if verdict == "graph_internal":
        return (
            f"capture 中该 source marker 出现 {capture_marker_count} 次；decode 中 "
            f"GRAPH_REPLAY 出现 {decode_graph_replay_count} 次时，source marker 为 0，"
            f"但对应 NPU 通信事件仍出现 {decode_device_event_count} 条。也就是说，"
            "重放时 Python 没有再次发起这项通信，通信工作由已捕获的图带出，因此判为图内。"
        )
    if verdict == "graph_external":
        return (
            f"decode 中已有 {decode_graph_replay_count} 次 GRAPH_REPLAY，同时该 "
            f"source marker 仍执行 {decode_marker_count} 次。也就是说，图在重放，"
            "Python 仍另外发起这项通信，因此判为图外。"
        )
    if verdict == "graph_external_prefill_only":
        return (
            f"该 source marker 只在 prefill 中出现 {prefill_marker_count} 次，decode "
            "重放中没有执行。prefill 本身不属于 decode 图，因此这项通信判为图外。"
        )
    return (
        "现有字段不足以同时证明 source 身份和图归属；该项必须保留为未确认，"
        "不能按事件名直接猜测。"
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")[:96]


def _dynamic_specs(
    begin_records: list[dict[str, Any]], profile_scan: dict[str, Any]
) -> list[dict[str, Any]]:
    known_keys = {
        _source_key(spec["scope"], spec["source"], spec["op"])
        for spec in (*EXPECTED_SOURCES, *OPTIONAL_DEEPEP_SOURCES)
    }
    observed_keys = {
        _source_key(record.get("scope"), record.get("source"), record.get("op"))
        for record in begin_records
    }
    for phase in profile_scan["phases"].values():
        for marker in phase["marker_counts"]:
            if not marker.startswith(PROFILE_MARKER_PREFIX):
                continue
            parts = marker[len(PROFILE_MARKER_PREFIX) :].split("::", 2)
            if len(parts) == 3:
                observed_keys.add(_source_key(*parts))

    specs = []
    for scope, source, op in sorted(observed_keys - known_keys):
        matching_records = [
            record
            for record in begin_records
            if _source_key(record.get("scope"), record.get("source"), record.get("op"))
            == (scope, source, op)
        ]
        specs.append(
            {
                "id": "observed_" + _slug(f"{scope}_{source}_{op}"),
                "name_cn": f"运行时发现：{source} / {op}",
                "scope": scope,
                "source": source,
                "op": op,
                "plane_cn": "按运行记录识别",
                "required_backend": None,
                "device_event_patterns": tuple(
                    {
                        *_patterns_from_records(matching_records),
                        *_patterns_for_op(op),
                    }
                ),
            }
        )
    return specs


def _build_row(
    run_dir: Path,
    spec: dict[str, Any],
    begin_records: list[dict[str, Any]],
    return_keys: set[tuple[Any, Any]],
    profile_scan: dict[str, Any],
    *,
    required: bool,
) -> dict[str, Any]:
    key = _source_key(spec["scope"], spec["source"], spec["op"])
    records = [
        record
        for record in begin_records
        if _source_key(record.get("scope"), record.get("source"), record.get("op"))
        == key
    ]
    successful_records = [
        record
        for record in records
        if (record.get("pid"), record.get("seq")) in return_keys
    ]
    marker = _profile_marker(spec)
    phases = profile_scan["phases"]
    capture_marker_count = phases["capture"]["marker_counts"].get(marker, 0)
    prefill_marker_count = phases["prefill"]["marker_counts"].get(marker, 0)
    decode_marker_count = phases["decode"]["marker_counts"].get(marker, 0)
    patterns = sorted(
        {
            *spec.get("device_event_patterns", ()),
            *_patterns_from_records(records),
        }
    )
    decode_device_event_count, decode_device_examples = _matching_device_evidence(
        phases["decode"], patterns
    )
    verdict = _classify_graph_membership(
        capture_marker_count,
        prefill_marker_count,
        decode_marker_count,
        decode_device_event_count,
        phases["decode"]["graph_replay_count"],
    )
    ranks = sorted(
        {
            int(record["global_rank"])
            for record in successful_records
            if isinstance(record.get("global_rank"), int)
        }
    )
    backends = sorted(
        {
            str(record["backend"])
            for record in records
            if record.get("backend") is not None
        }
    )
    required_backend = spec.get("required_backend")
    operation_valid = bool(successful_records)
    if required_backend is not None:
        operation_valid = operation_valid and required_backend in backends
    operation_valid = operation_valid and ranks == [0, 1, 2, 3]

    evidence_path = str(run_dir / "communication-matrix-summary.json")
    meaning = _meaning_cn(
        verdict,
        capture_marker_count=capture_marker_count,
        prefill_marker_count=prefill_marker_count,
        decode_marker_count=decode_marker_count,
        decode_device_event_count=decode_device_event_count,
        decode_graph_replay_count=phases["decode"]["graph_replay_count"],
    )
    return {
        "id": spec["id"],
        "communication_cn": spec["name_cn"],
        "plane_cn": spec["plane_cn"],
        "source": spec["source"],
        "scope": spec["scope"],
        "op": spec["op"],
        "required_for_fixed_config": required,
        "operation": {
            "structured_begin_count": len(records),
            "structured_successful_return_count": len(successful_records),
            "validated_global_ranks": ranks,
            "backends": backends,
            "tensor_devices": _tensor_devices(records),
            "valid_for_fixed_four_rank_config": operation_valid,
        },
        "graph_membership": {
            "verdict": verdict,
            "profile_marker": marker,
            "device_event_patterns": patterns,
            "evidence_result_path": evidence_path,
            "evidence_field_values": {
                "profile.capture.source_marker_count": capture_marker_count,
                "profile.prefill.source_marker_count": prefill_marker_count,
                "profile.decode.graph_replay_count": phases["decode"][
                    "graph_replay_count"
                ],
                "profile.decode.source_marker_count": decode_marker_count,
                "profile.decode.matched_device_event_count": (
                    decode_device_event_count
                ),
            },
            "source_marker_trace_paths": {
                phase_name: phases[phase_name]["marker_paths"].get(marker, [])
                for phase_name in ("capture", "prefill", "decode")
            },
            "matched_device_event_examples": decode_device_examples,
            "meaning_cn": meaning,
        },
    }


def _read_server_info(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "stage-0-manifest" / "resolved-server-info.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _event_inventory(
    rows: list[dict[str, Any]], profile_scan: dict[str, Any]
) -> dict[str, Any]:
    mapped_patterns = {
        pattern.lower()
        for row in rows
        for pattern in row["graph_membership"]["device_event_patterns"]
    }
    unknown = Counter()
    harness_only = Counter()
    by_phase = {}
    for phase_name, phase in profile_scan["phases"].items():
        phase_unknown = Counter()
        for name, count in phase["communication_event_counts"].items():
            lowered = name.lower()
            if any(pattern in lowered for pattern in HARNESS_ONLY_EVENT_PATTERNS):
                harness_only[name] += count
            elif not any(pattern in lowered for pattern in mapped_patterns):
                unknown[name] += count
                phase_unknown[name] += count
        by_phase[phase_name] = dict(phase_unknown)
    return {
        "unknown_event_name_counts": dict(unknown),
        "unknown_event_name_counts_by_phase": by_phase,
        "harness_only_profile_coordination_event_counts": dict(harness_only),
        "raw_event_occurrences_are_not_collective_call_counts": True,
    }


def analyze(run_dir: Path) -> dict[str, Any]:
    server_log = run_dir / "server.log"
    if not server_log.is_file():
        raise FileNotFoundError(server_log)

    records, malformed_count = _read_collective_records(server_log)
    begin_records = [record for record in records if record.get("event") == "BEGIN"]
    return_keys = {
        (record.get("pid"), record.get("seq"))
        for record in records
        if record.get("event") == "RETURN"
    }
    profile_scan = _scan_profiles(run_dir)
    optional_specs = []
    observed_keys = {
        _source_key(record.get("scope"), record.get("source"), record.get("op"))
        for record in begin_records
    }
    for spec in OPTIONAL_DEEPEP_SOURCES:
        if _source_key(spec["scope"], spec["source"], spec["op"]) in observed_keys:
            optional_specs.append(spec)
    dynamic_specs = _dynamic_specs(begin_records, profile_scan)
    rows = [
        _build_row(
            run_dir,
            spec,
            begin_records,
            return_keys,
            profile_scan,
            required=True,
        )
        for spec in EXPECTED_SOURCES
    ]
    rows.extend(
        _build_row(
            run_dir,
            spec,
            begin_records,
            return_keys,
            profile_scan,
            required=False,
        )
        for spec in (*optional_specs, *dynamic_specs)
    )
    event_inventory = _event_inventory(rows, profile_scan)
    fixed_expected_ranks = [0, 1, 2, 3]
    required_rows_complete = all(
        row["operation"]["valid_for_fixed_four_rank_config"]
        and row["graph_membership"]["verdict"]
        in {"graph_internal", "graph_external", "graph_external_prefill_only"}
        for row in rows
        if row["required_for_fixed_config"]
    )
    discovered_rows_complete = all(
        row["graph_membership"]["verdict"] != "inconclusive"
        for row in rows
        if not row["required_for_fixed_config"]
    )
    profile_ranks_complete = (
        profile_scan["phases"]["capture"]["profiled_ranks"] == fixed_expected_ranks
        and profile_scan["phases"]["decode"]["profiled_ranks"] == fixed_expected_ranks
    )
    inventory_complete = bool(
        required_rows_complete
        and discovered_rows_complete
        and profile_ranks_complete
        and not profile_scan["parse_errors"]
        and malformed_count == 0
        and not event_inventory["unknown_event_name_counts"]
    )
    server_info = _read_server_info(run_dir)
    return {
        "run_dir": str(run_dir),
        "safe_summary_path": str(run_dir / "communication-matrix-summary.json"),
        "raw_artifacts_leave_node": False,
        "scope_cn": (
            "固定 4×Atlas A3 配置的一次健康请求：graph capture、prefill、"
            "decode graph replay、调度同步，以及同一窗口内触发的 EPLB rebalance。"
        ),
        "fixed_config": {
            "tp_size": server_info.get("tp_size"),
            "dp_size": server_info.get("dp_size"),
            "ep_size": server_info.get("ep_size"),
            "attn_cp_size": server_info.get("attn_cp_size"),
            "enable_dp_attention": server_info.get("enable_dp_attention"),
            "enable_dp_lm_head": server_info.get("enable_dp_lm_head"),
            "moe_a2a_backend": "deepep",
            "deepep_mode": "low_latency",
        },
        "coverage": {
            "inventory_complete": inventory_complete,
            "required_rows_complete": required_rows_complete,
            "discovered_rows_complete": discovered_rows_complete,
            "profile_ranks_complete": profile_ranks_complete,
            "expected_ranks": fixed_expected_ranks,
            "capture_profiled_ranks": profile_scan["phases"]["capture"][
                "profiled_ranks"
            ],
            "decode_profiled_ranks": profile_scan["phases"]["decode"]["profiled_ranks"],
            "trace_parse_error_count": len(profile_scan["parse_errors"]),
            "malformed_structured_record_count": malformed_count,
            "unknown_profile_communication_event_name_count": len(
                event_inventory["unknown_event_name_counts"]
            ),
        },
        "matrix": rows,
        "profile": {
            "phases": profile_scan["phases"],
            "parse_errors": profile_scan["parse_errors"],
        },
        "profile_communication_event_inventory": event_inventory,
        "trace_records": {
            "parsed_record_count": len(records),
            "begin_record_count": len(begin_records),
            "malformed_record_count": malformed_count,
        },
        "excluded_by_fixed_config": [
            {
                "communication_cn": "LM-head vocab gather",
                "reason_cn": (
                    "enable_dp_lm_head=true，但该配置 effective attention TP=1，"
                    "词表没有跨 rank 分片，因此没有这项通信。"
                ),
            },
            {
                "communication_cn": "attention-TP 层间 collective",
                "reason_cn": (
                    "TP=DP=4 且 attn_cp_size=1，所以 effective attention TP=1；"
                    "attention 侧不存在需要跨 rank 合并的 TP 分片。"
                ),
            },
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    summary = analyze(args.run_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.require_complete and not summary["coverage"]["inventory_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
