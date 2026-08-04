# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Opt-in metadata tracing for large TP and related EPLB communication.

The tracer deliberately avoids tensor values and device synchronization. A
``RETURN`` record only means that the Python call returned; it does not prove
that asynchronous device work completed.
"""

from __future__ import annotations

import contextvars
import functools
import itertools
import json
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

import torch
import torch.distributed

TRACE_ENV = "SGLANG_BIG_TP_COLLECTIVE_TRACE"
TRACE_MAX_RECORDS_ENV = "SGLANG_BIG_TP_COLLECTIVE_TRACE_MAX_RECORDS"
TRACE_MAX_RECORDS_PER_SOURCE_ENV = (
    "SGLANG_BIG_TP_COLLECTIVE_TRACE_MAX_RECORDS_PER_SOURCE"
)
TRACE_PREFIX = "[SGLANG_BIG_TP_COLL] "
PROFILE_MARKER_PREFIX = "SGLANG_COLLECTIVE::"
_MAX_TENSOR_METADATA_PER_RECORD = 16

_TRACE_ENABLED = os.getenv(TRACE_ENV, "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_TRACE_MAX_RECORDS = int(os.getenv(TRACE_MAX_RECORDS_ENV, "0"))
_TRACE_MAX_RECORDS_PER_SOURCE = int(os.getenv(TRACE_MAX_RECORDS_PER_SOURCE_ENV, "0"))
_TRACE_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "sglang_big_tp_collective_trace_depth", default=0
)
_TRACE_SEQUENCE = itertools.count(1)
_TRACE_RECORD_COUNT = 0
_TRACE_RECORD_COUNT_BY_SOURCE: dict[str, int] = {}


def trace_enabled() -> bool:
    """Return whether tracing was enabled before this process started."""
    return _TRACE_ENABLED


def is_big_tp_group(group: Any) -> bool:
    """Identify the outer TP coordinator, including the PD-mux prefill variant."""
    unique_name = str(getattr(group, "unique_name", ""))
    return unique_name.startswith(("tp:", "pdmux_prefill_tp:"))


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
    }


def _collect_tensor_metadata(
    value: Any, output: list[dict[str, Any]], stats: dict[str, int]
) -> None:
    if isinstance(value, torch.Tensor):
        stats["count"] += 1
        if len(output) < _MAX_TENSOR_METADATA_PER_RECORD:
            output.append(_tensor_metadata(value))
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_tensor_metadata(item, output, stats)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_tensor_metadata(item, output, stats)


def _safe_backend(process_group: Any) -> Optional[str]:
    if process_group is None:
        return None
    try:
        return str(torch.distributed.get_backend(process_group))
    except Exception:
        return None


def _safe_dist_rank(process_group: Any = None) -> Optional[int]:
    try:
        if not torch.distributed.is_initialized():
            return None
        return torch.distributed.get_rank(process_group)
    except Exception:
        return None


def _safe_dist_world_size(process_group: Any = None) -> Optional[int]:
    try:
        if not torch.distributed.is_initialized():
            return None
        return torch.distributed.get_world_size(process_group)
    except Exception:
        return None


def _safe_active_mask(group: Any) -> Optional[list[int]]:
    active_ranks_cpu = getattr(group, "active_ranks_cpu", None)
    if not isinstance(active_ranks_cpu, torch.Tensor) or not active_ranks_cpu.is_cpu:
        return None
    # Reading this CPU control tensor is safe and does not synchronize a device.
    return [int(value) for value in active_ranks_cpu.tolist()]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return repr(value)


def _emit(record: dict[str, Any]) -> None:
    print(
        TRACE_PREFIX + json.dumps(record, sort_keys=True, ensure_ascii=False),
        flush=True,
    )


def _profile_marker_name(scope: str, source: str, op: str) -> str:
    return f"{PROFILE_MARKER_PREFIX}{scope}::{source}::{op}"


@contextmanager
def trace_collective(
    op: str,
    *,
    coordinator: Any = None,
    process_group: Any = None,
    scope: str = "big_tp",
    source: Optional[str] = None,
    tensors: Any = None,
    extra: Optional[dict[str, Any]] = None,
    require_big_tp_group: bool = False,
) -> Iterator[None]:
    """Trace one communication call without inspecting tensor contents.

    ``scope`` distinguishes exact outer-TP calls from WORLD/EPLB paths that
    happen to cover the same ranks for PP=1. Nested calls are suppressed so a
    high-level ``all_gather`` and its internal ``all_gather_into_tensor`` do not
    produce duplicate records.
    """
    global _TRACE_RECORD_COUNT

    source_key = source or "<unspecified>"
    should_instrument = _TRACE_ENABLED
    should_instrument = should_instrument and (
        not require_big_tp_group or is_big_tp_group(coordinator)
    )
    should_instrument = should_instrument and _TRACE_DEPTH.get() == 0
    if not should_instrument:
        yield
        return

    should_emit = _TRACE_MAX_RECORDS <= 0 or _TRACE_RECORD_COUNT < _TRACE_MAX_RECORDS
    should_emit = should_emit and (
        _TRACE_MAX_RECORDS_PER_SOURCE <= 0
        or _TRACE_RECORD_COUNT_BY_SOURCE.get(source_key, 0)
        < _TRACE_MAX_RECORDS_PER_SOURCE
    )
    profile_marker = _profile_marker_name(scope, source_key, op)
    sequence = next(_TRACE_SEQUENCE) if should_emit else None
    tensor_metadata: list[dict[str, Any]] = []
    tensor_stats = {"count": 0}
    if should_emit:
        _TRACE_RECORD_COUNT += 1
        _TRACE_RECORD_COUNT_BY_SOURCE[source_key] = (
            _TRACE_RECORD_COUNT_BY_SOURCE.get(source_key, 0) + 1
        )
        _collect_tensor_metadata(tensors, tensor_metadata, tensor_stats)

    base_record: dict[str, Any] = {}
    if should_emit:
        if process_group is None and coordinator is not None:
            process_group = getattr(coordinator, "device_group", None)
        ranks = getattr(coordinator, "ranks", None)
        rank_in_group = getattr(coordinator, "rank_in_group", None)
        group_name = getattr(coordinator, "unique_name", None)
        world_size = getattr(coordinator, "world_size", None)
        if world_size is None:
            world_size = _safe_dist_world_size(process_group)
        if ranks is None and world_size is not None:
            ranks = list(range(world_size))

        base_record = {
            "seq": sequence,
            "pid": os.getpid(),
            "profile_marker": profile_marker,
            "scope": scope,
            "source": source,
            "op": op,
            "group": group_name,
            "ranks": ranks,
            "group_size": world_size,
            "global_rank": _safe_dist_rank(),
            "rank_in_group": (
                rank_in_group
                if rank_in_group is not None
                else _safe_dist_rank(process_group)
            ),
            "backend": _safe_backend(process_group),
            "active_mask_cpu": _safe_active_mask(coordinator),
            "tensors": tensor_metadata,
            "tensor_count": tensor_stats["count"],
            "tensor_metadata_truncated": tensor_stats["count"] - len(tensor_metadata),
            "extra": _json_safe(extra or {}),
        }
    token = _TRACE_DEPTH.set(_TRACE_DEPTH.get() + 1)
    started_at = time.perf_counter_ns() if should_emit else 0
    if should_emit:
        _emit({"event": "BEGIN", **base_record})
    try:
        # Keep this marker active even after the text-record quota is
        # exhausted. The scheduler can call an MLP-sync collective thousands
        # of times before the requested profiling window starts; the profiler
        # itself records this scope only while it is active.
        with torch.profiler.record_function(profile_marker):
            yield
    except BaseException as error:
        if should_emit:
            _emit(
                {
                    "event": "ERROR",
                    **base_record,
                    "elapsed_us": (time.perf_counter_ns() - started_at) // 1000,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        raise
    else:
        if should_emit:
            _emit(
                {
                    "event": "RETURN",
                    **base_record,
                    "elapsed_us": (time.perf_counter_ns() - started_at) // 1000,
                    "completion": "python_call_returned_device_completion_not_implied",
                }
            )
    finally:
        _TRACE_DEPTH.reset(token)


def trace_big_tp_collective(
    op: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a GroupCoordinator method that communicates on the outer TP."""

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        # Preserve the exact production call path unless the process explicitly
        # opted in before importing SGLang. This avoids adding wrappers or graph
        # breaks to hot collective methods when tracing is disabled.
        if not _TRACE_ENABLED:
            return function

        @functools.wraps(function)
        def wrapped(group: Any, *args: Any, **kwargs: Any) -> Any:
            with trace_collective(
                op,
                coordinator=group,
                scope="big_tp",
                source=f"GroupCoordinator.{function.__name__}",
                tensors=(args, kwargs),
                require_big_tp_group=True,
            ):
                return function(group, *args, **kwargs)

        return wrapped

    return decorate
