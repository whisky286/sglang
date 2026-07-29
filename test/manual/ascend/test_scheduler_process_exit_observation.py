#!/usr/bin/env python3
"""A6: observe current SGLang behavior after one Scheduler is SIGKILLed.

This is a destructive manual experiment. It intentionally kills one Scheduler
process in an already running DP-attention service and records which parent and
sibling processes remain alive, plus the HTTP/control-plane responses.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psutil
import requests

SCHEDULER_TITLE_RE = re.compile(r"^sglang::scheduler(?:_|$)")
DP_RANK_RE = re.compile(r"(?:^|_)DP(\d+)(?:_|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--target-rank", type=int, required=True)
    parser.add_argument("--observe-seconds", type=float, default=10)
    parser.add_argument("--request-timeout", type=float, default=5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--confirm-destructive",
        action="store_true",
        help="Required acknowledgement that this test kills a Scheduler process.",
    )
    return parser.parse_args()


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_title(proc: psutil.Process) -> str:
    try:
        return " ".join(proc.cmdline())
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        return ""


def process_is_alive(pid: int) -> bool:
    try:
        status = psutil.Process(pid).status()
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    return status not in (psutil.STATUS_DEAD, psutil.STATUS_ZOMBIE)


def describe_process(pid: int, role: str) -> dict:
    try:
        proc = psutil.Process(pid)
        status = proc.status()
        return {
            "role": role,
            "pid": pid,
            "ppid": proc.ppid(),
            "alive": status not in (psutil.STATUS_DEAD, psutil.STATUS_ZOMBIE),
            "status": status,
            "title": process_title(proc),
        }
    except psutil.NoSuchProcess:
        return {
            "role": role,
            "pid": pid,
            "ppid": None,
            "alive": False,
            "status": "gone",
            "title": "",
        }
    except (psutil.AccessDenied, psutil.ZombieProcess) as exc:
        return {
            "role": role,
            "pid": pid,
            "ppid": None,
            "alive": False,
            "status": type(exc).__name__,
            "title": "",
        }


def scheduler_dp_rank(title: str) -> int | None:
    if not SCHEDULER_TITLE_RE.match(title):
        return None
    match = DP_RANK_RE.search(title)
    return int(match.group(1)) if match is not None else None


def find_target_scheduler(target_rank: int) -> psutil.Process:
    matches = []
    for proc in psutil.process_iter(["pid"]):
        title = process_title(proc)
        if scheduler_dp_rank(title) == target_rank:
            matches.append(proc)
    if len(matches) != 1:
        details = [{"pid": proc.pid, "title": process_title(proc)} for proc in matches]
        raise RuntimeError(
            f"Expected exactly one Scheduler for DP rank {target_rank}, "
            f"found {len(matches)}: {details}"
        )
    return matches[0]


def find_launch_server_ancestor(
    scheduler: psutil.Process, expected_port: int | None
) -> tuple[psutil.Process, psutil.Process]:
    dpc = scheduler.parent()
    if dpc is None or "sglang::data_parallel_controller" not in process_title(dpc):
        raise RuntimeError(
            f"Scheduler parent is not the DataParallelController: "
            f"pid={dpc.pid if dpc else None} title={process_title(dpc) if dpc else ''}"
        )
    main = dpc.parent()
    if main is None:
        raise RuntimeError("DataParallelController has no live parent process")
    main_title = process_title(main)
    if "sglang.launch_server" not in main_title:
        raise RuntimeError(
            f"DPC parent does not look like launch_server: "
            f"pid={main.pid} title={main_title}"
        )
    if expected_port is not None:
        port = str(expected_port)
        cmdline = main.cmdline()
        uses_expected_port = (
            any(
                arg == port and index > 0 and cmdline[index - 1] == "--port"
                for index, arg in enumerate(cmdline)
            )
            or f"--port={port}" in cmdline
        )
        if not uses_expected_port:
            raise RuntimeError(
                f"launch_server pid={main.pid} does not use expected port "
                f"{expected_port}"
            )
    return dpc, main


def build_tracked_processes(
    main: psutil.Process,
    dpc: psutil.Process,
    target: psutil.Process,
) -> dict[int, str]:
    tracked = {
        main.pid: "launch_server",
        dpc.pid: "data_parallel_controller",
        target.pid: "target_scheduler",
    }
    for proc in main.children(recursive=True):
        if proc.pid in tracked:
            continue
        title = process_title(proc)
        rank = scheduler_dp_rank(title)
        if rank is not None:
            tracked[proc.pid] = f"scheduler_dp{rank}"
        elif "sglang::detokenizer" in title:
            tracked[proc.pid] = "detokenizer"
        else:
            tracked[proc.pid] = "other_child"
    return tracked


def snapshot_processes(tracked: dict[int, str]) -> dict:
    return {
        "timestamp_utc": timestamp_utc(),
        "processes": [
            describe_process(pid, role)
            for pid, role in sorted(tracked.items(), key=lambda item: item[1])
        ],
    }


def request_json(
    session: requests.Session,
    base_url: str,
    path: str,
    timeout: float,
) -> dict:
    started = time.perf_counter()
    try:
        response = session.get(f"{base_url}{path}", timeout=timeout)
        try:
            body = response.json()
        except ValueError:
            body = response.text
        return {
            "timestamp_utc": timestamp_utc(),
            "path": path,
            "http_status": response.status_code,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "body": body,
            "error": None,
        }
    except requests.RequestException as exc:
        return {
            "timestamp_utc": timestamp_utc(),
            "path": path,
            "http_status": None,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "body": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def wait_for_target_exit(pid: int, timeout: float = 5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_is_alive(pid):
            return True
        time.sleep(0.05)
    return not process_is_alive(pid)


def main() -> None:
    args = parse_args()
    if not args.confirm_destructive:
        raise RuntimeError(
            "This test kills a live Scheduler. Re-run with --confirm-destructive."
        )
    if args.target_rank < 0:
        raise ValueError("--target-rank must be non-negative")
    if args.observe_seconds < 0:
        raise ValueError("--observe-seconds must be non-negative")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be positive")

    parsed_url = urlparse(args.base_url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.hostname:
        raise ValueError("--base-url must be an HTTP(S) URL")
    base_url = args.base_url.rstrip("/")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target = find_target_scheduler(args.target_rank)
    dpc, launch_server = find_launch_server_ancestor(target, parsed_url.port)
    tracked = build_tracked_processes(launch_server, dpc, target)

    before = snapshot_processes(tracked)
    write_json(output_dir / "processes_before.json", before)

    with requests.Session() as session:
        session.trust_env = False
        status_before = request_json(
            session,
            base_url,
            "/fault_tolerance/status",
            args.request_timeout,
        )
        write_json(output_dir / "status_before.json", status_before)

        kill_event = {
            "timestamp_utc": timestamp_utc(),
            "signal": "SIGKILL",
            "target_original_rank": args.target_rank,
            "target_pid": target.pid,
            "dpc_pid": dpc.pid,
            "launch_server_pid": launch_server.pid,
        }
        write_json(output_dir / "kill_event.json", kill_event)
        os.kill(target.pid, signal.SIGKILL)
        target_exited = wait_for_target_exit(target.pid)

        time.sleep(0.5)
        immediate = snapshot_processes(tracked)
        write_json(output_dir / "processes_immediate.json", immediate)

        health_after = request_json(session, base_url, "/health", args.request_timeout)
        write_json(output_dir / "health_after.json", health_after)
        model_info_after = request_json(
            session, base_url, "/model_info", args.request_timeout
        )
        write_json(output_dir / "model_info_after.json", model_info_after)
        status_after = request_json(
            session,
            base_url,
            "/fault_tolerance/status",
            args.request_timeout,
        )
        write_json(output_dir / "status_after.json", status_after)

    time.sleep(args.observe_seconds)
    final = snapshot_processes(tracked)
    write_json(output_dir / "processes_final.json", final)

    dpc_alive_immediate = next(
        item["alive"]
        for item in immediate["processes"]
        if item["role"] == "data_parallel_controller"
    )
    main_alive_immediate = next(
        item["alive"]
        for item in immediate["processes"]
        if item["role"] == "launch_server"
    )
    dpc_alive_final = next(
        item["alive"]
        for item in final["processes"]
        if item["role"] == "data_parallel_controller"
    )
    main_alive_final = next(
        item["alive"] for item in final["processes"] if item["role"] == "launch_server"
    )

    if not main_alive_final:
        outcome = "whole-service-exited"
    elif not dpc_alive_final:
        outcome = "dpc-exited-main-still-visible"
    else:
        outcome = "dpc-and-main-survived-observation-window"

    errors = []
    if status_before.get("http_status") != 200:
        errors.append("fault-tolerance status was not healthy before SIGKILL")
    if not target_exited:
        errors.append("target Scheduler did not exit or become a zombie")

    result = {
        "experiment": "A6-scheduler-process-exit-observation",
        "timestamp_utc": timestamp_utc(),
        "completed": not errors,
        "errors": errors,
        "target_original_rank": args.target_rank,
        "target_pid": target.pid,
        "outcome": outcome,
        "target_exited": target_exited,
        "dpc_alive_immediate": dpc_alive_immediate,
        "launch_server_alive_immediate": main_alive_immediate,
        "dpc_alive_final": dpc_alive_final,
        "launch_server_alive_final": main_alive_final,
        "health_after": {
            "http_status": health_after.get("http_status"),
            "error": health_after.get("error"),
        },
        "model_info_after": {
            "http_status": model_info_after.get("http_status"),
            "error": model_info_after.get("error"),
        },
        "fault_tolerance_status_after": {
            "http_status": status_after.get("http_status"),
            "error": status_after.get("error"),
        },
        "artifacts": [
            "processes_before.json",
            "status_before.json",
            "kill_event.json",
            "processes_immediate.json",
            "health_after.json",
            "model_info_after.json",
            "status_after.json",
            "processes_final.json",
        ],
    }
    write_json(output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
