"""Fault-inject and verify a disposable local NPU FT server.

Only scheduler descendants of the explicitly supplied server PID are killed.
The HTTP server remains running on failure so its logs can be inspected.
--scale-down-only also works with the pre-recovery serving implementation.
"""

import argparse
import importlib.metadata
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


def request_json(session, base_url, method, path, *, timeout_sec, event, **kwargs):
    """Log before blocking; never retry a possibly accepted FT mutation."""
    event("http_request_started", method=method, path=path, timeout_sec=timeout_sec)
    started = time.monotonic()
    response = None
    try:
        response = session.request(
            method, base_url.rstrip("/") + path, timeout=(5, timeout_sec), **kwargs
        )
        response.raise_for_status()
        result = response.json()
    except Exception as exc:
        details = {}
        if response is not None and response.status_code >= 400:
            # HTTPError.__str__ only contains the status and URL. Preserve the
            # scheduler abort reason instead of losing it behind a generic 503.
            body = response.text
            details = {
                "status_code": response.status_code,
                "response_text": body[:4096],
                "response_truncated": len(body) > 4096,
            }
        event(
            "http_request_failed",
            method=method,
            path=path,
            elapsed_sec=round(time.monotonic() - started, 3),
            error=str(exc),
            **details,
        )
        raise
    event(
        "http_request_finished",
        method=method,
        path=path,
        elapsed_sec=round(time.monotonic() - started, 3),
    )
    return result


def schedulers(server_pid):
    import psutil

    found = {}
    for process in psutil.Process(server_pid).children(recursive=True):
        try:
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                continue
            command = process.cmdline()
            title = command[0] if command else ""
            if not title.startswith("sglang::scheduler_"):
                continue
            rank = re.search(r"(?:^|_)DP(\d+)(?:_|$)", title)
            if rank:
                rank = int(rank.group(1))
                if rank in found:
                    raise RuntimeError(f"more than one scheduler owns DP rank {rank}")
                found[rank] = process
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--fault-rank", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument(
        "--scale-down-only",
        action="store_true",
        help="verify one shrink and survivor generation without requiring rank recovery",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument(
        "--expected-text", help="optional substring required in every generation"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/npu-recovery-serving.json")
    )
    parser.add_argument("--api-key-env", default="SGLANG_API_KEY")
    args = parser.parse_args()
    if urlsplit(args.base_url).hostname not in ("localhost", "127.0.0.1", "::1"):
        parser.error("the fault injection server must be local")
    if args.cycles <= 0 or args.timeout <= 0:
        parser.error("cycles and timeout must be positive")
    if args.scale_down_only and args.cycles != 1:
        parser.error("--scale-down-only requires --cycles 1")
    report = {
        "success": False,
        "mode": "scale_down_only" if args.scale_down_only else "shrink_recover",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "events": [],
    }

    def save_report():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        temporary.replace(args.output)

    def event(stage, **details):
        entry = {
            "stage": stage,
            "at": datetime.now(timezone.utc).isoformat(),
            **details,
        }
        report["events"].append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        save_report()

    def request(method, path, **kwargs):
        # A whole FT transaction may contain several timeout-bounded phases.
        # Read-only queries must not inherit that forty-minute wait budget.
        timeout = (
            4 * args.timeout + 30
            if path == "/fault_tolerance/apply"
            else args.timeout
            if path == "/generate"
            else 30
        )
        return request_json(
            session,
            args.base_url,
            method,
            path,
            timeout_sec=timeout,
            event=event,
            **kwargs,
        )

    def wait_ready():
        deadline = time.monotonic() + args.timeout
        next_progress = time.monotonic() + 10
        status = None
        while time.monotonic() < deadline:
            try:
                response = session.get(args.base_url.rstrip("/") + "/health", timeout=5)
                if response.status_code == 200:
                    event("health_ready")
                    return
                status = response.status_code
            except requests.RequestException as exc:
                status = str(exc)
            if not psutil.pid_exists(args.server_pid):
                raise RuntimeError("server process exited during startup")
            if time.monotonic() >= next_progress:
                event("waiting_for_health", last_status=status)
                next_progress = time.monotonic() + 10
            time.sleep(1)
        raise TimeoutError("server health readiness timed out")

    def generate(stage, rank):
        result = request(
            "POST",
            "/generate",
            json={
                "text": args.prompt,
                "routed_dp_rank": rank,
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
        )
        if (
            not result.get("text")
            or result.get("meta_info", {}).get("completion_tokens", 0) <= 0
        ):
            raise RuntimeError(f"generation returned no completion: {result}")
        reason = result.get("meta_info", {}).get("finish_reason", {})
        if isinstance(reason, dict) and reason.get("type") == "abort":
            raise RuntimeError(f"generation aborted: {result}")
        if (
            args.expected_text
            and args.expected_text.casefold() not in result["text"].casefold()
        ):
            raise RuntimeError(f"generation failed expected-text check: {result}")
        event(stage, routed_rank=rank, response=result)

    try:
        event("test_started", server_pid=args.server_pid, base_url=args.base_url)
        event("loading_test_dependencies")
        import psutil
        import requests

        session = requests.Session()
        session.trust_env = False
        if os.getenv(args.api_key_env):
            session.headers["Authorization"] = f"Bearer {os.environ[args.api_key_env]}"
        event("collecting_environment")
        report["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        report["versions"] = {}
        for name in ("torch", "torch-npu", "sgl-kernel-npu", "deep-ep"):
            report["versions"][name] = importlib.metadata.version(name)
        event("waiting_for_health", timeout_sec=args.timeout)
        wait_ready()
        if args.scale_down_only:
            # The September baseline has no recovery flag and its server_info
            # fan-out predates the fix. Use the FT status endpoint to discover
            # its cohort without changing that baseline's launch configuration.
            status = request("GET", "/fault_tolerance/status")
            rows = status["ranks"]
            if not rows or any(row["state"] != "healthy" for row in rows):
                raise RuntimeError("scale-down baseline requires all ranks healthy")
            expected_ranks = {row["rank"] for row in rows}
        else:
            info = request("GET", "/server_info")
            if not info.get("enable_npu_ft_recovery"):
                raise RuntimeError(
                    "server was not started with --enable-npu-ft-recovery"
                )
            expected_ranks = set(range(info["dp_size"]))
        if len(expected_ranks) < 2:
            raise ValueError("scale-down verification requires at least two ranks")
        if args.fault_rank not in expected_ranks:
            raise ValueError("fault-rank is outside the serving cohort")
        for cycle in range(args.cycles):
            before = schedulers(args.server_pid)
            if set(before) != expected_ranks:
                raise RuntimeError(
                    f"unexpected scheduler cohort before injection: {sorted(before)}"
                )
            identities = {
                rank: (process.pid, process.create_time())
                for rank, process in before.items()
            }
            event(
                "cohort_before_fault",
                cycle=cycle,
                pids={rank: value[0] for rank, value in identities.items()},
            )
            generate("baseline_generation", args.fault_rank)
            # Recheck parent/creation identity immediately before killing.
            victim = before[args.fault_rank]
            if (
                schedulers(args.server_pid)[args.fault_rank].create_time()
                != identities[args.fault_rank][1]
            ):
                raise RuntimeError("scheduler identity changed before fault injection")
            victim.kill()
            event("scheduler_killed", cycle=cycle, rank=args.fault_rank, pid=victim.pid)
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                status = request("GET", "/fault_tolerance/status")
                if any(
                    row["rank"] == args.fault_rank and row["state"] == "dead"
                    for row in status["ranks"]
                ):
                    break
                time.sleep(0.2)
            else:
                raise TimeoutError("controller did not observe scheduler exit")
            reduced = request(
                "POST",
                "/fault_tolerance/apply",
                json={
                    "instruction": "scale_down",
                    "params": {"ranks": [args.fault_rank], "timeout": args.timeout},
                },
            )
            if not reduced.get("success"):
                raise RuntimeError(f"scale-down failed: {reduced}")
            event("scale_down_committed", cycle=cycle, response=reduced)
            generate("survivor_generation", min(expected_ranks - {args.fault_rank}))
            if args.scale_down_only:
                after = schedulers(args.server_pid)
                if set(after) != expected_ranks - {args.fault_rank} or any(
                    (process.pid, process.create_time()) != identities[rank]
                    for rank, process in after.items()
                ):
                    raise RuntimeError("survivor scheduler identities changed")
                event(
                    "scale_down_verified",
                    pids={rank: process.pid for rank, process in after.items()},
                )
                break
            restored = request(
                "POST",
                "/fault_tolerance/apply",
                json={
                    "instruction": "recover",
                    "params": {"ranks": [args.fault_rank], "timeout": args.timeout},
                },
            )
            if not restored.get("success") or any(
                row["state"] != "healthy" for row in restored["ranks"]
            ):
                raise RuntimeError(f"recovery failed: {restored}")
            after = schedulers(args.server_pid)
            if set(after) != expected_ranks:
                raise RuntimeError("restored scheduler cohort is incomplete")
            for rank, process in after.items():
                identity = (process.pid, process.create_time())
                if (identity != identities[rank]) != (rank == args.fault_rank):
                    raise RuntimeError(
                        f"unexpected process replacement for rank {rank}"
                    )
            event(
                "recover_committed",
                cycle=cycle,
                response=restored,
                pids={rank: process.pid for rank, process in after.items()},
            )
            generate("recovered_rank_generation", args.fault_rank)
        report["success"] = True
        report["scope"] = (
            "local model serving; shrink and survivor generation only; rank recovery not exercised"
            if args.scale_down_only
            else "local eager model serving; scheduler replacement, shrink/recover and forced routing to the restored rank"
        )
    except Exception as exc:
        report["error"] = str(exc)
        raise
    except KeyboardInterrupt:
        report["error"] = "test interrupted; inspect the last stage before retrying"
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_report()
        print(
            json.dumps({"success": report["success"], "report": str(args.output)}),
            flush=True,
        )


if __name__ == "__main__":
    main()
