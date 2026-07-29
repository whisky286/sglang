#!/usr/bin/env python3
"""A5: compare healthy MLP-sync all-gather with CPU control aggregation."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: float,
    payload: dict | None = None,
) -> dict:
    started = time.perf_counter()
    try:
        response = session.request(method, url, json=payload, timeout=timeout)
        try:
            body = response.json()
        except ValueError:
            body = response.text
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "http_status": response.status_code,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "body": body,
            "error": None,
        }
    except requests.RequestException as exc:
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "http_status": None,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "body": None,
            "error": str(exc),
        }


def all_ranks_paused(status: dict) -> bool:
    body = status.get("body")
    ranks = body.get("ranks") if isinstance(body, dict) else None
    return (
        status.get("http_status") == 200
        and isinstance(ranks, list)
        and bool(ranks)
        and all(rank.get("available") and rank.get("engine_paused") for rank in ranks)
    )


def all_ranks_healthy(status: dict) -> bool:
    body = status.get("body")
    ranks = body.get("ranks") if isinstance(body, dict) else None
    return (
        status.get("http_status") == 200
        and body.get("service_state") == "HEALTHY"
        and isinstance(ranks, list)
        and bool(ranks)
        and all(
            rank.get("available") and not rank.get("engine_paused") for rank in ranks
        )
    )


def wait_for_pause(session: requests.Session, base_url: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last_status = {}
    while time.monotonic() < deadline:
        last_status = request_json(
            session,
            "GET",
            f"{base_url}/fault_tolerance/status",
            timeout=min(5, timeout),
        )
        if all_ranks_paused(last_status):
            return last_status
        time.sleep(0.1)
    return last_status


def wait_for_health(session: requests.Session, base_url: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last_status = {}
    while time.monotonic() < deadline:
        last_status = request_json(
            session,
            "GET",
            f"{base_url}/fault_tolerance/status",
            timeout=min(5, timeout),
        )
        if all_ranks_healthy(last_status):
            return last_status
        time.sleep(0.1)
    return last_status


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    errors = []
    pause = {}
    paused_status = {}
    parity = {}
    resume = {}
    resumed_status = {}

    with requests.Session() as session:
        session.trust_env = False
        pause = request_json(
            session,
            "POST",
            f"{base_url}/pause_generation",
            timeout=args.timeout,
            payload={"mode": "in_place"},
        )
        write_json(output_dir / "pause.json", pause)
        if pause.get("http_status") != 200:
            errors.append("pause_generation did not return HTTP 200")
        else:
            paused_status = wait_for_pause(session, base_url, args.timeout)
            write_json(output_dir / "paused_status.json", paused_status)
            if not all_ranks_paused(paused_status):
                errors.append("not every original Scheduler reached paused state")
            else:
                parity = request_json(
                    session,
                    "POST",
                    f"{base_url}/fault_tolerance/probe_metadata_parity",
                    timeout=args.timeout,
                    payload={},
                )
                write_json(output_dir / "parity.json", parity)
                parity_body = parity.get("body")
                if parity.get("http_status") != 200:
                    errors.append("metadata parity probe did not return HTTP 200")
                if not isinstance(parity_body, dict) or not parity_body.get("success"):
                    errors.append("metadata parity probe did not report success")
                elif not all(
                    comparison.get("matched")
                    for comparison in parity_body.get("comparisons", [])
                ):
                    errors.append("at least one rank reported a metadata mismatch")

        if pause.get("http_status") == 200:
            resume = request_json(
                session,
                "POST",
                f"{base_url}/continue_generation",
                timeout=args.timeout,
                payload={"torch_empty_cache": False},
            )
            write_json(output_dir / "resume.json", resume)
            if resume.get("http_status") != 200:
                errors.append("continue_generation did not return HTTP 200")
            resumed_status = wait_for_health(session, base_url, args.timeout)
            write_json(output_dir / "resumed_status.json", resumed_status)
            if not all_ranks_healthy(resumed_status):
                errors.append("service did not return to HEALTHY after the probe")

    result = {
        "experiment": "A5-mlp-sync-metadata-parity",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "errors": errors,
        "mlp_sync_group": (
            parity.get("body", {}).get("mlp_sync_group")
            if isinstance(parity.get("body"), dict)
            else None
        ),
        "pause_http_status": pause.get("http_status"),
        "parity_http_status": parity.get("http_status"),
        "resume_http_status": resume.get("http_status"),
    }
    write_json(output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
