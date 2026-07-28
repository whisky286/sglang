#!/usr/bin/env python3
"""Send a mixed healthy-state workload for the Scheduler skip A/B experiment.

The server must already be running. This script does not inject a fault and
does not call any fault-tolerance endpoint.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


CASES = (
    ("The capital of France is", 1),
    ("Write one short sentence about distributed systems.", 4),
    ("Explain why synchronization is needed between workers.", 16),
    ("Describe a safe recovery sequence for a distributed service.", 32),
    ("The capital of Japan is", 1),
    ("Give three words related to scheduling.", 4),
    ("Explain the difference between prefill and decode.", 16),
    ("Describe how an idle worker should behave in a batch.", 32),
    ("The result of one plus one is", 1),
    ("Write a four-token continuation.", 4),
    ("Explain why stale metadata can be dangerous.", 16),
    ("Describe a deterministic test for concurrent inference.", 32),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--label", choices=("baseline", "skip"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compare-with")
    parser.add_argument("--timeout", type=float, default=180)
    return parser.parse_args()


def send_one(
    base_url: str,
    timeout: float,
    case_id: int,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    started = time.perf_counter()
    try:
        # Do not inherit HTTP_PROXY/HTTPS_PROXY for a localhost request.
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(
                f"{base_url.rstrip('/')}/generate",
                json=payload,
                timeout=timeout,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        try:
            body = response.json()
        except ValueError:
            body = response.text
        return {
            "case_id": case_id,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "http_status": response.status_code,
            "elapsed_ms": elapsed_ms,
            "body": body,
            "error": None,
        }
    except Exception as exc:
        return {
            "case_id": case_id,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "http_status": None,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "body": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def normalized_output(result: dict[str, Any]) -> tuple[Any, Any]:
    body = result.get("body")
    if not isinstance(body, dict):
        return None, None
    return body.get("text"), body.get("output_ids")


def compare_results(
    current: list[dict[str, Any]], reference_path: Path
) -> dict[str, Any]:
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    reference = {
        item["case_id"]: item for item in reference_payload.get("results", [])
    }
    mismatches = []
    for item in current:
        expected = reference.get(item["case_id"])
        if expected is None:
            mismatches.append(
                {"case_id": item["case_id"], "reason": "missing reference case"}
            )
            continue
        if normalized_output(item) != normalized_output(expected):
            mismatches.append(
                {
                    "case_id": item["case_id"],
                    "reason": "generated output differs",
                    "baseline": normalized_output(expected),
                    "current": normalized_output(item),
                }
            )
    return {
        "reference": str(reference_path),
        "passed": not mismatches,
        "mismatches": mismatches,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                send_one,
                args.base_url,
                args.timeout,
                case_id,
                prompt,
                max_new_tokens,
            )
            for case_id, (prompt, max_new_tokens) in enumerate(CASES)
        ]
        results = [future.result() for future in futures]
    results.sort(key=lambda item: item["case_id"])

    errors = [
        {
            "case_id": item["case_id"],
            "http_status": item["http_status"],
            "error": item["error"],
        }
        for item in results
        if item["http_status"] != 200 or item["error"] is not None
    ]
    comparison = (
        compare_results(results, Path(args.compare_with))
        if args.compare_with
        else None
    )
    passed = not errors and (comparison is None or comparison["passed"])
    payload = {
        "experiment": "scheduler-all-gather-skip-understanding",
        "label": args.label,
        "healthy_state_only": True,
        "fault_tolerance_enabled": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "passed": passed,
        "errors": errors,
        "comparison": comparison,
        "results": results,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "label": args.label,
                "result_path": str(result_path),
                "errors": errors,
                "comparison": comparison,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
