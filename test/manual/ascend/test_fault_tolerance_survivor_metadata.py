#!/usr/bin/env python3
"""A4 probe for survivor-only Scheduler metadata aggregation over CPU control."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--active-mask", default="1,0,1,1")
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_active_mask(value: str) -> list[int]:
    try:
        mask = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("--active-mask must be a comma-separated 0/1 list") from exc
    if not mask or any(item not in (0, 1) for item in mask):
        raise ValueError("--active-mask must be a comma-separated 0/1 list")
    return mask


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    active_mask = parse_active_mask(args.active_mask)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        session.trust_env = False
        response = session.post(
            f"{args.base_url.rstrip('/')}/fault_tolerance/probe_survivor_metadata",
            json={"active_mask": active_mask},
            timeout=args.timeout,
        )
    try:
        body = response.json()
    except ValueError:
        body = response.text

    expected_active_ranks = [
        original_rank
        for original_rank, is_active in enumerate(active_mask)
        if is_active
    ]
    errors = []
    if response.status_code != 200:
        errors.append(f"HTTP status is {response.status_code}, expected 200")
    if not isinstance(body, dict) or not body.get("success"):
        errors.append("Probe response did not report success")
    if isinstance(body, dict):
        if body.get("target_original_ranks") != expected_active_ranks:
            errors.append("Control plane targeted ranks do not match active_mask")
        if body.get("acknowledged_ranks") != expected_active_ranks:
            errors.append("Acknowledged ranks do not match active survivors")
        slots = body.get("slots")
        if not isinstance(slots, list) or len(slots) != len(active_mask):
            errors.append("Result is not an original-rank fixed-width slot view")
        else:
            for original_rank, is_active in enumerate(active_mask):
                slot = slots[original_rank]
                if slot.get("original_rank") != original_rank:
                    errors.append(f"Slot {original_rank} lost original-rank ordering")
                expected_source = "Scheduler" if is_active else "IDLE/fallback"
                if slot.get("source") != expected_source:
                    errors.append(
                        f"Slot {original_rank} source is {slot.get('source')}, "
                        f"expected {expected_source}"
                    )
                if not is_active and slot.get("metadata") != {
                    "num_running_requests": 0,
                    "num_waiting_requests": 0,
                    "last_batch_forward_mode": "IDLE",
                }:
                    errors.append(
                        f"Inactive slot {original_rank} is not the IDLE fallback"
                    )

    result = {
        "experiment": "A4-survivor-only-cpu-metadata",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "errors": errors,
        "http_status": response.status_code,
        "active_mask": active_mask,
        "body": body,
    }
    write_json(output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
