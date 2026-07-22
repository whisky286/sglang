#!/usr/bin/env python3
"""Run experiment A1 against an already-running four-rank server."""

import argparse
import datetime
import json
import pathlib
import time
import urllib.error
import urllib.request


def query_status(base_url: str, timeout: float) -> tuple[int, dict, float]:
    start = time.monotonic()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/fault_tolerance/status",
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = response.status
            body = json.load(response)
    except urllib.error.HTTPError as error:
        status_code = error.code
        body = json.load(error)
    elapsed_ms = (time.monotonic() - start) * 1000
    return status_code, body, elapsed_ms


def validate(status_code: int, body: dict) -> list[str]:
    errors = []
    if status_code != 200:
        errors.append(f"expected HTTP 200, got {status_code}")
    if body.get("success") is not True:
        errors.append("status response success is not true")
    if body.get("service_state") != "HEALTHY":
        errors.append(
            f"expected service_state HEALTHY, got {body.get('service_state')!r}"
        )
    if body.get("original_ranks") != [0, 1, 2, 3]:
        errors.append(
            f"expected original_ranks [0, 1, 2, 3], got {body.get('original_ranks')!r}"
        )

    ranks = body.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != 4:
        errors.append(f"expected four rank records, got {ranks!r}")
    else:
        for expected_rank, rank_status in enumerate(ranks):
            if rank_status.get("original_rank") != expected_rank:
                errors.append(
                    f"rank record {expected_rank} has original_rank "
                    f"{rank_status.get('original_rank')!r}"
                )
            if rank_status.get("available") is not True:
                errors.append(f"original rank {expected_rank} is not available")
            if rank_status.get("engine_paused") is not False:
                errors.append(f"original rank {expected_rank} is unexpectedly paused")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observations = []
    errors = []
    for index in range(1, 3):
        status_code, body, elapsed_ms = query_status(args.base_url, args.timeout)
        observation = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "http_status": status_code,
            "elapsed_ms": elapsed_ms,
            "body": body,
        }
        observations.append(observation)
        errors.extend(
            f"query {index}: {error}" for error in validate(status_code, body)
        )
        with (output_dir / f"status-{index}.json").open("w") as file:
            json.dump(observation, file, indent=2, ensure_ascii=False)

    command_ids = [item["body"].get("command_id") for item in observations]
    if None in command_ids or command_ids[0] == command_ids[1]:
        errors.append("repeated status queries did not use distinct command_id values")

    result = {
        "experiment": "A1-fault-tolerance-status",
        "passed": not errors,
        "errors": errors,
        "command_ids": command_ids,
    }
    with (output_dir / "result.json").open("w") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
