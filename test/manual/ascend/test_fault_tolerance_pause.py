#!/usr/bin/env python3
"""Run experiment A2 against an already-running four-rank server."""

import argparse
import datetime
import json
import pathlib
import time
import urllib.error
import urllib.request
import uuid


def request_json(
    url: str, *, method: str, payload: dict | None, timeout: float
) -> tuple[int, object, float]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = response.status
            raw_body = response.read()
    except urllib.error.HTTPError as error:
        status_code = error.code
        raw_body = error.read()
    except (urllib.error.URLError, TimeoutError) as error:
        status_code = 0
        raw_body = json.dumps({"client_error": str(error)}).encode("utf-8")
    elapsed_ms = (time.monotonic() - start) * 1000
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        body = raw_body.decode("utf-8", errors="replace")
    return status_code, body, elapsed_ms


def observation(status_code: int, body: object, elapsed_ms: float) -> dict:
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "http_status": status_code,
        "elapsed_ms": elapsed_ms,
        "body": body,
    }


def write_json(output_dir: pathlib.Path, name: str, value: dict) -> None:
    with (output_dir / name).open("w") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def validate(
    *,
    request_id: str,
    target_rank: int,
    arm: dict,
    trigger: dict,
    status: dict,
    rejected: dict,
) -> list[str]:
    errors = []
    arm_body = arm["body"]
    if arm["http_status"] != 200 or not isinstance(arm_body, dict):
        errors.append(f"arm request failed: {arm!r}")
    else:
        if arm_body.get("success") is not True:
            errors.append(f"arm response was unsuccessful: {arm_body!r}")
        if arm_body.get("original_rank") != target_rank:
            errors.append(f"arm response targeted the wrong rank: {arm_body!r}")
        if arm_body.get("request_id") != request_id:
            errors.append(f"arm response targeted the wrong request: {arm_body!r}")
        if arm_body.get("acknowledged_ranks") != [target_rank]:
            errors.append(f"arm ack did not come only from rank {target_rank}")

    if trigger["http_status"] != 503:
        errors.append(
            f"injected request should fail with HTTP 503, got {trigger!r}"
        )

    status_body = status["body"]
    if status["http_status"] != 200 or not isinstance(status_body, dict):
        errors.append(f"status request failed: {status!r}")
    else:
        if status_body.get("service_state") != "PAUSED":
            errors.append(f"expected PAUSED status, got {status_body!r}")
        if status_body.get("admission_closed") is not True:
            errors.append("fault-tolerance admission gate is not closed")
        ranks = status_body.get("ranks")
        if not isinstance(ranks, list) or len(ranks) != 4:
            errors.append(f"expected four paused rank records, got {ranks!r}")
        elif any(rank.get("engine_paused") is not True for rank in ranks):
            errors.append(f"not every original rank is paused: {ranks!r}")

        fault = status_body.get("last_fault") or {}
        if fault.get("original_rank") != target_rank:
            errors.append(f"fault was not reported only by rank {target_rank}: {fault!r}")
        if fault.get("request_id") != request_id:
            errors.append(f"fault reported the wrong request: {fault!r}")

        transition = status_body.get("last_transition") or {}
        if transition.get("command") != "pause":
            errors.append(f"missing pause transition: {transition!r}")
        if transition.get("state") != "SUCCEEDED":
            errors.append(f"pause transition did not succeed: {transition!r}")
        if transition.get("acknowledged_ranks") != [0, 1, 2, 3]:
            errors.append(f"pause did not collect all four acks: {transition!r}")
        if not transition.get("command_id"):
            errors.append(f"pause transition has no command_id: {transition!r}")

    if rejected["http_status"] != 503:
        errors.append(
            "a new request during fault-tolerance pause should fail immediately "
            f"with HTTP 503, got {rejected!r}"
        )
    if rejected["elapsed_ms"] > 2000:
        errors.append(
            "a new request during pause was not rejected within 2000 ms: "
            f"{rejected['elapsed_ms']:.2f} ms"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--target-rank", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url.rstrip("/")
    request_id = f"a2-recoverable-error-{uuid.uuid4().hex}"

    arm = observation(
        *request_json(
            f"{base_url}/fault_tolerance/inject_recoverable_error",
            method="POST",
            payload={
                "original_rank": args.target_rank,
                "request_id": request_id,
            },
            timeout=args.timeout,
        )
    )
    write_json(output_dir, "arm.json", arm)

    trigger = observation(
        *request_json(
            f"{base_url}/generate",
            method="POST",
            payload={
                "text": "A2 recoverable error trigger",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                "rid": request_id,
                "routed_dp_rank": args.target_rank,
            },
            timeout=args.timeout,
        )
    )
    write_json(output_dir, "trigger.json", trigger)

    deadline = time.monotonic() + args.timeout
    status = None
    while time.monotonic() < deadline:
        status = observation(
            *request_json(
                f"{base_url}/fault_tolerance/status",
                method="GET",
                payload=None,
                timeout=args.timeout,
            )
        )
        body = status["body"]
        if (
            status["http_status"] == 200
            and isinstance(body, dict)
            and body.get("service_state") == "PAUSED"
            and (body.get("last_transition") or {}).get("state") == "SUCCEEDED"
        ):
            break
        time.sleep(0.1)
    assert status is not None
    write_json(output_dir, "status.json", status)

    rejected = observation(
        *request_json(
            f"{base_url}/generate",
            method="POST",
            payload={
                "text": "This request must be rejected while paused",
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "rid": f"a2-rejected-{uuid.uuid4().hex}",
            },
            timeout=args.timeout,
        )
    )
    write_json(output_dir, "rejected.json", rejected)

    errors = validate(
        request_id=request_id,
        target_rank=args.target_rank,
        arm=arm,
        trigger=trigger,
        status=status,
        rejected=rejected,
    )
    result = {
        "experiment": "A2-recoverable-error-and-coordinated-pause",
        "passed": not errors,
        "errors": errors,
        "request_id": request_id,
        "target_original_rank": args.target_rank,
        "arm_command_id": (
            arm["body"].get("command_id") if isinstance(arm["body"], dict) else None
        ),
        "pause_command_id": (
            (status["body"].get("last_transition") or {}).get("command_id")
            if isinstance(status["body"], dict)
            else None
        ),
    }
    write_json(output_dir, "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
