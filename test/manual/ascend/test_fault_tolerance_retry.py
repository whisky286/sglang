#!/usr/bin/env python3
"""Run experiment A3 against a server paused by experiment A2."""

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
    before: dict,
    apply: dict,
    status: dict,
    generated: dict,
) -> list[str]:
    errors = []
    before_body = before["body"]
    if before["http_status"] != 200 or not isinstance(before_body, dict):
        errors.append(f"pre-retry status request failed: {before!r}")
    else:
        if before_body.get("service_state") != "PAUSED":
            errors.append(f"A3 must begin in PAUSED state: {before_body!r}")
        if before_body.get("admission_closed") is not True:
            errors.append("admission was not closed before retry")
        ranks = before_body.get("ranks")
        if not isinstance(ranks, list) or len(ranks) != 4:
            errors.append(f"expected four rank records before retry: {ranks!r}")
        elif any(rank.get("engine_paused") is not True for rank in ranks):
            errors.append(f"not every rank was paused before retry: {ranks!r}")

    apply_body = apply["body"]
    if apply["http_status"] != 200 or not isinstance(apply_body, dict):
        errors.append(f"retry apply request failed: {apply!r}")
    else:
        if apply_body.get("success") is not True:
            errors.append(f"retry apply was unsuccessful: {apply_body!r}")
        if apply_body.get("action") != "retry":
            errors.append(f"unexpected apply action: {apply_body!r}")
        if apply_body.get("acknowledged_ranks") != [0, 1, 2, 3]:
            errors.append(f"retry did not collect all four acks: {apply_body!r}")
        if apply_body.get("missing_ranks") != []:
            errors.append(f"retry has missing ranks: {apply_body!r}")
        if apply_body.get("failed_ranks") != []:
            errors.append(f"retry has failed ranks: {apply_body!r}")

    status_body = status["body"]
    if status["http_status"] != 200 or not isinstance(status_body, dict):
        errors.append(f"post-retry status request failed: {status!r}")
    else:
        if status_body.get("service_state") != "HEALTHY":
            errors.append(f"expected HEALTHY after retry: {status_body!r}")
        if status_body.get("admission_closed") is not False:
            errors.append("admission was not reopened after all retry acks")
        ranks = status_body.get("ranks")
        if not isinstance(ranks, list) or len(ranks) != 4:
            errors.append(f"expected four rank records after retry: {ranks!r}")
        elif any(rank.get("engine_paused") is not False for rank in ranks):
            errors.append(f"not every rank resumed after retry: {ranks!r}")

        transition = status_body.get("last_transition") or {}
        if transition.get("command") != "retry":
            errors.append(f"missing retry transition: {transition!r}")
        if transition.get("state") != "SUCCEEDED":
            errors.append(f"retry transition did not succeed: {transition!r}")
        if transition.get("command_id") != (
            apply_body.get("command_id") if isinstance(apply_body, dict) else None
        ):
            errors.append("retry command_id differs between apply and status")

    if generated["http_status"] != 200:
        errors.append(f"post-retry inference failed: {generated!r}")
    elif not isinstance(generated["body"], dict):
        errors.append(f"post-retry inference returned a non-object: {generated!r}")
    else:
        meta_info = generated["body"].get("meta_info") or {}
        returned_request_id = meta_info.get("id")
        if returned_request_id is not None and returned_request_id != request_id:
            errors.append(
                "post-retry response belongs to a different request: "
                f"{returned_request_id!r}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--control-timeout", type=float, default=15.0)
    parser.add_argument("--generate-timeout", type=float, default=120.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url.rstrip("/")

    before = observation(
        *request_json(
            f"{base_url}/fault_tolerance/status",
            method="GET",
            payload=None,
            timeout=args.control_timeout,
        )
    )
    write_json(output_dir, "before.json", before)

    apply = observation(
        *request_json(
            f"{base_url}/fault_tolerance/apply",
            method="POST",
            payload={"action": "retry"},
            timeout=args.control_timeout,
        )
    )
    write_json(output_dir, "apply.json", apply)

    deadline = time.monotonic() + args.control_timeout
    status = None
    while time.monotonic() < deadline:
        status = observation(
            *request_json(
                f"{base_url}/fault_tolerance/status",
                method="GET",
                payload=None,
                timeout=args.control_timeout,
            )
        )
        body = status["body"]
        if (
            status["http_status"] == 200
            and isinstance(body, dict)
            and body.get("service_state") == "HEALTHY"
            and (body.get("last_transition") or {}).get("state") == "SUCCEEDED"
        ):
            break
        time.sleep(0.1)
    assert status is not None
    write_json(output_dir, "status.json", status)

    request_id = f"a3-post-retry-{uuid.uuid4().hex}"
    generated = observation(
        *request_json(
            f"{base_url}/generate",
            method="POST",
            payload={
                "text": "The capital city of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "rid": request_id,
            },
            timeout=args.generate_timeout,
        )
    )
    write_json(output_dir, "generated.json", generated)

    errors = validate(
        request_id=request_id,
        before=before,
        apply=apply,
        status=status,
        generated=generated,
    )
    apply_body = apply["body"]
    result = {
        "experiment": "A3-original-topology-retry",
        "passed": not errors,
        "errors": errors,
        "retry_command_id": (
            apply_body.get("command_id") if isinstance(apply_body, dict) else None
        ),
        "post_retry_request_id": request_id,
    }
    write_json(output_dir, "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
