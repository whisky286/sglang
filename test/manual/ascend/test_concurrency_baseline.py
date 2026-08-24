"""Baseline concurrent-inference test for Ascend manual validation.

This script intentionally does not inject faults and never calls the fault-tolerance
scale-down API. It is meant to isolate whether concurrent traffic alone is stable
before running the in-flight scale-down scenarios in test_fault_tolerance_suite.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import math
import statistics
import threading
import time
from dataclasses import dataclass
from typing import List

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("concurrency_baseline")


@dataclass
class RequestResult:
    worker_id: int
    request_id: int
    latency_sec: float
    status_code: int
    error: str = ""


def _generate_single(
    session: requests.Session,
    base_url: str,
    worker_id: int,
    request_id: int,
    *,
    max_new_tokens: int,
    timeout: float,
) -> RequestResult:
    start = time.monotonic()
    try:
        response = session.post(
            f"{base_url}/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": f"Say hello {worker_id}-{request_id}:",
                    }
                ],
                "max_tokens": max_new_tokens,
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        latency = time.monotonic() - start
        if response.status_code == 200:
            return RequestResult(worker_id, request_id, latency, 200)
        return RequestResult(
            worker_id,
            request_id,
            latency,
            response.status_code,
            error=response.text[:500],
        )
    except Exception as exc:
        return RequestResult(
            worker_id,
            request_id,
            time.monotonic() - start,
            0,
            error=str(exc),
        )


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def run_concurrency_baseline(
    base_url: str,
    *,
    concurrency: int,
    duration_secs: float,
    max_new_tokens: int,
    request_timeout: float,
) -> int:
    if concurrency <= 0:
        raise ValueError("--concurrency must be greater than 0")
    if duration_secs <= 0:
        raise ValueError("--duration-secs must be greater than 0")

    # Fail fast before starting concurrent traffic.
    with requests.Session() as session:
        warmup = _generate_single(
            session,
            base_url,
            worker_id=-1,
            request_id=0,
            max_new_tokens=max_new_tokens,
            timeout=request_timeout,
        )
    if warmup.status_code != 200:
        raise RuntimeError(
            f"Warmup request failed: status={warmup.status_code}, error={warmup.error}"
        )

    logger.info(
        "Starting concurrency-only baseline: concurrency=%d, duration=%.1fs",
        concurrency,
        duration_secs,
    )
    logger.info("No fault injection or scale-down will be performed.")

    stop_event = threading.Event()
    start_barrier = threading.Barrier(concurrency + 1)

    def worker_loop(worker_id: int) -> List[RequestResult]:
        local_results: List[RequestResult] = []
        request_id = 0
        with requests.Session() as session:
            start_barrier.wait()
            while not stop_event.is_set():
                local_results.append(
                    _generate_single(
                        session,
                        base_url,
                        worker_id,
                        request_id,
                        max_new_tokens=max_new_tokens,
                        timeout=request_timeout,
                    )
                )
                request_id += 1
        return local_results

    started_at = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker_loop, i) for i in range(concurrency)]
        start_barrier.wait()
        time.sleep(duration_secs)
        stop_event.set()
        results = [result for future in futures for result in future.result()]
    elapsed = time.monotonic() - started_at

    success_200 = sum(result.status_code == 200 for result in results)
    paused_503 = sum(result.status_code == 503 for result in results)
    transport_errors = sum(result.status_code == 0 for result in results)
    other_http_errors = len(results) - success_200 - paused_503 - transport_errors
    latencies = [result.latency_sec for result in results]

    print("\n" + "=" * 72)
    print("Concurrency-only baseline result")
    print(f"  concurrency:       {concurrency}")
    print(f"  requested duration:{duration_secs:.2f} s")
    print(f"  elapsed:           {elapsed:.2f} s")
    print(f"  total requests:    {len(results)}")
    print(f"  200 OK:            {success_200}")
    print(f"  503 paused:        {paused_503}")
    print(f"  other HTTP errors: {other_http_errors}")
    print(f"  transport errors:  {transport_errors}")
    if elapsed > 0:
        print(f"  throughput:        {len(results) / elapsed:.2f} req/s")
    if latencies:
        print(f"  latency avg:       {statistics.mean(latencies):.3f} s")
        print(f"  latency p50:       {_percentile(latencies, 0.50):.3f} s")
        print(f"  latency p95:       {_percentile(latencies, 0.95):.3f} s")
        print(f"  latency p99:       {_percentile(latencies, 0.99):.3f} s")
        print(f"  latency max:       {max(latencies):.3f} s")
    print("=" * 72)

    failures = [result for result in results if result.status_code != 200]
    if failures:
        print("\nFirst failures:")
        for result in failures[:10]:
            print(
                f"  worker={result.worker_id} request={result.request_id} "
                f"status={result.status_code} latency={result.latency_sec:.3f}s "
                f"error={result.error!r}"
            )
        return 1

    logger.info("Concurrency-only baseline PASSED: all requests returned HTTP 200.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test concurrent SGLang inference without fault injection or scale-down"
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:30000",
        help="SGLang server base URL",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent request workers",
    )
    parser.add_argument(
        "--duration-secs",
        type=float,
        default=20.0,
        help="How long to sustain concurrent traffic",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum generated tokens per request",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=10.0,
        help="Per-request timeout in seconds",
    )
    args = parser.parse_args()

    return run_concurrency_baseline(
        args.base_url.rstrip("/"),
        concurrency=args.concurrency,
        duration_secs=args.duration_secs,
        max_new_tokens=args.max_new_tokens,
        request_timeout=args.request_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
