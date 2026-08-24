"""Comprehensive Ascend MC2 Fault-Tolerance & Scale-Down Test Suite.

This script implements the complete fault-injection validation plan covering:
1. Idle scale-down (direct API scale-down vs. incident scale-down)
2. In-flight dynamic scale-down under concurrent inference load
3. FT strategy comparison (pause vs. continue)
4. Mixed fault injection (application exception + watchdog SIGKILL)
5. Tensor Parallelism TP > 1 fault tolerance & DP-unit isolation
6. Multi-victim & cascading sequential scale-down (4 -> 3 -> 2)
7. SIGKILL during one active inference request -> pause -> scale-down recovery
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ft_test_suite")


# Log patterns for Ascend NPU MC2 recovery verification
MC2_LOG_PATTERN = re.compile(
    r"\[NPU FT\].*MC2.*rank=(?P<rank>\d+).*"
    r"data_ptr=(?P<data_ptr>\d+).*values=\[(?P<values>[^]]*)\]"
)
PROCESS_GROUP_LOG_PATTERN = re.compile(
    r"\[NPU FT\] rebuilt graph-external process groups: "
    r"generation=(?P<generation>\d+) original_rank=(?P<rank>\d+) "
    r"compact_rank=(?P<compact_rank>\d+) "
    r"active_original_ranks=\[(?P<active_ranks>[^]]*)\]"
)
DEVICE_STOP_LOG_PATTERN = re.compile(
    r"\[NPU FT\] stopping survivor device before communication-domain "
    r"rebuild: rank=(?P<rank>\d+) device_id=(?P<device_id>\d+)"
)
DEVICE_RESTART_LOG_PATTERN = re.compile(
    r"\[NPU FT\] restarted survivor device without rebuilding graph "
    r"resources: rank=(?P<rank>\d+) device_id=(?P<device_id>\d+)"
)
SCHEDULER_PROCESS_TITLE_PATTERN = re.compile(
    r"(?:^|\s)sglang::scheduler_DP(?P<dp_rank>\d+)(?:_|\s|$)"
)


@dataclass
class ExperimentReport:
    test_case: str
    timestamp: str
    strategy: str = "unknown"
    victim_ranks: List[int] = field(default_factory=list)
    warmup_text: str = ""
    post_recovery_text: str = ""
    exact_match: bool = False
    scale_down_latency_sec: float = 0.0
    traffic_total: int = 0
    traffic_200_ok: int = 0
    traffic_503_paused: int = 0
    traffic_errors: int = 0
    device_stops_detected: int = 0
    device_restarts_detected: int = 0
    group_rebuilds_detected: int = 0
    verdict: str = "PENDING"
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def save_reports(self, output_dir: Path) -> Tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"report_{self.test_case}_{time_tag}.json"
        md_path = output_dir / f"report_{self.test_case}_{time_tag}.md"

        json_path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

        md_content = f"""# SGLang NPU FT 故障注入测试报告

- **测试用例**: `{self.test_case}`
- **执行时间**: `{self.timestamp}`
- **测试判定 (Verdict)**: **{self.verdict}**
- **FT 策略**: `{self.strategy}`
- **受害 Rank**: `{self.victim_ranks}`
- **缩容重构耗时**: `{self.scale_down_latency_sec:.2f} s`
- **输出文本一致性 (Exact Match)**: `{'✅ PASS' if self.exact_match else '❌ FAIL'}`

## 1. 流量统计 (In-Flight Traffic Stats)
| 总请求数 | 200 OK (成功) | 503 Paused (熔断) | 异常/超时错误 |
| :--- | :--- | :--- | :--- |
| `{self.traffic_total}` | `{self.traffic_200_ok}` | `{self.traffic_503_paused}` | `{self.traffic_errors}` |

## 2. NPU 驱动与通信域重建审计
- **设备停止调用 (stop_device)**: `{self.device_stops_detected}` 次
- **设备重启调用 (restart_device)**: `{self.device_restarts_detected}` 次
- **存活通信域重建 (rebuilt groups)**: `{self.group_rebuilds_detected}` 次

## 3. 生成内容对比
- **基线输出 (Warmup)**:
```text
{self.warmup_text}
```
- **缩容后输出 (Post-Recovery)**:
```text
{self.post_recovery_text}
```
"""
        if self.error_message:
            md_content += f"\n## ⚠️ 错误详情\n```text\n{self.error_message}\n```\n"

        md_path.write_text(md_content)
        return json_path, md_path


@dataclass
class RequestResult:
    request_id: int
    start_time: float
    end_time: float
    status_code: int
    text: str = ""
    error: str = ""


@dataclass
class InFlightTrafficStats:
    total_requests: int = 0
    success_200: int = 0
    paused_503: int = 0
    other_errors: int = 0
    results: List[RequestResult] = field(default_factory=list)


def _http_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: float = 10.0,
    payload: Optional[Dict[str, Any]] = None,
) -> Any:
    response = session.request(method, url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _get_ft_status(session: requests.Session, base_url: str) -> Dict[str, Any]:
    return _http_json(session, "GET", f"{base_url}/fault_tolerance/status", timeout=5.0)


def _rank_states(status: Dict[str, Any]) -> Dict[int, str]:
    return {int(item["rank"]): str(item["state"]) for item in status.get("ranks", [])}


def _find_scheduler_pids() -> Dict[int, int]:
    """Discover local DP rank -> PID mapping by searching /proc command lines."""
    rank_pids: Dict[int, int] = {}
    proc_root = Path("/proc")
    if not proc_root.exists():
        return rank_pids

    for pid_dir in proc_root.iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmdline = (
                (pid_dir / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
                .strip()
            )
            match = SCHEDULER_PROCESS_TITLE_PATTERN.search(cmdline)
            if match:
                dp_rank = int(match.group("dp_rank"))
                rank_pids[dp_rank] = int(pid_dir.name)
        except (OSError, PermissionError):
            continue
    return rank_pids


def _generate_single(
    session: requests.Session,
    base_url: str,
    prompt: str,
    *,
    req_id: int = 0,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    timeout: float = 30.0,
) -> RequestResult:
    start = time.monotonic()
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }
    try:
        resp = session.post(url, json=payload, timeout=timeout)
        end = time.monotonic()
        if resp.status_code == 200:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return RequestResult(req_id, start, end, 200, text=text)
        else:
            return RequestResult(
                req_id, start, end, resp.status_code, error=resp.text
            )
    except Exception as exc:
        end = time.monotonic()
        return RequestResult(req_id, start, end, 0, error=str(exc))


def _generate_streaming_request(
    base_url: str,
    prompt: str,
    first_chunk_event: threading.Event,
    *,
    req_id: int = 0,
    max_new_tokens: int = 512,
    timeout: float = 120.0,
) -> RequestResult:
    """Run one streaming request and signal after the first generated chunk arrives."""
    start = time.monotonic()
    text_parts: List[str] = []
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    try:
        with requests.Session() as stream_session:
            with stream_session.post(
                url, json=payload, stream=True, timeout=timeout
            ) as resp:
                if resp.status_code != 200:
                    return RequestResult(
                        req_id,
                        start,
                        time.monotonic(),
                        resp.status_code,
                        error=resp.text,
                    )

                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_text = line[5:].strip()
                    if data_text == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_text)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content") or ""
                    if content:
                        text_parts.append(content)
                        first_chunk_event.set()

                return RequestResult(
                    req_id,
                    start,
                    time.monotonic(),
                    200,
                    text="".join(text_parts),
                )
    except Exception as exc:
        return RequestResult(
            req_id,
            start,
            time.monotonic(),
            0,
            text="".join(text_parts),
            error=str(exc),
        )


def _wait_for_incident(
    session: requests.Session,
    base_url: str,
    victim_ranks: List[int],
    *,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_status: Optional[Dict[str, Any]] = None
    target_set = set(victim_ranks)
    while time.monotonic() < deadline:
        try:
            last_status = _get_ft_status(session, base_url)
            states = _rank_states(last_status)
            if all(states.get(r) in {"dead", "unhealthy"} for r in target_set):
                return last_status
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(
        f"Ranks {victim_ranks} did not reach expected incident state in {timeout}s; last={last_status}"
    )


def _trigger_scale_down(
    session: requests.Session,
    base_url: str,
    removed_ranks: List[int],
    *,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    logger.info(f"Triggering scale-down for ranks {removed_ranks}...")
    url = f"{base_url}/fault_tolerance/apply"
    resp = _http_json(
        session,
        "POST",
        url,
        payload={
            "instruction": "scale_down",
            "params": {"ranks": removed_ranks, "timeout": int(timeout)},
        },
        timeout=timeout,
    )
    logger.info(f"Scale-down response: {resp}")
    return resp


def _verify_server_log_rebuild(
    log_path: Path, expected_generation: int
) -> Tuple[int, int, int]:
    if not log_path.exists():
        logger.warning(f"Log file {log_path} not found; skipping log assertions.")
        return 0, 0, 0

    text = log_path.read_text(errors="replace")
    stop_matches = DEVICE_STOP_LOG_PATTERN.findall(text)
    restart_matches = DEVICE_RESTART_LOG_PATTERN.findall(text)
    rebuild_matches = [
        m.groupdict()
        for m in PROCESS_GROUP_LOG_PATTERN.finditer(text)
        if int(m.group("generation")) == expected_generation
    ]

    logger.info(
        f"Log audit [Gen {expected_generation}]: stops={len(stop_matches)}, "
        f"restarts={len(restart_matches)}, group_rebuilds={len(rebuild_matches)}"
    )
    if not rebuild_matches:
        logger.warning(
            f"No process group rebuild log found for generation {expected_generation}!"
        )
    return len(stop_matches), len(restart_matches), len(rebuild_matches)


# ==============================================================================
# Scenario 1: Idle Scale-Down (Direct API vs. Incident Scale-Down)
# ==============================================================================
def run_exp1_idle_scale_down(
    session: requests.Session,
    base_url: str,
    victim_rank: int,
    log_path: Path,
    *,
    direct_api: bool = False,
) -> ExperimentReport:
    logger.info("=== [EXP-1] Starting Idle Scale-Down Test ===")
    report = ExperimentReport(
        test_case="idle_scale_down",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        victim_ranks=[victim_rank],
    )
    status = _get_ft_status(session, base_url)
    report.strategy = str(status.get("strategy", "unknown"))

    # 1. Warmup Baseline
    warmup = _generate_single(
        session, base_url, "Count from 1 to 5:", max_new_tokens=16
    )
    assert warmup.status_code == 200, f"Warmup failed: {warmup}"
    report.warmup_text = warmup.text
    logger.info(f"Warmup baseline output: {warmup.text!r}")

    t_start_scale = time.monotonic()
    if direct_api:
        logger.info("Directly triggering scale_down without killing process...")
        _trigger_scale_down(session, base_url, [victim_rank])
    else:
        # Discover PID and kill
        pids = _find_scheduler_pids()
        victim_pid = pids.get(victim_rank)
        assert victim_pid is not None, f"Could not find PID for DP rank {victim_rank}"
        logger.info(f"Killing victim DP rank {victim_rank} (PID {victim_pid})...")
        os.kill(victim_pid, signal.SIGKILL)
        _wait_for_incident(session, base_url, [victim_rank])
        _trigger_scale_down(session, base_url, [victim_rank])
    report.scale_down_latency_sec = time.monotonic() - t_start_scale

    time.sleep(2.0)
    # 2. Verify Post-Scale-Down Inference
    post_req = _generate_single(
        session, base_url, "Count from 1 to 5:", max_new_tokens=16
    )
    assert (
        post_req.status_code == 200
    ), f"Post-scale-down request failed: {post_req.error}"
    report.post_recovery_text = post_req.text
    report.exact_match = (post_req.text == warmup.text)
    logger.info(f"Post-scale-down output: {post_req.text!r}")
    assert (
        report.exact_match
    ), f"Output mismatch: expected {warmup.text!r}, got {post_req.text!r}"

    stops, restarts, rebuilds = _verify_server_log_rebuild(
        log_path, expected_generation=1
    )
    report.device_stops_detected = stops
    report.device_restarts_detected = restarts
    report.group_rebuilds_detected = rebuilds
    report.verdict = "PASS"
    logger.info("=== [EXP-1] Idle Scale-Down Test PASSED ===")
    return report


# ==============================================================================
# Scenario 2: In-Flight Dynamic Scale-Down under Concurrent Load
# ==============================================================================
def run_exp2_inflight_scale_down(
    session: requests.Session,
    base_url: str,
    victim_rank: int,
    log_path: Path,
    *,
    concurrency: int = 10,
    duration_secs: float = 20.0,
) -> ExperimentReport:
    logger.info("=== [EXP-2] Starting In-Flight Dynamic Scale-Down Test ===")
    report = ExperimentReport(
        test_case="inflight_scale_down",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        victim_ranks=[victim_rank],
    )
    status = _get_ft_status(session, base_url)
    report.strategy = str(status.get("strategy", "unknown"))

    warmup = _generate_single(
        session, base_url, "Baseline check before traffic:", max_new_tokens=16
    )
    report.warmup_text = warmup.text

    stats = InFlightTrafficStats()
    stop_event = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency + 2)

    def worker_loop(worker_id: int):
        req_counter = 0
        while not stop_flag:
            req_id = worker_id * 1000 + req_counter
            req_counter += 1
            res = _generate_single(
                session,
                base_url,
                f"Say hello {req_id}:",
                req_id=req_id,
                max_new_tokens=16,
                timeout=10.0,
            )
            stats.results.append(res)
            stats.total_requests += 1
            if res.status_code == 200:
                stats.success_200 += 1
            elif res.status_code == 503:
                stats.paused_503 += 1
            else:
                stats.other_errors += 1
            time.sleep(0.1)

    stop_flag = False
    futures = [stop_event.submit(worker_loop, i) for i in range(concurrency)]

    logger.info("Traffic started. Waiting 5s before injecting fault...")
    time.sleep(5.0)

    pids = _find_scheduler_pids()
    victim_pid = pids.get(victim_rank)
    assert victim_pid is not None, f"Could not find PID for DP rank {victim_rank}"
    logger.info(
        f"Injecting SIGKILL to victim DP rank {victim_rank} (PID {victim_pid})..."
    )
    os.kill(victim_pid, signal.SIGKILL)

    _wait_for_incident(session, base_url, [victim_rank])
    logger.info("Incident detected by watchdog. Waiting 3s then scaling down...")
    time.sleep(3.0)

    t_scale_start = time.monotonic()
    _trigger_scale_down(session, base_url, [victim_rank])
    report.scale_down_latency_sec = time.monotonic() - t_scale_start
    logger.info("Scale down completed. Letting traffic run for 5 more seconds...")
    time.sleep(5.0)

    stop_flag = True
    stop_event.shutdown(wait=True)

    report.traffic_total = stats.total_requests
    report.traffic_200_ok = stats.success_200
    report.traffic_503_paused = stats.paused_503
    report.traffic_errors = stats.other_errors

    logger.info(
        f"In-Flight Traffic Stats: Total={stats.total_requests}, "
        f"200_OK={stats.success_200}, 503_PAUSED={stats.paused_503}, Errors={stats.other_errors}"
    )
    # Post recovery clean check
    post_check = _generate_single(
        session, base_url, "Baseline check before traffic:", max_new_tokens=16
    )
    assert post_check.status_code == 200, f"Final check failed: {post_check}"
    report.post_recovery_text = post_check.text
    report.exact_match = (post_check.text == warmup.text)

    stops, restarts, rebuilds = _verify_server_log_rebuild(
        log_path, expected_generation=1
    )
    report.device_stops_detected = stops
    report.device_restarts_detected = restarts
    report.group_rebuilds_detected = rebuilds
    report.verdict = "PASS"
    logger.info("=== [EXP-2] In-Flight Dynamic Scale-Down Test PASSED ===")
    return report


# ==============================================================================
# Scenario 3: Strategy Comparison (Continue vs. Pause)
# ==============================================================================
def run_exp3_continue_isolation(
    session: requests.Session,
    base_url: str,
    victim_rank: int,
    log_path: Path,
) -> ExperimentReport:
    logger.info("=== [EXP-3] Starting Continue Strategy Isolation Test ===")
    report = ExperimentReport(
        test_case="strategy_continue_isolation",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        victim_ranks=[victim_rank],
    )
    status = _get_ft_status(session, base_url)
    report.strategy = str(status.get("strategy", "unknown"))
    assert (
        report.strategy == "continue"
    ), f"Server must be launched with --fault-tolerance-on-error-strategy continue, got {status}"

    warmup = _generate_single(
        session, base_url, "Warmup check for continue:", max_new_tokens=16
    )
    report.warmup_text = warmup.text

    pids = _find_scheduler_pids()
    victim_pid = pids.get(victim_rank)
    assert victim_pid is not None, f"Could not find PID for DP rank {victim_rank}"

    logger.info(
        f"Killing victim rank {victim_rank} under continue strategy (PID {victim_pid})..."
    )
    os.kill(victim_pid, signal.SIGKILL)
    _wait_for_incident(session, base_url, [victim_rank])

    logger.info("Verifying that non-faulty DP ranks continue serving without 503...")
    success_count = 0
    total_probes = 10
    for i in range(total_probes):
        res = _generate_single(
            session, base_url, f"Prompt {i}", max_new_tokens=8, timeout=5.0
        )
        if res.status_code == 200:
            success_count += 1
    report.traffic_total = total_probes
    report.traffic_200_ok = success_count
    logger.info(f"Received {success_count}/{total_probes} successful responses during incident.")
    assert success_count > 0, "No requests succeeded during continue incident state!"

    t_scale_start = time.monotonic()
    _trigger_scale_down(session, base_url, [victim_rank])
    report.scale_down_latency_sec = time.monotonic() - t_scale_start

    post_check = _generate_single(
        session, base_url, "Warmup check for continue:", max_new_tokens=16
    )
    report.post_recovery_text = post_check.text
    report.exact_match = (post_check.text == warmup.text)

    stops, restarts, rebuilds = _verify_server_log_rebuild(
        log_path, expected_generation=1
    )
    report.device_stops_detected = stops
    report.device_restarts_detected = restarts
    report.group_rebuilds_detected = rebuilds
    report.verdict = "PASS"
    logger.info("=== [EXP-3] Continue Strategy Isolation Test PASSED ===")
    return report


# ==============================================================================
# Scenario 4: Mixed Fault Injection (Application Exception + SIGKILL)
# ==============================================================================
def run_exp4_mixed_fault_injection(
    session: requests.Session,
    base_url: str,
    soft_victim_rank: int,
    hard_victim_rank: int,
    log_path: Path,
) -> ExperimentReport:
    logger.info("=== [EXP-4] Starting Mixed Fault Injection Test ===")
    report = ExperimentReport(
        test_case="mixed_fault_injection",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        victim_ranks=[soft_victim_rank, hard_victim_rank],
    )
    status = _get_ft_status(session, base_url)
    report.strategy = str(status.get("strategy", "unknown"))

    warmup = _generate_single(
        session, base_url, "Mixed fault baseline check:", max_new_tokens=16
    )
    report.warmup_text = warmup.text

    # 1. Soft fault injection via API (inject a scheduler exception -> unhealthy)
    logger.info(
        f"Injecting soft exception to rank {soft_victim_rank} via API endpoint..."
    )
    inject_resp = _http_json(
        session,
        "POST",
        f"{base_url}/fault_tolerance/apply",
        payload={
            "instruction": "inject_fault",
            "params": {"ranks": [soft_victim_rank]},
        },
        timeout=5.0,
    )
    logger.info(f"Inject fault response: {inject_resp}")
    assert inject_resp.get("success"), f"inject_fault did not succeed: {inject_resp}"

    # 2. Hard fault injection via SIGKILL
    pids = _find_scheduler_pids()
    hard_pid = pids.get(hard_victim_rank)
    assert (
        hard_pid is not None
    ), f"Could not find PID for hard victim DP rank {hard_victim_rank}"
    logger.info(
        f"Injecting hard SIGKILL to rank {hard_victim_rank} (PID {hard_pid})..."
    )
    os.kill(hard_pid, signal.SIGKILL)

    # 3. Wait for both to be captured in status
    status = _wait_for_incident(
        session, base_url, [soft_victim_rank, hard_victim_rank]
    )
    states = _rank_states(status)
    logger.info(f"Mixed incident states: {states}")
    assert states.get(soft_victim_rank) in {"unhealthy", "dead"}
    assert states.get(hard_victim_rank) == "dead"

    # 4. Scale down both victims simultaneously
    t_scale_start = time.monotonic()
    _trigger_scale_down(session, base_url, [soft_victim_rank, hard_victim_rank])
    report.scale_down_latency_sec = time.monotonic() - t_scale_start

    # 5. Verify 2-rank survivor cluster
    res = _generate_single(
        session, base_url, "Mixed fault baseline check:", max_new_tokens=16
    )
    assert res.status_code == 200, f"Post mixed scale-down failed: {res}"
    report.post_recovery_text = res.text
    report.exact_match = (res.text == warmup.text)

    stops, restarts, rebuilds = _verify_server_log_rebuild(
        log_path, expected_generation=1
    )
    report.device_stops_detected = stops
    report.device_restarts_detected = restarts
    report.group_rebuilds_detected = rebuilds
    report.verdict = "PASS"
    logger.info("=== [EXP-4] Mixed Fault Injection Test PASSED ===")
    return report


# ==============================================================================
# Scenario 5: Tensor Parallelism TP > 1 (e.g. TP=2, DP=2)
# ==============================================================================
def run_exp5_tp_parallel_scale_down(
    session: requests.Session,
    base_url: str,
    victim_dp_rank: int,
    log_path: Path,
) -> ExperimentReport:
    logger.info(
        f"=== [EXP-5] Starting TP > 1 Parallel Scale-Down Test (Victim DP {victim_dp_rank}) ==="
    )
    report = ExperimentReport(
        test_case="tp_parallel_scale_down",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        victim_ranks=[victim_dp_rank],
    )
    status = _get_ft_status(session, base_url)
    report.strategy = str(status.get("strategy", "unknown"))

    warmup = _generate_single(session, base_url, "TP test warmup:", max_new_tokens=16)
    assert warmup.status_code == 200, f"TP warmup failed: {warmup}"
    report.warmup_text = warmup.text

    pids = _find_scheduler_pids()
    victim_pid = pids.get(victim_dp_rank)
    assert (
        victim_pid is not None
    ), f"Could not find PID for DP rank {victim_dp_rank} across TP workers"

    logger.info(
        f"Killing TP worker in DP rank {victim_dp_rank} (PID {victim_pid})..."
    )
    os.kill(victim_pid, signal.SIGKILL)

    _wait_for_incident(session, base_url, [victim_dp_rank])
    t_scale_start = time.monotonic()
    _trigger_scale_down(session, base_url, [victim_dp_rank])
    report.scale_down_latency_sec = time.monotonic() - t_scale_start

    post_req = _generate_single(
        session, base_url, "TP test warmup:", max_new_tokens=16
    )
    assert (
        post_req.status_code == 200
    ), f"Post TP scale-down request failed: {post_req}"
    report.post_recovery_text = post_req.text
    report.exact_match = (post_req.text == warmup.text)

    stops, restarts, rebuilds = _verify_server_log_rebuild(
        log_path, expected_generation=1
    )
    report.device_stops_detected = stops
    report.device_restarts_detected = restarts
    report.group_rebuilds_detected = rebuilds
    report.verdict = "PASS"
    logger.info("=== [EXP-5] TP > 1 Parallel Scale-Down Test PASSED ===")
    return report


# ==============================================================================
# Scenario 6: Cascading Sequential Scale-Down (4 -> 3 -> 2)
# ==============================================================================
def run_exp6_cascading_scale_down(
    session: requests.Session,
    base_url: str,
    victim_ranks: List[int],
    log_path: Path,
) -> ExperimentReport:
    logger.info(
        f"=== [EXP-6] Starting Cascading Scale-Down Test (Steps: {victim_ranks}) ==="
    )
    report = ExperimentReport(
        test_case="cascading_scale_down",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        victim_ranks=victim_ranks,
    )
    status = _get_ft_status(session, base_url)
    report.strategy = str(status.get("strategy", "unknown"))

    warmup = _generate_single(
        session, base_url, "Cascading test baseline:", max_new_tokens=16
    )
    assert warmup.status_code == 200
    report.warmup_text = warmup.text

    t_scale_total = 0.0
    for step, victim_rank in enumerate(victim_ranks, start=1):
        logger.info(
            f"--- Cascading Step {step}: Killing rank {victim_rank} ---"
        )
        pids = _find_scheduler_pids()
        victim_pid = pids.get(victim_rank)
        assert (
            victim_pid is not None
        ), f"Step {step}: PID for DP rank {victim_rank} not found"

        os.kill(victim_pid, signal.SIGKILL)
        _wait_for_incident(session, base_url, [victim_rank])
        t_step_start = time.monotonic()
        _trigger_scale_down(session, base_url, [victim_rank])
        t_scale_total += (time.monotonic() - t_step_start)

        # Verify generation after each step
        res = _generate_single(
            session,
            base_url,
            "Cascading test baseline:",
            max_new_tokens=16,
        )
        assert (
            res.status_code == 200
        ), f"Cascading step {step} generation failed: {res}"
        report.post_recovery_text = res.text
        report.exact_match = (res.text == warmup.text)
        _verify_server_log_rebuild(log_path, expected_generation=step)
        logger.info(f"--- Cascading Step {step} Complete (Generation {step}) ---")

    report.scale_down_latency_sec = t_scale_total
    stops, restarts, rebuilds = _verify_server_log_rebuild(
        log_path, expected_generation=len(victim_ranks)
    )
    report.device_stops_detected = stops
    report.device_restarts_detected = restarts
    report.group_rebuilds_detected = rebuilds
    report.verdict = "PASS"
    logger.info("=== [EXP-6] Cascading Scale-Down Test PASSED ===")
    return report


# ==============================================================================
# Scenario 7: Kill a Rank During One Active Inference -> Pause -> Scale-Down
# ==============================================================================
def run_exp7_inflight_request_pause_scale_down(
    session: requests.Session,
    base_url: str,
    victim_rank: int,
    log_path: Path,
    *,
    max_new_tokens: int = 512,
    stream_start_timeout: float = 30.0,
    inflight_completion_timeout: float = 15.0,
) -> ExperimentReport:
    logger.info("=== [EXP-7] Starting In-Flight Request Pause + Scale-Down Test ===")
    report = ExperimentReport(
        test_case="inflight_request_pause_scale_down",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        victim_ranks=[victim_rank],
    )

    status = _get_ft_status(session, base_url)
    report.strategy = str(status.get("strategy", "unknown"))
    assert report.strategy == "pause", (
        "Server must be launched with --fault-tolerance-on-error-strategy pause, "
        f"got {status}"
    )

    warmup_prompt = "Reply with exactly FT_OK and nothing else."
    warmup = _generate_single(
        session, base_url, warmup_prompt, max_new_tokens=8, timeout=30.0
    )
    assert warmup.status_code == 200, f"Warmup failed: {warmup}"
    report.warmup_text = warmup.text

    pids = _find_scheduler_pids()
    victim_pid = pids.get(victim_rank)
    assert victim_pid is not None, f"Could not find PID for DP rank {victim_rank}"

    first_chunk_event = threading.Event()
    inflight_result: Dict[str, RequestResult] = {}

    def run_inflight_request() -> None:
        inflight_result["result"] = _generate_streaming_request(
            base_url,
            (
                "Write a detailed numbered explanation of how distributed inference "
                "works. Keep generating until you reach the token limit."
            ),
            first_chunk_event,
            req_id=7000,
            max_new_tokens=max_new_tokens,
            timeout=max(120.0, inflight_completion_timeout + 30.0),
        )

    request_thread = threading.Thread(
        target=run_inflight_request,
        name="ft-inflight-request",
        daemon=True,
    )
    request_thread.start()

    wait_started = time.monotonic()
    stream_started = first_chunk_event.wait(timeout=stream_start_timeout)
    first_chunk_wait_sec = time.monotonic() - wait_started
    if not stream_started:
        early_result = inflight_result.get("result")
        raise TimeoutError(
            "The streaming request did not produce a first chunk before the timeout; "
            f"result={early_result}"
        )
    assert request_thread.is_alive(), (
        "Streaming request completed before fault injection; increase "
        "--inflight-max-new-tokens"
    )

    logger.info(
        "Streaming request is in flight (first chunk received after %.2fs). "
        "Killing DP rank %d (PID %d)...",
        first_chunk_wait_sec,
        victim_rank,
        victim_pid,
    )
    kill_time = time.monotonic()
    os.kill(victim_pid, signal.SIGKILL)

    incident_status = _wait_for_incident(session, base_url, [victim_rank])
    incident_detect_latency = time.monotonic() - kill_time
    logger.info(
        "Incident detected after %.2fs. Probing admission to verify pause...",
        incident_detect_latency,
    )

    pause_probe = _generate_single(
        session,
        base_url,
        "Pause admission probe",
        req_id=7001,
        max_new_tokens=1,
        timeout=5.0,
    )
    report.traffic_total = 1
    if pause_probe.status_code == 503:
        report.traffic_503_paused = 1
    elif pause_probe.status_code == 200:
        report.traffic_200_ok = 1
    else:
        report.traffic_errors = 1
    assert pause_probe.status_code == 503, (
        "Pause strategy did not reject a new request with HTTP 503 after the rank "
        f"incident: status={pause_probe.status_code}, error={pause_probe.error!r}"
    )

    logger.info("Pause confirmed by HTTP 503. Triggering scale down...")
    t_scale_start = time.monotonic()
    scale_response = _trigger_scale_down(session, base_url, [victim_rank])
    report.scale_down_latency_sec = time.monotonic() - t_scale_start

    request_thread.join(timeout=inflight_completion_timeout)
    request_still_running = request_thread.is_alive()
    original_result = inflight_result.get("result")

    post_check = _generate_single(
        session, base_url, warmup_prompt, max_new_tokens=8, timeout=30.0
    )
    assert post_check.status_code == 200, f"Post-scale-down request failed: {post_check}"
    report.post_recovery_text = post_check.text
    report.exact_match = post_check.text == warmup.text

    report.details.update(
        {
            "victim_pid": victim_pid,
            "stream_started_before_kill": True,
            "first_chunk_wait_sec": first_chunk_wait_sec,
            "incident_detect_latency_sec": incident_detect_latency,
            "incident_status": incident_status,
            "pause_probe_status": pause_probe.status_code,
            "pause_probe_error": pause_probe.error,
            "scale_down_response": scale_response,
            "inflight_request_completed_after_scale_down": not request_still_running,
            "inflight_request_status": (
                original_result.status_code if original_result is not None else None
            ),
            "inflight_request_error": (
                original_result.error if original_result is not None else "still running"
            ),
            "inflight_partial_text": (
                original_result.text if original_result is not None else ""
            ),
        }
    )

    stops, restarts, rebuilds = _verify_server_log_rebuild(
        log_path, expected_generation=1
    )
    report.device_stops_detected = stops
    report.device_restarts_detected = restarts
    report.group_rebuilds_detected = rebuilds
    report.verdict = "PASS"
    logger.info(
        "=== [EXP-7] In-Flight Request Pause + Scale-Down Test PASSED "
        "(in-flight completed=%s) ===",
        not request_still_running,
    )
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Ascend MC2 Fault-Tolerance Test Suite"
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:30000",
        help="SGLang server base URL",
    )
    parser.add_argument(
        "--test-case",
        choices=[
            "idle_scale_down",
            "inflight_scale_down",
            "strategy_continue_isolation",
            "mixed_fault_injection",
            "tp_parallel_scale_down",
            "cascading_scale_down",
            "inflight_request_pause_scale_down",
        ],
        required=True,
        help="Test case scenario to execute",
    )
    parser.add_argument(
        "--victim-rank",
        type=int,
        default=3,
        help="Victim DP rank to kill/scale down",
    )
    parser.add_argument(
        "--soft-victim-rank",
        type=int,
        default=1,
        help="Victim DP rank for soft exception fault",
    )
    parser.add_argument(
        "--hard-victim-rank",
        type=int,
        default=2,
        help="Victim DP rank for hard SIGKILL fault",
    )
    parser.add_argument(
        "--cascading-ranks",
        type=int,
        nargs="+",
        default=[3, 2],
        help="Ordered list of victim ranks for cascading scale down",
    )
    parser.add_argument(
        "--direct-api",
        action="store_true",
        help="Directly invoke scale_down API without killing process (for EXP-1)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Concurrency for in-flight traffic test",
    )
    parser.add_argument(
        "--inflight-max-new-tokens",
        type=int,
        default=512,
        help="Token budget for the single streaming request in EXP-7",
    )
    parser.add_argument(
        "--stream-start-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for the first streaming chunk before EXP-7 kills a rank",
    )
    parser.add_argument(
        "--inflight-completion-timeout",
        type=float,
        default=15.0,
        help="Seconds to observe whether the original EXP-7 request completes after scale-down",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("/tmp/sglang-npu-ft.log"),
        help="Path to SGLang server log file for audit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./test_reports"),
        help="Directory to automatically save structured test reports (JSON & Markdown)",
    )

    args = parser.parse_args()
    session = requests.Session()
    report: Optional[ExperimentReport] = None

    try:
        if args.test_case == "idle_scale_down":
            report = run_exp1_idle_scale_down(
                session,
                args.base_url,
                args.victim_rank,
                args.log_path,
                direct_api=args.direct_api,
            )
        elif args.test_case == "inflight_scale_down":
            report = run_exp2_inflight_scale_down(
                session,
                args.base_url,
                args.victim_rank,
                args.log_path,
                concurrency=args.concurrency,
            )
        elif args.test_case == "strategy_continue_isolation":
            report = run_exp3_continue_isolation(
                session, args.base_url, args.victim_rank, args.log_path
            )
        elif args.test_case == "mixed_fault_injection":
            report = run_exp4_mixed_fault_injection(
                session,
                args.base_url,
                args.soft_victim_rank,
                args.hard_victim_rank,
                args.log_path,
            )
        elif args.test_case == "tp_parallel_scale_down":
            report = run_exp5_tp_parallel_scale_down(
                session, args.base_url, args.victim_rank, args.log_path
            )
        elif args.test_case == "cascading_scale_down":
            report = run_exp6_cascading_scale_down(
                session, args.base_url, args.cascading_ranks, args.log_path
            )
        elif args.test_case == "inflight_request_pause_scale_down":
            report = run_exp7_inflight_request_pause_scale_down(
                session,
                args.base_url,
                args.victim_rank,
                args.log_path,
                max_new_tokens=args.inflight_max_new_tokens,
                stream_start_timeout=args.stream_start_timeout,
                inflight_completion_timeout=args.inflight_completion_timeout,
            )
    except Exception as exc:
        logger.error(f"Test case {args.test_case} failed with exception: {exc}", exc_info=True)
        if report is None:
            report = ExperimentReport(
                test_case=args.test_case,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        report.verdict = "FAIL"
        report.error_message = str(exc)
        raise
    finally:
        if report is not None:
            json_p, md_p = report.save_reports(args.output_dir)
            print("\n" + "=" * 60)
            print(f"📊 [Test Report Saved]")
            print(f"  • JSON Report:     {json_p.resolve()}")
            print(f"  • Markdown Report: {md_p.resolve()}")
            print(f"  • Final Verdict:   {report.verdict}")
            print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
