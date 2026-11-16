"""CPU regression for the Shell experiment controller; never launches an engine.

All signals and HTTP calls in control-path tests are Shell functions with fixture
outputs. Runtime automation itself does not depend on this Python test file.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4]
BENCH = ROOT / "test/manual/ep"


@unittest.skipUnless(shutil.which("jq") and shutil.which("bash"), "requires jq/bash")
class TestNpuFtBench(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        (self.path / "clock").write_text("10.25 0\n")
        (self.path / "input.json").write_text("[1,2]")

    def shell(self, code, *, check=True):
        setup = """
set -euo pipefail
source "$SCRIPT_DIR/npu_ft_bench_lib.sh"
CASE_DIR="$FIXTURE"; INPUT_IDS="$FIXTURE/input.json"
FT_CLOCK_FILE="$FIXTURE/clock"
BASE_URL=http://127.0.0.1:31000
FAULT_RANK=1; CONTROL_TIMEOUT=2; STATUS_TIMEOUT=3; REQUEST_TIMEOUT=3
FAULT_AFTER_TOKENS=1
"""
        env = dict(os.environ, SCRIPT_DIR=str(BENCH), FIXTURE=str(self.path))
        result = subprocess.run(
            ["bash", "-c", setup + code],
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )
        if check and result.returncode != 0:
            self.fail(f"Shell exited {result.returncode}:\n{result.stderr}")
        return result

    def jq(self, expression, value):
        result = subprocess.run(
            ["jq", "-c", "-L", str(BENCH), 'include "npu_ft_bench_metrics"; ' + expression],
            input=json.dumps(value),
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        )
        return json.loads(result.stdout)

    @staticmethod
    def driver_functions():
        source = (BENCH / "run_npu_ft_bench.sh").read_text()
        functions = source[
            source.index("ft_stop_job() {") : source.index("\nfor round in")
        ]
        # Import function bodies without installing the driver's lifecycle traps.
        return (
            "\n".join(
                line for line in functions.splitlines() if not line.startswith("trap ")
            )
            + "\n"
        )

    @staticmethod
    def status(states):
        return {"ranks": [{"rank": r, "state": s} for r, s in enumerate(states)]}

    def test_status_requires_all_three_unhealthy_and_target_dead(self):
        scenarios = [
            (["unhealthy", "dead", "unhealthy", "unhealthy"], True),
            (["unhealthy", "dead", "healthy", "unhealthy"], False),
            (["unhealthy", "unhealthy", "unhealthy", "unhealthy"], False),
            (["healthy"] * 4, False),
        ]
        for states, expected in scenarios:
            with self.subTest(states=states):
                self.assertEqual(
                    self.jq('status_matches("incident";1)', self.status(states)),
                    expected,
                )

    def test_status_rejects_missing_duplicate_and_wrong_schema(self):
        fixtures = [
            {"engines": [{"id": 1, "status": "dead"}]},
            {"ranks": [{"rank": 1, "state": "dead"}]},
            {"ranks": [{"rank": 1, "state": "dead"}] * 4},
            self.status(["unknown"] * 4),
        ]
        for fixture in fixtures:
            with self.subTest(fixture=fixture):
                self.assertFalse(self.jq('status_matches("incident";1)', fixture))

    def test_wait_polls_past_partial_unhealthy(self):
        (self.path / "partial.json").write_text(
            json.dumps(self.status(["unhealthy", "dead", "healthy", "unhealthy"]))
        )
        (self.path / "ready.json").write_text(
            json.dumps(self.status(["unhealthy", "dead", "unhealthy", "unhealthy"]))
        )
        self.shell("""
ft_now() { echo 1; }
sleep() { :; }
count=0
ft_status() {
  count=$((count+1))
  if [ "$count" -eq 1 ]; then cp "$FIXTURE/partial.json" "$1";
  else cp "$FIXTURE/ready.json" "$1"; fi
}
ft_wait_status incident
[ "$count" -eq 2 ]
""")

    def test_control_does_not_post_before_status_gate(self):
        result = self.shell("""
ft_wait_status() { return 1; }
curl() { touch "$FIXTURE/posted"; }
ft_apply scale_down scale-down
""", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.path / "posted").exists())

    def test_status_timeout_with_one_healthy_rank_never_posts(self):
        (self.path / "partial.json").write_text(
            json.dumps(self.status(["unhealthy", "dead", "healthy", "unhealthy"]))
        )
        result = self.shell("""
echo 0 > "$FIXTURE/counter"
ft_now() { local n; n=$(cat "$FIXTURE/counter"); echo $((n+1)) > "$FIXTURE/counter"; echo "$n"; }
sleep() { :; }
ft_status() { cp "$FIXTURE/partial.json" "$1"; }
curl() { touch "$FIXTURE/posted"; }
ft_apply scale_down scale-down
""", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.path / "posted").exists())
        self.assertIn("timeout waiting for incident", result.stderr)

    def test_control_timeout_never_retries(self):
        result = self.shell("""
ft_wait_status() { :; }
curl() { echo call >> "$FIXTURE/posts"; return 28; }
ft_apply scale_down scale-down
""", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.path / "posts").read_text().splitlines(), ["call"])
        self.assertIn("NOT be resent", result.stderr)

    def test_http_202_is_not_treated_as_committed(self):
        result = self.shell("""
ft_wait_status() { :; }
curl() {
  while [ "$#" -gt 0 ]; do
    if [ "$1" = -o ]; then echo '{"success":true}' > "$2"; shift; fi
    shift
  done
  printf 202
}
ft_apply scale_down scale-down
""", check=False)
        self.assertNotEqual(result.returncode, 0)

    def test_successful_control_uses_current_branch_payload_once(self):
        (self.path / "ready.json").write_text(
            json.dumps(self.status(["unhealthy", "dead", "unhealthy", "unhealthy"]))
        )
        self.shell("""
ft_status() { cp "$FIXTURE/ready.json" "$1"; }
curl() {
  local output= payload=
  while [ "$#" -gt 0 ]; do
    case "$1" in -o) output=$2; shift ;; --data-binary) payload=${2#@}; shift ;; esac
    shift
  done
  jq -e '.instruction=="scale_down" and .params.ranks==[1] and .params.timeout==2' "$payload" >/dev/null
  echo call >> "$FIXTURE/posts"
  echo '{"success":true,"message":"fault_tolerance_scale_down_committed"}' > "$output"
  printf 200
}
ft_apply scale_down scale-down
""")
        self.assertEqual((self.path / "posts").read_text().splitlines(), ["call"])

    def test_descendant_selection_ignores_other_instance(self):
        result = self.shell("""
printf '%s\n' '301 300 S sglang::scheduler_DP1_TP1_EP1' \
 '201 200 S sglang::scheduler_DP1_TP1_EP1' '200 100 S controller' \
 '100 1 S server' '300 1 S other-server' | ft_descendants 100
""")
        self.assertIn("201 200", result.stdout)
        self.assertNotIn("301 300", result.stdout)

    def test_scheduler_discovery_requires_unique_owned_rank_and_exact_topology(self):
        rows = ["100 1 S server", "200 100 S controller"]
        rows += [f"{201+r} 200 S sglang::scheduler_DP{r}_TP{r}_EP{r}" for r in range(4)]
        rows += ["300 1 S other-server", "301 300 S sglang::scheduler_DP1_TP1_EP1"]
        for scenario in ("valid", "duplicate", "wrong-topology"):
            with self.subTest(scenario=scenario):
                fixture = list(rows)
                if scenario == "duplicate":
                    fixture.append("401 200 S sglang::scheduler_DP1_TP1_EP1")
                if scenario == "wrong-topology":
                    fixture[3] = "202 200 S sglang::scheduler_DP1_TP2_EP1"
                (self.path / "ps.txt").write_text("\n".join(fixture) + "\n")
                result = self.shell("""
SERVER_PID=100; SERVER_TICKS=111
ft_identity_matches() { :; }
ft_start_ticks() { echo 111; }
ps() { cat "$FIXTURE/ps.txt"; }
ft_schedulers "$FIXTURE/pids.json"
""", check=False)
                if scenario == "valid":
                    self.assertEqual(result.returncode, 0, result.stderr)
                    data = json.loads((self.path / "pids.json").read_text())
                    self.assertEqual([p["pid"] for p in data], [201, 202, 203, 204])
                else:
                    self.assertNotEqual(result.returncode, 0)

    def test_start_ticks_handles_spaces_and_parentheses_in_comm(self):
        proc = self.path / "proc/123"
        proc.mkdir(parents=True)
        proc.joinpath("stat").write_text(
            "123 (a (worker) name) S " + " ".join(["0"] * 18 + ["4321"]) + "\n"
        )
        result = self.shell('FT_PROC_ROOT="$FIXTURE/proc"; ft_start_ticks 123')
        self.assertEqual(result.stdout.strip(), "4321")

    def test_zombie_is_not_considered_live_identity(self):
        proc = self.path / "proc/123"
        proc.mkdir(parents=True)
        proc.joinpath("stat").write_text(
            "123 (worker) Z " + " ".join(["0"] * 18 + ["4321"]) + "\n"
        )
        result = self.shell(
            'FT_PROC_ROOT="$FIXTURE/proc"; ft_identity_matches 123 4321', check=False
        )
        self.assertNotEqual(result.returncode, 0)

    def test_no_kill_if_trigger_already_terminal_or_pid_changed(self):
        for scenario in ("terminal", "changed"):
            with self.subTest(scenario=scenario):
                work = self.path / scenario
                work.mkdir()
                result = self.shell("""
CASE_DIR="$FIXTURE/%s"
ft_request() { :; }
ft_track_job() { :; }
ft_wait_progress() { %s; }
ft_schedulers() {
  if [ "${1##*/}" = pids-at-kill.json ]; then id=%s; else id=201; fi
  jq -n --argjson id "$id" '[range(0;4)|{rank:.,pid:(if .==1 then $id else .+300 end)}]' > "$1"
}
ft_start_ticks() { echo 12345; }
ft_identity_matches() { :; }
kill() { touch "$FIXTURE/kill-called"; }
ft_inject_fault "$CASE_DIR/stream"
""" % (
                    scenario,
                    'mkdir -p "$1"; touch "$1/terminal"' if scenario == "terminal" else ":",
                    202 if scenario == "changed" else 201,
                ), check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.path / "kill-called").exists())

    def test_ended_stream_does_not_trigger_fault(self):
        stream = self.path / "ended"
        stream.mkdir()
        (stream / "terminal").touch()
        (stream / "progress").write_text("1536\n")
        result = self.shell('ft_wait_progress "$FIXTURE/ended" 1', check=False)
        self.assertNotEqual(result.returncode, 0)

    def test_fault_signal_follows_progress_and_identity_checks(self):
        self.shell("""
ft_request() { mkdir -p "$1"; echo 1 > "$1/progress"; }
ft_track_job() { :; }
ft_schedulers() { jq -n '[range(0;4)|{rank:.,pid:(200+.)}]' > "$1"; }
ft_start_ticks() { echo 12345; }
ft_identity_matches() { :; }
ft_wait_status() { echo "$1" > "$FIXTURE/waited"; }
kill() { printf '%s %s\n' "$1" "$2" > "$FIXTURE/signal"; }
ft_inject_fault "$FIXTURE/trigger"
""")
        self.assertEqual((self.path / "signal").read_text().strip(), "-KILL 201")
        self.assertEqual((self.path / "waited").read_text().strip(), "incident")

    def request_result(self, finish="length", count=4, **overrides):
        value = {
            "sent": 1, "ended": 4, "planned": 1, "expected": 4, "http": 200,
            "curl_exit": 0, "collector_exit": 0, "input_tokens": 2,
            "events": [
                {"at": 2, "count": 1, "finish": "-", "prompt": 2, "error": False},
                {"at": 3, "count": count, "finish": finish, "prompt": 2, "error": False},
            ],
        }
        value.update(overrides)
        return self.jq("summarize_request", value)

    def test_stream_terminal_classification(self):
        cases = [
            ({}, "normal"),
            ({"finish": "abort"}, "aborted"),
            ({"finish": "-"}, "missing_terminal"),
            ({"count": 2}, "wrong_length"),
            ({"http": 503}, "rejected"),
            ({"curl_exit": 28}, "timeout"),
            ({"collector_exit": 1}, "collector_error"),
            ({"count": 0}, "collector_error"),
        ]
        for kwargs, outcome in cases:
            with self.subTest(outcome=outcome):
                self.assertEqual(self.request_result(**kwargs)["outcome"], outcome)

    def test_chunked_tokens_and_tpot(self):
        result = self.request_result()
        self.assertEqual(result["increments"], [{"at": 2, "count": 1}, {"at": 3, "count": 3}])
        self.assertEqual(result["ttft"], 1)
        self.assertAlmostEqual(result["tpot"], 1 / 3)

    def test_window_excludes_failed_tokens_and_uses_send_cohort(self):
        ok = self.request_result()
        failed = self.request_result(finish="abort")
        result = self.jq("summarize_window(2;4;10;10;1)", [ok, failed])
        self.assertEqual(result["sent"], 0)  # Sent before the measurement window.
        self.assertEqual(result["output_tokens_per_second"], 2)  # Four normal tokens / 2 s.
        result = self.jq("summarize_window(1;4;10;10;1)", [ok, failed])
        self.assertEqual(result["success_rate"], 0.5)
        self.assertFalse(result["qualified"])

    def test_window_marks_client_lag_and_missing_results(self):
        directory = self.path / "load/requests/request-0"
        directory.mkdir(parents=True)
        result = self.request_result()
        result["send_lag"] = 1
        directory.joinpath("result.json").write_text(json.dumps(result))
        (self.path / "load/scheduled-count").write_text("1\n")
        functions = self.driver_functions()
        self.shell(functions + """
MIN_SAMPLES=1; MAX_SEND_LAG=0.25; TTFT_LIMIT=10; TPOT_LIMIT=10
window_summary "$FIXTURE/load" 1 4
""")
        summary = json.loads((self.path / "load/summary.json").read_text())
        self.assertFalse(summary["client_timing_valid"])
        self.assertFalse(summary["qualified"])
        (self.path / "load/scheduled-count").write_text("2\n")
        missing = self.shell(functions + """
MIN_SAMPLES=1; MAX_SEND_LAG=0.25; TTFT_LIMIT=10; TPOT_LIMIT=10
window_summary "$FIXTURE/load" 1 4
""", check=False)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("collector lost results", missing.stderr)

    def test_capacity_phase_aggregation(self):
        self.shell(self.driver_functions() + """
OUTPUT="$FIXTURE"; TTFT_LIMIT=10; TPOT_LIMIT=10
server_start() { :; }; server_stop() { :; }; ft_generation() { :; }
ft_inject_fault() { :; }; ft_apply() { :; }; ft_wait_status() { :; }
ft_finish_fault_stream() { :; }; idle_before_next_phase() { :; }; ft_recover() { :; }
capacity_phase() {
  local rate=2
  [ "$1" != three ] || rate=1
  mkdir -p "$CASE_DIR/$1/rate-$rate"
  jq -n --arg phase "$1" --argjson rate "$rate" \
    '{phase:$phase,rate:$rate,qualified:true}' > "$CASE_DIR/$1/rate-$rate/point.json"
}
capacity 1
""")
        summary = json.loads((self.path / "capacity-1/summary.json").read_text())
        self.assertEqual(summary["capacity_retention"], 0.5)
        self.assertEqual(summary["max_verified_request_rates"]["four_rejoined"], 2)

    def test_root_summary_retains_individual_rounds(self):
        source = (BENCH / "run_npu_ft_bench.sh").read_text()
        expression = source.rsplit("jq -s '", 1)[1].split("' \\", 1)[0]
        values = [
            {"arm": "restart", "recovery_seconds": 100},
            {"arm": "restart", "recovery_seconds": 120},
            {"arm": "shrink", "recovery_seconds": 10},
            {"capacity_retention": 0.75},
        ]
        result = self.jq(expression, values)
        # The actual driver uses jq -s; the array fixture above has the same shape.
        self.assertEqual(result["recovery_seconds"]["restart"]["median"], 110)
        self.assertEqual(result["recovery_seconds"]["shrink"]["n"], 1)
        self.assertEqual(len(result["cases"]), 4)

    def test_real_shell_collector_preserves_http_200_abort(self):
        for finish, expected in (("length", "normal"), ("abort", "aborted")):
            with self.subTest(finish=finish):
                self.shell("""
curl() {
  printf 'data: {"meta_info":{"completion_tokens":1,"prompt_tokens":2,"finish_reason":null}}\n\n'
  printf 'data: {"meta_info":{"completion_tokens":4,"prompt_tokens":2,"finish_reason":{"type":"%s"}}}\n\n'
  printf 'data: [DONE]\n\n\n__HTTP_STATUS__:200\n'
}
ft_request "$FIXTURE/%s" 4 1 10 10
""" % (finish, finish))
                result = json.loads((self.path / finish / "result.json").read_text())
                self.assertEqual(result["outcome"], expected)
                self.assertEqual(result["http"], 200)
                self.assertTrue((self.path / finish / "stream.sse").read_text())

    def test_real_shell_collector_rejects_malformed_stream_and_http_503(self):
        for scenario in ("malformed", "rejected"):
            with self.subTest(scenario=scenario):
                body = (
                    "data: {broken-json}"
                    if scenario == "malformed"
                    else '{"error":"paused"}'
                )
                http = 200 if scenario == "malformed" else 503
                self.shell("""
curl() { printf '%%s\n\n__HTTP_STATUS__:%s\n' '%s'; }
ft_request "$FIXTURE/%s" 4 1 10 10
""" % (http, body, scenario))
                result = json.loads((self.path / scenario / "result.json").read_text())
                self.assertNotEqual(result["outcome"], "normal")
                if scenario == "rejected":
                    self.assertEqual(result["outcome"], "rejected")


if __name__ == "__main__":
    unittest.main()
