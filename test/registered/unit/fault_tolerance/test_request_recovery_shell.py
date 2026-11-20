"""Exercise the Shell acceptance checks without running or killing a server."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4]
LIB = ROOT / "test/manual/ep/ft_request_recovery_lib.sh"
SCRIPT = ROOT / "test/manual/ep/run_ft_request_recovery.sh"


class TestRequestRecoveryShell(CustomTestCase):
    def shell(self, code, *args):
        return subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; shift; ' + code,
                "test",
                str(LIB),
                *map(str, args),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    @staticmethod
    def chunk(ids, count, *, finish=None, attempt=0, rid="public", text=""):
        return {
            "output_ids": ids,
            "text": text,
            "meta_info": {
                "id": rid,
                "completion_tokens": count,
                "recovery_attempts": attempt,
                "finish_reason": finish,
            },
        }

    @unittest.skipUnless(
        shutil.which("jq"), "jq is required for Shell acceptance checks"
    )
    def test_stream_gate_rejects_duplicate_missing_and_foreign_output(self):
        first = self.chunk([10, 11], 2, text="AB")
        last = self.chunk([12], 3, finish={"type": "length"}, attempt=1, text="C")
        cases = [
            ([first, last], True),
            ([{**first, "text": ""}, last], False),
            (
                [first, self.chunk([11, 12], 3, finish={"type": "length"}, attempt=1)],
                False,
            ),
            (
                [
                    self.chunk([10], 1),
                    self.chunk([12], 2, finish={"type": "length"}, attempt=1),
                ],
                False,
            ),
            (
                [first, {**last, "meta_info": {**last["meta_info"], "id": "wrong"}}],
                False,
            ),
            ([first, last, last], False),
            ([first, self.chunk([12], 3, finish={"type": "length"}, attempt=0)], False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            prefix = path / "prefix.json"
            prefix.write_text(json.dumps([first]))
            data = path / "data.json"
            for chunks, valid in cases:
                with self.subTest(chunks=chunks):
                    data.write_text(json.dumps(chunks))
                    result = self.shell(
                        'ft_check_stream "$1" "$2" public 3 1', data, prefix
                    )
                    self.assertEqual(result.returncode == 0, valid, result.stderr)

    @unittest.skipUnless(
        shutil.which("jq"), "jq is required for Shell acceptance checks"
    )
    def test_sse_reader_ignores_done_and_incomplete_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stream.sse"
            first = self.chunk([10], 1)
            path.write_text(
                "data: " + json.dumps(first) + '\n\ndata: [DONE]\n\ndata: {"partial":'
            )
            result = self.shell('ft_sse_json "$1"', path)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), [first])
            result = self.shell('ft_sse_json "$1"', path.parent / "not-created-yet")
            self.assertEqual(json.loads(result.stdout), [])

    def test_signal_gate_never_signals_invalid_or_unowned_pid(self):
        code = """
        ft_same_process() { [ "$1" = 123 ] && [ "$2" = 99 ]; }
        ft_descends_from() { [ "$1" = 123 ] && [ "$2" = 100 ]; }
        ps() { printf '%s\n' sglang::scheduler_DP1_TP1_EP1; }
        kill() { printf 'SIGNAL %s\n' "$*"; }
        ft_signal_rank "$1" "$2" "$3" "$4"
        """
        for args in (
            ("0", "99", "1", "100"),
            ("-1", "99", "1", "100"),
            ("abc", "99", "1", "100"),
            ("123", "98", "1", "100"),
            ("123", "99", "1", "101"),
            ("123", "99", "2", "100"),
        ):
            with self.subTest(args=args):
                result = self.shell(code, *args)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("SIGNAL", result.stdout)
        result = self.shell(code, "123", "99", "1", "100")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SIGNAL -KILL 123", result.stdout)

    def test_help_and_bash_syntax_need_no_accelerator(self):
        for path in (LIB, SCRIPT):
            result = subprocess.run(
                ["bash", "-n", str(path)], check=False, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            ["bash", str(SCRIPT), "--help"], check=False, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("NPU uses MC2, CUDA uses Mooncake", result.stdout)


if __name__ == "__main__":
    unittest.main()
