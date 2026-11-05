import ast
import asyncio
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.fault_tolerance.request_journal import (
    EXECUTION_PREFIX,
    RequestJournal,
    RequestJournalFull,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4]


def method(
    name,
    namespace,
    path="python/sglang/srt/managers/tokenizer_manager.py",
    cls="TokenizerManager",
):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == cls)
    fn = next(n for n in node.body if getattr(n, "name", None) == name)
    fn.decorator_list = []
    nodes = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        fn,
    ]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            path,
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def payload(rid="public", limit=8):
    value = SimpleNamespace(
        rid=rid,
        input_ids=[1, 2],
        sampling_params=SimpleNamespace(max_new_tokens=limit),
        time_stats=Mock(),
        mm_inputs=None,
        session_id=None,
    )
    value.wrap_pickle_fields = lambda: None
    return value


class TestRequestJournal(CustomTestCase):
    def test_snapshot_and_dispatch_have_independent_cpu_data(self):
        class Timing:
            def __deepcopy__(self, memo):
                raise AssertionError("must not copy live tracing objects")

        original = payload()
        original.time_stats = Timing()
        journal = RequestJournal()
        wire = journal.start(original, membership_epoch=4)
        self.assertTrue(wire.rid.startswith(EXECUTION_PREFIX))
        self.assertEqual(original.rid, "public")
        self.assertIs(wire.time_stats, original.time_stats)
        self.assertEqual(journal.resolve(wire.rid), "public")
        original.input_ids.append(3)
        original.sampling_params.max_new_tokens = 100
        wire.input_ids.append(4)
        snapshot = journal.records["public"].snapshot
        self.assertEqual(snapshot.input_ids, [1, 2])
        self.assertEqual(snapshot.sampling_params.max_new_tokens, 8)
        self.assertIsNone(snapshot.time_stats)

    def test_reused_public_id_cannot_receive_old_execution_results(self):
        journal = RequestJournal()
        first = journal.start(payload(), membership_epoch=0)
        journal.finish("public")
        second = journal.start(payload(), membership_epoch=2)
        self.assertNotEqual(first.rid, second.rid)
        self.assertIsNone(journal.resolve(first.rid))
        self.assertIsNone(journal.resolve("public"))
        self.assertEqual(journal.resolve(second.rid), "public")
        self.assertEqual(journal.status()["stale_messages_dropped"], 2)
        self.assertEqual(journal.resolve("health-check"), "health-check")

    def test_quota_refusal_and_terminal_release(self):
        for requests, tokens in ((1, 100), (10, 10)):
            with self.subTest(requests=requests, tokens=tokens):
                journal = RequestJournal(max_requests=requests, max_tokens=tokens)
                journal.start(payload(), membership_epoch=0)
                with self.assertRaises(RequestJournalFull):
                    journal.start(payload("second"), membership_epoch=0)
                self.assertEqual(len(journal.records), 1)
                self.assertEqual(journal.reserved_tokens, 10)
                journal.finish("public")
                journal.finish("public")
                self.assertEqual(journal.reserved_tokens, 0)
                self.assertFalse(journal._executions)

    def test_cancel_retains_alias_until_terminal_ack(self):
        journal = RequestJournal()
        a = journal.start(payload("a"), membership_epoch=0)
        journal.start(payload("b"), membership_epoch=0)
        journal.cancel("a")
        self.assertTrue(journal.records["a"].cancelled)
        self.assertFalse(journal.records["b"].cancelled)
        self.assertEqual(journal.resolve(a.rid), "a")
        journal.cancel("", all_requests=True)
        self.assertTrue(journal.records["b"].cancelled)

    def test_prefix_budget_never_silently_truncates_a_replayable_record(self):
        journal = RequestJournal()
        journal.start(payload(limit=2), membership_epoch=0)
        journal.commit_output("public", [10])
        journal.commit_output("public", [20])
        self.assertEqual(journal.records["public"].output_ids, [10, 20])
        journal.commit_output("public", [30])
        self.assertFalse(journal.records["public"].replayable)
        self.assertIsNone(journal.records["public"].snapshot)
        self.assertEqual(journal.records["public"].output_ids, [])
        journal.finish("public")
        self.assertEqual(journal.reserved_tokens, 0)

    def test_unsupported_and_duplicate_inputs_leave_no_partial_record(self):
        journal = RequestJournal()
        with self.assertRaises(ValueError):
            journal.validate_request_ids(["ordinary", EXECUTION_PREFIX + "batch"])
        with self.assertRaises(ValueError):
            journal.start(payload(EXECUTION_PREFIX + "user"), membership_epoch=0)
        for field in (
            "mm_inputs",
            "session_id",
            "custom_logit_processor",
            "input_embeds",
        ):
            value = payload()
            setattr(value, field, object())
            self.assertIs(journal.start(value, membership_epoch=0), value)
        self.assertEqual(journal.status()["active_requests"], 0)
        wire = journal.start(payload(), membership_epoch=0)
        with self.assertRaises(ValueError):
            journal.start(payload(), membership_epoch=0)
        self.assertEqual(journal.execution_id("public"), wire.rid)

    def test_sender_uses_private_id_and_keeps_public_request_unchanged(self):
        send = method(
            "_send_one_request",
            {"HEALTH_CHECK_RID_PREFIX": "health", "wrap_shm_features": lambda obj: obj},
        )
        journal = RequestJournal()
        worker = SimpleNamespace(
            request_journal=journal,
            fault_tolerance=SimpleNamespace(state=SimpleNamespace(membership_epoch=8)),
            _dispatch_to_scheduler=Mock(),
        )
        original = payload()
        send(worker, original, record_request=True)
        wire = worker._dispatch_to_scheduler.call_args.args[0]
        self.assertEqual(original.rid, "public")
        self.assertEqual(journal.resolve(wire.rid), "public")
        self.assertEqual(journal.records["public"].membership_epoch, 8)
        self.assertIsNone(journal.records["public"].snapshot.time_stats)

    def test_user_abort_targets_wire_id_and_cleanup_releases_record(self):
        abort = method(
            "abort_request",
            {
                "AbortReq": lambda **kw: SimpleNamespace(**kw),
                "logger": logging.getLogger(__name__),
            },
        )
        cleanup = method("_discard_pending_req_states", {})
        journal = RequestJournal()
        wire = journal.start(payload(), membership_epoch=0)
        obj = SimpleNamespace(rid="public", is_single=True)
        worker = SimpleNamespace(
            request_journal=journal,
            rid_to_state={"public": SimpleNamespace(obj=obj)},
            server_args=SimpleNamespace(tokenizer_worker_num=1),
            enable_metrics=False,
            _dispatch_to_scheduler=Mock(),
        )
        abort(worker, "public")
        self.assertEqual(worker._dispatch_to_scheduler.call_args.args[0].rid, wire.rid)
        self.assertTrue(journal.records["public"].cancelled)
        cleanup(worker, SimpleNamespace(rid="public", is_single=True))
        self.assertIn("public", worker.rid_to_state)
        self.assertEqual(journal.execution_id("public"), wire.rid)
        cleanup(worker, obj)
        self.assertFalse(worker.rid_to_state)
        self.assertEqual(journal.reserved_tokens, 0)

    def test_journal_flag_requires_original_rank_recovery(self):
        handle = method(
            "_handle_fault_tolerance",
            {},
            path="python/sglang/srt/server_args.py",
            cls="ServerArgs",
        )
        with self.assertRaisesRegex(ValueError, "requires --enable-npu-ft-recovery"):
            handle(
                SimpleNamespace(
                    enable_npu_ft_request_journal=True, enable_npu_ft_recovery=False
                )
            )

    def test_health_and_batched_warmup_can_dispatch_at_journal_capacity(self):
        send = method(
            "_send_one_request",
            {"HEALTH_CHECK_RID_PREFIX": "health", "wrap_shm_features": lambda obj: obj},
        )
        journal = RequestJournal(max_requests=1)
        journal.start(payload(), membership_epoch=0)
        worker = SimpleNamespace(request_journal=journal, _dispatch_to_scheduler=Mock())
        send(worker, payload("health-123"), record_request=True)
        send(worker, payload("warmup"))
        self.assertEqual(
            [call.args[0].rid for call in worker._dispatch_to_scheduler.call_args_list],
            ["health-123", "warmup"],
        )
        self.assertEqual(journal.status()["active_requests"], 1)

    def test_late_abort_cannot_finish_new_execution_with_same_public_id(self):
        receive = method(
            "_handle_abort_req",
            {
                "is_health_check_generate_req": lambda obj: False,
                "logger": logging.getLogger(__name__),
            },
        )
        journal = RequestJournal()
        old = journal.start(payload(), membership_epoch=0)
        journal.finish("public")
        new = journal.start(payload(), membership_epoch=2)
        state = SimpleNamespace(
            finished=False,
            obj=SimpleNamespace(stream=False),
            time_stats=Mock(),
            output_ids=[],
            get_text=lambda: "",
            out_list=[],
            event=Mock(),
        )
        worker = SimpleNamespace(
            request_journal=journal,
            rid_to_state={"public": state},
            server_args=SimpleNamespace(weight_version="default"),
        )
        receive(
            worker,
            SimpleNamespace(rid=old.rid, abort_message=None, finished_reason=None),
        )
        self.assertFalse(state.finished)
        state.event.set.assert_not_called()
        receive(
            worker,
            SimpleNamespace(rid=new.rid, abort_message=None, finished_reason=None),
        )
        self.assertTrue(state.finished)
        self.assertEqual(state.out_list[0]["meta_info"]["id"], "public")
        state.event.set.assert_called_once()
        self.assertEqual(journal.reserved_tokens, 0)


class TestJournalOutput(unittest.IsolatedAsyncioTestCase, CustomTestCase):
    async def test_late_finished_batch_does_not_touch_reused_public_request(self):
        class Output(SimpleNamespace):
            pass

        receive = method(
            "_handle_batch_output",
            {
                "unwrap_from_pickle": lambda value: value,
                "BatchStrOutput": Output,
                "BatchTokenIDOutput": type("TokenOutput", (), {}),
                "asyncio": asyncio,
            },
        )
        journal = RequestJournal()
        old = journal.start(payload(), membership_epoch=0)
        journal.finish("public")
        journal.start(payload(), membership_epoch=2)
        state = SimpleNamespace(finished=False, output_ids=[10])
        worker = SimpleNamespace(
            request_journal=journal,
            server_args=SimpleNamespace(batch_notify_size=1),
            rid_to_state={"public": state},
        )
        await receive(
            worker, Output(rids=[old.rid], time_stats=None, customized_info=None)
        )
        self.assertFalse(state.finished)
        self.assertEqual(state.output_ids, [10])
        self.assertEqual(len(journal.records), 1)

    async def test_current_batch_retains_public_id_and_cleans_journal(self):
        class Output(SimpleNamespace):
            def __getattr__(self, name):
                return None

        other = type("OtherOutput", (), {})
        receive = method(
            "_handle_batch_output",
            {
                "unwrap_from_pickle": lambda value: value,
                "BatchStrOutput": Output,
                "BatchTokenIDOutput": other,
                "BatchEmbeddingOutput": other,
                "GenerateReqInput": type("Generate", (), {}),
                "asyncio": asyncio,
            },
        )
        journal = RequestJournal()
        wire = journal.start(payload(), membership_epoch=0)
        text = []
        state = SimpleNamespace(
            obj=SimpleNamespace(stream=False, return_logprob=False),
            finished=False,
            output_ids=[],
            last_output_offset=0,
            prompt_token_ids=None,
            customized_info_accumulated={},
            append_text=text.append,
            get_text=lambda: "".join(text),
            time_stats=Mock(
                first_token_time=0, trace_ctx=SimpleNamespace(tracing_enable=False)
            ),
            out_list=[],
            event=Mock(),
        )
        worker = SimpleNamespace(
            request_journal=journal,
            rid_to_state={"public": state},
            server_args=SimpleNamespace(
                batch_notify_size=1,
                weight_version="default",
                speculative_algorithm=None,
            ),
            enable_metrics=False,
            enable_lora=False,
            dump_requests_folder=None,
            crash_dump_folder=None,
        )
        await receive(
            worker,
            Output(
                rids=[wire.rid],
                time_stats=None,
                customized_info=None,
                finished_reasons=[{"type": "length", "length": 2}],
                prompt_tokens=[2],
                retraction_counts=[0],
                reasoning_tokens=[0],
                completion_tokens=[2],
                cached_tokens=[0],
                output_strs=["ok"],
                output_ids=[[11, 12]],
            ),
        )
        self.assertEqual(state.out_list[0]["meta_info"]["id"], "public")
        self.assertEqual(state.output_ids, [11, 12])
        state.event.set.assert_called_once()
        self.assertEqual(journal.reserved_tokens, 0)
        self.assertFalse(worker.rid_to_state)


if __name__ == "__main__":
    unittest.main()
