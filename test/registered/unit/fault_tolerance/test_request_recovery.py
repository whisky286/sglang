"""CPU tests for request migration, decoding boundaries and production hooks."""

import ast
import asyncio
import dataclasses
import logging
import time
import typing
import unittest
from array import array
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import msgspec
from sglang.srt.fault_tolerance.controller import FaultToleranceState
from sglang.srt.fault_tolerance.request_recovery import (
    EXECUTION_PREFIX,
    RecoveryCapacityError,
    RequestRecovery,
    RequestRecoveryCoordinator,
    decode_checkpoint,
    restore_request,
    validate_recovery_config,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4]


def extract(path, names, namespace, cls=None):
    """Exercise real methods without importing accelerator-only dependencies."""
    tree = ast.parse((ROOT / path).read_text())
    body = (
        tree.body
        if cls is None
        else next(n for n in tree.body if getattr(n, "name", None) == cls).body
    )
    nodes = [n for n in body if getattr(n, "name", None) in names]
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
    namespace.setdefault("__name__", __name__)
    exec(  # noqa: S102 - execute selected repository AST nodes with test doubles.
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        *nodes,
                    ],
                    type_ignores=[],
                )
            ),
            path,
            "exec",
        ),
        namespace,
    )
    return namespace


def payload(rid="public", limit=16):
    return SimpleNamespace(
        rid=rid,
        input_ids=[1, 2],
        input_text="prompt",
        sampling_params=SamplingParams(temperature=0, max_new_tokens=limit),
        routed_dp_rank=None,
        time_stats=Mock(),
        wrap_pickle_fields=lambda: None,
    )


def checkpoint(ids=(65,), pending=0):
    return {"decode_ids": list(ids), "read_offset": len(ids), "pending_chars": pending}


class TestRequestRecovery(CustomTestCase):
    def test_snapshot_preserves_inputs_without_copying_live_tracing(self):
        class LiveTiming:
            def __deepcopy__(self, memo):
                raise AssertionError("live tracing state must not be copied")

        original = payload()
        original.time_stats = LiveTiming()
        journal = RequestRecovery()
        wire = journal.start(original, epoch=0, weight_key=("model", "v1"))
        original.input_ids.append(3)
        wire.input_ids.append(4)
        original.sampling_params.max_new_tokens = 99
        record = journal.records["public"]
        self.assertEqual(record.payload.input_ids, [1, 2])
        self.assertEqual(record.payload.sampling_params.max_new_tokens, 16)
        self.assertIsNone(record.payload.time_stats)
        self.assertIs(wire.time_stats, original.time_stats)
        self.assertNotEqual(wire.rid, original.rid)

    def test_dead_rank_can_be_replayed_before_its_first_output(self):
        journal = RequestRecovery()
        first = journal.start(payload(), epoch=0, weight_key=("m", "v1"))
        self.assertEqual(journal.freeze(), [first.rid])
        wire = journal.resume(
            "public", epoch=1, weight_key=("m", "v1"), time_stats=None
        )
        self.assertEqual(wire.input_ids, [1, 2])
        self.assertEqual(wire.ft_resume_output_ids, [])
        self.assertEqual(wire.ft_request_epoch, 1)
        self.assertIsNone(journal.resolve(first.rid))
        self.assertEqual(journal.resolve(wire.rid), "public")

    def test_prefix_and_original_budget_survive_two_failures(self):
        now = [0.0]
        journal = RequestRecovery(clock=lambda: now[0])
        first = journal.start(payload(), epoch=0, weight_key=("m", "v1"))
        journal.commit_output("public", [65, 66], checkpoint(), 2)
        journal.freeze()
        now[0] = 5
        second = journal.resume(
            "public", epoch=1, weight_key=("m", "v1"), time_stats=None
        )
        journal.commit_output("public", [67], checkpoint(), 3)
        journal.freeze()
        now[0] = 8
        third = journal.resume(
            "public", epoch=2, weight_key=("m", "v1"), time_stats=None
        )
        self.assertEqual(third.ft_resume_output_ids, [65, 66, 67])
        self.assertEqual(third.sampling_params.max_new_tokens, 16)
        self.assertEqual(journal.records["public"].recovery_wait, 8)
        self.assertEqual(journal.records["public"].deadline, 1800)
        self.assertIsNone(journal.resolve(first.rid))
        self.assertIsNone(journal.resolve(second.rid))
        self.assertEqual(journal.resolve(third.rid), "public")
        journal.freeze()
        with self.assertRaisesRegex(ValueError, "retry limit"):
            journal.resume("public", epoch=3, weight_key=("m", "v1"), time_stats=None)

    def test_deadline_wait_and_weight_change_fail_independently(self):
        for kind in ("deadline", "wait", "weight"):
            with self.subTest(kind=kind):
                now = [0.0]
                journal = RequestRecovery(
                    clock=lambda current=now: current[0],
                    request_timeout=20,
                    recovery_timeout=10,
                )
                journal.start(payload(), epoch=0, weight_key=("m", "v1"))
                journal.freeze()
                now[0] = {"deadline": 21, "wait": 11, "weight": 1}[kind]
                key = ("m", "v2") if kind == "weight" else ("m", "v1")
                with self.assertRaisesRegex(
                    ValueError,
                    {"deadline": "deadline", "wait": "wait", "weight": "version"}[kind],
                ):
                    journal.resume("public", epoch=1, weight_key=key, time_stats=None)
                self.assertEqual(journal.records["public"].deadline, 20)

    def test_capacity_is_reserved_and_released_once(self):
        journal = RequestRecovery(max_requests=1, max_tokens=18)
        wire = journal.start(payload(), epoch=0, weight_key=("m", None))
        with self.assertRaises(RecoveryCapacityError):
            journal.start(payload("second"), epoch=0, weight_key=("m", None))
        journal.finish("public")
        journal.finish("public")
        self.assertEqual(journal.reserved_tokens, 0)
        self.assertIsNone(journal.resolve(wire.rid))
        new = journal.start(payload(), epoch=1, weight_key=("m", None))
        self.assertNotEqual(wire.rid, new.rid)

    def test_bad_boundaries_do_not_commit_partial_output(self):
        for ids, cp, count in (
            ([65], checkpoint(), 2),
            ([65], checkpoint(pending=-1), 1),
        ):
            journal = RequestRecovery()
            journal.start(payload(), epoch=0, weight_key=("m", None))
            with self.assertRaises(ValueError):
                journal.commit_output("public", ids, cp, count)
            self.assertEqual(journal.records["public"].output_ids, [])

    def test_unsupported_requests_leave_no_record(self):
        cases = [
            ("routed_dp_rank", 1),
            ("return_logprob", True),
            ("input_embeds", [[1]]),
            ("session_id", "s"),
            ("mm_inputs", object()),
            ("sampling_params", SamplingParams(temperature=1)),
            ("sampling_params", SamplingParams(temperature=0, stop="END")),
            ("sampling_params", SamplingParams(temperature=0, repetition_penalty=1.2)),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=str(value)):
                obj = payload()
                setattr(obj, field, value)
                journal = RequestRecovery()
                with self.assertRaises(ValueError):
                    journal.start(obj, epoch=0, weight_key=("m", None))
                self.assertEqual(journal.status()["active_requests"], 0)

    def test_restored_req_keeps_prompt_output_boundary_and_offsets(self):
        wire = payload()
        wire.ft_resume_output_ids = [65, 66]
        wire.ft_decode_checkpoint = checkpoint()
        wire.ft_recovery_attempt = 1
        req = SimpleNamespace(
            sampling_params=wire.sampling_params,
            origin_input_ids=[1, 2],
            init_incremental_detokenize=lambda: ([2, 65, 66], 1),
        )
        restore_request(req, wire)
        self.assertEqual(list(req.output_ids), [65, 66])
        self.assertEqual(req.origin_input_ids, [1, 2])
        self.assertEqual(req.send_token_offset, 2)
        self.assertEqual(req.send_decode_id_offset, 3)
        wire.ft_resume_output_ids = list(range(16))
        with self.assertRaises(ValueError):
            restore_request(req, wire)

    def test_config_rejects_unimplemented_modes(self):
        args = SimpleNamespace(
            enable_ft_request_recovery=True,
            enable_fault_tolerance=True,
            fault_tolerance_on_error_strategy="pause",
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            disable_radix_cache=True,
            skip_tokenizer_init=False,
            disaggregation_mode="null",
            nnodes=1,
            ft_request_recovery_timeout=600,
            ft_request_timeout=1800,
            ft_request_recovery_max_requests=128,
            ft_request_recovery_max_tokens=262144,
            ft_request_recovery_max_retries=2,
        )
        validate_recovery_config(args)
        for field, value in (
            ("disable_overlap_schedule", False),
            ("nnodes", 2),
            ("enable_fault_tolerance", False),
            ("ft_request_timeout", float("nan")),
        ):
            modified = SimpleNamespace(**vars(args))
            setattr(modified, field, value)
            with self.assertRaises(ValueError):
                validate_recovery_config(modified)

    def test_real_ipc_struct_msgpack_roundtrip(self):
        # Compile the actual struct declarations with CPU-only annotation aliases.
        path = ROOT / "python/sglang/srt/managers/io_struct.py"
        tree = ast.parse(path.read_text())
        nodes = [
            n
            for n in tree.body
            if getattr(n, "name", None) in {
                "BaseReq",
                "BaseBatchReq",
                "TokenizedGenerateReqInput",
                "BatchTokenIDOutput",
                "BatchStrOutput",
                "FTDecodeRetireReq",
            }
        ]
        ns = {
            **vars(typing),
            "msgspec": msgspec,
            "array": array,
            "SamplingParams": SamplingParams,
            "torch": SimpleNamespace(Tensor=typing.Any),
            "__name__": __name__,
        }
        for node in nodes:
            for field in node.body:
                if isinstance(field, ast.AnnAssign):
                    for name in ast.walk(field.annotation):
                        if (
                            isinstance(name, ast.Name)
                            and name.id not in ns
                            and name.id
                            not in ("str", "int", "float", "bool", "bytes", "object")
                        ):
                            ns[name.id] = typing.Any
        exec(  # noqa: S102 - compile the real IPC declarations with CPU aliases.
            compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns
        )
        cls = ns["TokenizedGenerateReqInput"]
        original = cls(
            rid="public",
            input_text="prompt",
            input_ids=array("q", [1, 2]),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=SamplingParams(temperature=0, max_new_tokens=16),
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=True,
        )
        journal = RequestRecovery()
        wire = journal.start(original, epoch=0, weight_key=("m", "v1"))
        journal.commit_output("public", [65], checkpoint(), 1)
        journal.freeze()
        wire = journal.resume(
            "public", epoch=1, weight_key=("m", "v1"), time_stats=None
        )
        encoded = msgspec.msgpack.encode(wire, enc_hook=list)
        decoded = msgspec.msgpack.decode(
            encoded, type=cls, dec_hook=lambda typ, obj: array("q", obj)
        )
        self.assertEqual(decoded.ft_resume_output_ids, [65])
        self.assertEqual(decoded.ft_decode_checkpoint, checkpoint())
        self.assertEqual(decoded.ft_request_epoch, 1)
        self.assertEqual(list(decoded.input_ids), [1, 2])
        for name in ("BatchTokenIDOutput", "BatchStrOutput"):
            self.assertIn("ft_decode_checkpoints", ns[name].__struct_fields__)
        retire_cls = ns["FTDecodeRetireReq"]
        retired = msgspec.msgpack.decode(
            msgspec.msgpack.encode(retire_cls(execution_ids=["old"])),
            type=retire_cls,
        )
        self.assertEqual(retired.execution_ids, ["old"])


class TestDecodeRecovery(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        ns = {
            "dataclasses": dataclasses,
            "OrderedDict": OrderedDict,
            "DETOKENIZER_MAX_STATES": 65536,
        }
        extract(
            "python/sglang/utils.py", {"find_printable_text", "_is_chinese_char"}, ns
        )
        extract(
            "python/sglang/srt/managers/detokenizer_manager.py", {"DecodeStatus"}, ns
        )
        extract(
            "python/sglang/srt/managers/detokenizer_manager.py",
            {
                "_decode_batch_token_id_output",
                "_clamp_decode_ids",
                "trim_matched_stop",
                "handle_ft_decode_retire",
            },
            ns,
            cls="DetokenizerManager",
        )
        cls.ns = ns

    def decoder(self):
        obj = SimpleNamespace(
            decode_status={},
            _ft_retired={},
            vocab_size=256,
            disable_tokenizer_batch_decode=True,
            is_tool_call_parser_gpt_oss=False,
            tokenizer=SimpleNamespace(
                decode=lambda ids, **kw: bytes(t for t in ids if t).decode(
                    "utf8", errors="replace"
                )
            ),
        )
        obj._clamp_decode_ids = self.ns["_clamp_decode_ids"]
        obj.trim_matched_stop = lambda *args: self.ns["trim_matched_stop"](obj, *args)
        return obj

    def feed(self, obj, rid, ids, *, initial=False, cp=None, finish=False):
        value = SimpleNamespace(
            rids=[rid],
            decoded_texts=[""],
            decode_ids=[[0, *ids] if initial else list(ids)],
            read_offsets=[1],
            finished_reasons=[{"type": "length"} if finish else None],
            no_stop_trim=[False],
            skip_special_tokens=[True],
            spaces_between_special_tokens=[True],
            ft_decode_checkpoints=[cp],
        )
        return self.ns["_decode_batch_token_id_output"](obj, value)[0]

    def test_resume_at_every_chinese_emoji_and_printable_tail_boundary(self):
        ids = list("Hello 你好🙂 world!".encode())
        for split in range(1, len(ids)):
            with self.subTest(split=split):
                old = self.decoder()
                prefix = self.feed(old, "old", ids[:split], initial=True)
                cp = decode_checkpoint(old.decode_status["old"])
                # The old decoder can run ahead; only this saved checkpoint is authoritative.
                self.feed(old, "old", ids[split : split + 1])
                resumed = self.decoder()
                suffix = self.feed(resumed, "new", ids[split:], cp=cp, finish=True)
                self.assertEqual(prefix + suffix, "Hello 你好🙂 world!")

    def test_repeated_recovery_preserves_pending_unicode(self):
        ids = list("A 你好🙂Z".encode())
        decoder = self.decoder()
        result = self.feed(decoder, "a", ids[:3], initial=True)
        for rid, start, end in (("b", 3, 6), ("c", 6, 9), ("d", 9, len(ids))):
            cp = decode_checkpoint(next(iter(decoder.decode_status.values())))
            decoder = self.decoder()
            result += self.feed(
                decoder, rid, ids[start:end], cp=cp, finish=end == len(ids)
            )
        self.assertEqual(result, "A 你好🙂Z")

    def test_fresh_scheduler_req_sends_only_the_suffix_to_restored_decoder(self):
        ns = {"INIT_INCREMENTAL_DETOKENIZATION_OFFSET": 5}
        extract(
            "python/sglang/srt/managers/schedule_batch.py",
            {"init_incremental_detokenize"},
            ns,
            cls="Req",
        )

        class CPUReq:
            origin_input_ids_unpadded = array("q", [0] * 9)
            surr_offset = None
            read_offset = None

            @property
            def output_ids_through_stop(self):
                return self.output_ids

        ids = list("Hi 你🙂".encode())
        for split in range(1, len(ids)):
            with self.subTest(split=split):
                old = self.decoder()
                prefix = self.feed(old, "old", ids[:split], initial=True)
                cp = decode_checkpoint(old.decode_status["old"])
                wire = payload()
                wire.ft_resume_output_ids = ids[:split]
                wire.ft_decode_checkpoint = cp
                wire.ft_recovery_attempt = 1
                req = CPUReq()
                req.sampling_params = wire.sampling_params
                req.init_incremental_detokenize = lambda current=req: ns[
                    "init_incremental_detokenize"
                ](current)
                restore_request(req, wire)
                # The real Req method supplies the streamer's decoding token buffer.
                req.output_ids.extend(ids[split:])
                decode_ids, _ = req.init_incremental_detokenize()
                suffix_ids = decode_ids[req.send_decode_id_offset :]
                self.assertEqual(list(suffix_ids), ids[split:])
                resumed = self.decoder()
                result = self.feed(resumed, "new", suffix_ids, cp=cp, finish=True)
                self.assertEqual(prefix + result, "Hi 你🙂")

    def test_retired_decode_state_is_not_resurrected_by_late_output(self):
        decoder = self.decoder()
        self.feed(decoder, "old", [65], initial=True)
        self.ns["handle_ft_decode_retire"](
            decoder, SimpleNamespace(execution_ids=["old"])
        )
        self.assertNotIn("old", decoder.decode_status)
        self.assertEqual(self.feed(decoder, "old", [66]), "")
        self.assertNotIn("old", decoder.decode_status)


class Output(SimpleNamespace):
    def __getattr__(self, name):
        return None


class TestTokenizerRecovery(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        ns = {
            "asyncio": asyncio,
            "HEALTH_CHECK_RID_PREFIX": "HEALTH_CHECK",
            "AbortReq": Output,
            "BatchStrOutput": Output,
            "GenerateReqInput": type("GenerateReqInput", (), {}),
            "BatchEmbeddingOutput": type("Embedding", (), {}),
            "BatchTokenIDOutput": type("TokenOutput", (), {}),
            "is_health_check_generate_req": lambda req: req.rid.startswith(
                "HEALTH_CHECK"
            ),
            "unwrap_from_pickle": lambda obj: obj,
            "wrap_shm_features": lambda obj: obj,
            "logger": logging.getLogger(__name__),
            "_slice_streaming_output_meta_info": lambda *args: None,
        }
        extract(
            "python/sglang/srt/managers/tokenizer_manager.py",
            {
                "_send_one_request",
                "_handle_batch_output",
                "_handle_abort_req",
                "abort_request",
                "_discard_pending_req_states",
                "create_abort_task",
            },
            ns,
            cls="TokenizerManager",
        )
        cls.ns = ns

    async def run_scenario(
        self, cancel=False, initially_paused=False, stream=True, before_first=False
    ):
        manager = SimpleNamespace(
            model_path="m",
            rid_to_state={},
            incremental_streaming_output=True,
            enable_metrics=False,
            enable_lora=False,
            dump_requests_folder="",
            crash_dump_folder="",
            server_args=SimpleNamespace(
                weight_version="v1",
                tokenizer_worker_num=1,
                batch_notify_size=8,
                speculative_algorithm=None,
                ft_request_recovery_max_requests=128,
                ft_request_recovery_max_tokens=262144,
                ft_request_recovery_max_retries=2,
                ft_request_recovery_timeout=600,
                ft_request_timeout=1800,
            ),
            fault_tolerance=SimpleNamespace(state=SimpleNamespace(membership_epoch=0)),
            _dispatch_to_scheduler=Mock(),
        )
        paused = [initially_paused]
        manager.fault_tolerance.should_reject_admission = lambda: paused[0]
        recovery = RequestRecoveryCoordinator(manager, retire=Mock())
        manager.request_recovery = recovery
        manager._send_one_request = lambda obj, **kw: self.ns["_send_one_request"](
            manager, obj, **kw
        )
        manager._handle_abort_req = lambda obj: self.ns["_handle_abort_req"](
            manager, obj
        )
        obj = SimpleNamespace(
            rid="public",
            stream=stream,
            return_logprob=False,
            log_metrics=False,
            is_single=True,
        )
        state = SimpleNamespace(
            obj=obj,
            output_ids=[],
            text="",
            out_list=[],
            event=asyncio.Event(),
            finished=False,
            last_output_offset=0,
            customized_info_accumulated={},
            prompt_token_ids=None,
            time_stats=Mock(first_token_time=0.0),
        )
        state.time_stats.trace_ctx.tracing_enable = False
        state.append_text = lambda text: setattr(state, "text", state.text + text)
        state.get_text = lambda: state.text
        manager.rid_to_state["public"] = state
        manager._send_one_request(payload())
        first = recovery.journal.records["public"].execution_id
        if not initially_paused:
            if not before_first:
                await self.ns["_handle_batch_output"](
                    manager, self.output(first, [65], "A", 1)
                )
            paused[0] = True
            recovery.freeze()
            # Both stale output and stale abort must leave the original state intact.
            await self.ns["_handle_batch_output"](
                manager, self.output(first, [66], "B", 2)
            )
            manager._handle_abort_req(Output(rid=first))
            self.assertIs(manager.rid_to_state["public"], state)
            self.assertEqual(state.output_ids, [] if before_first else [65])
        if cancel:
            self.ns["abort_request"](manager, "public")
            self.assertTrue(state.finished)
            self.assertTrue(state.event.is_set())
            self.assertEqual(recovery.journal.reserved_tokens, 0)
            count = manager._dispatch_to_scheduler.call_count
            paused[0] = False
            await recovery.resume()
            self.assertEqual(manager._dispatch_to_scheduler.call_count, count)
            if initially_paused:
                manager._dispatch_to_scheduler.assert_not_called()
        else:
            paused[0] = False
            manager.fault_tolerance.state.membership_epoch = 1
            await recovery.resume()
            second = recovery.journal.records["public"].execution_id
            self.assertNotEqual(first, second)
            await self.ns["_handle_batch_output"](
                manager,
                self.output(
                    second,
                    [65, 66] if before_first else [66],
                    "AB" if before_first else "B",
                    2,
                    finish=True,
                ),
            )
            self.assertEqual(state.text, "AB")
            self.assertEqual(state.output_ids, [65, 66])
            self.assertEqual(
                [out["meta_info"]["id"] for out in state.out_list],
                ["public"] * (1 if before_first or not stream else 2),
            )
            self.assertEqual(state.out_list[-1]["meta_info"]["recovery_attempts"], 1)
            self.assertNotIn("public", manager.rid_to_state)
            self.assertEqual(recovery.journal.reserved_tokens, 0)
        if recovery._expiry_task:
            recovery._expiry_task.cancel()
            await asyncio.gather(recovery._expiry_task, return_exceptions=True)

    @staticmethod
    def output(rid, ids, text, count, finish=False):
        return Output(
            rids=[rid],
            time_stats=None,
            customized_info=None,
            finished_reasons=[{"type": "length"} if finish else None],
            output_ids=[ids],
            output_strs=[text],
            completion_tokens=[count],
            prompt_tokens=[2],
            reasoning_tokens=[0],
            cached_tokens=[0],
            retraction_counts=[0],
            ft_decode_checkpoints=[None if finish else checkpoint()],
        )

    def test_original_stream_state_survives_migration_and_stale_messages(self):
        with patch.dict(
            "sys.modules",
            {"sglang.srt.managers.io_struct": SimpleNamespace(AbortReq=Output)},
        ):
            asyncio.run(self.run_scenario())

    def test_cancel_while_paused_does_not_wait_for_dead_scheduler(self):
        with patch.dict(
            "sys.modules",
            {"sglang.srt.managers.io_struct": SimpleNamespace(AbortReq=Output)},
        ):
            asyncio.run(self.run_scenario(cancel=True))

    def test_original_nonstream_wait_returns_one_final_result(self):
        with patch.dict(
            "sys.modules",
            {"sglang.srt.managers.io_struct": SimpleNamespace(AbortReq=Output)},
        ):
            asyncio.run(self.run_scenario(stream=False))

    def test_original_wait_survives_fault_before_first_output(self):
        with patch.dict(
            "sys.modules",
            {"sglang.srt.managers.io_struct": SimpleNamespace(AbortReq=Output)},
        ):
            asyncio.run(self.run_scenario(before_first=True))

    def test_cancel_during_tokenization_pause_never_sends_empty_abort(self):
        with patch.dict(
            "sys.modules",
            {"sglang.srt.managers.io_struct": SimpleNamespace(AbortReq=Output)},
        ):
            asyncio.run(self.run_scenario(cancel=True, initially_paused=True))

    def test_background_cleanup_created_before_dispatch_matches_request_identity(self):
        class Tasks:
            def add_task(self, fn):
                self.fn = fn

        self.ns["BackgroundTasks"] = Tasks

        async def scenario(reused):
            obj = SimpleNamespace(rid="public", is_single=True)
            manager = SimpleNamespace(rid_to_state={}, abort_request=Mock())
            tasks = self.ns["create_abort_task"](manager, obj)
            manager.rid_to_state["public"] = SimpleNamespace(
                obj=SimpleNamespace(rid="public") if reused else obj
            )
            with patch.object(asyncio, "sleep", AsyncMock()):
                await tasks.fn()
            self.assertEqual(manager.abort_request.call_count, 0 if reused else 1)

        asyncio.run(scenario(False))
        asyncio.run(scenario(True))


class TestSchedulerReplayHooks(CustomTestCase):
    def test_old_requests_are_retired_without_releasing_npu_state_early(self):
        class Batch:
            def __init__(self, reqs, **kwargs):
                self.reqs = reqs

        class Finish(Output):
            def to_json(self):
                return vars(self)

        def req(rid, allocated):
            value = Output(
                rid=rid,
                output_ids=[65],
                origin_input_ids=[1, 2],
                kv_committed_len=4,
                req_pool_idx=1 if allocated else None,
                kv=object() if allocated else None,
                finished_reason=None,
            )
            value.finished = lambda: value.finished_reason is not None
            return value

        ns = {
            "ScheduleBatch": Batch,
            "FINISH_ABORT": Finish,
            "AbortReq": Output,
            "HTTPStatus": SimpleNamespace(SERVICE_UNAVAILABLE=503),
            "logger": logging.getLogger(__name__),
            "release_kv_cache": Mock(),
        }
        extract(
            "python/sglang/srt/managers/scheduler.py",
            {"_ft_discard_inflight_window", "_ft_release_deferred_kv_cache"},
            ns,
            cls="Scheduler",
        )
        running, waiting = req("running", True), req("waiting", False)
        worker = SimpleNamespace(
            server_args=SimpleNamespace(enable_ft_request_recovery=True),
            cur_batch_for_debug=Batch([running]),
            last_batch=None,
            running_batch=Batch([running]),
            waiting_queue=[waiting],
            chunked_req=running,
            tree_cache=object(),
            ipc_channels=SimpleNamespace(send_to_tokenizer=Mock()),
        )
        self.assertTrue(
            ns["_ft_discard_inflight_window"](worker, "fault", defer_kv_release=True)
        )
        ns["release_kv_cache"].assert_not_called()
        self.assertEqual(worker.waiting_queue, [])
        self.assertEqual(worker.running_batch.reqs, [])
        self.assertIsNone(worker.chunked_req)
        self.assertEqual(
            set(worker._ft_deferred_kv_release_reqs), {"running", "waiting"}
        )
        ns["_ft_release_deferred_kv_cache"](worker)
        self.assertEqual(ns["release_kv_cache"].call_count, 1)
        self.assertIs(ns["release_kv_cache"].call_args.args[0], running)
        self.assertEqual(worker._ft_deferred_kv_release_reqs, {})

    def test_scheduler_rejects_late_old_epoch_work_and_work_while_paused(self):
        ns = {"AbortReq": Output}
        extract(
            "python/sglang/srt/managers/scheduler.py",
            {"handle_generate_request"},
            ns,
            cls="Scheduler",
        )
        for epoch, paused in ((0, False), (1, True)):
            worker = SimpleNamespace(
                server_args=SimpleNamespace(enable_ft_request_recovery=True),
                _ft_request_epoch=1,
                _engine_paused=paused,
                ipc_channels=SimpleNamespace(send_to_tokenizer=Mock()),
            )
            wire = SimpleNamespace(rid=EXECUTION_PREFIX + "old", ft_request_epoch=epoch)
            ns["handle_generate_request"](worker, wire)
            output = worker.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
            self.assertEqual(output.finished_reason["err_type"], "SchedulerFault")

    def test_scheduler_only_advances_epoch_after_device_recovery_succeeds(self):
        ns = {
            "FaultToleranceCommandReqOutput": Output,
            "_is_npu": True,
            "logger": logging.getLogger(__name__),
        }
        extract(
            "python/sglang/srt/managers/scheduler.py",
            {"handle_fault_tolerance_command"},
            ns,
            cls="Scheduler",
        )
        for fail in (False, True):
            worker = SimpleNamespace(
                ps=SimpleNamespace(dp_rank=0, attn_tp_rank=0, attn_cp_rank=0),
                server_args=SimpleNamespace(
                    enable_ft_request_recovery=True, elastic_ep_backend="mc2"
                ),
                _engine_paused=False,
                _ft_request_epoch=1,
                _ft_pause_deadline=123,
                tp_worker=SimpleNamespace(model_runner=object()),
            )

            def discard(*args, _worker=worker, **kwargs):
                self.assertTrue(_worker._engine_paused)
                self.assertEqual(_worker._ft_request_epoch, 1)
                return True

            worker._ft_discard_inflight_window = discard
            worker._recover_npu_fault_tolerance_scale_down = Mock(
                side_effect=RuntimeError("recovery failed") if fail else None
            )
            result = ns["handle_fault_tolerance_command"](
                worker,
                Output(
                    command="scale_down",
                    target_ranks=[0],
                    request_id="command",
                    active_mask=[True, False, True, True],
                    request_epoch=2,
                ),
            )
            self.assertEqual(result.success, not fail)
            self.assertEqual(worker._ft_request_epoch, 1 if fail else 2)
            self.assertEqual(worker._engine_paused, fail)


class TestRecoveryControl(CustomTestCase):
    async def scenario(self, fail=False):
        ns = {
            "asyncio": asyncio,
            "dataclasses": dataclasses,
            "time": time,
            "logging": logging,
            "logger": logging.getLogger(__name__),
            "FaultToleranceState": FaultToleranceState,
            "ft_failure": lambda message: {"success": False, "message": message},
            "WATCHDOG_LEASE_TIMEOUT_SEC": 5,
            "FaultToleranceCommandReqInput": Output,
        }
        extract(
            "python/sglang/srt/fault_tolerance/manager.py",
            {"FaultToleranceManager", "PendingFTCommand"},
            ns,
        )
        events = []
        manager = ns["FaultToleranceManager"](
            server_args=SimpleNamespace(
                dp_size=4,
                tp_size=4,
                fault_tolerance_on_error_strategy="pause",
                fault_tolerance_timeout=1,
                enable_ft_request_recovery=True,
            ),
            send_to_scheduler=AsyncMock(),
            send_to_dpc=AsyncMock(),
            on_incident=Mock(),
        )
        manager.bind_event_loop(asyncio.get_running_loop())
        manager.handle_rank_fault(Output(rank=0, message="failure", request_epoch=0))
        manager.on_incident.assert_called_once()
        manager._shutdown_dp_processes = AsyncMock()
        manager._send_command_collect = AsyncMock(return_value={0, 2, 3})

        async def publish(*args):
            events.append("route_committed")
            if fail:
                raise RuntimeError("route ack missing")

        async def recovered():
            self.assertEqual(manager.state.membership_epoch, 1)
            self.assertEqual(events, ["route_committed"])
            events.append("replay")

        manager._publish_route_dp_mask = publish
        manager.on_recovered = AsyncMock(side_effect=recovered)
        manager._failstop = Mock(side_effect=RuntimeError("failstop"))
        if fail:
            with self.assertRaisesRegex(RuntimeError, "failstop"):
                await manager.apply(
                    {"instruction": "scale_down", "params": {"ranks": [1]}}
                )
            manager.on_recovered.assert_not_awaited()
            self.assertEqual(manager.state.membership_epoch, 0)
        else:
            code, body = await manager.apply(
                {"instruction": "scale_down", "params": {"ranks": [1]}}
            )
            self.assertEqual(code, 200)
            self.assertEqual(body["membership_epoch"], 1)
            self.assertEqual(events, ["route_committed", "replay"])
            manager.handle_rank_fault(Output(rank=0, message="late", request_epoch=0))
            self.assertEqual(manager.state.unhealthy_dp_ranks, set())
            manager.observe_active_ranks(
                Output(status=[True, False, False, True], request_epoch=0)
            )
            self.assertEqual(manager.state.unhealthy_dp_ranks, set())

    def test_replay_only_after_route_commit_and_stale_faults_are_ignored(self):
        asyncio.run(self.scenario())

    def test_failed_route_commit_never_replays_requests(self):
        asyncio.run(self.scenario(fail=True))


if __name__ == "__main__":
    unittest.main()
