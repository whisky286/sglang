"""Healthy membership drain and terminal cleanup after an NPU worker exit."""

import ast
import asyncio
import logging
import unittest
from contextlib import nullcontext
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from sglang.srt.fault_tolerance.manager import FaultToleranceManager
from sglang.srt.fault_tolerance.request_drain import (
    RequestDrainError,
    drain_requests,
    scheduler_has_pending_requests,
)
from sglang.srt.managers.io_struct import (
    AbortReq,
    FaultToleranceCommandReqOutput,
    FaultToleranceRankFaultOutput,
    ProcessActiveRanksOutput,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4]


def load_method(path, class_name, method_name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if getattr(n, "name", None) == class_name)
    method = next(n for n in cls.body if getattr(n, "name", None) == method_name)
    method.decorator_list = []
    nodes = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        method,
    ]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            path,
            "exec",
        ),
        namespace,
    )
    return namespace[method_name]


def make_manager(pending=None, on_incident=None):
    manager = FaultToleranceManager(
        server_args=SimpleNamespace(
            device="npu",
            elastic_ep_backend="mc2",
            enable_npu_ft_recovery=True,
            dp_size=4,
            tp_size=4,
            fault_tolerance_timeout=5,
            fault_tolerance_on_error_strategy="pause",
        ),
        send_to_scheduler=AsyncMock(),
        send_to_dpc=AsyncMock(),
        pending_requests=pending,
        on_incident=on_incident,
    )
    manager.bind_event_loop(asyncio.get_running_loop())
    manager.state.expected_dp_mask = [True, False, True, True]
    manager.state.process_alive_global_rank_mask[1] = False
    manager.state.membership_epoch = 1
    manager._watchdog_leases[0] = (0, (0, 1, 2, 3))
    manager._route_dp_mask = [True, False, True, True]
    return manager


class TestRequestDrain(unittest.IsolatedAsyncioTestCase, CustomTestCase):
    async def test_request_closed_during_tokenization_is_not_dispatched(self):
        class HttpError(Exception):
            def __init__(self, *, status_code, detail):
                self.status_code = status_code

        class Context:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def wait_for(self, predicate):
                self.assertion = predicate()

        class Generate:
            rid = "before-first-token"
            routed_dp_rank = None
            is_single = True
            normalize_batch_and_arguments = Mock()

        generate = load_method(
            "python/sglang/srt/managers/tokenizer_manager.py",
            "TokenizerManager",
            "generate_request",
            {
                "GenerateReqInput": Generate,
                "fastapi": SimpleNamespace(HTTPException=HttpError),
            },
        )
        worker = SimpleNamespace(
            auto_create_handle_loop=Mock(),
            _set_default_priority=Mock(),
            fault_tolerance=SimpleNamespace(should_reject_admission=lambda: False),
            rid_to_state={},
            tokenizer=None,
            server_args=SimpleNamespace(language_only=False),
            request_logger=SimpleNamespace(log_received_request=Mock()),
            is_pause_cond=Context(),
            is_pause=False,
            model_update_lock=SimpleNamespace(reader_lock=Context()),
            _validate_and_resolve_lora=AsyncMock(),
            _send_one_request=Mock(),
            _discard_pending_req_states=Mock(),
        )
        worker._init_req_state = lambda obj, request: worker.rid_to_state.update(
            {obj.rid: object()}
        )

        async def tokenize(obj):
            worker.rid_to_state.clear()  # The incident callback ended this request.
            return SimpleNamespace(rid=obj.rid)

        worker._tokenize_one_request = tokenize
        with self.assertRaises(HttpError) as error:
            await anext(generate(worker, Generate()))
        self.assertEqual(error.exception.status_code, 503)
        worker._send_one_request.assert_not_called()

    async def test_waits_for_frontend_then_all_schedulers(self):
        pending = Mock(side_effect=[True, False, False, False])
        check = AsyncMock(side_effect=[False, True])
        await drain_requests(
            pending=pending, check_idle=check, has_incident=lambda: False, timeout=1
        )
        self.assertEqual(check.await_count, 2)

    async def test_pending_or_unresponsive_scheduler_has_finite_timeout(self):
        for busy_frontend in (True, False):
            with self.subTest(busy_frontend=busy_frontend):

                async def blocked(remaining):
                    await asyncio.sleep(1)

                with self.assertRaisesRegex(RequestDrainError, "recover_drain_timeout"):
                    await drain_requests(
                        pending=lambda: busy_frontend,
                        check_idle=blocked,
                        has_incident=lambda: False,
                        timeout=0.01,
                    )

    async def test_incident_and_cancel_before_switch_release_admission(self):
        for mode in ("timeout", "incident", "cancel", "check_failed"):
            with self.subTest(mode=mode):
                manager = make_manager(lambda: mode != "check_failed")
                manager._failstop = Mock(
                    side_effect=AssertionError("must not fail-stop")
                )

                async def failed_probe(**kwargs):
                    self.assertTrue(manager.should_reject_admission())
                    raise RuntimeError("probe failed")

                manager._send_command_collect = AsyncMock(side_effect=failed_probe)
                manager._publish_route_dp_mask = AsyncMock()
                task = asyncio.create_task(
                    manager.apply(
                        {
                            "instruction": "recover",
                            "params": {"ranks": [1], "drain_timeout": 0.02},
                        }
                    )
                )
                await asyncio.sleep(0)
                if not task.done():
                    self.assertTrue(manager.should_reject_admission())
                if mode == "incident":
                    manager.state.observe_rank_fault(0)
                if mode == "cancel":
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                else:
                    status, body = await task
                    self.assertEqual(
                        status,
                        {"timeout": 408, "incident": 409, "check_failed": 503}[mode],
                    )
                    self.assertFalse(body["success"])
                self.assertFalse(manager.state.ft_operation_in_progress)
                self.assertEqual(manager.state.membership_epoch, 1)
                self.assertIsNone(manager._recovering_epoch)
                manager.send_to_dpc.assert_not_awaited()
                manager._publish_route_dp_mask.assert_not_awaited()
                manager._failstop.assert_not_called()
                self.assertEqual(manager.should_reject_admission(), mode == "incident")

    async def test_accepted_request_finishes_before_pause_or_spawn(self):
        requests = {"long": object()}
        manager = make_manager(lambda: bool(requests))
        events = []

        async def command(**kwargs):
            events.append(kwargs["command"])
            self.assertFalse(requests)
            self.assertTrue(manager.should_reject_admission())
            return set(kwargs["target_ranks"])

        manager._send_command_collect = AsyncMock(side_effect=command)
        manager._publish_route_dp_mask = AsyncMock()
        task = asyncio.create_task(
            manager.apply({"instruction": "recover", "params": {"ranks": [1]}})
        )
        await asyncio.sleep(0.01)
        self.assertEqual(events, [])
        manager.send_to_dpc.assert_not_awaited()
        requests.clear()
        self.assertEqual((await task)[0], 200)
        self.assertEqual(events[:2], ["recover_check_idle", "recover_pause"])
        self.assertEqual(manager.state.membership_epoch, 2)
        self.assertFalse(manager.should_reject_admission())

    async def test_collect_idle_uses_every_rank_message_and_ignores_duplicate(self):
        manager = make_manager()

        async def send(request):
            for rank, message in ((0, "idle"), (0, "busy"), (2, "busy"), (3, "idle")):
                manager.handle_command_output(
                    FaultToleranceCommandReqOutput(
                        request_id=request.request_id,
                        rank=rank,
                        success=True,
                        message=message,
                    )
                )

        manager.send_to_scheduler = AsyncMock(side_effect=send)
        idle = await manager._send_command_collect(
            command="recover_check_idle", target_ranks=[0, 2, 3], timeout_sec=1
        )
        self.assertEqual(idle, {0, 3})
        self.assertFalse(manager._pending_commands)

    async def test_invalid_drain_timeout_does_not_touch_service(self):
        for value in (True, 0, -1, "1", float("nan"), float("inf")):
            with self.subTest(value=value):
                manager = make_manager(lambda: False)
                self.assertEqual(
                    (
                        await manager.apply(
                            {
                                "instruction": "recover",
                                "params": {"ranks": [1], "drain_timeout": value},
                            }
                        )
                    )[0],
                    400,
                )
                self.assertFalse(manager.state.ft_operation_in_progress)

    async def test_real_incident_closes_frontend_but_removed_or_stale_exit_does_not(
        self,
    ):
        callback = Mock()
        manager = make_manager(on_incident=callback)
        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[1], active=False, membership_epoch=1)
        )
        callback.assert_not_called()
        manager._process_event_floor_epoch = 2
        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[0], active=False, membership_epoch=1)
        )
        callback.assert_not_called()
        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[0], active=False, membership_epoch=2)
        )
        callback.assert_called_once()
        manager.handle_rank_fault(
            FaultToleranceRankFaultOutput(rank=2, message="collective")
        )
        self.assertEqual(callback.call_count, 2)


class TestFrontendAndSchedulerDrain(CustomTestCase):
    def test_scale_down_closes_batches_before_device_and_queues_after_device(self):
        handle = load_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "_recover_npu_fault_tolerance_scale_down",
            {"AbortReq": AbortReq},
        )
        events = []
        runner = SimpleNamespace(
            recover_npu_device_for_fault_tolerance_scale_down=lambda: events.append(
                "device"
            ),
            apply_fault_tolerance_scale_down=lambda *args, **kwargs: events.append(
                "membership"
            ),
            run_npu_fault_tolerance_dummy_batch=lambda *args: events.append("dummy"),
            synchronize_npu_fault_tolerance_health_gate=lambda: events.append("health"),
        )

        def close_batches(reason, *, defer_kv_release):
            self.assertTrue(defer_kv_release)
            events.append("detach_batches")
            return True

        def cancel_queues(request):
            self.assertTrue(request.abort_all)
            events.append("cancel_queues")

        worker = SimpleNamespace(
            server_args=SimpleNamespace(enable_npu_ft_recovery=True),
            tp_worker=SimpleNamespace(model_runner=runner),
            _ft_discard_inflight_window=close_batches,
            abort_request=cancel_queues,
            device_module=SimpleNamespace(StreamContext=lambda stream: nullcontext()),
            schedule_stream=SimpleNamespace(synchronize=lambda: events.append("sync")),
            forward_stream=SimpleNamespace(wait_stream=lambda stream: None),
            forward_stream_ctx=nullcontext(),
            _ft_release_deferred_kv_cache=lambda: events.append("release"),
        )
        handle(worker, [True, False, True, True], membership_epoch=1)
        self.assertEqual(
            events,
            [
                "detach_batches",
                "device",
                "cancel_queues",
                "membership",
                "release",
                "sync",
                "dummy",
                "health",
            ],
        )

    def test_scheduler_checks_every_queue_without_device_access(self):
        for location in (
            "waiting",
            "grammar",
            "running",
            "last",
            "current",
            "result",
            "chunked",
            "kv",
        ):
            with self.subTest(location=location):
                req = SimpleNamespace(
                    finished=lambda: location == "kv",
                    req_pool_idx=3 if location == "kv" else None,
                )
                batch = SimpleNamespace(reqs=[req])
                self.assertTrue(
                    scheduler_has_pending_requests(
                        queues=(
                            [req] if location == "waiting" else [],
                            [req] if location == "grammar" else [],
                        ),
                        batches=tuple(
                            batch if location == pos else None
                            for pos in ("running", "last", "current")
                        ),
                        result_queue=[(batch, None)] if location == "result" else [],
                        chunked_req=req if location in ("chunked", "kv") else None,
                    )
                )
        self.assertFalse(
            scheduler_has_pending_requests(
                queues=([], []),
                batches=(
                    SimpleNamespace(reqs=[SimpleNamespace(finished=lambda: True)]),
                ),
                result_queue=[],
                chunked_req=None,
            )
        )

    def test_fault_cleanup_including_dead_owner_and_late_abort_is_terminal_once(self):
        abort = load_method(
            "python/sglang/srt/managers/tokenizer_manager.py",
            "TokenizerManager",
            "_handle_abort_req",
            {
                "is_health_check_generate_req": lambda obj: False,
                "logger": logging.getLogger(__name__),
            },
        )
        cleanup = load_method(
            "python/sglang/srt/managers/tokenizer_manager.py",
            "TokenizerManager",
            "_finish_npu_faulted_requests",
            {"AbortReq": AbortReq, "HTTPStatus": HTTPStatus},
        )
        for incremental in (False, True):
            with self.subTest(incremental=incremental):
                states = {
                    rid: SimpleNamespace(
                        obj=SimpleNamespace(stream=True, return_logprob=False),
                        finished=False,
                        time_stats=SimpleNamespace(
                            set_finished_time=Mock(), get_e2e_latency=lambda: 1
                        ),
                        output_ids=[10, 20],
                        get_text=lambda: "prefix",
                        out_list=[],
                        event=Mock(),
                    )
                    for rid in ("dead-owner", "survivor")
                }
                worker = SimpleNamespace(
                    rid_to_state=dict(states),
                    incremental_streaming_output=incremental,
                    server_args=SimpleNamespace(weight_version="default"),
                )
                worker._handle_abort_req = lambda obj: abort(worker, obj)
                cleanup(worker, "fault")
                self.assertEqual(worker.rid_to_state, {})
                worker._handle_abort_req(AbortReq(rid="dead-owner"))
                for state in states.values():
                    state.event.set.assert_called_once()
                    self.assertEqual(len(state.out_list), 1)
                    result = state.out_list[0]
                    self.assertEqual(
                        result["meta_info"]["finish_reason"]["status_code"], 503
                    )
                    self.assertEqual(result["meta_info"]["completion_tokens"], 2)
                    self.assertEqual(result["output_ids"], [] if incremental else [20])
                    self.assertEqual(result["text"], "" if incremental else "prefix")

    def test_idle_probe_does_not_pause_scheduler_or_touch_device(self):
        handle = load_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "_handle_npu_recovery_command",
            {"FaultToleranceCommandReqOutput": FaultToleranceCommandReqOutput},
        )
        worker = SimpleNamespace(
            server_args=SimpleNamespace(enable_npu_ft_recovery=True),
            ps=SimpleNamespace(dp_rank=0),
            waiting_queue=[],
            grammar_manager=SimpleNamespace(grammar_queue=[]),
            running_batch=None,
            last_batch=None,
            cur_batch_for_debug=None,
            chunked_req=None,
            _engine_paused=False,
        )
        result = handle(
            worker, SimpleNamespace(command="recover_check_idle", request_id="probe")
        )
        self.assertTrue(result.success)
        self.assertEqual(result.message, "idle")
        self.assertFalse(worker._engine_paused)


if __name__ == "__main__":
    unittest.main()
