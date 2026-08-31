import ast
from collections import deque
from contextlib import nullcontext
from http import HTTPStatus
import logging
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Optional, Tuple
import unittest
from unittest.mock import Mock, patch

REPO_ROOT = Path(__file__).resolve().parents[4]


class Struct:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FaultToleranceCommandReqInput(Struct):
    pass


class FaultToleranceCommandReqOutput(Struct):
    pass


class FaultToleranceDPCShutdownReqInput(Struct):
    pass


class FaultToleranceRankFaultOutput(Struct):
    pass


class ProcessActiveRanksOutput(Struct):
    pass


class WatchdogHeartbeatOutput(Struct):
    pass


class AbortReq(Struct):
    pass


class FinishAbort:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_json(self):
        return self.__dict__


class FakeTensor:
    def __init__(self, value):
        self.value = list(value)

    def copy_(self, other):
        self.value = list(other.value)
        return self

    def detach(self):
        return self

    def cpu(self):
        return self


class FakeReq:
    def __init__(self, rid, *, origin_input_ids=None, output_ids=None, committed=0):
        self.rid = rid
        self.finished_reason = None
        self.origin_input_ids = origin_input_ids or []
        self.output_ids = output_ids or []
        self.kv_committed_len = committed

    def finished(self):
        return self.finished_reason is not None


class FakeBatch:
    def __init__(self, reqs, batch_is_full=True):
        self.reqs = list(reqs)
        self.batch_is_full = batch_is_full


def load_class_methods(path, class_name, method_names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in method_names
    ]
    module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in method_names}


class Sender:
    def __init__(self):
        self.sent = []
        self.options = []
        self.endpoint = None
        self.closed = False

    def send_pyobj(self, value, flags=0):
        self.sent.append(value)

    def send_output(self, value, *args):
        self.sent.append((value, *args))

    def setsockopt(self, option, value):
        self.options.append((option, value))

    def connect(self, endpoint):
        self.endpoint = endpoint

    def close(self, linger=None):
        self.closed = True


class FakeContext:
    def __init__(self, sender):
        self.sender = sender
        self.terminated = False

    def socket(self, socket_type):
        return self.sender

    def term(self):
        self.terminated = True


class TestSchedulerFaultToleranceControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scheduler = load_class_methods(
            REPO_ROOT / "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            {
                "_run_event_loop_fault_tolerance",
                "_rebuild_npu_fault_tolerance_control_runtime",
                "_recover_npu_fault_tolerance_scale_down",
                "handle_fault_tolerance_command",
                "_check_ft_pause_deadline",
                "_ft_discard_inflight_window",
                "_ft_release_deferred_kv_cache",
                "_process_next_overlap_result",
            },
            {
                "AbortReq": AbortReq,
                "FINISH_ABORT": FinishAbort,
                "FaultToleranceCommandReqInput": FaultToleranceCommandReqInput,
                "FaultToleranceCommandReqOutput": FaultToleranceCommandReqOutput,
                "FaultToleranceRankFaultOutput": FaultToleranceRankFaultOutput,
                "HTTPStatus": HTTPStatus,
                "Optional": Optional,
                "ScheduleBatch": FakeBatch,
                "Tuple": Tuple,
                "logger": logging.getLogger(__name__),
                "notify_node_main_process_failure": Mock(),
                "release_kv_cache": lambda *args, **kwargs: None,
                "time": SimpleNamespace(monotonic=lambda: 100.0, sleep=Mock()),
                "_is_npu": False,
            },
        )
        cls.run_ft_loop = staticmethod(scheduler["_run_event_loop_fault_tolerance"])
        cls.rebuild_streams = staticmethod(
            scheduler["_rebuild_npu_fault_tolerance_control_runtime"]
        )
        cls.recover_scale_down = staticmethod(
            scheduler["_recover_npu_fault_tolerance_scale_down"]
        )
        cls.handle_command = staticmethod(scheduler["handle_fault_tolerance_command"])
        cls.check_deadline = staticmethod(scheduler["_check_ft_pause_deadline"])
        cls.discard = staticmethod(scheduler["_ft_discard_inflight_window"])
        cls.release_deferred_kv = staticmethod(
            scheduler["_ft_release_deferred_kv_cache"]
        )
        cls.process_overlap = staticmethod(scheduler["_process_next_overlap_result"])

        server_args = load_class_methods(
            REPO_ROOT / "python/sglang/srt/server_args.py",
            "ServerArgs",
            {"_handle_fault_tolerance"},
            {
                "is_npu": lambda: True,
                "logger": logging.getLogger(__name__),
                "os": os,
            },
        )
        cls.handle_fault_tolerance_args = staticmethod(
            server_args["_handle_fault_tolerance"]
        )

        cls.dpc_globals = {
            "FaultToleranceCommandReqInput": FaultToleranceCommandReqInput,
            "FaultToleranceDPCShutdownReqInput": FaultToleranceDPCShutdownReqInput,
            "ProcessActiveRanksOutput": ProcessActiveRanksOutput,
            "WatchdogHeartbeatOutput": WatchdogHeartbeatOutput,
            "FT_WATCHDOG_SEND_TIMEOUT_MS": 1000,
            "logger": logging.getLogger(__name__),
            "sock_send": lambda socket, value, flags=0: socket.send_pyobj(value, flags),
            "zmq": SimpleNamespace(
                Context=None,
                PUSH="push",
                LINGER="linger",
                SNDHWM="sndhwm",
                IMMEDIATE="immediate",
                SNDTIMEO="sndtimeo",
                IPV6="ipv6",
                NOBLOCK=1,
                Again=RuntimeError,
            ),
        }
        dpc = load_class_methods(
            REPO_ROOT / "python/sglang/srt/managers/data_parallel_controller.py",
            "DataParallelController",
            {
                "send_fault_tolerance_command",
                "shutdown_dp",
                "_get_watchdog_sender",
                "_handle_scheduler_process_exit",
                "_watchdog_heartbeat",
                "_report_initial_watchdog_heartbeat",
                "_report_watchdog_heartbeat",
                "_close_watchdog_sender",
                "_report_process_active_ranks",
            },
            cls.dpc_globals,
        )
        cls.dpc_methods = dpc

        cls.fake_torch = SimpleNamespace(
            device=lambda device_type, device_id: (device_type, device_id),
            npu=SimpleNamespace(set_device=Mock(), synchronize=Mock()),
        )
        model_runner = load_class_methods(
            REPO_ROOT / "python/sglang/srt/model_executor/model_runner.py",
            "ModelRunner",
            {
                "recover_npu_device_for_fault_tolerance_scale_down",
                "run_npu_fault_tolerance_dummy_batch",
            },
            {
                "ElasticEPStateManager": SimpleNamespace(
                    instance=lambda: SimpleNamespace(npu_mc2_elastic_info=None)
                ),
                "logger": logging.getLogger(__name__),
                "torch": cls.fake_torch,
            },
        )
        cls.recover_npu = staticmethod(
            model_runner["recover_npu_device_for_fault_tolerance_scale_down"]
        )
        cls.run_npu_dummy = staticmethod(
            model_runner["run_npu_fault_tolerance_dummy_batch"]
        )

        base_runner = load_class_methods(
            REPO_ROOT / "python/sglang/srt/model_executor/runner/base_runner.py",
            "BaseRunner",
            {"run_dummy_via_model_runner"},
            {
                "Optional": Optional,
                "set_dp_buffer_len": Mock(),
                "set_is_extend_in_batch": Mock(),
                "torch": SimpleNamespace(inference_mode=lambda: nullcontext()),
            },
        )
        cls.run_dummy_dispatch = staticmethod(
            base_runner["run_dummy_via_model_runner"]
        )

    def make_scheduler(self, *, leader=True):
        forward_stream = SimpleNamespace(
            stream_id=10, synchronize=Mock(), wait_stream=Mock()
        )
        old_copy_stream = SimpleNamespace(stream_id=8, synchronize=Mock())
        old_schedule_stream = SimpleNamespace(stream_id=9, synchronize=Mock())
        model_runner = SimpleNamespace(
            forward_stream=forward_stream,
            apply_fault_tolerance_scale_down=Mock(),
            recover_npu_device_for_fault_tolerance_scale_down=Mock(),
            run_npu_fault_tolerance_dummy_batch=Mock(),
            synchronize_npu_fault_tolerance_health_gate=Mock(),
        )
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(
                dp_rank=1,
                attn_tp_rank=0 if leader else 1,
                attn_cp_rank=0,
            ),
            tp_worker=SimpleNamespace(model_runner=model_runner),
            server_args=SimpleNamespace(elastic_ep_backend="mc2"),
            device_module=SimpleNamespace(
                Stream=Mock(),
                stream=lambda stream: nullcontext(),
                StreamContext=lambda stream: nullcontext(),
            ),
            forward_stream=forward_stream,
            forward_stream_ctx=nullcontext(),
            copy_stream=old_copy_stream,
            copy_stream_ctx=nullcontext(),
            schedule_stream=old_schedule_stream,
            model_worker=SimpleNamespace(
                war_fastpath_runner=SimpleNamespace(
                    war_fastpath_read_done_event=object()
                )
            ),
            hisparse_coordinator=None,
            enable_unified_memory=False,
            future_map=SimpleNamespace(publish_ready=object(), _publish_fresh=True),
            _engine_paused=True,
            _ft_pause_deadline=130.0,
            _ft_discard_inflight_window=Mock(return_value=True),
            _ft_release_deferred_kv_cache=Mock(),
        )
        scheduler._rebuild_npu_fault_tolerance_control_runtime = lambda: (
            self.rebuild_streams(scheduler)
        )
        scheduler._recover_npu_fault_tolerance_scale_down = (
            lambda active_mask, request_id: self.recover_scale_down(
                scheduler, active_mask, request_id
            )
        )
        return scheduler

    def test_retry_restores_last_mask_without_replacing_tensors(self):
        state = SimpleNamespace(
            active_ranks=FakeTensor([1, 0]),
            active_ranks_cpu=FakeTensor([1, 0]),
            last_active_ranks=FakeTensor([1, 1]),
        )
        manager = SimpleNamespace(instance=lambda: state)
        module = ModuleType("sglang.srt.elastic_ep.elastic_ep")
        module.ElasticEPStateManager = manager
        modules = {
            "sglang": ModuleType("sglang"),
            "sglang.srt": ModuleType("sglang.srt"),
            "sglang.srt.elastic_ep": ModuleType("sglang.srt.elastic_ep"),
            "sglang.srt.elastic_ep.elastic_ep": module,
        }
        scheduler = self.make_scheduler()
        request = FaultToleranceCommandReqInput(
            request_id="r", command="retry", target_ranks=[1], active_mask=None
        )

        with patch.dict(sys.modules, modules):
            output = self.handle_command(scheduler, request)

        self.assertEqual(state.active_ranks.value, [1, 1])
        self.assertEqual(state.active_ranks_cpu.value, [1, 1])
        self.assertFalse(scheduler._engine_paused)
        self.assertIsNone(scheduler._ft_pause_deadline)
        self.assertEqual(output.message, "retried")

    def test_scale_down_is_one_command_and_unpauses(self):
        scheduler = self.make_scheduler()
        request = FaultToleranceCommandReqInput(
            request_id="s",
            command="scale_down",
            target_ranks=[1],
            active_mask=[True, False],
        )

        output = self.handle_command(scheduler, request)

        apply_scale_down = (
            scheduler.tp_worker.model_runner.apply_fault_tolerance_scale_down
        )
        apply_scale_down.assert_called_once_with([True, False])
        self.assertEqual(output.message, "scaled down")
        self.assertFalse(scheduler._engine_paused)

    def test_npu_scale_down_recovers_releases_deferred_kv_then_applies(self):
        events = []
        scheduler = self.make_scheduler()
        model_runner = scheduler.tp_worker.model_runner
        model_runner.recover_npu_device_for_fault_tolerance_scale_down.side_effect = (
            lambda: events.append("recover_device")
        )
        model_runner.run_npu_fault_tolerance_dummy_batch.side_effect = (
            lambda active_mask: events.append(("dummy_batch", active_mask))
        )
        model_runner.synchronize_npu_fault_tolerance_health_gate.side_effect = (
            lambda: events.append("health_sync")
        )
        scheduler._rebuild_npu_fault_tolerance_control_runtime = lambda: (
            events.append("rebuild_control_runtime")
        )
        scheduler.forward_stream.wait_stream.side_effect = lambda stream: events.append(
            "forward_wait"
        )
        scheduler.schedule_stream.synchronize.side_effect = lambda: events.append(
            "schedule_sync"
        )
        model_runner.apply_fault_tolerance_scale_down.side_effect = lambda mask: (
            events.append(("scale_down", mask))
        )
        scheduler._ft_release_deferred_kv_cache.side_effect = lambda: events.append(
            "release_deferred_kv"
        )
        request = FaultToleranceCommandReqInput(
            request_id="s",
            command="scale_down",
            target_ranks=[1],
            active_mask=[True, False],
        )

        self.handle_command.__globals__["_is_npu"] = True
        try:
            with self.assertLogs(__name__, level="INFO") as captured:
                output = self.handle_command(scheduler, request)
        finally:
            self.handle_command.__globals__["_is_npu"] = False

        self.assertEqual(
            events,
            [
                "recover_device",
                "rebuild_control_runtime",
                ("scale_down", [True, False]),
                "release_deferred_kv",
                "schedule_sync",
                "forward_wait",
                ("dummy_batch", [True, False]),
                "health_sync",
            ],
        )
        self.assertEqual(output.message, "scaled down")
        log_text = "\n".join(captured.output)
        expected_log_steps = [
            "step=receive_scale_down phase=complete",
            "step=recover_device phase=begin",
            "step=recover_device phase=complete",
            "step=rebuild_control_runtime phase=begin",
            "step=rebuild_control_runtime phase=complete",
            "step=apply_elastic_scale_down phase=begin",
            "step=apply_elastic_scale_down phase=complete",
            "step=release_deferred_kv phase=begin",
            "step=release_deferred_kv phase=complete",
            "step=schedule_stream_synchronize phase=begin",
            "step=schedule_stream_synchronize phase=complete",
            "step=device_probe phase=skipped",
            "step=forward_stream_handoff phase=begin",
            "step=forward_stream_handoff phase=complete",
            "step=dummy_forward phase=begin",
            "step=dummy_forward phase=complete",
            "step=final_device_synchronize phase=begin",
            "step=final_device_synchronize phase=complete",
            "step=commit_unpause phase=begin",
            "step=commit_unpause phase=complete",
            "step=ack_ready phase=complete",
        ]
        positions = [log_text.index(step) for step in expected_log_steps]
        self.assertEqual(positions, sorted(positions))

    def test_retry_does_not_recover_device(self):
        state = SimpleNamespace(
            active_ranks=FakeTensor([1, 0]),
            active_ranks_cpu=FakeTensor([1, 0]),
            last_active_ranks=FakeTensor([1, 1]),
        )
        module = ModuleType("sglang.srt.elastic_ep.elastic_ep")
        module.ElasticEPStateManager = SimpleNamespace(instance=lambda: state)
        modules = {
            "sglang": ModuleType("sglang"),
            "sglang.srt": ModuleType("sglang.srt"),
            "sglang.srt.elastic_ep": ModuleType("sglang.srt.elastic_ep"),
            "sglang.srt.elastic_ep.elastic_ep": module,
        }
        scheduler = self.make_scheduler()
        request = FaultToleranceCommandReqInput(
            request_id="r", command="retry", target_ranks=[1], active_mask=None
        )

        self.handle_command.__globals__["_is_npu"] = True
        try:
            with patch.dict(sys.modules, modules):
                self.handle_command(scheduler, request)
        finally:
            self.handle_command.__globals__["_is_npu"] = False

        recover = scheduler.tp_worker.model_runner.recover_npu_device_for_fault_tolerance_scale_down
        recover.assert_not_called()

    def test_npu_scale_down_returns_failed_ack_when_deferred_kv_release_fails(self):
        scheduler = self.make_scheduler()
        scheduler._ft_release_deferred_kv_cache.side_effect = RuntimeError(
            "release failed"
        )
        request = FaultToleranceCommandReqInput(
            request_id="s",
            command="scale_down",
            target_ranks=[1],
            active_mask=[True, False],
        )

        self.handle_command.__globals__["_is_npu"] = True
        try:
            output = self.handle_command(scheduler, request)
        finally:
            self.handle_command.__globals__["_is_npu"] = False

        self.assertFalse(output.success)
        self.assertTrue(scheduler._engine_paused)
        apply_scale_down = (
            scheduler.tp_worker.model_runner.apply_fault_tolerance_scale_down
        )
        apply_scale_down.assert_called_once_with([True, False])
        scheduler.tp_worker.model_runner.run_npu_fault_tolerance_dummy_batch.assert_not_called()

    def test_npu_scale_down_returns_failed_ack_when_dummy_batch_fails(self):
        scheduler = self.make_scheduler()
        scheduler.tp_worker.model_runner.run_npu_fault_tolerance_dummy_batch.side_effect = (
            RuntimeError("507015")
        )
        request = FaultToleranceCommandReqInput(
            request_id="s",
            command="scale_down",
            target_ranks=[1],
            active_mask=[True, False],
        )

        self.handle_command.__globals__["_is_npu"] = True
        try:
            output = self.handle_command(scheduler, request)
        finally:
            self.handle_command.__globals__["_is_npu"] = False

        self.assertFalse(output.success)
        self.assertIn("507015", output.message)
        self.assertTrue(scheduler._engine_paused)

    def test_nonleader_executes_without_ack(self):
        scheduler = self.make_scheduler(leader=False)
        request = FaultToleranceCommandReqInput(
            request_id="s",
            command="scale_down",
            target_ranks=[1],
            active_mask=[True, False],
        )
        self.assertIsNone(self.handle_command(scheduler, request))
        apply_scale_down = (
            scheduler.tp_worker.model_runner.apply_fault_tolerance_scale_down
        )
        apply_scale_down.assert_called_once()

    def test_exception_self_pause_starts_deadline_before_reporting(self):
        events = []

        def dispatch(_):
            if not events:
                events.append("fault")
                raise RuntimeError("boom")
            raise KeyboardInterrupt()

        self.run_ft_loop.__globals__["dispatch_event_loop"] = dispatch
        sender = SimpleNamespace(send_output=lambda *_: events.append("report"))
        scheduler = SimpleNamespace(
            _ft_discard_inflight_window=lambda exc: events.append("discarded") or True,
            ipc_channels=SimpleNamespace(send_to_tokenizer=sender),
            ps=SimpleNamespace(dp_rank=0),
            server_args=SimpleNamespace(
                fault_tolerance_on_error_strategy="pause",
                fault_tolerance_pause_timeout=30,
            ),
            _engine_paused=False,
            _ft_pause_deadline=None,
        )
        with self.assertRaises(KeyboardInterrupt):
            self.run_ft_loop(scheduler)
        self.assertTrue(scheduler._engine_paused)
        self.assertEqual(scheduler._ft_pause_deadline, 130.0)
        self.assertEqual(events, ["fault", "discarded", "report"])

    def test_gpu_continue_discards_and_reenters_normal_loop(self):
        events = []

        def dispatch(_):
            if not events:
                events.append("fault")
                raise RuntimeError("boom")
            raise KeyboardInterrupt()

        self.run_ft_loop.__globals__["dispatch_event_loop"] = dispatch
        self.run_ft_loop.__globals__["_is_npu"] = False
        scheduler = SimpleNamespace(
            _ft_discard_inflight_window=lambda exc: events.append("discarded") or True,
            ipc_channels=SimpleNamespace(
                send_to_tokenizer=SimpleNamespace(
                    send_output=lambda *_: events.append("report")
                )
            ),
            ps=SimpleNamespace(dp_rank=0),
            server_args=SimpleNamespace(
                fault_tolerance_on_error_strategy="continue",
                fault_tolerance_pause_timeout=30,
            ),
            _engine_paused=False,
            _ft_pause_deadline=None,
        )

        with self.assertRaises(KeyboardInterrupt):
            self.run_ft_loop(scheduler)

        self.assertEqual(events, ["fault", "discarded", "report"])

    def test_npu_mc2_discards_host_state_but_defers_kv_release(self):
        events = []
        dispatched = False

        def dispatch(_):
            nonlocal dispatched
            if dispatched:
                raise KeyboardInterrupt()
            dispatched = True
            events.append("fault")
            raise RuntimeError("mlp-sync failed")

        self.run_ft_loop.__globals__["dispatch_event_loop"] = dispatch
        self.run_ft_loop.__globals__["_is_npu"] = True

        def discard(exc, *, defer_kv_release=False):
            events.append(("discarded", str(exc), defer_kv_release))
            return True

        scheduler = SimpleNamespace(
            _ft_discard_inflight_window=discard,
            ipc_channels=SimpleNamespace(
                send_to_tokenizer=SimpleNamespace(
                    send_output=lambda *_: events.append("report")
                )
            ),
            ps=SimpleNamespace(dp_rank=0),
            schedule_stream=object(),
            server_args=SimpleNamespace(
                elastic_ep_backend="mc2",
                fault_tolerance_on_error_strategy="pause",
                fault_tolerance_pause_timeout=30,
            ),
            _engine_paused=False,
            _ft_pause_deadline=None,
        )

        try:
            with self.assertRaises(KeyboardInterrupt):
                self.run_ft_loop(scheduler)
        finally:
            self.run_ft_loop.__globals__["_is_npu"] = False

        self.assertEqual(
            events,
            [
                "fault",
                "report",
                ("discarded", "mlp-sync failed", True),
            ],
        )
        self.assertTrue(scheduler._engine_paused)

    def test_npu_continue_strategy_is_normalized_during_startup(self):
        module = ModuleType("sglang.srt.fault_tolerance.controller")
        module.is_ft_supported_config = lambda server_args: (True, "")
        server_args = SimpleNamespace(
            enable_fault_tolerance=True,
            fault_tolerance_on_error_strategy="continue",
            elastic_ep_backend="mc2",
            fault_tolerance_communication_abort_timeout=10,
        )

        with patch.dict(
            sys.modules,
            {"sglang.srt.fault_tolerance.controller": module},
        ):
            with patch.dict(os.environ, {}, clear=True):
                self.handle_fault_tolerance_args(server_args)
                self.assertEqual(os.environ["TASK_QUEUE_ENABLE"], "0")
                self.assertEqual(os.environ["HCCL_EVENT_TIMEOUT"], "10")
                self.assertEqual(os.environ["HCCL_EXEC_TIMEOUT"], "9")
                self.assertEqual(os.environ["ACL_DEVICE_SYNC_TIMEOUT"], "10")
                self.assertEqual(os.environ["ACL_STREAM_TIMEOUT"], "10000")

        self.assertEqual(server_args.fault_tolerance_on_error_strategy, "pause")

    def test_fault_tolerance_reenters_shared_dispatch_loop(self):
        events = []
        dispatched = False

        def dispatch(_):
            nonlocal dispatched
            if not dispatched:
                dispatched = True
                events.append("fault")
                raise RuntimeError("boom")
            events.append("shared_dispatch")
            raise KeyboardInterrupt()

        self.run_ft_loop.__globals__["dispatch_event_loop"] = dispatch
        scheduler = SimpleNamespace(
            _ft_discard_inflight_window=lambda exc: events.append("discarded") or True,
            ipc_channels=SimpleNamespace(
                send_to_tokenizer=SimpleNamespace(
                    send_output=lambda *_: events.append("report")
                )
            ),
            ps=SimpleNamespace(dp_rank=0),
            server_args=SimpleNamespace(
                fault_tolerance_on_error_strategy="pause",
                fault_tolerance_pause_timeout=30,
            ),
            _engine_paused=False,
            _ft_pause_deadline=None,
        )

        self.run_ft_loop.__globals__["_is_npu"] = False
        with self.assertRaises(KeyboardInterrupt):
            self.run_ft_loop(scheduler)

        self.assertEqual(
            events,
            [
                "fault",
                "discarded",
                "report",
                "shared_dispatch",
            ],
        )

    def test_control_runtime_rebuild_preserves_streams_and_events(self):
        scheduler = self.make_scheduler()
        old_schedule_stream = scheduler.schedule_stream
        old_copy_stream = scheduler.copy_stream
        old_copy_stream_ctx = scheduler.copy_stream_ctx
        old_forward_stream = scheduler.forward_stream
        old_forward_stream_ctx = scheduler.forward_stream_ctx
        old_war_event = (
            scheduler.model_worker.war_fastpath_runner.war_fastpath_read_done_event
        )
        old_publish_event = scheduler.future_map.publish_ready
        decode_graph_runner = object()
        prefill_graph_runner = object()
        scheduler.tp_worker.model_runner.decode_cuda_graph_runner = decode_graph_runner
        scheduler.tp_worker.model_runner.prefill_cuda_graph_runner = (
            prefill_graph_runner
        )

        with self.assertLogs(__name__, level="INFO") as captured:
            self.rebuild_streams(scheduler)

        self.assertIs(scheduler.schedule_stream, old_schedule_stream)
        self.assertIs(scheduler.copy_stream, old_copy_stream)
        self.assertIs(scheduler.copy_stream_ctx, old_copy_stream_ctx)
        self.assertIs(scheduler.forward_stream, old_forward_stream)
        self.assertIs(scheduler.forward_stream_ctx, old_forward_stream_ctx)
        self.assertIs(
            scheduler.tp_worker.model_runner.forward_stream, old_forward_stream
        )
        self.assertIs(
            scheduler.tp_worker.model_runner.decode_cuda_graph_runner,
            decode_graph_runner,
        )
        self.assertIs(
            scheduler.tp_worker.model_runner.prefill_cuda_graph_runner,
            prefill_graph_runner,
        )
        self.assertIs(
            scheduler.model_worker.war_fastpath_runner.war_fastpath_read_done_event,
            old_war_event,
        )
        self.assertIs(scheduler.future_map.publish_ready, old_publish_event)
        self.assertTrue(scheduler.future_map._publish_fresh)
        scheduler.device_module.Stream.assert_not_called()
        log_text = "\n".join(captured.output)
        self.assertIn("step=rebuild_forward_stream phase=skipped", log_text)
        self.assertIn("step=rebuild_copy_stream phase=skipped", log_text)
        self.assertIn("step=rebuild_schedule_stream phase=skipped", log_text)
        self.assertIn("forward_stream=10", log_text)
        self.assertIn("step=reset_readiness_events phase=skipped", log_text)
        self.assertIn("reason=ablation", log_text)
        self.assertIn("graphs=preserved", log_text)

    def test_rebuild_preserves_long_lived_forward_stream_consumers(self):
        scheduler = self.make_scheduler()
        old_forward_stream = scheduler.forward_stream
        hisparse_coordinator = SimpleNamespace(set_decode_producer_stream=Mock())
        allocator = SimpleNamespace(
            forward_stream=old_forward_stream,
            full_attn_allocator=SimpleNamespace(forward_stream=old_forward_stream),
            mamba_allocator=SimpleNamespace(forward_stream=old_forward_stream),
            swa_attn_allocator=SimpleNamespace(forward_stream=old_forward_stream),
        )
        scheduler.hisparse_coordinator = hisparse_coordinator
        scheduler.enable_unified_memory = True
        scheduler.token_to_kv_pool_allocator = allocator

        self.rebuild_streams(scheduler)

        hisparse_coordinator.set_decode_producer_stream.assert_not_called()
        self.assertIs(scheduler.forward_stream, old_forward_stream)
        self.assertIs(
            scheduler.tp_worker.model_runner.forward_stream, old_forward_stream
        )
        self.assertIs(allocator.forward_stream, old_forward_stream)
        self.assertIs(
            allocator.full_attn_allocator.forward_stream, old_forward_stream
        )
        self.assertIs(
            allocator.mamba_allocator.forward_stream, old_forward_stream
        )
        self.assertIs(
            allocator.swa_attn_allocator.forward_stream, old_forward_stream
        )

    def test_npu_scale_down_restarts_without_artificial_delay(self):
        calls = []
        npu = SimpleNamespace(
            current_device=lambda: 3,
            stop_device=lambda device_id: calls.append(("stop", device_id)) or 0,
            restart_device=lambda device_id: calls.append(("restart", device_id)) or 0,
        )
        torch_npu = ModuleType("torch_npu")
        torch_npu.npu = npu
        torch_npu._C = SimpleNamespace()
        torch_npu.distributed = SimpleNamespace(
            reinit_process_group=lambda *args: calls.append(("reinit", *args))
        )
        runner = SimpleNamespace(gpu_id=3, ps=SimpleNamespace(dp_rank=3))
        self.fake_torch.npu.set_device = lambda device: calls.append(("set", device))
        self.fake_torch.npu.synchronize = lambda: calls.append(("synchronize",))

        with patch.dict(sys.modules, {"torch_npu": torch_npu}):
            with self.assertLogs(__name__, level="INFO") as captured:
                self.recover_npu(runner)

        self.assertEqual(
            calls,
            [
                ("set", ("npu", 3)),
                ("stop", 3),
                ("restart", 3),
                ("reinit", None, False),
                ("synchronize",),
            ],
        )
        log_text = "\n".join(captured.output)
        expected_log_steps = [
            "step=bind_device phase=begin",
            "step=bind_device phase=complete",
            "step=stop_device phase=begin",
            "step=stop_device phase=complete",
            "step=recover_torch_npu_stream_pool phase=skipped",
            "step=restart_device phase=begin",
            "step=restart_device phase=complete",
            "step=reinit_process_group phase=begin",
            "step=reinit_process_group phase=complete",
            "step=post_reinit_synchronize phase=begin",
            "step=post_reinit_synchronize phase=complete",
        ]
        positions = [log_text.index(step) for step in expected_log_steps]
        self.assertEqual(positions, sorted(positions))

    def test_npu_scale_down_rebalances_before_updating_mc2_elastic_info(self):
        path = REPO_ROOT / "python/sglang/srt/model_executor/model_runner.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        model_runner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
        )
        method = next(
            node
            for node in model_runner.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "apply_fault_tolerance_scale_down"
        )
        npu_branch = next(
            node
            for node in method.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "is_npu_ft_mc2"
            and "update_npu_mc2_elastic_info" in ast.unparse(node)
        )
        branch_calls = [
            node
            for statement in npu_branch.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
        ]
        rebalance = next(
            call
            for call in branch_calls
            if isinstance(call.func, ast.Attribute)
            and call.func.attr == "rebalance"
        )
        update_elastic_info = next(
            call
            for call in branch_calls
            if isinstance(call.func, ast.Attribute)
            and call.func.attr == "update_npu_mc2_elastic_info"
        )

        self.assertLess(rebalance.lineno, update_elastic_info.lineno)
        self.assertTrue(
            any(
                keyword.arg == "force"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in rebalance.keywords
            )
        )

    def test_npu_dummy_uses_normal_model_runner_dispatch(self):
        output = SimpleNamespace(can_run_graph=True)
        decode_graph = object()
        prefill_graph = object()
        eager_runner = SimpleNamespace(
            run_dummy_via_model_runner=Mock(return_value=output)
        )
        runner = SimpleNamespace(
            ps=SimpleNamespace(dp_rank=1),
            eager_runner=eager_runner,
            decode_cuda_graph_runner=decode_graph,
            prefill_cuda_graph_runner=prefill_graph,
        )

        self.run_npu_dummy(runner, [False, True, True, True])

        eager_runner.run_dummy_via_model_runner.assert_called_once_with(
            batch_size=1,
            active_mask=[False, True, True, True],
        )
        self.assertIs(runner.decode_cuda_graph_runner, decode_graph)
        self.assertIs(runner.prefill_cuda_graph_runner, prefill_graph)

    def test_dummy_dispatch_calls_model_runner_forward(self):
        class FillableBuffer:
            def __init__(self):
                self.value = None

            def __getitem__(self, _key):
                return self

            def fill_(self, value):
                self.value = value

        forward_batch = SimpleNamespace(
            dp_local_start_pos=object(),
            dp_local_num_tokens=object(),
            dp_padding_mode=SimpleNamespace(is_max_len=lambda: False),
        )
        pp_proxy_tensors = object()
        output = SimpleNamespace(can_run_graph=True)
        decode_graph = object()
        prefill_graph = object()
        model_runner = SimpleNamespace(
            forward=Mock(return_value=output),
            decode_cuda_graph_runner=decode_graph,
            prefill_cuda_graph_runner=prefill_graph,
        )
        seq_lens = FillableBuffer()
        seq_lens_cpu = FillableBuffer()
        buffers = SimpleNamespace(
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
        )
        runner = SimpleNamespace(
            model_runner=model_runner,
            _alloc_dummy_decode_buffers=Mock(return_value=buffers),
            _prepare_dummy_forward_batch=Mock(
                return_value=(forward_batch, pp_proxy_tensors, 3, 1, [0, 1, 1, 1])
            ),
        )

        result = self.run_dummy_dispatch(runner, 1, [False, True, True, True])

        self.assertIs(result, output)
        self.assertEqual(seq_lens.value, 1)
        self.assertEqual(seq_lens_cpu.value, 1)
        model_runner.forward.assert_called_once_with(
            forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        runner._prepare_dummy_forward_batch.assert_called_once_with(
            1,
            buffers=buffers,
            active_mask=[False, True, True, True],
        )
        self.assertIs(model_runner.decode_cuda_graph_runner, decode_graph)
        self.assertIs(model_runner.prefill_cuda_graph_runner, prefill_graph)

    def test_pause_deadline_notifies_node_main_once(self):
        notify = self.check_deadline.__globals__["notify_node_main_process_failure"]
        notify.reset_mock()
        scheduler = SimpleNamespace(
            _ft_pause_deadline=100.0,
            server_args=SimpleNamespace(fault_tolerance_pause_timeout=30),
            ps=SimpleNamespace(dp_rank=0),
        )
        self.check_deadline(scheduler)
        self.check_deadline(scheduler)
        notify.assert_called_once_with()

    def test_exception_discards_overlap_window_once_per_request(self):
        shared = FakeReq("shared")
        current = FakeReq(
            "current", origin_input_ids=list(range(10)), output_ids=[10], committed=12
        )
        previous = FakeBatch([shared])
        running = FakeBatch([shared, current])
        sender = Sender()
        released = []
        self.discard.__globals__["release_kv_cache"] = lambda req, *args, **kwargs: (
            released.append(req.rid)
        )
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(dp_rank=1),
            cur_batch_for_debug=FakeBatch([shared, current]),
            last_batch=previous,
            result_queue=deque([(previous, object())]),
            running_batch=running,
            chunked_req=current,
            tree_cache=object(),
            ipc_channels=SimpleNamespace(send_to_tokenizer=sender),
        )

        self.assertTrue(self.discard(scheduler, RuntimeError("boom")))

        self.assertCountEqual(released, ["shared", "current"])
        self.assertEqual(current.kv_committed_len, 11)
        self.assertEqual(scheduler.running_batch.reqs, [])
        self.assertEqual(scheduler.result_queue, deque())

    def test_deferred_kv_release_runs_only_after_explicit_recovery_step(self):
        req = FakeReq(
            "current", origin_input_ids=list(range(10)), output_ids=[10], committed=12
        )
        running = FakeBatch([req])
        sender = Sender()
        release = Mock()
        self.discard.__globals__["release_kv_cache"] = release
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(dp_rank=1),
            cur_batch_for_debug=running,
            last_batch=running,
            result_queue=deque(),
            running_batch=running,
            chunked_req=None,
            tree_cache=object(),
            ipc_channels=SimpleNamespace(send_to_tokenizer=sender),
        )

        self.assertTrue(
            self.discard(
                scheduler,
                RuntimeError("mlp-sync failed"),
                defer_kv_release=True,
            )
        )

        release.assert_not_called()
        self.assertEqual(
            list(scheduler._ft_deferred_kv_release_reqs),
            ["current"],
        )
        self.assertEqual(scheduler.running_batch.reqs, [])

        self.release_deferred_kv(scheduler)

        release.assert_called_once_with(
            req,
            scheduler.tree_cache,
            is_insert=False,
            allow_non_spec_overallocated=True,
        )
        self.assertEqual(scheduler._ft_deferred_kv_release_reqs, {})

    def test_failed_discard_keeps_inflight_window_for_diagnosis(self):
        req = FakeReq("current")
        running = FakeBatch([req])
        self.discard.__globals__["release_kv_cache"] = Mock(
            side_effect=RuntimeError("release failed")
        )
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(dp_rank=1),
            cur_batch_for_debug=running,
            last_batch=running,
            result_queue=deque(),
            running_batch=running,
            chunked_req=None,
            tree_cache=object(),
            ipc_channels=SimpleNamespace(send_to_tokenizer=Sender()),
        )

        self.assertFalse(self.discard(scheduler, RuntimeError("boom")))
        self.assertIs(scheduler.running_batch, running)
        self.assertEqual(scheduler.running_batch.reqs, [req])

    def test_discard_fails_when_post_release_pool_invariant_fails(self):
        running = FakeBatch([])
        checker = SimpleNamespace(
            _check_all_pools=Mock(return_value=(True, ["missing one page"]))
        )
        observer = SimpleNamespace(get_pool_stats=Mock(return_value=object()))
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(dp_rank=1),
            cur_batch_for_debug=running,
            last_batch=running,
            result_queue=deque(),
            running_batch=running,
            chunked_req=None,
            tree_cache=object(),
            ipc_channels=SimpleNamespace(send_to_tokenizer=Sender()),
            invariant_checker=checker,
            pool_stats_observer=observer,
        )

        self.assertFalse(self.discard(scheduler, RuntimeError("boom")))
        checker._check_all_pools.assert_called_once()

    def make_dpc(self):
        sender = Sender()
        context = FakeContext(sender)
        self.dpc_globals["zmq"].Context = lambda: context
        dpc = SimpleNamespace(
            workers=[Sender(), Sender()],
            scheduler_procs=[],
            scheduler_process_dp_ranks=[0, 1, 1],
            scheduler_process_global_ranks=[0, 2, 3],
            server_args=SimpleNamespace(node_rank=1),
            port_args=SimpleNamespace(tokenizer_ipc_name="tcp://node0:1"),
            send_to_tokenizer=Sender(),
            ft_control_endpoint="tcp://node1:2",
            _watchdog_context=None,
            _watchdog_sender=None,
        )
        dpc._get_watchdog_sender = lambda: self.dpc_methods["_get_watchdog_sender"](dpc)
        dpc._watchdog_heartbeat = lambda: self.dpc_methods["_watchdog_heartbeat"](dpc)
        return dpc, sender

    def test_watchdog_reports_global_rank_and_endpoint(self):
        dpc, sender = self.make_dpc()
        proc = SimpleNamespace(pid=123)
        self.dpc_methods["_handle_scheduler_process_exit"](dpc, 1, proc, "scheduler")
        self.dpc_methods["_report_watchdog_heartbeat"](dpc)

        down, heartbeat = sender.sent
        self.assertEqual(down.ranks, [2])
        self.assertEqual(heartbeat.ranks, [0, 2, 3])
        self.assertEqual(heartbeat.control_endpoint, "tcp://node1:2")

    def test_shutdown_kills_every_local_member_of_target_dp(self):
        dpc, _ = self.make_dpc()
        dpc.scheduler_procs = [
            SimpleNamespace(is_alive=lambda: True, kill=Mock()) for _ in range(3)
        ]
        request = FaultToleranceDPCShutdownReqInput(target_dp_ranks=[1])

        self.dpc_methods["shutdown_dp"](dpc, request)

        dpc.scheduler_procs[0].kill.assert_not_called()
        dpc.scheduler_procs[1].kill.assert_called_once()
        dpc.scheduler_procs[2].kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
