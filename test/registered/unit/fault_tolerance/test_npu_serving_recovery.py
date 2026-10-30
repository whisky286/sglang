"""Control transaction and failure gates for original-cohort NPU recovery."""

import ast
import asyncio
import copy
import importlib.util
import io
import json
import logging
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch
from sglang.srt.fault_tolerance.manager import FaultToleranceManager
from sglang.srt.fault_tolerance.npu_recovery import (
    RecoveryGroups,
    recovery_config_error,
    validate_health_output,
)
from sglang.srt.fault_tolerance.recovery import RankRecoveryPlan
from sglang.srt.managers.io_struct import (
    FaultToleranceCommandReqInput,
    FaultToleranceCommandReqOutput,
    FaultToleranceDPCRecoverReqInput,
    ProcessActiveRanksOutput,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4]


def load_method(path, class_name, method_name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node for node in cls.body if getattr(node, "name", None) == method_name
    )
    method.decorator_list = []
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        method,
                    ],
                    type_ignores=[],
                )
            ),
            path,
            "exec",
        ),
        namespace,
    )
    return namespace[method_name]


def make_manager():
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
    )
    manager.bind_event_loop(asyncio.get_running_loop())
    manager.state.expected_dp_mask = [True, True, False, True]
    manager.state.process_alive_global_rank_mask[2] = False
    manager.state.membership_epoch = 1
    manager._route_dp_mask = [True, True, False, True]
    manager._watchdog_leases[0] = (0, (0, 1, 2, 3))
    return manager


class TestNpuRecoveryControl(unittest.IsolatedAsyncioTestCase, CustomTestCase):
    async def test_routes_commit_only_after_all_member_validation(self):
        manager = make_manager()
        events = []

        async def command(**kwargs):
            self.assertTrue(manager.should_reject_admission())
            self.assertFalse(manager.state.expected_dp_mask[2])
            events.append(kwargs["command"])
            if kwargs["command"] == "recover_prepare":
                self.assertEqual(
                    kwargs["previous_active_mask"], [True, True, False, True]
                )
                self.assertEqual(kwargs["target_ranks"], [0, 1, 3])
            if kwargs["command"] in ("recover_validate", "recover_commit"):
                self.assertEqual(kwargs["target_ranks"], [0, 1, 2, 3])
            self.assertEqual(kwargs["membership_epoch"], 2)

        async def publish(mask, timeout):
            self.assertIn("recover_validate", events)
            self.assertNotIn("recover_commit", events)
            self.assertEqual(mask, [True] * 4)
            events.append("route")
            manager._route_dp_mask = list(mask)

        manager._send_command_collect = AsyncMock(side_effect=command)
        manager._publish_route_dp_mask = AsyncMock(side_effect=publish)
        status, body = await manager.apply(
            {"instruction": "recover", "params": {"ranks": [2]}}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["recovered_ranks"], [2])
        self.assertEqual(
            events,
            [
                "recover_pause",
                "recover_prepare",
                "recover_validate",
                "route",
                "recover_commit",
            ],
        )
        self.assertFalse(manager.should_reject_admission())
        self.assertTrue(all(manager.state.expected_dp_mask))
        self.assertEqual(manager._process_event_floor_epoch, 2)
        request = manager.send_to_dpc.await_args.args[1]
        self.assertEqual(request.membership_epoch, 2)
        self.assertEqual(request.target_dp_ranks, [2])

    async def test_failed_health_gate_does_not_publish_or_commit(self):
        manager = make_manager()

        async def command(**kwargs):
            if kwargs["command"] == "recover_validate":
                raise RuntimeError("dummy forward failed")

        manager._send_command_collect = AsyncMock(side_effect=command)
        manager._publish_route_dp_mask = AsyncMock()
        manager._failstop = Mock(side_effect=RuntimeError("fail-stop"))
        with self.assertRaisesRegex(RuntimeError, "fail-stop"):
            await manager.apply({"instruction": "recover", "params": {"ranks": [2]}})
        manager._publish_route_dp_mask.assert_not_awaited()
        self.assertTrue(manager.should_reject_admission())
        self.assertFalse(manager.state.expected_dp_mask[2])
        self.assertNotIn(
            "recover_commit",
            [
                call.kwargs["command"]
                for call in manager._send_command_collect.await_args_list
            ],
        )

    async def test_rejects_disabled_partial_or_unhealthy_recovery_before_pause(self):
        for mode in ("disabled", "partial", "healthy_rank", "unhealthy", "no_watchdog"):
            with self.subTest(mode=mode):
                manager = make_manager()
                ranks = [2]
                if mode == "disabled":
                    manager.server_args.enable_npu_ft_recovery = False
                elif mode == "partial":
                    manager.state.expected_dp_mask[3] = False
                elif mode == "healthy_rank":
                    ranks = [1]
                elif mode == "unhealthy":
                    manager.state.unhealthy_dp_ranks.add(0)
                else:
                    manager._watchdog_leases.clear()
                manager._send_command_collect = AsyncMock()
                code, body = await manager.apply(
                    {"instruction": "recover", "params": {"ranks": ranks}}
                )
                self.assertIn(code, (400, 409))
                self.assertFalse(body["success"])
                manager._send_command_collect.assert_not_awaited()

    async def test_exit_after_health_ack_cannot_be_overwritten_by_finish(self):
        manager = make_manager()

        async def command(**kwargs):
            if kwargs["command"] == "recover_commit":
                manager.observe_process_active_ranks(
                    ProcessActiveRanksOutput(
                        ranks=[2], active=False, membership_epoch=2
                    )
                )

        manager._send_command_collect = AsyncMock(side_effect=command)
        manager._publish_route_dp_mask = AsyncMock()
        manager._failstop = Mock(side_effect=RuntimeError("fail-stop"))
        with self.assertRaisesRegex(RuntimeError, "fail-stop"):
            await manager.apply({"instruction": "recover", "params": {"ranks": [2]}})
        self.assertFalse(manager.state.expected_dp_mask[2])
        self.assertTrue(manager.state.ft_operation_in_progress)

    async def test_npu_process_and_native_ready_do_not_auto_open_route(self):
        from sglang.srt.managers.io_struct import ActiveRanksOutput

        manager = make_manager()
        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[2], active=True)
        )
        result = manager.observe_active_ranks(ActiveRanksOutput(status=[True] * 4))
        self.assertIsNone(result)
        self.assertFalse(manager.state.expected_dp_mask[2])
        self.assertFalse(manager._route_dp_mask[2])

    async def test_old_incarnation_exit_cannot_kill_restored_rank(self):
        manager = make_manager()
        manager._process_event_floor_epoch = 2
        manager.state.process_alive_global_rank_mask[2] = True
        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[2], active=False, membership_epoch=0)
        )
        self.assertTrue(manager.state.process_alive_global_rank_mask[2])
        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[2], active=False, membership_epoch=2)
        )
        self.assertFalse(manager.state.process_alive_global_rank_mask[2])

    async def test_malformed_requests_do_not_start_a_transaction(self):
        for params in (
            {"ranks": [2, 2]},
            {"ranks": [True]},
            {"ranks": [4]},
            {"ranks": [2], "timeout": float("nan")},
            {"ranks": [2], "timeout": True},
        ):
            with self.subTest(params=params):
                manager = make_manager()
                code, _ = await manager.apply(
                    {"instruction": "recover", "params": params}
                )
                self.assertEqual(code, 400)
                self.assertFalse(manager.state.ft_operation_in_progress)
                manager.send_to_dpc.assert_not_awaited()


class TestNpuRecoveryRuntime(CustomTestCase):
    def test_dpc_control_requests_reach_each_local_broadcast_leader(self):
        path = "python/sglang/srt/managers/data_parallel_controller.py"
        tree = ast.parse((ROOT / path).read_text())
        cls = next(
            node
            for node in tree.body
            if getattr(node, "name", None) == "DataParallelController"
        )
        init = next(
            node for node in cls.body if getattr(node, "name", None) == "__init__"
        )
        # Execute the actual launch/control setup branch without starting workers.
        setup = next(
            node
            for node in init.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Attribute)
            and node.test.attr == "enable_dp_attention"
        )
        send = Mock()
        forward = load_method(
            path, "DataParallelController", "send_control_message", {"sock_send": send}
        )
        for dp_attention, local, joiner, ft in (
            (True, False, False, False),
            (True, True, False, False),
            (True, False, True, False),
            (True, False, False, True),
            (False, False, False, False),
        ):
            with self.subTest(
                dp_attention=dp_attention, local=local, joiner=joiner, ft=ft
            ):
                args = SimpleNamespace(
                    enable_dp_attention=dp_attention,
                    enable_dp_attention_local_control_broadcast=local,
                    is_ep_scale_joiner=joiner,
                    enable_fault_tolerance=ft,
                    tp_size=4,
                )
                controller = SimpleNamespace(
                    launch_dp_attention_schedulers=Mock(),
                    launch_dp_schedulers=Mock(),
                    workers=[Mock() for _ in range(4)],
                )
                exec(
                    compile(ast.Module(body=[setup], type_ignores=[]), path, "exec"),
                    {"self": controller, "server_args": args, "port_args": Mock()},
                )
                for active in ([0, 1, 2, 3], [0, 1, 3]):
                    send.reset_mock()
                    controller._active_workers = active
                    request = Mock()
                    forward(controller, request)
                    expected = (
                        active if not dp_attention or local or joiner or ft else [0]
                    )
                    self.assertEqual(
                        [call.args[0] for call in send.call_args_list],
                        [controller.workers[rank] for rank in expected],
                    )

    def test_health_gate_rejects_missing_nan_or_infinite_logits(self):
        for logits in (
            None,
            torch.empty(0),
            torch.tensor([float("nan")]),
            torch.tensor([float("inf")]),
        ):
            with self.subTest(logits=logits):
                with self.assertRaisesRegex(RuntimeError, "logits"):
                    validate_health_output(
                        SimpleNamespace(
                            logits_output=SimpleNamespace(next_token_logits=logits)
                        )
                    )
        validate_health_output(
            SimpleNamespace(
                logits_output=SimpleNamespace(
                    next_token_logits=torch.tensor([1.0, -2.0])
                )
            )
        )

    def test_supported_config_and_each_required_gate(self):
        args = SimpleNamespace(
            device="npu",
            enable_fault_tolerance=True,
            elastic_ep_backend="mc2",
            nnodes=1,
            tp_size=4,
            dp_size=4,
            ep_size=4,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            moe_a2a_backend="deepep",
            deepep_mode="low_latency",
        )
        with patch.dict(
            os.environ,
            {
                "DEEP_USE_MODE": "default",
                "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "0",
                "SGLANG_USE_MESSAGE_QUEUE_BROADCASTER": "0",
            },
        ):
            self.assertIsNone(recovery_config_error(args))
            for field, value in {
                "nnodes": 2,
                "ep_size": 2,
                "disable_cuda_graph": False,
                "disable_overlap_schedule": False,
                "deepep_mode": "auto",
                "quantization": "modelslim",
                "pre_warm_nccl": True,
                "tokenizer_worker_num": 2,
                "ep_dispatch_algorithm": "lp",
            }.items():
                with (
                    self.subTest(field=field),
                    patch.object(args, field, value, create=True),
                ):
                    self.assertIsNotNone(recovery_config_error(args))

    def test_group_keys_are_epoch_scoped_and_global_rank_mapping_is_preserved(self):
        from torch.distributed.distributed_c10d import _world

        plan = RankRecoveryPlan(2, (True, True, False, True), (2,))
        context = RecoveryGroups(Mock(), plan, 2, 30)
        group = Mock()
        with (
            patch(
                "sglang.srt.utils.init_custom_process_group", return_value=group
            ) as create,
            patch("torch.distributed.PrefixStore"),
            patch.dict(_world.pg_group_ranks),
        ):
            self.assertIs(context.group("attention_tp", [2], "gloo"), group)
            self.assertEqual(_world.pg_group_ranks[group], {2: 0})
            self.assertIs(context.group("attention_tp", [2], "gloo"), group)
            create.assert_called_once()
            self.assertIn(
                "npu-ft_1111_2_restore", create.call_args.kwargs["group_name"]
            )

    def test_commit_requires_validation_for_the_same_epoch(self):
        namespace = {
            "FaultToleranceCommandReqOutput": FaultToleranceCommandReqOutput,
            "logger": logging.getLogger(__name__),
        }
        method = load_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "_handle_npu_recovery_command",
            namespace,
        )
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(enable_npu_ft_recovery=True, override=Mock()),
            ps=SimpleNamespace(dp_rank=0),
            _engine_paused=True,
            _ft_recovery_epoch=2,
            _ft_recovery_validated=False,
        )
        for validated, epoch, success in (
            (False, 2, False),
            (True, 1, False),
            (True, 2, True),
        ):
            with self.subTest(validated=validated, epoch=epoch):
                scheduler._ft_recovery_validated = validated
                response = method(
                    scheduler,
                    FaultToleranceCommandReqInput(
                        request_id="commit",
                        command="recover_commit",
                        target_ranks=[0],
                        membership_epoch=epoch,
                    ),
                )
                self.assertEqual(response.success, success)
                self.assertEqual(scheduler._engine_paused, not success)

    def test_dpc_startup_poll_does_not_block_control_forwarding(self):
        sender = Mock()
        namespace = {
            "time": SimpleNamespace(monotonic=lambda: 10),
            "sock_send": sender,
            "ProcessActiveRanksOutput": ProcessActiveRanksOutput,
        }
        method = load_method(
            "python/sglang/srt/managers/data_parallel_controller.py",
            "DataParallelController",
            "poll_npu_recovery_startup",
            namespace,
        )
        reader = Mock()
        reader.poll.return_value = False
        controller = SimpleNamespace(
            _npu_recovery_readers={2: (reader, 20)},
            send_to_tokenizer=Mock(),
            _watchdog_event_epoch=2,
        )
        method(controller)
        reader.recv.assert_not_called()
        sender.assert_not_called()
        reader.poll.return_value = True
        reader.recv.return_value = {"status": "ready"}
        method(controller)
        self.assertEqual(controller._npu_recovery_readers, {})
        self.assertEqual(sender.call_args.args[1].membership_epoch, 2)

    def test_dpc_replaces_only_dead_rank_and_restarts_watchdog(self):
        class Args(SimpleNamespace):
            def override(self, source, **kwargs):
                self.__dict__.update(kwargs)

        args = Args(enable_npu_ft_recovery=True, fault_tolerance_timeout=30)
        old = [Mock() for _ in range(4)]
        for i, process in enumerate(old):
            process.is_alive.return_value = i != 2
        reader, writer, replacement, watchdog = Mock(), Mock(), Mock(), Mock()
        process_factory, watchdog_factory = (
            Mock(return_value=replacement),
            Mock(return_value=watchdog),
        )
        send = Mock()
        namespace = {
            "copy": copy,
            "mp": SimpleNamespace(
                Pipe=Mock(return_value=(reader, writer)), Process=process_factory
            ),
            "time": SimpleNamespace(monotonic=lambda: 10),
            "numa_utils": SimpleNamespace(
                configure_subprocess=lambda *args: nullcontext()
            ),
            "SubprocessWatchdog": watchdog_factory,
            "FT_WATCHDOG_POLL_INTERVAL": 1,
        }
        method = load_method(
            "python/sglang/srt/managers/data_parallel_controller.py",
            "DataParallelController",
            "recover_dp",
            namespace,
        )
        controller = SimpleNamespace(
            server_args=args,
            port_args=SimpleNamespace(
                fault_tolerance_metadata_ipc_name="tcp://127.0.0.1:1234"
            ),
            scheduler_process_dp_ranks=list(range(4)),
            scheduler_process_global_ranks=list(range(4)),
            scheduler_procs=list(old),
            _scheduler_launch_specs={
                2: (args, Mock(), 2, 2, 0, 0, 2, 0, 2, Mock(), 2, 2, 2)
            },
            _scheduler_watchdog=Mock(),
            _npu_recovery_readers={},
            run_scheduler_process_func=Mock(),
            _handle_scheduler_process_exit=Mock(),
            _report_watchdog_heartbeat=Mock(),
            _close_watchdog_sender=Mock(),
            _report_initial_watchdog_heartbeat=send,
        )
        request = FaultToleranceDPCRecoverReqInput(
            target_dp_ranks=[2],
            membership_epoch=2,
            previous_active_mask=[True, True, False, True],
        )
        # A live process is rejected before touching the watchdog or handles.
        old[2].is_alive.return_value = True
        with self.assertRaisesRegex(ValueError, "live scheduler"):
            method(controller, request)
        process_factory.assert_not_called()
        old[2].is_alive.return_value = False
        method(controller, request)
        self.assertIs(controller.scheduler_procs[2], replacement)
        for rank in (0, 1, 3):
            self.assertIs(controller.scheduler_procs[rank], old[rank])
        new_args = process_factory.call_args.kwargs["args"]
        self.assertEqual(new_args[0]._npu_ft_recovery["plan"]["membership_epoch"], 2)
        self.assertEqual(new_args[2:4], (2, 2))
        self.assertIs(new_args[9], writer)
        replacement.start.assert_called_once()
        reader.recv.assert_not_called()
        watchdog.start.assert_called_once()
        self.assertEqual(controller._watchdog_event_epoch, 2)
        self.assertIn(2, controller._npu_recovery_readers)

    def test_buffer_retirement_updates_dispatcher_and_drops_old_handles(self):
        class Dispatcher:
            pass

        dispatcher = Dispatcher()
        old_group, new_group = Mock(), Mock()
        impl = SimpleNamespace(group=old_group, handle=Mock(), packed_recv_count=Mock())
        dispatcher._low_latency_dispatcher = impl
        dispatcher._stage = "in-flight"
        dispatcher._dispatch_intermediate_state = (Mock(),)
        state = SimpleNamespace(buffer=Mock(), dispatch_mode="low_latency")
        old_buffer = state.buffer
        owner = SimpleNamespace(_state=lambda: state)
        namespace = {
            "_is_npu": True,
            "DeepEPDispatcher": Dispatcher,
            "_Stage": SimpleNamespace(INITIAL="initial"),
        }
        method = load_method(
            "python/sglang/srt/layers/moe/token_dispatcher/deepep.py",
            "DeepEPBuffer",
            "retire_for_npu_recovery",
            namespace,
        )
        with patch(
            "sglang.srt.distributed.parallel_state.get_moe_ep_group",
            return_value=SimpleNamespace(device_group=new_group),
        ):
            retired = []
            method(
                owner,
                SimpleNamespace(
                    modules=lambda: [
                        SimpleNamespace(
                            dispatcher=SimpleNamespace(_inners=[dispatcher])
                        )
                    ]
                ),
                retired,
            )
        self.assertIsNone(state.buffer)
        self.assertEqual(retired, [old_buffer])
        self.assertIs(impl.group, new_group)
        self.assertIsNone(impl.handle)
        self.assertFalse(hasattr(dispatcher, "_dispatch_intermediate_state"))
        self.assertEqual(dispatcher._stage, "initial")


class TestNpuRecoveryProbeDiagnostics(CustomTestCase):
    def load_probe(self):
        path = ROOT / "test/manual/ep/verify_npu_recovery_serving.py"
        spec = importlib.util.spec_from_file_location(
            "npu_recovery_serving_probe", path
        )
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        return probe

    def test_scale_down_only_supports_old_server_and_rejects_failed_generation(self):
        for mode in (
            "success",
            "generation_failure",
            "replacement",
            "unhealthy",
            "single_rank",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                probe = self.load_probe()
                alive = {
                    rank: SimpleNamespace(
                        pid=100 + rank,
                        create_time=Mock(return_value=10 + rank),
                        kill=Mock(),
                    )
                    for rank in range(4)
                }
                victim = alive[2]
                victim.kill.side_effect = lambda: alive.pop(2)
                commands = []
                paths = []

                def request(method, url, **kwargs):
                    path = probe.urlsplit(url).path
                    paths.append(path)
                    if path == "/fault_tolerance/status":
                        result = {
                            "ranks": [
                                {
                                    "rank": rank,
                                    "state": (
                                        "unhealthy"
                                        if mode == "unhealthy" and rank == 0
                                        else "healthy"
                                        if rank in alive
                                        else "dead"
                                    ),
                                }
                                for rank in range(1 if mode == "single_rank" else 4)
                            ]
                        }
                    elif path == "/fault_tolerance/apply":
                        commands.append(kwargs["json"]["instruction"])
                        self.assertEqual(commands[-1], "scale_down")
                        result = {"success": True}
                    elif path == "/generate":
                        if commands:
                            self.assertEqual(kwargs["json"]["routed_dp_rank"], 0)
                            if mode == "generation_failure":
                                raise TimeoutError("survivor forward timed out")
                            if mode == "replacement":
                                alive[0] = SimpleNamespace(
                                    pid=999, create_time=Mock(return_value=99)
                                )
                        result = {
                            "text": "Paris",
                            "meta_info": {"completion_tokens": 1},
                        }
                    else:
                        raise AssertionError(f"unexpected old-server endpoint: {path}")
                    return SimpleNamespace(
                        status_code=200,
                        raise_for_status=Mock(),
                        json=lambda: result,
                    )

                session = SimpleNamespace(
                    headers={},
                    get=Mock(return_value=SimpleNamespace(status_code=200)),
                    request=Mock(side_effect=request),
                )
                report_path = Path(directory) / "report.json"
                with (
                    patch.dict(
                        sys.modules,
                        {
                            "psutil": SimpleNamespace(pid_exists=lambda _: True),
                            "requests": SimpleNamespace(
                                Session=lambda: session, RequestException=TimeoutError
                            ),
                        },
                    ),
                    patch.object(
                        probe, "schedulers", side_effect=lambda _: dict(alive)
                    ),
                    patch.object(
                        probe.importlib.metadata, "version", return_value="test"
                    ),
                    patch.object(
                        probe.subprocess, "check_output", return_value="46147ef9bc"
                    ),
                    patch.object(
                        sys,
                        "argv",
                        [
                            "probe",
                            "--server-pid",
                            "123",
                            "--scale-down-only",
                            "--cycles",
                            "1",
                            "--output",
                            str(report_path),
                        ],
                    ),
                    redirect_stdout(io.StringIO()),
                ):
                    if mode == "success":
                        probe.main()
                    else:
                        error = {
                            "generation_failure": "survivor forward timed out",
                            "replacement": "survivor scheduler identities changed",
                            "unhealthy": "requires all ranks healthy",
                            "single_rank": "requires at least two ranks",
                        }[mode]
                        with self.assertRaisesRegex(
                            (RuntimeError, TimeoutError, ValueError), error
                        ):
                            probe.main()
                report = json.loads(report_path.read_text())
                self.assertEqual(report["mode"], "scale_down_only")
                self.assertEqual(report["success"], mode == "success")
                self.assertEqual(report["git_commit"], "46147ef9bc")
                self.assertNotIn("/server_info", paths)
                self.assertEqual(
                    commands,
                    [] if mode in ("unhealthy", "single_rank") else ["scale_down"],
                )
                if mode == "success":
                    self.assertIn("rank recovery not exercised", report["scope"])
                    self.assertEqual(
                        report["events"][-1]["stage"], "scale_down_verified"
                    )
                    victim.kill.assert_called_once()
                if mode in ("unhealthy", "single_rank"):
                    victim.kill.assert_not_called()

    def test_scale_down_only_rejects_multiple_cycles_before_injection(self):
        probe = self.load_probe()
        with (
            patch.object(
                sys, "argv", ["probe", "--server-pid", "123", "--scale-down-only"]
            ),
            patch.object(probe, "schedulers") as schedulers,
            redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as result:
                probe.main()
        self.assertEqual(result.exception.code, 2)
        schedulers.assert_not_called()

    def test_logs_request_before_blocking_and_propagates_failure_without_retry(self):
        probe = self.load_probe()
        events = []

        def event(stage, **details):
            events.append({"stage": stage, **details})

        def blocked(*args, **kwargs):
            self.assertEqual(events[-1]["stage"], "http_request_started")
            self.assertEqual(kwargs["timeout"], (5, 30))
            raise TimeoutError("server_info stalled")

        session = SimpleNamespace(request=Mock(side_effect=blocked))
        with self.assertRaisesRegex(TimeoutError, "server_info stalled"):
            probe.request_json(
                session,
                "http://localhost:31000",
                "GET",
                "/server_info",
                timeout_sec=30,
                event=event,
            )
        session.request.assert_called_once()
        self.assertEqual(events[-1]["stage"], "http_request_failed")
        self.assertEqual(events[-1]["path"], "/server_info")

    def test_http_failure_preserves_bounded_response_body_without_retry(self):
        probe = self.load_probe()
        reason = '{"detail":"Request discarded during fault-tolerance recovery: MLP sync failed"}'
        for body in (reason, reason + "x" * 5000):
            with self.subTest(length=len(body)):
                response = SimpleNamespace(
                    status_code=503,
                    text=body,
                    raise_for_status=Mock(
                        side_effect=RuntimeError("503 Service Unavailable")
                    ),
                    json=Mock(),
                )
                session = SimpleNamespace(request=Mock(return_value=response))
                events = []
                with self.assertRaisesRegex(RuntimeError, "503"):
                    probe.request_json(
                        session,
                        "http://localhost:31000",
                        "POST",
                        "/generate",
                        timeout_sec=600,
                        event=lambda stage, **details: events.append(
                            {"stage": stage, **details}
                        ),
                        json={"text": "test"},
                    )
                failed = events[-1]
                self.assertEqual(failed["stage"], "http_request_failed")
                self.assertEqual(failed["status_code"], 503)
                self.assertTrue(failed["response_text"].startswith(reason))
                self.assertLessEqual(len(failed["response_text"]), 4096)
                self.assertEqual(failed["response_truncated"], len(body) > 4096)
                session.request.assert_called_once()
                response.json.assert_not_called()

    def test_stalled_metadata_has_live_report_and_cannot_inject_a_fault(self):
        probe = self.load_probe()
        output = io.StringIO()
        session = Mock()
        session.get.return_value.status_code = 200
        psutil = SimpleNamespace(Process=Mock(), pid_exists=Mock(return_value=True))

        def create_session():
            self.assertIn('"stage": "test_started"', output.getvalue())
            return session

        requests = SimpleNamespace(
            Session=Mock(side_effect=create_session), RequestException=TimeoutError
        )
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"

            def blocked(*args, **kwargs):
                self.assertEqual(args[1], "http://127.0.0.1:30000/server_info")
                self.assertEqual(kwargs["timeout"], (5, 30))
                live = json.loads(report_path.read_text())
                self.assertEqual(live["events"][-1]["stage"], "http_request_started")
                self.assertFalse(live["success"])
                raise TimeoutError("metadata stalled")

            session.request.side_effect = blocked
            with (
                patch.dict(sys.modules, {"requests": requests, "psutil": psutil}),
                patch.object(probe.importlib.metadata, "version", return_value="test"),
                patch.object(
                    probe.subprocess, "check_output", return_value="test-commit"
                ),
                patch.object(probe, "schedulers") as schedulers,
                patch.object(
                    sys,
                    "argv",
                    ["probe", "--server-pid", "123", "--output", str(report_path)],
                ),
                redirect_stdout(output),
            ):
                with self.assertRaisesRegex(TimeoutError, "metadata stalled"):
                    probe.main()
            final = json.loads(report_path.read_text())
            self.assertEqual(final["error"], "metadata stalled")
            self.assertFalse(final["success"])
            self.assertIn("finished_at", final)
            schedulers.assert_not_called()
            psutil.Process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
