# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A controller that dispatches requests to multiple data parallel workers."""

import copy
import faulthandler
import logging
import multiprocessing as mp
import signal
import threading
import time
from enum import Enum, auto
from typing import Callable, List, Optional

import psutil
import setproctitle
import zmq

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.io_struct import (
    AbortReq,
    ActiveRanksOutput,
    ActiveRanksUpdateReqOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    BlockReqInput,
    ElasticScaleUpdateReq,
    FaultToleranceCommandReqInput,
    FaultToleranceDPCRecoverReqInput,
    FaultToleranceDPCShutdownReqInput,
    ProcessActiveRanksOutput,
    ProfileReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    WatchdogHeartbeatOutput,
    sock_recv,
    sock_send,
    unwrap_from_pickle,
    wrap_as_pickle,
)
from sglang.srt.managers.load_snapshot import create_load_snapshot_reader
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.observability.cpu_monitor import start_cpu_monitor_thread
from sglang.srt.observability.req_time_stats import DPControllerReqTimeStats
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.server_args import (
    DP_ATTENTION_HANDSHAKE_PORT_DELTA,
    PortArgs,
    ServerArgs,
)
from sglang.srt.utils import numa_utils
from sglang.srt.utils.common import (
    configure_logger,
    kill_itself_when_parent_died,
    maybe_reindex_device_id,
)
from sglang.srt.utils.network import (
    NetworkAddress,
    bind_port,
    get_local_ip_auto,
    get_zmq_socket,
    get_zmq_socket_on_host,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.watchdog import SubprocessWatchdog, Watchdog
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)

SCHEDULER_PIDS_ARG = "scheduler_pids"
FT_WATCHDOG_POLL_INTERVAL = 1.0
FT_WATCHDOG_SEND_TIMEOUT_MS = 1000


class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    FOLLOW_BOOTSTRAP_ROOM = auto()
    TOTAL_REQUESTS = auto()
    TOTAL_TOKENS = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


class DPBudget:
    def __init__(self, dp_size: int):
        self.dp_size = dp_size
        self.total_requests = [0] * dp_size
        self.total_tokens = [0] * dp_size
        self.last_timestamp = [0.0] * dp_size

    def update_budget(self, loads):
        """Update budget from shm snapshots, skipping stale reads."""
        for load in loads:
            if load.timestamp == self.last_timestamp[load.dp_rank]:
                continue
            self.last_timestamp[load.dp_rank] = load.timestamp
            self.total_requests[load.dp_rank] = (
                load.num_running_reqs + load.num_waiting_reqs
            )
            self.total_tokens[load.dp_rank] = load.num_total_tokens

    def dispatch(self, method: LoadBalanceMethod, estimated_tokens: int = 0):
        if method == LoadBalanceMethod.TOTAL_REQUESTS:
            target_rank = self.total_requests.index(min(self.total_requests))
        elif method == LoadBalanceMethod.TOTAL_TOKENS:
            # Use total_requests as a tie-breaker when total_tokens are equal
            target_rank = min(
                range(self.dp_size),
                key=lambda i: (self.total_tokens[i], self.total_requests[i]),
            )
        else:
            return None

        # Increment the load of that worker by one as a heuristic
        self.total_requests[target_rank] += 1
        self.total_tokens[target_rank] += estimated_tokens
        return target_rank


class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        run_scheduler_process_func: Callable,
    ) -> None:
        # Parse args
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )
        self.run_scheduler_process_func = run_scheduler_process_func

        # Init inter-process communication
        self.context = zmq.Context(1 + server_args.dp_size)
        self.recv_from_tokenizer = None
        if server_args.node_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                self.context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
        self.send_to_tokenizer = None
        self.recv_from_ft_controller = None
        self.ft_control_endpoint = None
        if (
            server_args.device == "npu"
            and server_args.enable_fault_tolerance
            and server_args.elastic_ep_backend == "mc2"
        ):
            from sglang.srt.fault_tolerance.npu_communication import create_npu_ft_store

            self.npu_ft_store = create_npu_ft_store(
                port_args.fault_tolerance_metadata_ipc_name
            )
        if server_args.enable_fault_tolerance:
            self.send_to_tokenizer = get_zmq_socket(
                self.context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )
            host = get_local_ip_auto(fallback="127.0.0.1")
            port, self.recv_from_ft_controller = get_zmq_socket_on_host(
                self.context, zmq.PULL, host=host
            )
            self.ft_control_endpoint = NetworkAddress(host, port).to_tcp()

        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM: self.follow_bootstrap_room_scheduler,
            LoadBalanceMethod.TOTAL_REQUESTS: self.total_requests_scheduler,
            LoadBalanceMethod.TOTAL_TOKENS: self.total_tokens_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]
        self.refresh_load_budget_on_dispatch = self.load_balance_method in (
            LoadBalanceMethod.TOTAL_REQUESTS,
            LoadBalanceMethod.TOTAL_TOKENS,
        )

        self.launch_dp_size: int = server_args.dp_size
        self.max_dp_size: int = server_args.max_ep_size or server_args.dp_size
        assert self.max_dp_size >= self.launch_dp_size, (
            f"--max-ep-size ({self.max_dp_size}) must be >= "
            f"--dp ({self.launch_dp_size})."
        )

        self.dp_active: List[bool] = [True] * self.launch_dp_size + [False] * (
            self.max_dp_size - self.launch_dp_size
        )

        self.dp_budget = DPBudget(server_args.dp_size)
        self.load_snapshot_reader = create_load_snapshot_reader(
            server_args,
            port_args,
            caller="DataParallelController",
        )
        self._last_refresh_time = 0.0

        # To protect changing env vars to set CUDA_VISIBLE_DEVICES.
        self.env_lock = threading.Lock()

        # Launch data parallel workers
        self.scheduler_procs = []
        self.scheduler_process_dp_ranks: List[int] = []
        self.scheduler_process_global_ranks: List[int] = []
        self._scheduler_launch_specs = {}
        self._npu_recovery_readers = {}
        self._watchdog_event_epoch = 0
        self.workers: List[Optional[zmq.Socket]] = [None] * self.max_dp_size
        self.status: List[bool] = list(self.dp_active)
        self._active_workers: List[int] = list(range(self.launch_dp_size))
        self._active_count_cache: int = self.launch_dp_size
        self._watchdog_context = None
        self._watchdog_sender = None

        if server_args.enable_dp_attention:
            self.launch_dp_attention_schedulers(server_args, port_args)
            # When local control broadcast is enabled, send control messages to
            # every DP group leader (attn_tp_rank=0) so each leader broadcasts
            # within its own attn_tp_group instead of the full tp_group.
            # Otherwise fall back to the original behaviour: send to only the
            # first leader, which then broadcasts over the full tp_group.
            # Match SchedulerRequestReceiver: FT and scale joiners no longer
            # relay control requests through the full TP group.
            local_ctrl = (
                server_args.enable_dp_attention_local_control_broadcast
                or server_args.is_ep_scale_joiner
                or server_args.enable_fault_tolerance
            )
            self.control_message_step = 1 if local_ctrl else server_args.tp_size
        else:
            self.launch_dp_schedulers(server_args, port_args)
            self.control_message_step = 1

        self._scheduler_watchdog = None
        if server_args.enable_fault_tolerance and self.scheduler_procs:
            self._scheduler_watchdog = SubprocessWatchdog(
                processes=self.scheduler_procs,
                process_names=[
                    f"scheduler_rank_{rank}"
                    for rank in self.scheduler_process_global_ranks
                ],
                on_exit=self._handle_scheduler_process_exit,
                on_poll=self._report_watchdog_heartbeat,
                on_thread_stop=self._close_watchdog_sender,
                interval=FT_WATCHDOG_POLL_INTERVAL,
                fail_stop_on_exit=False,
                report_clean_exit=True,
            )
            self._report_initial_watchdog_heartbeat()
            self._scheduler_watchdog.start()
            if server_args.ep_join_mode == "recover":
                self._report_process_active_ranks(active=True)

        self.init_dispatcher()

        self.soft_watchdog = Watchdog.create(
            debug_name="DataParallelController",
            watchdog_timeout=server_args.soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_DP_CONTROLLER.get(),
        )

        if server_args.enable_metrics:
            start_cpu_monitor_thread("data_parallel_controller")

    def send_to_all_workers(self, obj):
        for i, worker in enumerate(self.workers):
            if worker is not None and self.status[i]:
                sock_send(worker, obj)

    def send_control_message(self, obj):
        for i in self._active_workers[:: self.control_message_step]:
            worker = self.workers[i]
            if worker is not None:
                sock_send(worker, obj)

    def send_fault_tolerance_command(self, obj: FaultToleranceCommandReqInput):
        # FT commands are DP-scoped. Each target socket belongs to that DP's
        # attention leader. Do not gate command delivery on route status because
        # recovery may need to resume a DP while its route is still inactive.
        for rank in obj.target_ranks:
            worker = self.workers[rank]
            if worker is None:
                raise ValueError(f"DP rank {rank} has no scheduler socket")
            sock_send(worker, obj)

    def shutdown_dp(self, obj: FaultToleranceDPCShutdownReqInput):
        targets = set(obj.target_dp_ranks)
        for proc, dp_rank in zip(
            self.scheduler_procs, self.scheduler_process_dp_ranks
        ):
            if dp_rank in targets and proc.is_alive():
                proc.kill()

    def recover_dp(self, obj: FaultToleranceDPCRecoverReqInput):
        if not self.server_args.enable_npu_ft_recovery:
            raise ValueError("NPU recovery was not enabled at launch")
        targets = set(obj.target_dp_ranks)
        indices = [
            i for i, rank in enumerate(self.scheduler_process_dp_ranks) if rank in targets
        ]
        if {self.scheduler_process_dp_ranks[i] for i in indices} != targets:
            raise ValueError("recovery contains a rank not owned by this DPC")
        if any(self.scheduler_procs[i].is_alive() for i in indices):
            raise ValueError("cannot replace a live scheduler")
        # The watchdog keeps per-index exit markers. Restart it after replacing
        # handles, so a second failure of the same original rank is observable.
        self._scheduler_watchdog.stop()
        self._watchdog_event_epoch = obj.membership_epoch
        for index in indices:
            rank = self.scheduler_process_global_ranks[index]
            self.scheduler_procs[index].join(timeout=1)
            args = list(self._scheduler_launch_specs[rank])
            replacement_args = copy.deepcopy(self.server_args)
            replacement_args.override(
                "npu_ft.recover_startup",
                _npu_ft_recovery={
                    "endpoint": self.port_args.fault_tolerance_metadata_ipc_name,
                    "plan": {
                        "membership_epoch": obj.membership_epoch,
                        "previous_active_mask": obj.previous_active_mask,
                        "recovering_ranks": sorted(targets),
                    },
                },
            )
            reader, writer = mp.Pipe(duplex=False)
            args[0], args[9] = replacement_args, writer
            process = mp.Process(target=self.run_scheduler_process_func, args=tuple(args))
            with numa_utils.configure_subprocess(replacement_args, args[2]):
                process.start()
            writer.close()
            self.scheduler_procs[index] = process
            self._npu_recovery_readers[rank] = (
                reader,
                time.monotonic() + self.server_args.fault_tolerance_timeout,
            )
        # Never wait for model startup here: survivors' prepare commands must
        # still be forwarded by this event loop while joiners construct PGs.
        self._scheduler_watchdog = SubprocessWatchdog(
            processes=self.scheduler_procs,
            process_names=[
                f"scheduler_rank_{rank}" for rank in self.scheduler_process_global_ranks
            ],
            on_exit=self._handle_scheduler_process_exit,
            on_poll=self._report_watchdog_heartbeat,
            on_thread_stop=self._close_watchdog_sender,
            interval=FT_WATCHDOG_POLL_INTERVAL,
            fail_stop_on_exit=False,
            report_clean_exit=True,
        )
        self._report_initial_watchdog_heartbeat()
        self._scheduler_watchdog.start()

    def poll_npu_recovery_startup(self):
        for rank, (reader, deadline) in list(self._npu_recovery_readers.items()):
            if reader.poll():
                try:
                    info = reader.recv()
                finally:
                    reader.close()
                    del self._npu_recovery_readers[rank]
                if info.get("status") != "ready":
                    raise RuntimeError(f"NPU replacement failed to initialize: {info}")
                sock_send(
                    self.send_to_tokenizer,
                    ProcessActiveRanksOutput(
                        ranks=[rank],
                        active=True,
                        membership_epoch=self._watchdog_event_epoch,
                    ),
                )
            elif time.monotonic() >= deadline:
                raise TimeoutError(f"NPU replacement scheduler {rank} startup timed out")

    def _handle_scheduler_process_exit(self, index, proc, name):
        dp_rank = self.scheduler_process_dp_ranks[index]
        global_rank = self.scheduler_process_global_ranks[index]
        logger.warning(
            "Scheduler global rank %s for DP rank %s exited (pid=%s)",
            global_rank,
            dp_rank,
            proc.pid,
        )
        sock_send(
            self._get_watchdog_sender(),
            ProcessActiveRanksOutput(
                ranks=[global_rank],
                active=False,
                membership_epoch=getattr(self, "_watchdog_event_epoch", 0),
            ),
        )

    def _get_watchdog_sender(self):
        if self._watchdog_sender is not None:
            return self._watchdog_sender

        context = zmq.Context()
        self._watchdog_context = context
        sender = context.socket(zmq.PUSH)
        self._watchdog_sender = sender
        sender.setsockopt(zmq.LINGER, 0)
        sender.setsockopt(zmq.SNDHWM, 1)
        sender.setsockopt(zmq.IMMEDIATE, 1)
        sender.setsockopt(zmq.SNDTIMEO, FT_WATCHDOG_SEND_TIMEOUT_MS)
        if "[" in self.port_args.tokenizer_ipc_name:
            sender.setsockopt(zmq.IPV6, 1)
        sender.connect(self.port_args.tokenizer_ipc_name)
        return sender

    def _watchdog_heartbeat(self):
        return WatchdogHeartbeatOutput(
            node_rank=self.server_args.node_rank,
            ranks=sorted(self.scheduler_process_global_ranks),
            control_endpoint=self.ft_control_endpoint,
        )

    def _report_initial_watchdog_heartbeat(self):
        sock_send(self.send_to_tokenizer, self._watchdog_heartbeat())

    def _report_watchdog_heartbeat(self):
        try:
            sock_send(
                self._get_watchdog_sender(),
                self._watchdog_heartbeat(),
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            logger.debug("Dropping watchdog heartbeat because tokenizer is unavailable")

    def _close_watchdog_sender(self):
        sender = self._watchdog_sender
        context = self._watchdog_context
        self._watchdog_sender = None
        self._watchdog_context = None
        try:
            if sender is not None:
                sender.close(linger=0)
        finally:
            if context is not None:
                context.term()

    def _report_process_active_ranks(self, *, active: bool) -> None:
        sock_send(
            self.send_to_tokenizer,
            ProcessActiveRanksOutput(
                ranks=sorted(self.scheduler_process_global_ranks),
                active=active,
            ),
        )

    def handle_load_update_req(self, obj):
        self.dp_budget.update_budget(obj)

    def update_active_ranks(self, ranks: ActiveRanksOutput):
        if len(ranks.status) != self.max_dp_size:
            logger.warning(
                "[DPC] update_active_ranks: status len=%d != max_dp_size=%d; "
                "ignoring update",
                len(ranks.status),
                self.max_dp_size,
            )
            return
        if self.server_args.elastic_ep_backend is not None:
            self.status = [
                self.dp_active[i] and bool(ranks.status[i])
                for i in range(self.max_dp_size)
            ]
        else:
            self.status = list(ranks.status)
        self._refresh_active_workers()

        if ranks.request_id is not None and self.send_to_tokenizer is not None:
            sock_send(
                self.send_to_tokenizer,
                ActiveRanksUpdateReqOutput(
                    request_id=ranks.request_id,
                    success=True,
                    message="active ranks updated",
                )
            )

    def add_elastic_workers(self, slot_offset: int, slot_count: int):
        """Activate a range of pre-bound worker slots."""
        end = slot_offset + slot_count
        if end > self.max_dp_size:
            raise ValueError(
                f"[Elastic EP] add_elastic_workers: slot_offset={slot_offset} + "
                f"slot_count={slot_count} exceeds max_dp_size={self.max_dp_size}. "
                f"Restart with a larger --max-ep-size."
            )

        for slot in range(slot_offset, end):
            if self.dp_active[slot]:
                logger.debug(
                    "[Elastic EP] add_elastic_workers: slot %d already active; "
                    "skipping",
                    slot,
                )
                continue
            assert self.workers[slot] is not None, (
                f"[Elastic EP] add_elastic_workers: slot {slot} was not "
                f"pre-bound at launch; expected a primary-bound PUSH socket."
            )
            self.dp_active[slot] = True
            self.status[slot] = True

        self._refresh_active_workers()
        logger.debug(
            "[Elastic EP] DataParallelController activated slots %s "
            "(active=%d / max=%d)",
            list(range(slot_offset, end)),
            self._active_count_cache,
            self.max_dp_size,
        )

    def _refresh_active_workers(self) -> None:
        self._active_workers = [
            i for i, active in enumerate(self.dp_active) if active and self.status[i]
        ]
        self._active_count_cache = len(self._active_workers)

    def refresh_load_budget(self):
        # Throttle to at most once per 20ms.  When a burst of requests
        # arrives, dispatching_with_trace() calls this before every
        # dispatch.  Each call reads the latest scheduler snapshot and
        # overwrites the speculative +1 increments that DPBudget.dispatch()
        # added for previously dispatched requests in this burst.  Without
        # throttling, the budget resets to the (stale) scheduler-reported
        # value on every request, causing the entire burst to land on a
        # single DP rank.  The 20ms interval lets the burst complete
        # using speculative counters, then refreshes from the real
        # scheduler load for the next batch.
        now = time.perf_counter()
        if now - self._last_refresh_time < 0.02:
            return
        self._last_refresh_time = now
        self.dp_budget.update_budget(self.load_snapshot_reader.read_all())

    def dispatching_with_trace(self, req: Req, refresh_load_budget: bool = True):
        if refresh_load_budget and self.refresh_load_budget_on_dispatch:
            self.refresh_load_budget()

        time_stats = DPControllerReqTimeStats.new_from_obj(
            unwrap_from_pickle(req.time_stats)
        )

        time_stats.set_dp_dispatch_time()
        req.time_stats = wrap_as_pickle(time_stats)
        if self.server_args.enable_fault_tolerance and not self._active_workers:
            self._reject_req(req, "no active DP rank")
        else:
            self.dispatching(req)
        req.time_stats = time_stats
        req.time_stats.set_dp_dispatch_finish_time()

    def dispatch_batch_generate(self, batch_req: BatchTokenizedGenerateReqInput):
        if self.refresh_load_budget_on_dispatch:
            self.refresh_load_budget()
        for req in batch_req:
            self.dispatching_with_trace(req, refresh_load_budget=False)

    def dispatch_batch_embedding(self, batch_req: BatchTokenizedEmbeddingReqInput):
        if self.refresh_load_budget_on_dispatch:
            self.refresh_load_budget()
        for req in batch_req:
            self.dispatching_with_trace(req, refresh_load_budget=False)

    def init_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.dispatching_with_trace),
                (TokenizedEmbeddingReqInput, self.dispatching_with_trace),
                (BatchTokenizedGenerateReqInput, self.dispatch_batch_generate),
                (BatchTokenizedEmbeddingReqInput, self.dispatch_batch_embedding),
                (BlockReqInput, self.send_to_all_workers),
                (ProfileReq, self.send_to_all_workers),
                (ActiveRanksOutput, self.update_active_ranks),
                (
                    ElasticScaleUpdateReq,
                    lambda msg: self.add_elastic_workers(
                        msg.slot_offset, msg.slot_count
                    ),
                ),
                (FaultToleranceCommandReqInput, self.send_fault_tolerance_command),
                (FaultToleranceDPCShutdownReqInput, self.shutdown_dp),
                (FaultToleranceDPCRecoverReqInput, self.recover_dp),
            ]
        )
        self._request_dispatcher.add_fallback_fn(self.send_control_message)

    def launch_dp_schedulers(self, server_args, port_args):
        base_gpu_id = 0

        threads = []
        sockets = []
        ready_events = []
        for dp_rank in range(server_args.dp_size):
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name
            tmp_port_args.instance_id = port_args.instance_id

            # This port is checked free in PortArgs.init_new.
            # We hold it first so that the next dp worker gets a different port
            sockets.append(bind_port(tmp_port_args.nccl_port))

            ready_event = threading.Event()
            ready_events.append(ready_event)

            # Create a thread for each worker
            thread = threading.Thread(
                target=self.launch_tensor_parallel_group_thread,
                args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ready_event),
            )
            threads.append(thread)
            base_gpu_id += (
                server_args.tp_size * server_args.pp_size * server_args.gpu_id_step
            )

            if server_args.node_rank == 0:
                self.workers[dp_rank] = get_zmq_socket(
                    self.context,
                    zmq.PUSH,
                    tmp_port_args.scheduler_input_ipc_name,
                    True,
                )

        # Free all sockets before starting the threads to launch TP workers
        for sock in sockets:
            sock.close()

        # Start all threads
        for thread in threads:
            thread.start()
        for event in ready_events:
            event.wait()

    def launch_tensor_parallel_group_thread(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
        ready_event: threading.Event,
    ):
        self.launch_tensor_parallel_group(server_args, port_args, base_gpu_id, dp_rank)
        ready_event.set()

        # This thread cannot be closed because otherwise the `kill_itself_when_parent_died`
        # function in scheduler.py will kill the scheduler.
        while True:
            time.sleep(30 * 24 * 3600)

    def _broadcast_worker_ports(
        self, server_args: ServerArgs, worker_ports: Optional[List[int]] = None
    ) -> List[int]:
        """Broadcast worker ports from node 0 to all other nodes.

        Node 0 acts as the server, waiting for all other nodes to connect and
        sending them the pre-allocated worker ports. Other nodes act as clients,
        connecting to node 0 to receive their copy of the worker ports.

        Args:
            server_args: Server arguments containing node configuration.
            worker_ports: Pre-allocated worker ports to broadcast.

        Returns:
            List of worker ports (same on all nodes after broadcast).
        """
        is_joiner = server_args.is_ep_scale_joiner
        if server_args.dist_init_addr is None or is_joiner:
            na = NetworkAddress(
                server_args.host or "127.0.0.1",
                server_args.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA,
            )
        else:
            na = NetworkAddress.parse(server_args.dist_init_addr)
            na = NetworkAddress(na.host, na.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA)
        endpoint = na.to_tcp()

        if server_args.node_rank == 0:
            # Node 0: Broadcast worker ports to all other nodes
            return self._broadcast_ports_as_server(
                endpoint, server_args.nnodes - 1, worker_ports
            )
        else:
            # Other nodes: Receive worker ports from node 0
            return self._receive_ports_as_client(endpoint, server_args.node_rank)

    def _broadcast_ports_as_server(
        self, endpoint: str, expected_clients: int, worker_ports: List[int]
    ) -> List[int]:
        """Broadcast worker ports to all client nodes."""
        logger.debug(f"Broadcasting worker ports to {expected_clients} client nodes")
        logger.debug(f"Worker ports: {worker_ports}")

        rep_socket = get_zmq_socket(self.context, zmq.REP, endpoint, True)

        try:
            connected_clients = 0
            while connected_clients < expected_clients:
                # Wait for client handshake
                client_rank = sock_recv(rep_socket)
                logger.debug(f"Received handshake from node {client_rank}")

                # Send worker ports to client
                sock_send(rep_socket, wrap_as_pickle(worker_ports))
                connected_clients += 1
                logger.debug(
                    f"Sent worker ports to {connected_clients}/{expected_clients} nodes"
                )

            logger.debug("Worker port broadcast completed")
            return worker_ports
        finally:
            if self.server_args.elastic_ep_backend is None:
                rep_socket.close()
            else:
                threading.Thread(
                    target=self._reply_ports_as_server,
                    args=(rep_socket, worker_ports),
                    daemon=True,
                ).start()

    def _reply_ports_as_server(self, rep_socket: zmq.Socket, worker_ports: List[int]):
        """Background thread: serve the pre-bound worker-port list to
        late-arriving elastic joiners. Publishes port numbers only; the primary
        keeps ownership of every socket."""
        while True:
            try:
                client_rank = sock_recv(rep_socket)
            except Exception:
                logger.exception(
                    "Failed to recv/decode handshake in reply thread; continue"
                )
                continue
            logger.debug(f"Received handshake from node {client_rank}")

            # Send worker ports to client
            sock_send(rep_socket, wrap_as_pickle(worker_ports))
            logger.debug(f"Sent worker ports to node {client_rank}")

    def _receive_ports_as_client(self, endpoint: str, node_rank: int) -> List[int]:
        """Receive worker ports from the server node."""
        logger.debug(f"Connecting to node 0 to receive worker ports")

        req_socket = get_zmq_socket(self.context, zmq.REQ, endpoint, False)
        req_socket.setsockopt(zmq.RCVTIMEO, 600 * 1000)  # 10 minute timeout
        req_socket.setsockopt(zmq.SNDTIMEO, 600 * 1000)

        try:
            # Send handshake with our node rank
            sock_send(req_socket, wrap_as_pickle(str(node_rank)))

            # Receive worker ports
            worker_ports = sock_recv(req_socket)
            logger.debug(f"Received {len(worker_ports)} worker ports from node 0")
            return worker_ports
        except zmq.Again:
            logger.error("Timeout waiting for worker ports from node 0")
            raise RuntimeError(
                "Failed to receive worker ports from node 0 within timeout"
            )
        finally:
            req_socket.close()

    def _joiner_local_tp_span(self, server_args: ServerArgs) -> int:
        return server_args.tp_size

    def _joiner_slot_offset(self, server_args: ServerArgs) -> int:
        return server_args.ep_join_rank_offset

    def launch_dp_attention_schedulers(self, server_args: ServerArgs, port_args: PortArgs):
        if server_args.dist_init_addr is None:
            bind_host = "127.0.0.1"
        else:
            bind_host = NetworkAddress.parse(server_args.dist_init_addr).host

        worker_ports = []
        if server_args.is_ep_scale_joiner:
            # Scale joiners connect to their pre-bound primary worker sockets.
            primary = NetworkAddress.parse(server_args.dist_init_addr)
            primary_endpoint = NetworkAddress(
                primary.host, primary.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA
            ).to_tcp()
            all_ports = self._receive_ports_as_client(
                primary_endpoint, server_args.node_rank
            )
            offset = self._joiner_slot_offset(server_args)
            local_tp_span = self._joiner_local_tp_span(server_args)
            broadcasted_ports = all_ports[offset : offset + local_tp_span]
        elif server_args.node_rank == 0:
            # Elastic primaries reserve sockets for the maximum DP size.
            bind_count = (
                self.max_dp_size
                if server_args.elastic_ep_backend is not None
                else server_args.dp_size
            )
            for slot in range(bind_count):
                worker_port, worker_socket = get_zmq_socket_on_host(
                    self.context, zmq.PUSH, host=bind_host
                )
                worker_ports.append(worker_port)
                self.workers[slot] = worker_socket
                logger.debug(
                    "Assigned port %s to worker slot %s on host %s",
                    worker_port,
                    slot,
                    bind_host,
                )
            broadcasted_ports = self._broadcast_worker_ports(server_args, worker_ports)
        else:
            broadcasted_ports = self._broadcast_worker_ports(server_args, None)

        self.launch_tensor_parallel_group(
            server_args, port_args, 0, None, broadcasted_ports
        )

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: Optional[int],
        worker_ports: Optional[List[int]] = None,
    ):
        if not server_args.enable_dp_attention:
            logger.info(f"Launch DP{dp_rank} starting at GPU #{base_gpu_id}.")

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )

        scheduler_pipe_readers = []

        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        nnodes_per_pp_rank = max(server_args.nnodes // server_args.pp_size, 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank),
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank + 1),
        )

        nnodes_per_tp_group = nnodes_per_pp_rank
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        if server_args.is_ep_scale_joiner:
            # Scale joiners enumerate their full local TP span.
            tp_rank_range = range(server_args.tp_size)
            tp_size_per_node = server_args.tp_size
        else:
            tp_rank_range = range(
                tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
                tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
            )

        attn_cp_rank = 0
        moe_dp_rank = 0
        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                rank_port_args = port_args

                if server_args.enable_dp_attention:
                    # dp attention has different sharding logic
                    _, _, dp_rank, _ = compute_dp_attention_world_info(
                        server_args.enable_dp_attention,
                        tp_rank,
                        server_args.tp_size,
                        server_args.dp_size,
                        server_args.attn_cp_size,
                    )
                    # compute zmq ports for this dp rank
                    rank_port_args = PortArgs.init_new(
                        server_args, dp_rank, worker_ports
                    )
                    if server_args.is_ep_scale_joiner:
                        # Scale-joiner outputs return through the primary tokenizer.
                        primary_addr = NetworkAddress.parse(server_args.dist_init_addr)
                        primary_port_base = primary_addr.port + 1
                        rank_port_args.tokenizer_ipc_name = NetworkAddress(
                            primary_addr.host, primary_port_base
                        ).to_tcp()
                        rank_port_args.detokenizer_ipc_name = NetworkAddress(
                            primary_addr.host, primary_port_base + 1
                        ).to_tcp()
                    # Data parallelism reuses the tensor parallelism group,
                    # so all dp ranks should use the same nccl port.
                    rank_port_args.nccl_port = port_args.nccl_port
                    rank_port_args.instance_id = port_args.instance_id

                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    server_args.base_gpu_id
                    + base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                attn_dp_size = (
                    server_args.dp_size if server_args.enable_dp_attention else 1
                )

                # Parallelism hierarchy (outermost to innermost):
                # - Attention: Global(TP) -> DP -> ATTN_CP -> ATTN_TP (innermost)
                # - MoE: Global(TP) -> MOE_DP -> EP -> MOE_TP (innermost)
                attn_tp_size = (
                    server_args.tp_size // attn_dp_size // server_args.attn_cp_size
                )
                attn_cp_rank = (tp_rank // attn_tp_size) % server_args.attn_cp_size
                moe_dp_rank = tp_rank // (
                    server_args.tp_size // server_args.moe_dp_size
                )
                moe_ep_rank = (
                    tp_rank
                    % (server_args.tp_size // server_args.moe_dp_size)
                    // (
                        server_args.tp_size
                        // server_args.moe_dp_size
                        // server_args.ep_size
                    )
                )

                # Scheduler internals use local ranks; logs use global ranks.
                offset = server_args.ep_join_rank_offset
                display_tp_rank = tp_rank + offset
                display_moe_ep_rank = moe_ep_rank + offset
                display_dp_rank = dp_rank + offset if dp_rank is not None else None

                with self.env_lock, maybe_reindex_device_id(gpu_id) as gpu_id:
                    scheduler_args = (
                        server_args,
                        rank_port_args,
                        gpu_id,
                        tp_rank,
                        attn_cp_rank,
                        moe_dp_rank,
                        moe_ep_rank,
                        pp_rank,
                        dp_rank,
                        writer,
                        display_tp_rank,
                        display_dp_rank,
                        display_moe_ep_rank,
                    )
                    proc = mp.Process(
                        target=self.run_scheduler_process_func,
                        args=scheduler_args,
                    )
                    if server_args.enable_npu_ft_recovery:
                        self._scheduler_launch_specs[tp_rank] = scheduler_args
                    with (
                        memory_saver_adapter.configure_subprocess(),
                        numa_utils.configure_subprocess(server_args, gpu_id),
                    ):
                        proc.start()
                self.scheduler_procs.append(proc)
                if server_args.enable_fault_tolerance:
                    self.scheduler_process_dp_ranks.append(dp_rank)
                    rank_offset = (
                        server_args.ep_join_rank_offset
                        if server_args.is_ep_scale_joiner
                        else 0
                    )
                    self.scheduler_process_global_ranks.append(
                        rank_offset + server_args.tp_size * pp_rank + tp_rank
                    )
                scheduler_pipe_readers.append(reader)

        # Wait for model to finish loading
        scheduler_info = []
        for i in range(len(scheduler_pipe_readers)):
            scheduler_info.append(scheduler_pipe_readers[i].recv())

        self.max_total_num_tokens = scheduler_info[0]["max_total_num_tokens"]
        self.max_req_input_len = scheduler_info[0]["max_req_input_len"]

    def maybe_external_dp_rank_routing(self, req: Req):
        if req.routed_dp_rank is not None:
            rank = req.routed_dp_rank
            if (
                rank < 0
                or rank >= len(self.workers)
                or rank not in self._active_workers
                or self.workers[rank] is None
            ):
                if self.server_args.enable_fault_tolerance:
                    self._reject_req(req, f"routed_dp_rank={rank} is inactive")
                    return True
                raise ValueError(f"DP rank {rank} is not active.")
            logger.debug(f"Direct routing to DP rank {rank}")
            sock_send(self.workers[rank], req)
            return True
        return False

    def _reject_req(self, req: Req, message: str):
        logger.warning("Rejecting DP request %s: %s", getattr(req, "rid", ""), message)
        if self.send_to_tokenizer is not None:
            sock_send(
                self.send_to_tokenizer,
                AbortReq(rid=req.rid, abort_message=message)
            )

    def round_robin_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return

        active = self._active_workers
        if not active:
            raise RuntimeError("No active DP workers are available for routing.")
        attempts = 0
        while attempts < len(active):
            slot = active[self.round_robin_counter % len(active)]
            self.round_robin_counter = (self.round_robin_counter + 1) % len(active)
            if self.status[slot]:
                logger.debug(f"Choose worker {slot}")
                sock_send(self.workers[slot], req)
                return
            attempts += 1
        raise RuntimeError(
            f"Cannot route request: all {len(active)} active DP workers "
            "are unavailable."
        )

    def follow_bootstrap_room_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return

        assert req.bootstrap_room is not None, (
            "req.bootstrap_room should not be None. Do not send requests directly to "
            "prefill or decode instances; send to the router instead."
        )
        target_rank = req.bootstrap_room % len(self.workers)
        sock_send(self.workers[target_rank], req)

    def total_requests_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return
        target_worker = self.dp_budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        sock_send(self.workers[target_worker], req)

    def total_tokens_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return
        estimated_tokens = len(req.input_ids)
        target_worker = self.dp_budget.dispatch(
            LoadBalanceMethod.TOTAL_TOKENS, estimated_tokens=estimated_tokens
        )
        sock_send(self.workers[target_worker], req)

    def event_loop(self):
        poller = zmq.Poller()
        sockets = [
            socket
            for socket in (self.recv_from_tokenizer, self.recv_from_ft_controller)
            if socket is not None
        ]
        for socket in sockets:
            poller.register(socket, zmq.POLLIN)
        while True:
            self.soft_watchdog.feed()
            self.poll_npu_recovery_startup()
            ready = dict(poller.poll(timeout=100))
            for socket in sockets:
                if socket in ready:
                    self._request_dispatcher(sock_recv(socket))

def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
    run_scheduler_process_func: Callable = run_scheduler_process,
):
    setproctitle.setproctitle("sglang::data_parallel_controller")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    configure_logger(server_args)
    if server_args.enable_trace:
        process_tracing_init(
            server_args.otlp_traces_endpoint,
            "sglang",
            trace_modules=server_args.trace_modules,
        )
        thread_label = "DP Controller"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill DP Controller"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode DP Controller"
        trace_set_thread_info(thread_label)

    try:
        controller = DataParallelController(
            server_args, port_args, run_scheduler_process_func
        )
        scheduler_pids = [
            proc.pid for proc in controller.scheduler_procs if proc is not None
        ]
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
                SCHEDULER_PIDS_ARG: scheduler_pids,
            }
        )
        # The primary owns routing for the expanded scheduler set.
        if (
            server_args.node_rank == 0 and not server_args.is_ep_scale_joiner
        ) or server_args.enable_fault_tolerance:
            controller.event_loop()
        for proc in controller.scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DataParallelController hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
