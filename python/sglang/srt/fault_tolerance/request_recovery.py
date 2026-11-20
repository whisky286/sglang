"""CPU-owned request recovery; no device objects survive an execution attempt."""

import asyncio
import copy
import logging
import math
import time
import uuid
from array import array
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
EXECUTION_PREFIX = "__sglang_ft_attempt__"


def decode_checkpoint(state):
    """A bounded tail plus cursors, rather than a copy of the whole output."""
    return {
        "decode_ids": list(state.decode_ids[state.surr_offset :]),
        "read_offset": state.read_offset - state.surr_offset,
        "pending_chars": state.sent_offset - state.decoded_text_len,
    }


def restore_request(req, payload):
    """Rebuild CPU progress on a fresh Req, keeping prompt/output separate."""
    prefix = payload.ft_resume_output_ids
    if not prefix:
        return
    if len(prefix) >= req.sampling_params.max_new_tokens:
        raise ValueError("FT replay must not sample beyond the original output limit")
    if payload.ft_decode_checkpoint is None:
        raise ValueError("FT replay requires a decode checkpoint")
    req.output_ids = array("q", prefix)
    decode_ids, _ = req.init_incremental_detokenize()
    req.send_decode_id_offset = len(decode_ids)
    req.send_token_offset = len(prefix)
    req.ft_decode_checkpoint = payload.ft_decode_checkpoint
    req.retraction_count = payload.ft_recovery_attempt
    req.is_retracted = True
    req.retracted_stain = True


class RecoveryCapacityError(RuntimeError):
    pass


class RecoveryDeadlineError(RuntimeError):
    pass


def validate_recovery_config(args):
    if not getattr(args, "enable_ft_request_recovery", False):
        return
    required = {
        "enable_fault_tolerance": True,
        "fault_tolerance_on_error_strategy": "pause",
        "disable_cuda_graph": True,
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        "skip_tokenizer_init": False,
        "disaggregation_mode": "null",
        "nnodes": 1,
    }
    for name, value in required.items():
        if getattr(args, name, None) != value:
            raise ValueError(f"FT request recovery requires {name}={value!r}")
    for name in (
        "speculative_algorithm",
        "enable_lora",
        "enable_hierarchical_cache",
        "enable_hicache_storage",
        "enable_session_radix_cache",
        "dllm_algorithm",
    ):
        if getattr(args, name, None):
            raise ValueError(f"FT request recovery does not support {name}")
    for name in (
        "ft_request_recovery_timeout",
        "ft_request_timeout",
        "ft_request_recovery_max_requests",
        "ft_request_recovery_max_tokens",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.ft_request_recovery_max_retries < 0:
        raise ValueError("ft_request_recovery_max_retries must be nonnegative")


def validate_payload(payload):
    for name in (
        "mm_inputs",
        "input_embeds",
        "positional_embed_overrides",
        "lora_id",
        "session_id",
        "session_params",
        "custom_logit_processor",
        "token_type_ids",
        "return_logprob",
        "return_sampling_mask",
        "return_hidden_states",
        "return_routed_experts",
        "return_indexer_topk",
        "return_entropy",
        "require_reasoning",
    ):
        if getattr(payload, name, None):
            raise ValueError(f"FT request recovery does not support {name}")
    params = payload.sampling_params
    if params.top_k != 1:  # SamplingParams normalizes temperature=0 to top_k=1.
        raise ValueError("FT request recovery requires greedy sampling (temperature=0)")
    for name in (
        "stop",
        "stop_strs",
        "stop_regex",
        "stop_regex_strs",
        "stop_token_ids",
        "json_schema",
        "regex",
        "ebnf",
        "structural_tag",
        "logit_bias",
        "custom_params",
    ):
        if getattr(params, name, None):
            raise ValueError(f"FT request recovery does not support {name}")
    for name, value in (
        ("frequency_penalty", 0),
        ("presence_penalty", 0),
        ("repetition_penalty", 1),
        ("min_new_tokens", 0),
        ("n", 1),
    ):
        if getattr(params, name, value) != value:
            raise ValueError(f"FT request recovery requires {name}={value}")
    if payload.routed_dp_rank is not None:
        raise ValueError("FT request recovery does not support a pinned routed_dp_rank")
    limit = params.max_new_tokens
    if not isinstance(limit, int) or limit <= 0 or payload.input_ids is None:
        raise ValueError(
            "FT request recovery requires input IDs and max_new_tokens > 0"
        )


@dataclass
class RecoveryRecord:
    public_id: str
    payload: object
    weight_key: tuple
    reserved_tokens: int
    deadline: float
    execution_id: str = ""
    epoch: int = 0
    retries: int = 0
    output_ids: list = field(default_factory=list)
    decode_checkpoint: object = None
    paused_at: object = None
    recovery_wait: float = 0.0
    owner_dp_rank: object = None


class RequestRecovery:
    def __init__(
        self,
        *,
        max_requests=128,
        max_tokens=262144,
        max_retries=2,
        recovery_timeout=600,
        request_timeout=1800,
        clock=time.monotonic,
    ):
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.recovery_timeout = recovery_timeout
        self.request_timeout = request_timeout
        self.clock = clock
        self.records = {}
        self.executions = {}
        self.reserved_tokens = 0
        self.stale_messages_dropped = 0
        self.resumed_requests = 0
        self.recomputed_tokens = 0

    def start(self, payload, *, epoch, weight_key, paused=False, lifetime_elapsed=0):
        public_id = payload.rid
        if public_id.startswith(EXECUTION_PREFIX) or public_id in self.records:
            raise ValueError("Duplicate or reserved request ID")
        validate_payload(payload)
        cost = len(payload.input_ids) + payload.sampling_params.max_new_tokens
        if (
            len(self.records) >= self.max_requests
            or self.reserved_tokens + cost > self.max_tokens
        ):
            raise RecoveryCapacityError("FT request recovery capacity exceeded")
        snapshot = copy.copy(payload)
        for name in getattr(
            payload,
            "__struct_fields__",
            vars(payload) if hasattr(payload, "__dict__") else (),
        ):
            setattr(
                snapshot,
                name,
                None if name == "time_stats" else copy.deepcopy(getattr(payload, name)),
            )
        record = RecoveryRecord(
            public_id,
            snapshot,
            weight_key,
            cost,
            self.clock() + self.request_timeout - lifetime_elapsed,
            epoch=epoch,
        )
        self.records[public_id] = record
        self.reserved_tokens += cost
        if paused:
            record.paused_at = self.clock()
            return None
        return self._attempt(record, payload.time_stats)

    def _attempt(self, record, time_stats):
        record.execution_id = EXECUTION_PREFIX + uuid.uuid4().hex
        self.executions[record.execution_id] = record.public_id
        wire = copy.deepcopy(record.payload)
        wire.rid = record.execution_id
        wire.time_stats = time_stats
        wire.ft_request_epoch = record.epoch
        wire.ft_recovery_attempt = record.retries
        wire.ft_resume_output_ids = list(record.output_ids)
        wire.ft_decode_checkpoint = copy.deepcopy(record.decode_checkpoint)
        return wire

    def resolve(self, execution_id):
        if not execution_id.startswith(EXECUTION_PREFIX):
            if execution_id not in self.records:
                return execution_id
        elif execution_id in self.executions:
            return self.executions[execution_id]
        self.stale_messages_dropped += 1
        return None

    def freeze(self):
        retired = list(self.executions)
        self.executions.clear()
        for record in self.records.values():
            if record.paused_at is None:
                record.paused_at = self.clock()
        return retired

    def commit_output(
        self, public_id, token_ids, checkpoint, completion_tokens, owner_dp_rank=None
    ):
        record = self.records[public_id]
        count = len(record.output_ids) + len(token_ids)
        if (
            count != completion_tokens
            or count > record.payload.sampling_params.max_new_tokens
        ):
            raise ValueError("FT output token boundary mismatch")
        if checkpoint is not None and (
            not isinstance(checkpoint, dict)
            or not isinstance(checkpoint.get("decode_ids"), list)
            or not isinstance(checkpoint.get("read_offset"), int)
            or not isinstance(checkpoint.get("pending_chars"), int)
            or not 0
            <= checkpoint.get("read_offset", -1)
            <= len(checkpoint["decode_ids"])
            or checkpoint.get("pending_chars", -1) < 0
        ):
            raise ValueError("Invalid FT decode checkpoint")
        record.output_ids.extend(token_ids)
        record.decode_checkpoint = copy.deepcopy(checkpoint)
        record.owner_dp_rank = owner_dp_rank

    def failure(self, record, weight_key):
        now = self.clock()
        if now >= record.deadline:
            return "FT request deadline exceeded"
        waiting = record.recovery_wait + (
            now - record.paused_at if record.paused_at is not None else 0
        )
        if waiting >= self.recovery_timeout:
            return "FT cumulative recovery wait exceeded"
        if record.weight_key != weight_key:
            return "Model/weight version changed during FT request"
        return None

    def resume(self, public_id, *, epoch, weight_key, time_stats):
        record = self.records[public_id]
        if record.paused_at is None:
            return None
        reason = self.failure(record, weight_key)
        if reason:
            raise ValueError(reason)
        if record.retries >= self.max_retries:
            raise ValueError("FT request recovery retry limit exceeded")
        if record.output_ids and record.decode_checkpoint is None:
            raise ValueError("FT decode checkpoint is missing")
        if len(record.output_ids) >= record.payload.sampling_params.max_new_tokens:
            raise ValueError("FT request already reached its output limit")
        record.recovery_wait += self.clock() - record.paused_at
        record.paused_at = None
        record.epoch = epoch
        record.retries += 1
        record.owner_dp_rank = None
        self.resumed_requests += 1
        self.recomputed_tokens += len(record.payload.input_ids) + len(record.output_ids)
        return self._attempt(record, time_stats)

    def finish(self, public_id):
        record = self.records.pop(public_id, None)
        if record:
            self.executions.pop(record.execution_id, None)
            self.reserved_tokens -= record.reserved_tokens
        return record

    def status(self):
        return {
            "active_requests": len(self.records),
            "reserved_tokens": self.reserved_tokens,
            "paused_requests": sum(
                r.paused_at is not None for r in self.records.values()
            ),
            "resumed_requests": self.resumed_requests,
            "recomputed_tokens": self.recomputed_tokens,
            "stale_messages_dropped": self.stale_messages_dropped,
            "active": [
                {
                    "id": r.public_id,
                    "output_tokens": len(r.output_ids),
                    "dp_rank": r.owner_dp_rank,
                    "attempt": r.retries,
                    "epoch": r.epoch,
                    "paused": r.paused_at is not None,
                }
                for r in self.records.values()
            ],
        }


class RequestRecoveryCoordinator:
    """Tokenizer-side integration. All state changes run on its asyncio loop."""

    def __init__(self, manager, retire):
        self.manager = manager
        self.retire = retire
        args = manager.server_args
        self.journal = RequestRecovery(
            max_requests=args.ft_request_recovery_max_requests,
            max_tokens=args.ft_request_recovery_max_tokens,
            max_retries=args.ft_request_recovery_max_retries,
            recovery_timeout=args.ft_request_recovery_timeout,
            request_timeout=args.ft_request_timeout,
        )
        self._expiry_task = None

    def weight_key(self):
        return (self.manager.model_path, self.manager.server_args.weight_version)

    def prepare(self, payload):
        created = getattr(payload.time_stats, "created_time", None)
        elapsed = (
            max(0, time.perf_counter() - created)
            if isinstance(created, (int, float)) and created > 0
            else 0
        )
        if elapsed >= self.journal.request_timeout:
            raise RecoveryDeadlineError("FT request expired before dispatch")
        wire = self.journal.start(
            payload,
            epoch=self.manager.fault_tolerance.state.membership_epoch,
            weight_key=self.weight_key(),
            paused=self.manager.fault_tolerance.should_reject_admission(),
            lifetime_elapsed=elapsed,
        )
        if self._expiry_task is None:
            self._expiry_task = asyncio.create_task(self._expire())
        return wire

    def freeze(self):
        retired = self.journal.freeze()
        if retired:
            self.retire(retired)
            logger.warning("FT paused %d request(s)", len(retired))

    async def resume(self):
        manager = self.manager
        for rid in list(self.journal.records):
            if manager.fault_tolerance.should_reject_admission():
                return
            record = self.journal.records.get(rid)
            state = manager.rid_to_state.get(rid)
            if record is None or state is None or record.paused_at is None:
                continue
            try:
                wire = self.journal.resume(
                    rid,
                    epoch=manager.fault_tolerance.state.membership_epoch,
                    weight_key=self.weight_key(),
                    time_stats=state.time_stats,
                )
                if wire is not None:
                    logger.info(
                        "FT resuming request=%s attempt=%d epoch=%d prefix_tokens=%d",
                        rid,
                        record.retries,
                        record.epoch,
                        len(record.output_ids),
                    )
                    manager._send_one_request(wire, recovery_replay=True)
            except Exception as exc:  # noqa: BLE001 - one request must not stop the others.
                self.abort(rid, str(exc))
            await asyncio.sleep(
                0
            )  # Let cancellations/fault observations run between dispatches.

    def abort(self, public_id, reason, status_code=503):
        from sglang.srt.managers.io_struct import AbortReq

        record = self.journal.finish(public_id)
        if record is None:
            return False
        if record.execution_id:
            self.retire([record.execution_id])
        # The public ID resolves normally after finish; this is the sole final output.
        finish_reason = {
            "type": "abort",
            "message": reason,
            "err_type": "RequestRecoveryError",
        }
        if status_code is not None:
            finish_reason["status_code"] = status_code
        self.manager._handle_abort_req(
            AbortReq(rid=public_id, finished_reason=finish_reason)
        )
        if record.execution_id:
            self.manager._dispatch_to_scheduler(AbortReq(rid=record.execution_id))
        return True

    def finish(self, public_id):
        record = self.journal.finish(public_id)
        if record and record.execution_id:
            self.retire([record.execution_id])

    async def _expire(self):
        try:
            while self.journal.records:
                await asyncio.sleep(1)
                for rid, record in list(self.journal.records.items()):
                    reason = self.journal.failure(record, self.weight_key())
                    if reason:
                        self.abort(rid, reason)
        finally:
            self._expiry_task = None
