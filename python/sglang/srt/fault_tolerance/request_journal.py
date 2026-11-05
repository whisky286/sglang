"""Bounded CPU snapshots and private execution IDs for future request replay."""

import copy
import uuid
from dataclasses import dataclass, field

EXECUTION_PREFIX = "__sglang_ft_attempt__"


class RequestJournalFull(RuntimeError):
    pass


@dataclass
class RequestRecord:
    public_id: str
    execution_id: str
    membership_epoch: int
    snapshot: object
    reserved_tokens: int
    output_limit: int
    output_ids: list[int] = field(default_factory=list)
    cancelled: bool = False
    replayable: bool = True


class RequestJournal:
    def __init__(self, *, max_requests=128, max_tokens=262144):
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.records: dict[str, RequestRecord] = {}
        self._executions: dict[str, str] = {}
        self.reserved_tokens = 0
        self.stale_messages_dropped = 0

    def validate_request_ids(self, public_ids):
        if any(public_id.startswith(EXECUTION_PREFIX) for public_id in public_ids):
            raise ValueError("Request ID uses the reserved NPU FT execution namespace")

    def start(self, payload, *, membership_epoch):
        public_id = payload.rid
        self.validate_request_ids((public_id,))
        if public_id in self.records:
            raise ValueError("Request already has an active execution")
        for name in (
            "mm_inputs",
            "input_embeds",
            "positional_embed_overrides",
            "lora_id",
            "session_id",
            "session_params",
            "custom_logit_processor",
            "mm_data_mooncake",
        ):
            if getattr(payload, name, None) is not None:
                return payload
        output_limit = payload.sampling_params.max_new_tokens
        if (
            payload.input_ids is None
            or not isinstance(output_limit, int)
            or output_limit < 0
        ):
            return payload
        cost = len(payload.input_ids) + output_limit
        if (
            len(self.records) >= self.max_requests
            or self.reserved_tokens + cost > self.max_tokens
        ):
            raise RequestJournalFull("NPU FT request journal capacity exceeded")

        # Do not deepcopy tracing/timing objects, sockets or a live device state.
        # Supported plain-text IPC fields are copied before wrap_pickle_fields.
        snapshot = copy.copy(payload)
        field_names = getattr(payload, "__struct_fields__", None)
        if field_names is None:
            field_names = vars(payload)
        for name in field_names:
            setattr(
                snapshot,
                name,
                None if name == "time_stats" else copy.deepcopy(getattr(payload, name)),
            )
        execution_id = EXECUTION_PREFIX + uuid.uuid4().hex
        wire = copy.deepcopy(snapshot)
        wire.rid = execution_id
        wire.time_stats = payload.time_stats
        self.records[public_id] = RequestRecord(
            public_id, execution_id, membership_epoch, snapshot, cost, output_limit
        )
        self._executions[execution_id] = public_id
        self.reserved_tokens += cost
        return wire

    def resolve(self, execution_id):
        if execution_id.startswith(EXECUTION_PREFIX):
            public_id = self._executions.get(execution_id)
            if public_id is not None:
                return public_id
        elif execution_id not in self.records:
            return execution_id  # Health probes and legacy, untracked requests.
        self.stale_messages_dropped += 1
        return None

    def execution_id(self, public_id):
        record = self.records.get(public_id)
        return record.execution_id if record is not None else public_id

    def commit_output(self, public_id, token_ids):
        record = self.records.get(public_id)
        if record is None or not record.replayable:
            return
        if len(record.output_ids) + len(token_ids) > record.output_limit:
            # Never crash the result loop or silently save a truncated prefix.
            # Delivery continues, but a future replay policy must refuse it.
            record.replayable = False
            record.output_ids.clear()
            record.snapshot = None
            return
        record.output_ids.extend(token_ids)

    def cancel(self, public_id, *, all_requests=False):
        records = (
            self.records.values() if all_requests else [self.records.get(public_id)]
        )
        for record in records:
            if record is not None:
                record.cancelled = True

    def finish(self, public_id):
        record = self.records.pop(public_id, None)
        if record is not None:
            self._executions.pop(record.execution_id, None)
            self.reserved_tokens -= record.reserved_tokens

    def status(self):
        return {
            "active_requests": len(self.records),
            "reserved_tokens": self.reserved_tokens,
            "stale_messages_dropped": self.stale_messages_dropped,
        }
