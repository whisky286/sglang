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
"""Minimal control-plane state used by fault-tolerance experiments."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import uuid
from typing import Callable, Dict, FrozenSet, Iterable, Optional, Tuple

from sglang.srt.managers.io_struct import (
    FaultToleranceCommandReqInput,
    FaultToleranceCommandReqOutput,
    FaultToleranceRecoverableErrorOutput,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(slots=True)
class _PendingCommand:
    command: str
    target_original_ranks: FrozenSet[int]
    future: asyncio.Future
    outputs: Dict[int, FaultToleranceCommandReqOutput] = dataclasses.field(
        default_factory=dict
    )


class FaultToleranceManager:
    """Fan out a command and correlate one acknowledgement per original rank.

    Stages A1/A2 implement status, one-shot test injection, and coordinated
    pause. Topology, process groups, DeepEP state, and expert metadata are
    deliberately outside this object.
    """

    def __init__(
        self,
        *,
        original_world_size: int,
        command_timeout: float,
        dispatch_command: Callable[[FaultToleranceCommandReqInput], None],
    ) -> None:
        if original_world_size <= 0:
            raise ValueError("original_world_size must be greater than zero")
        if command_timeout <= 0:
            raise ValueError("command_timeout must be greater than zero")

        self.original_ranks = tuple(range(original_world_size))
        self.command_timeout = command_timeout
        self._dispatch_command = dispatch_command
        self._pending_commands: Dict[str, _PendingCommand] = {}
        self.admission_closed = False
        self.armed_injection: Optional[dict] = None
        self.last_fault: Optional[dict] = None
        self.last_transition: Optional[dict] = None
        self._pause_task: Optional[asyncio.Task] = None

    async def status(self) -> Tuple[int, dict]:
        """Query every original rank without changing the runtime state."""

        command_id, pending, timed_out, dispatch_error = await self._execute_command(
            command="status",
            target_original_ranks=self.original_ranks,
        )
        response = self._build_status_response(
            command_id=command_id,
            pending=pending,
            timed_out=timed_out,
            dispatch_error=dispatch_error,
        )
        logger.info(
            "[FaultTolerance] finish status command_id=%s http_status=%s "
            "received_ranks=%s",
            command_id,
            response[0],
            sorted(pending.outputs),
        )
        return response

    async def arm_recoverable_error(
        self, *, original_rank: int, request_id: str
    ) -> Tuple[int, dict]:
        """Arm a one-shot, caught exception on one original Scheduler rank."""

        if original_rank not in self.original_ranks:
            return 400, {
                "success": False,
                "last_error": f"original_rank {original_rank} is out of range",
            }
        if not request_id or not request_id.strip():
            return 400, {
                "success": False,
                "last_error": "request_id must not be empty",
            }
        if self.admission_closed:
            return 409, {
                "success": False,
                "last_error": "Inference admission is already closed",
            }
        if self.armed_injection is not None:
            return 409, {
                "success": False,
                "last_error": (
                    "A recoverable error injection is already armed for request "
                    f"{self.armed_injection['request_id']}"
                ),
            }

        self.armed_injection = {
            "original_rank": original_rank,
            "request_id": request_id,
        }
        command_id, pending, timed_out, dispatch_error = await self._execute_command(
            command="arm_recoverable_error",
            target_original_ranks=(original_rank,),
            request_id=request_id,
        )
        missing_ranks = sorted(
            pending.target_original_ranks.difference(pending.outputs)
        )
        failed_ranks = sorted(
            rank for rank, output in pending.outputs.items() if not output.success
        )
        success = (
            not missing_ranks
            and not failed_ranks
            and dispatch_error is None
            and not timed_out
        )
        errors = []
        if dispatch_error is not None:
            errors.append(dispatch_error)
        if timed_out:
            errors.append(
                "Timed out waiting for injection arm ack from original ranks "
                f"{missing_ranks}"
            )
        if failed_ranks:
            errors.append(
                f"Injection arm command failed on original ranks {failed_ranks}"
            )
        if not success:
            # A timeout cannot prove that the target Scheduler was not armed.
            # Fail closed so a later matching request cannot trigger an
            # uncoordinated exception.
            self.armed_injection = None
            self.admission_closed = True
            self.last_transition = {
                "command": "arm_recoverable_error",
                "command_id": command_id,
                "state": "FAILED",
                "acknowledged_ranks": sorted(pending.outputs),
                "missing_ranks": missing_ranks,
                "failed_ranks": failed_ranks,
                "last_error": "; ".join(errors) if errors else None,
            }
        return (200 if success else 503), {
            "success": success,
            "command_id": command_id,
            "original_rank": original_rank,
            "request_id": request_id,
            "acknowledged_ranks": sorted(pending.outputs),
            "missing_ranks": missing_ranks,
            "failed_ranks": failed_ranks,
            "last_error": "; ".join(errors) if errors else None,
        }

    async def apply_retry(self) -> Tuple[int, dict]:
        """Resume every original rank without changing topology or resources."""

        transition = self.last_transition or {}
        if not self.admission_closed:
            return 409, {
                "success": False,
                "action": "retry",
                "last_error": "Inference admission is not closed",
            }
        if self.armed_injection is not None:
            return 409, {
                "success": False,
                "action": "retry",
                "last_error": "A recoverable error injection is still armed",
            }
        if (
            transition.get("command") != "pause"
            or transition.get("state") != "SUCCEEDED"
        ):
            return 409, {
                "success": False,
                "action": "retry",
                "last_error": "All original ranks must confirm pause before retry",
            }

        self.last_transition = {
            "command": "retry",
            "state": "PENDING",
        }
        command_id, pending, timed_out, dispatch_error = await self._execute_command(
            command="retry",
            target_original_ranks=self.original_ranks,
        )
        missing_ranks = sorted(
            pending.target_original_ranks.difference(pending.outputs)
        )
        failed_ranks = sorted(
            rank for rank, output in pending.outputs.items() if not output.success
        )
        success = (
            not missing_ranks
            and not failed_ranks
            and dispatch_error is None
            and not timed_out
        )
        errors = []
        if dispatch_error is not None:
            errors.append(dispatch_error)
        if timed_out:
            errors.append(
                f"Timed out waiting for retry ack from original ranks {missing_ranks}"
            )
        if failed_ranks:
            errors.append(f"Retry command failed on original ranks {failed_ranks}")

        self.last_transition = {
            "command": "retry",
            "command_id": command_id,
            "state": "SUCCEEDED" if success else "FAILED",
            "acknowledged_ranks": sorted(pending.outputs),
            "missing_ranks": missing_ranks,
            "failed_ranks": failed_ranks,
            "last_error": "; ".join(errors) if errors else None,
        }
        if success:
            # Keep admission closed until every original rank has acknowledged
            # that its existing runtime is ready to run again.
            self.admission_closed = False

        logger.info(
            "[FaultTolerance] finish retry command_id=%s state=%s "
            "acknowledged_ranks=%s",
            command_id,
            self.last_transition["state"],
            self.last_transition["acknowledged_ranks"],
        )
        return (200 if success else 503), {
            "success": success,
            "action": "retry",
            "command_id": command_id,
            "acknowledged_ranks": sorted(pending.outputs),
            "missing_ranks": missing_ranks,
            "failed_ranks": failed_ranks,
            "last_error": "; ".join(errors) if errors else None,
        }

    def handle_recoverable_error(
        self, output: FaultToleranceRecoverableErrorOutput
    ) -> None:
        """Close admission immediately and coordinate a pause on every rank."""

        if output.original_rank not in self.original_ranks:
            logger.warning(
                "Ignore recoverable error from invalid original rank %s",
                output.original_rank,
            )
            return
        if self.armed_injection != {
            "original_rank": output.original_rank,
            "request_id": output.request_id,
        }:
            logger.warning(
                "Ignore recoverable error that does not match the armed injection "
                "event_id=%s original_rank=%s request_id=%s armed=%s",
                output.event_id,
                output.original_rank,
                output.request_id,
                self.armed_injection,
            )
            return

        self.armed_injection = None
        self.admission_closed = True
        self.last_fault = {
            "event_id": output.event_id,
            "original_rank": output.original_rank,
            "request_id": output.request_id,
            "message": output.message,
        }

        if self.last_transition is not None and self.last_transition.get("state") in (
            "PENDING",
            "SUCCEEDED",
        ):
            logger.info(
                "[FaultTolerance] admission already closed; ignore duplicate "
                "recoverable error event_id=%s original_rank=%s",
                output.event_id,
                output.original_rank,
            )
            return

        self.last_transition = {
            "command": "pause",
            "state": "PENDING",
            "trigger_event_id": output.event_id,
        }
        logger.error(
            "[FaultTolerance] recoverable error event_id=%s original_rank=%s "
            "request_id=%s; closing admission and pausing all original ranks",
            output.event_id,
            output.original_rank,
            output.request_id,
        )
        self._pause_task = asyncio.get_running_loop().create_task(
            self._pause_after_recoverable_error(output.event_id)
        )

    def get_admission_error(
        self, *, request_id: Optional[str], routed_dp_rank: Optional[int]
    ) -> Optional[str]:
        """Return a rejection reason without waiting on a pause condition."""

        if self.admission_closed:
            return "Inference is paused after a recoverable worker error."
        if self.armed_injection is None:
            return None
        if (
            request_id == self.armed_injection["request_id"]
            and routed_dp_rank == self.armed_injection["original_rank"]
        ):
            return None
        return (
            "A fault-injection experiment is armed; only its target request is allowed."
        )

    async def _pause_after_recoverable_error(self, event_id: str) -> None:
        command_id, pending, timed_out, dispatch_error = await self._execute_command(
            command="pause",
            target_original_ranks=self.original_ranks,
        )
        missing_ranks = sorted(
            pending.target_original_ranks.difference(pending.outputs)
        )
        failed_ranks = sorted(
            rank for rank, output in pending.outputs.items() if not output.success
        )
        success = (
            not missing_ranks
            and not failed_ranks
            and dispatch_error is None
            and not timed_out
        )
        errors = []
        if dispatch_error is not None:
            errors.append(dispatch_error)
        if timed_out:
            errors.append(
                f"Timed out waiting for pause ack from original ranks {missing_ranks}"
            )
        if failed_ranks:
            errors.append(f"Pause command failed on original ranks {failed_ranks}")

        self.last_transition = {
            "command": "pause",
            "command_id": command_id,
            "state": "SUCCEEDED" if success else "FAILED",
            "trigger_event_id": event_id,
            "acknowledged_ranks": sorted(pending.outputs),
            "missing_ranks": missing_ranks,
            "failed_ranks": failed_ranks,
            "last_error": "; ".join(errors) if errors else None,
        }
        logger.info(
            "[FaultTolerance] finish pause command_id=%s state=%s "
            "acknowledged_ranks=%s",
            command_id,
            self.last_transition["state"],
            self.last_transition["acknowledged_ranks"],
        )

    async def _execute_command(
        self,
        *,
        command: str,
        target_original_ranks: Iterable[int],
        request_id: Optional[str] = None,
    ) -> Tuple[str, _PendingCommand, bool, Optional[str]]:
        """Dispatch one command and collect one acknowledgement per target rank."""

        targets = frozenset(target_original_ranks)
        command_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        pending = _PendingCommand(
            command=command,
            target_original_ranks=targets,
            future=loop.create_future(),
        )
        self._pending_commands[command_id] = pending

        request = FaultToleranceCommandReqInput(
            command_id=command_id,
            command=command,
            target_original_ranks=sorted(targets),
            request_id=request_id,
        )
        logger.info(
            "[FaultTolerance] dispatch command=%s command_id=%s original_ranks=%s",
            command,
            command_id,
            sorted(targets),
        )

        dispatch_error: Optional[str] = None
        timed_out = False
        try:
            self._dispatch_command(request)
            await asyncio.wait_for(pending.future, timeout=self.command_timeout)
        except asyncio.TimeoutError:
            timed_out = True
        except Exception as exc:
            dispatch_error = f"Failed to dispatch {command} command: {exc}"
            logger.exception(dispatch_error)
        finally:
            self._pending_commands.pop(command_id, None)

        return command_id, pending, timed_out, dispatch_error

    def handle_command_output(self, output: FaultToleranceCommandReqOutput) -> None:
        """Record a valid acknowledgement; ignore stale or duplicate output."""

        pending = self._pending_commands.get(output.command_id)
        if pending is None:
            logger.debug(
                "Ignore stale fault-tolerance acknowledgement command_id=%s rank=%s",
                output.command_id,
                output.original_rank,
            )
            return
        if output.command != pending.command:
            logger.warning(
                "Ignore fault-tolerance acknowledgement with mismatched command "
                "command_id=%s expected=%s actual=%s rank=%s",
                output.command_id,
                pending.command,
                output.command,
                output.original_rank,
            )
            return
        if output.original_rank not in pending.target_original_ranks:
            logger.warning(
                "Ignore fault-tolerance acknowledgement from untargeted rank "
                "command_id=%s rank=%s",
                output.command_id,
                output.original_rank,
            )
            return
        if output.original_rank in pending.outputs:
            logger.debug(
                "Ignore duplicate fault-tolerance acknowledgement command_id=%s rank=%s",
                output.command_id,
                output.original_rank,
            )
            return

        pending.outputs[output.original_rank] = output
        if (
            len(pending.outputs) == len(pending.target_original_ranks)
            and not pending.future.done()
        ):
            pending.future.set_result(None)

    def _build_status_response(
        self,
        *,
        command_id: str,
        pending: _PendingCommand,
        timed_out: bool,
        dispatch_error: Optional[str],
    ) -> Tuple[int, dict]:
        missing_ranks = sorted(
            pending.target_original_ranks.difference(pending.outputs)
        )
        failed_ranks = sorted(
            rank for rank, output in pending.outputs.items() if not output.success
        )
        complete = not missing_ranks and not failed_ranks and dispatch_error is None

        rank_status = []
        for rank in self.original_ranks:
            output = pending.outputs.get(rank)
            rank_status.append(
                {
                    "original_rank": rank,
                    "available": bool(output is not None and output.success),
                    "engine_paused": (
                        output.engine_paused if output is not None else None
                    ),
                    "message": output.message if output is not None else "No ack",
                }
            )

        paused_values = {
            output.engine_paused
            for output in pending.outputs.values()
            if output.success
        }
        transition_state = (
            self.last_transition.get("state") if self.last_transition else None
        )
        transition_command = (
            self.last_transition.get("command") if self.last_transition else None
        )
        if self.admission_closed and transition_state == "FAILED":
            service_state = "FAIL_STOP"
        elif self.admission_closed and transition_state == "PENDING":
            service_state = "RESUMING" if transition_command == "retry" else "PAUSING"
        elif self.admission_closed and complete and paused_values != {True}:
            service_state = "INCONSISTENT"
        elif complete and paused_values == {False}:
            service_state = "HEALTHY"
        elif complete and paused_values == {True}:
            service_state = "PAUSED"
        elif complete:
            service_state = "INCONSISTENT"
        else:
            service_state = "UNKNOWN"

        errors = []
        if dispatch_error is not None:
            errors.append(dispatch_error)
        if timed_out:
            errors.append(
                "Timed out waiting for status ack from original ranks "
                f"{missing_ranks}"
            )
        if failed_ranks:
            errors.append(f"Status command failed on original ranks {failed_ranks}")

        body = {
            "success": complete,
            "service_state": service_state,
            "command_id": command_id,
            "original_ranks": list(self.original_ranks),
            "ranks": rank_status,
            "missing_ranks": missing_ranks,
            "failed_ranks": failed_ranks,
            "last_error": "; ".join(errors) if errors else None,
            "admission_closed": self.admission_closed,
            "armed_injection": (
                dict(self.armed_injection) if self.armed_injection else None
            ),
            "last_fault": dict(self.last_fault) if self.last_fault else None,
            "last_transition": (
                dict(self.last_transition) if self.last_transition else None
            ),
        }
        return (200 if complete else 503), body
