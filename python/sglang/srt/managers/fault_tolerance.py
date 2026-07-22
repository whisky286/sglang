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
from typing import Callable, Dict, FrozenSet, Optional, Tuple

from sglang.srt.managers.io_struct import (
    FaultToleranceCommandReqInput,
    FaultToleranceCommandReqOutput,
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

    Stage A1 implements only the read-only ``status`` command. Topology,
    process groups, DeepEP state, and expert metadata are deliberately outside
    this object.
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

    async def status(self) -> Tuple[int, dict]:
        """Query every original rank without changing the runtime state."""

        command_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        pending = _PendingCommand(
            command="status",
            target_original_ranks=frozenset(self.original_ranks),
            future=loop.create_future(),
        )
        self._pending_commands[command_id] = pending

        request = FaultToleranceCommandReqInput(
            command_id=command_id,
            command="status",
            target_original_ranks=list(self.original_ranks),
        )
        logger.info(
            "[FaultTolerance] dispatch status command_id=%s original_ranks=%s",
            command_id,
            list(self.original_ranks),
        )

        dispatch_error: Optional[str] = None
        timed_out = False
        try:
            self._dispatch_command(request)
            await asyncio.wait_for(pending.future, timeout=self.command_timeout)
        except asyncio.TimeoutError:
            timed_out = True
        except Exception as exc:
            dispatch_error = f"Failed to dispatch status command: {exc}"
            logger.exception(dispatch_error)
        finally:
            self._pending_commands.pop(command_id, None)

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
        if complete and paused_values == {False}:
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
        }
        return (200 if complete else 503), body
