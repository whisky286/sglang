"""CPU-only request quiescence before changing a healthy NPU membership."""

import asyncio
import time


class RequestDrainError(RuntimeError):
    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


async def drain_requests(*, pending, check_idle, has_incident, timeout):
    """Keep serving admitted requests while the caller holds admission closed."""
    deadline = time.monotonic() + timeout
    while True:
        if has_incident():
            raise RequestDrainError("recover_drain_interrupted_by_incident", 409)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RequestDrainError("recover_drain_timeout", 408)
        if not pending():
            try:
                idle = await asyncio.wait_for(check_idle(remaining), remaining)
            except asyncio.TimeoutError as exc:
                raise RequestDrainError("recover_drain_timeout", 408) from exc
            except Exception as exc:
                raise RequestDrainError("recover_drain_check_failed", 503) from exc
            if has_incident():
                raise RequestDrainError("recover_drain_interrupted_by_incident", 409)
            if idle and not pending():
                return
        await asyncio.sleep(min(0.05, max(0, deadline - time.monotonic())))


def scheduler_has_pending_requests(*, queues, batches, result_queue, chunked_req):
    """Read host metadata only, including finished requests with unreleased KV."""
    requests = [req for queue in queues for req in queue]
    for batch in (*batches, *(batch for batch, _ in result_queue)):
        if batch is not None:
            requests.extend(batch.reqs)
    if chunked_req is not None:
        requests.append(chunked_req)
    return any(
        not req.finished()
        or getattr(req, "req_pool_idx", None) is not None
        or getattr(req, "kv", None) is not None
        for req in requests
    )
