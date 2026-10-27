"""Probe fresh process-group construction across N -> N-1 -> N processes.

This uses new candidate HCCL groups; it does not claim that an existing MC2
buffer or captured graph can be rebound. See README_npu_rank_recovery.md.
"""

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist


def _candidate_group(store, *, prefix, backend, rank, world_size, timeout):
    # Match init_custom_process_group's members-only construction: [] bypasses
    # parent-WORLD membership, which may still contain the removed process.
    from torch.distributed.distributed_c10d import (
        Backend,
        _new_process_group_helper,
        _world,
    )

    version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    options = "backend_options" if version >= (2, 6) else "pg_options"
    group, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        Backend(backend),
        dist.PrefixStore(prefix, store),
        group_name=prefix.replace("/", "_"),
        timeout=timeout,
        **{options: None},
    )
    _world.pg_group_ranks[group] = {index: index for index in range(world_size)}
    return group


def _wait(work, timeout):
    if work.wait(timeout=timeout) is False:
        raise TimeoutError("candidate collective timed out")


def _worker(original_rank, port, backend, timeout_sec, channel):
    try:
        torch.set_num_threads(1)
        if backend == "hccl":
            import torch_npu  # noqa: F401 -- registers the HCCL backend

            torch.npu.set_device(original_rank)
        device = f"npu:{original_rank}" if backend == "hccl" else "cpu"
        timeout = timedelta(seconds=timeout_sec)
        store = dist.TCPStore(
            "127.0.0.1",
            port,
            is_master=False,
            timeout=timeout,
            wait_for_workers=False,
        )
        groups = []
        while (command := channel.recv()) is not None:
            mask = command["active_mask"]
            epoch = command["membership_epoch"]
            active = [rank for rank, enabled in enumerate(mask) if enabled]
            compact_rank = active.index(original_rank)
            prefix = (
                f"npu-ft/{''.join('1' if value else '0' for value in mask)}/{epoch}"
            )
            control = _candidate_group(
                store,
                prefix=f"{prefix}/mlp-sync",
                backend="gloo",
                rank=compact_rank,
                world_size=len(active),
                timeout=timeout,
            )
            data = _candidate_group(
                store,
                prefix=f"{prefix}/candidate-data",
                backend=backend,
                rank=compact_rank,
                world_size=len(active),
                timeout=timeout,
            )
            # Keep old groups alive: retiring groups containing dead ranks is
            # a separate backend contract, not part of this probe.
            groups.extend((control, data))
            local_rank = torch.tensor([original_rank], dtype=torch.int64)
            gathered = [torch.empty_like(local_rank) for _ in active]
            dist.all_gather(gathered, local_rank, group=control)
            observed = [int(value.item()) for value in gathered]
            if observed != active:
                raise RuntimeError(f"physical-rank mismatch: {observed} != {active}")

            value = torch.tensor([original_rank + 1.0], device=device)
            _wait(dist.all_reduce(value, group=data, async_op=True), timeout)
            if backend == "hccl":
                torch.npu.synchronize()
            total = value.item()
            if total != sum(rank + 1 for rank in active):
                raise RuntimeError(f"incorrect candidate all-reduce: {total}")

            # Direct PG P2P uses compact peer indices, with no default WORLD.
            payload = torch.tensor([float(epoch)], device=device)
            if compact_rank == 0:
                _wait(data.send([payload], len(active) - 1, 0), timeout)
            elif compact_rank == len(active) - 1:
                payload.zero_()
                _wait(data.recv([payload], 0, 0), timeout)
                if payload.item() != epoch:
                    raise RuntimeError("candidate P2P payload mismatch")
            dist.barrier(group=control)
            channel.send(
                {
                    "rank": original_rank,
                    "pid": os.getpid(),
                    "epoch": epoch,
                    "active_ranks": observed,
                    "sum": total,
                    "success": True,
                }
            )
    except Exception:
        channel.send(
            {"rank": original_rank, "success": False, "error": traceback.format_exc()}
        )
    finally:
        channel.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hccl", "gloo"), default="hccl")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--fault-rank", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    if (
        args.world_size < 3
        or not 0 <= args.fault_rank < args.world_size
        or args.timeout <= 0
    ):
        parser.error("require world-size >= 3, a valid fault-rank, and timeout > 0")

    from sglang.srt.fault_tolerance.recovery import RankRecoveryPlan

    context = mp.get_context("spawn")
    timeout = timedelta(seconds=args.timeout)
    store = dist.TCPStore(
        "127.0.0.1", 0, is_master=True, timeout=timeout, wait_for_workers=False
    )
    peers = {}
    all_processes = []

    def launch(rank):
        parent, child = context.Pipe()
        process = context.Process(
            target=_worker, args=(rank, store.port, args.backend, args.timeout, child)
        )
        process.start()
        child.close()
        peers[rank] = (process, parent)
        all_processes.append(process)

    def run_phase(mask, epoch):
        active = [rank for rank, enabled in enumerate(mask) if enabled]
        for rank in active:
            peers[rank][1].send({"active_mask": mask, "membership_epoch": epoch})
        deadline = time.monotonic() + args.timeout
        reports = []
        for rank in active:
            channel = peers[rank][1]
            if not channel.poll(max(0, deadline - time.monotonic())):
                raise TimeoutError(f"phase {epoch} waiting for original rank {rank}")
            report = channel.recv()
            if not report["success"]:
                raise RuntimeError(report["error"])
            reports.append(report)
        print(
            json.dumps({"backend": args.backend, "phase": epoch, "reports": reports}),
            flush=True,
        )
        return {report["rank"]: report["pid"] for report in reports}

    try:
        for rank in range(args.world_size):
            launch(rank)
        initial_mask = [True] * args.world_size
        initial_pids = run_phase(initial_mask, 0)
        failed, channel = peers.pop(args.fault_rank)
        failed.kill()
        failed.join(timeout=args.timeout)
        channel.close()
        if failed.is_alive():
            raise TimeoutError("removed process did not exit")
        reduced_mask = list(initial_mask)
        reduced_mask[args.fault_rank] = False
        run_phase(reduced_mask, 1)

        plan = RankRecoveryPlan(2, reduced_mask, (args.fault_rank,))
        launch(args.fault_rank)
        restored_pids = run_phase(plan.target_active_mask, plan.membership_epoch)
        if restored_pids[args.fault_rank] == initial_pids[args.fault_rank]:
            raise RuntimeError("recovery did not use a fresh process")
        if any(
            restored_pids[rank] != pid
            for rank, pid in initial_pids.items()
            if rank != args.fault_rank
        ):
            raise RuntimeError("surviving processes restarted")
        print(
            json.dumps(
                {
                    "success": True,
                    "scope": "candidate process groups only; MC2 and graph rebind remain unverified",
                }
            ),
            flush=True,
        )
    finally:
        for process, channel in peers.values():
            try:
                channel.close()
            finally:
                if process.is_alive():
                    process.terminate()
        for process in all_processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)


if __name__ == "__main__":
    main()
