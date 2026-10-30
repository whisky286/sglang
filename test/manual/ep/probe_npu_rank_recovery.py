"""Probe fresh process-group construction across N -> N-1 -> N processes.

This uses new candidate HCCL groups; it does not claim that an existing MC2
buffer or captured graph can be rebound. See README_npu_rank_recovery.md.
"""

import argparse
import importlib.metadata
import inspect
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


def _mc2_roundtrip(group, *, original_rank, config, device, synchronize):
    """Test a fresh low-latency buffer, with identity work in expert slots.

    Expert IDs here are compact within the candidate group. This deliberately
    does not exercise serving's original-rank elastic_info or old graph replay.
    """
    import deep_ep

    local_experts = config["local_experts"]
    num_tokens, hidden, topk = config["tokens"], config["hidden"], config["topk"]
    num_experts = local_experts * group.size()
    max_tokens = 2 * num_tokens
    rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        max_tokens, hidden, group.size(), num_experts
    )
    # This wrapper selects its strategy from DEEP_USE_MODE in Buffer.__init__;
    # its constructor strategy argument is overwritten by that environment.
    # Select only this fresh buffer's strategy, then restore the worker's env.
    inherited_mode = os.environ.get("DEEP_USE_MODE")
    try:
        os.environ["DEEP_USE_MODE"] = config["strategy"]
        buffer = deep_ep.Buffer(
            group,
            0,
            rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=local_experts,
            allow_mnnvl=False,
        )
    finally:
        if inherited_mode is None:
            os.environ.pop("DEEP_USE_MODE", None)
        else:
            os.environ["DEEP_USE_MODE"] = inherited_mode
    if getattr(buffer, "moe_all_to_all_group_name", None) == "":
        raise RuntimeError("fresh DeepEP buffer has an empty HCCL communicator name")
    strategy = getattr(buffer, "low_latency_strategy", None)
    get_name = getattr(strategy, "get_name", None)
    effective_strategy = get_name() if callable(get_name) else None
    if effective_strategy != config["strategy"]:
        raise RuntimeError(
            f"DeepEP strategy mismatch: requested {config['strategy']!r}, "
            f"got {effective_strategy!r} ({type(strategy).__name__})"
        )
    buffer_info = {
        "group_size": group.size(),
        "compact_rank": group.rank(),
        "requested_strategy": config["strategy"],
        "effective_strategy": effective_strategy,
        "strategy": type(strategy).__name__,
        "comm_alg": getattr(strategy, "comm_alg", None),
        "hccl_comm_name": getattr(buffer, "moe_all_to_all_group_name", None),
    }
    if group.rank() == 0:
        print(
            json.dumps({"stage": "MC2 buffer ready before dispatch", **buffer_info}),
            flush=True,
        )

    # Small integral values keep the identity result exact in BF16, even when
    # top-k contributions are accumulated in a different order.
    rows = torch.arange(num_tokens, device=device).view(-1, 1)
    columns = torch.arange(hidden, device=device).view(1, -1)
    x = (rows % 2 + columns % 2 + original_rank % 4).to(torch.bfloat16)
    choices = torch.arange(topk, device=device).view(1, -1)
    topk_ids = (
        group.rank() * local_experts + rows + choices * (local_experts + 1)
    ) % num_experts
    weights = torch.ones((num_tokens, topk), dtype=torch.float32, device=device)
    packed, counts, handle, _, _ = buffer.low_latency_dispatch(
        x,
        topk_ids,
        max_tokens,
        num_experts,
        use_fp8=False,
        topk_weights=weights,
        async_finish=False,
        return_recv_hook=False,
    )
    synchronize()
    if not isinstance(packed, torch.Tensor) or packed.dtype != torch.bfloat16:
        raise RuntimeError("BF16 dispatch did not return an unquantized tensor")
    # Identity expert computation: preserve the wrapper's packed layout.
    combined, _, _ = buffer.low_latency_combine(
        x=packed.clone(),
        topk_idx=topk_ids,
        topk_weights=weights,
        handle=handle,
        async_finish=False,
        return_recv_hook=False,
    )
    synchronize()
    expected = x * topk
    if combined.dtype != expected.dtype:
        raise RuntimeError(
            f"MC2 combine dtype mismatch: {combined.dtype} != {expected.dtype}"
        )
    if combined.shape != expected.shape:
        raise RuntimeError(
            f"MC2 combine shape mismatch: {combined.shape} != {expected.shape}"
        )
    error = (combined.float() - expected.float()).abs().max().item()
    if not torch.equal(combined, expected):
        raise RuntimeError(
            f"MC2 identity dispatch/combine mismatch: max_abs_error={error}"
        )
    return buffer, {
        **buffer_info,
        "success": True,
        "num_experts": num_experts,
        "local_experts": local_experts,
        "max_abs_error": error,
        "received_counts": counts.detach().cpu().tolist(),
        "wrapper": str(deep_ep.__file__),
    }


def _mc2_environment():
    import deep_ep
    import torch_npu

    versions = {"torch": torch.__version__, "torch_npu": torch_npu.__version__}
    for name in ("sgl-kernel-npu", "deep-ep"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "versions": versions,
        "wrapper": str(deep_ep.__file__),
        "dispatch_signature": str(
            inspect.signature(deep_ep.Buffer.low_latency_dispatch)
        ),
        "combine_signature": str(inspect.signature(deep_ep.Buffer.low_latency_combine)),
        "DEEP_USE_MODE": os.getenv("DEEP_USE_MODE"),
    }


def _worker(original_rank, port, backend, timeout_sec, channel, mc2_config=None):
    stage = "initialization"
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
        buffers = []
        while (command := channel.recv()) is not None:
            mask = command["active_mask"]
            epoch = command["membership_epoch"]
            active = [rank for rank, enabled in enumerate(mask) if enabled]
            compact_rank = active.index(original_rank)
            stage = f"epoch {epoch}: candidate groups"
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
            mc2_report = None
            if mc2_config is not None:
                stage = f"epoch {epoch}: fresh DeepEP buffer and identity roundtrip"
                if compact_rank == 0:
                    print(
                        json.dumps({"stage": stage, "environment": _mc2_environment()}),
                        flush=True,
                    )
                buffer, mc2_report = _mc2_roundtrip(
                    data,
                    original_rank=original_rank,
                    config=mc2_config,
                    device=device,
                    synchronize=torch.npu.synchronize,
                )
                buffers.append(buffer)
            stage = f"epoch {epoch}: final control barrier"
            dist.barrier(group=control)
            channel.send(
                {
                    "rank": original_rank,
                    "pid": os.getpid(),
                    "epoch": epoch,
                    "active_ranks": observed,
                    "sum": total,
                    "success": True,
                    **({"mc2": mc2_report} if mc2_config is not None else {}),
                }
            )
    except Exception:
        channel.send(
            {
                "rank": original_rank,
                "success": False,
                "stage": stage,
                "error": traceback.format_exc(),
            }
        )
    finally:
        channel.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hccl", "gloo"), default="hccl")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--fault-rank", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument(
        "--mc2",
        action="store_true",
        help="also test a fresh DeepEP low-latency buffer at every phase",
    )
    parser.add_argument("--mc2-hidden", type=int, default=2048)
    parser.add_argument("--mc2-local-experts", type=int, default=32)
    parser.add_argument("--mc2-tokens", type=int, default=16)
    parser.add_argument("--mc2-topk", type=int, default=8)
    parser.add_argument(
        "--mc2-strategy",
        choices=("default", "ops"),
        default="default",
        help="fresh-buffer strategy without elastic_info, independent of inherited "
        "DEEP_USE_MODE; default uses custom C++ ops (ops rejected EP=4 in this "
        "probe on the tested CANN 9.0 stack; serving with elastic_info differs)",
    )
    args = parser.parse_args()
    if (
        args.world_size < 3
        or not 0 <= args.fault_rank < args.world_size
        or args.timeout <= 0
    ):
        parser.error("require world-size >= 3, a valid fault-rank, and timeout > 0")
    mc2_config = None
    if args.mc2:
        if args.backend != "hccl":
            parser.error("--mc2 requires --backend hccl")
        if (
            min(args.mc2_hidden, args.mc2_local_experts, args.mc2_tokens, args.mc2_topk)
            <= 0
        ):
            parser.error("MC2 dimensions must be positive")
        if args.mc2_topk > min(args.mc2_local_experts, 32):
            parser.error("MC2 probe requires topk <= min(local-experts, 32)")
        mc2_config = {
            "strategy": args.mc2_strategy,
            "hidden": args.mc2_hidden,
            "local_experts": args.mc2_local_experts,
            "tokens": args.mc2_tokens,
            "topk": args.mc2_topk,
        }

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
            target=_worker,
            args=(rank, store.port, args.backend, args.timeout, child, mc2_config),
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
                raise RuntimeError(
                    f"rank {rank}, {report.get('stage')}:\n{report['error']}"
                )
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
                    "scope": (
                        "candidate process groups and fresh DeepEP low-latency buffers; "
                        "original-rank elastic_info, existing buffers, graph replay and model serving remain unverified"
                        if args.mc2
                        else "candidate process groups only; MC2 and graph rebind remain unverified"
                    ),
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
