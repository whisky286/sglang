# Ascend MC2 fault-tolerance smoke test

This test targets one Atlas A3 node with four Scheduler ranks, Qwen3-30B-A3B,
and the fixed original topology `TP=DP=EP=4`. It validates scale-down after a
real Scheduler process exit. It does not exercise scale-up/recover and must not
depend on NPU graph recapture.

Before starting SGLang, apply
`patches/ascend/sgl-kernel-npu-mc2-elastic-info.patch` to the matching
`sgl-kernel-npu` checkout and rebuild/install its DeepEP package. The patch
passes the optional tensor through both the default C++ strategy and the
`DEEP_USE_MODE=ops` torch-npu strategy down to the ACLNN `elasticInfo` input.
For this EP=4 test, select `DEEP_USE_MODE=default`: on the torch-npu 2.10/CANN
9.0 stack, the `ops` path reaches `aclnnMoeDistributeDispatchV4`, whose tiling
requires `epWorldSize` to be a multiple of 16 and rejects EP=4.
For an original `EP=16` topology, `DEEP_USE_MODE=ops` is supported. When
`elastic_info` is present, the patch makes the ops dispatch/combine pair use
`fullmesh_v1` instead of its normal `hierarchy` default because the dynamic
scale-down tiling path rejects `elasticInfo` with hierarchical communication.
Calls without `elastic_info` retain the original strategy default.
If the earlier combined MC2 patch is already present in a kernel checkout,
apply `patches/ascend/sgl-kernel-npu-mc2-ops-elastic-fullmesh.patch` on top.
Fresh checkouts only need the combined
`patches/ascend/sgl-kernel-npu-mc2-elastic-info.patch`, which already includes
the same correction.
It is rebased and `git apply --check`-validated against the `2026.7.2` source
tag (`7a396def6d0d7ce85e940549a366351ce1d7821b`), including that version's
`use_mxfp4` DeepEP argument. Regenerate the patch rather than applying it with
fuzz when using a different DeepEP source commit.

Start a disposable server with the target topology (adapt model and log paths):

```bash
DEEP_USE_MODE=ops python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port 30000 \
  --device npu \
  --tp-size 4 \
  --dp-size 4 \
  --ep-size 4 \
  --moe-dense-tp-size 1 \
  --moe-dp-size 1 \
  --attn-cp-size 1 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --enable-eplb \
  --eplb-algorithm elasticity_aware \
  --ep-dispatch-algorithm dynamic \
  --ep-num-redundant-experts 128 \
  --cuda-graph-bs-decode 1 2 4 8 \
  --disable-radix-cache \
  --mem-fraction-static 0.5 \
  --max-running-requests 8 \
  --max-total-tokens 4096 \
  --context-length 1024 \
  --watchdog-timeout 30 \
  --nnodes 1 \
  --node-rank 0 \
  --trust-remote-code \
  --elastic-ep-backend mc2 \
  --enable-fault-tolerance \
  --fault-tolerance-on-error-strategy pause \
  --fault-tolerance-timeout 600 \
  2>&1 | tee artifacts/npu_ft/sglang-npu-ft.log
```

Qwen3-30B-A3B has 128 logical experts. With one of four ranks removed, each
survivor needs 43 fixed expert slots, so launch-time capacity must be
`43 * 4 = 172` physical slots: 128 base experts plus 44 redundant experts.
This is fixed storage allocated before graph capture, not post-failure graph
reconstruction. Startup fails fast if even one-rank scale-down would leave fewer
physical slots than logical experts.

At scale-down, the decode-graph MC2 dispatch/combine window is intentionally not
rebuilt. Its fixed-address `elastic_info` is committed in place only after the
expert layout is ready. All graph-external domains are rebuilt from the
survivors through the controller-hosted rendezvous store:

- Scheduler/control objects and optional PrefillDelayer negotiation use the new
  compact-rank Gloo group after rebuild.
- MLP-sync metadata uses a new compact-rank Scheduler HCCL group.
- EPLB statistics and expert P2P use a separate rebuilt HCCL group; peers are
  translated from immutable original ranks to compact survivor ranks.
- The precompile barrier selects the rebuilt Scheduler HCCL group if invoked
  after scale-down. The required `--deepep-mode low_latency` keeps prefill and
  decode on the MC2 low-latency path, so DeepEP normal prefill communication is
  not enabled in this FT configuration.

Before the service becomes ready, every original rank also prewarms the
MLP-sync Gloo group. This prevents a survivor from attempting that group's lazy
four-rank full-mesh initialization after another rank has already failed. A
pre-scale-down MLP-sync collective uses a bounded wait so its Scheduler owner
thread can return to the FT request loop and consume a queued scale-down
command instead of starving the survivor-only rebuild.

Before rebuilding those domains, every survivor calls
`torch_npu.npu.stop_device(local_device_id)` followed immediately by
`restart_device(local_device_id)` on the Scheduler/ModelRunner device-owner
thread. This aborts unfinished device work and resets TorchNPU's HCCL watchdog
state left by the failed rank. The restart call deliberately uses its default
mode: do not pass `rebuild_all_resource(s)=True`, because that mode can rebuild
streams and mark existing tensors unsafe. The default path does not request a
new graph capture or replace the fixed MC2 buffers. The manual test verifies the
stop/restart/rebuild/elastic-info ordering and then exercises the same captured
decode graph with deterministic requests.

Expert restoration has a strict priority order for every destination slot:
reuse an unchanged or duplicate local physical expert, copy it from another
survivor over the rebuilt HCCL group, and use DRAM backup or checkpoint reload
only when no survivor owns that logical expert.

Find a non-controller Scheduler PID and its original DP rank, then run:

```bash
python test/manual/ascend/test_fault_tolerance_mc2_scale_down.py \
  --victim-rank 1 \
  --victim-pid <scheduler-pid-for-rank-1> \
  --server-log artifacts/npu_ft/sglang-npu-ft.log
```

The test requires all four ranks to be healthy first. It sends a deterministic
baseline request, kills exactly the supplied Scheduler PID, waits for the FT
incident, applies sparse scale-down, and sends three more requests. With
`--server-log`, it additionally proves that the MC2 `elastic_info` device
address did not change, the victim is `-1` in the original-to-effective table,
the reverse table remains fixed-width, and every survivor logged the same
rebuilt process-group membership plus the expected compact-rank mapping.

Use `--wait-for-existing-incident` instead of `--victim-pid` when the process
failure is injected externally.

---

## Multi-Scenario Fault-Injection Test Suite

To run the extended fault-injection test scenarios, use the **two-terminal workflow**:
- **Terminal 1**: Launch the SGLang server with the target topology and FT strategy.
- **Terminal 2**: Run the client-side test orchestrator after the server is ready.

### 1. Standard Topology + Pause Strategy (for EXP-1, EXP-2, EXP-4, EXP-6)

**[Terminal 1] Launch Command**:
```bash
DEEP_USE_MODE=ops python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port 30000 \
  --device npu \
  --tp-size 4 --dp-size 4 --ep-size 4 \
  --moe-dense-tp-size 1 --moe-dp-size 1 --attn-cp-size 1 \
  --enable-dp-attention --enable-dp-lm-head \
  --moe-a2a-backend deepep --deepep-mode low_latency \
  --enable-eplb --eplb-algorithm elasticity_aware --ep-dispatch-algorithm dynamic \
  --ep-num-redundant-experts 128 \
  --cuda-graph-bs-decode 1 2 4 8 \
  --disable-radix-cache \
  --mem-fraction-static 0.5 \
  --max-running-requests 8 \
  --max-total-tokens 4096 \
  --context-length 1024 \
  --watchdog-timeout 30 \
  --nnodes 1 \
  --node-rank 0 \
  --trust-remote-code \
  --elastic-ep-backend mc2 \
  --enable-fault-tolerance \
  --fault-tolerance-on-error-strategy pause \
  --fault-tolerance-timeout 600 \
  2>&1 | tee artifacts/npu_ft/sglang-npu-ft.log
```

**[Terminal 2] Test Execution**:
```bash
# EXP-1: Idle baseline scale-down
python test/manual/ascend/test_fault_tolerance_suite.py --test-case idle_scale_down --victim-rank 3

# EXP-2: In-flight dynamic scale-down under concurrent traffic (10 QPS)
python test/manual/ascend/test_fault_tolerance_suite.py --test-case inflight_scale_down --victim-rank 3 --concurrency 10

# EXP-4: Mixed soft exception + hard SIGKILL multi-rank scale-down
python test/manual/ascend/test_fault_tolerance_suite.py --test-case mixed_fault_injection --soft-victim-rank 1 --hard-victim-rank 2

# EXP-6: Multi-step cascading scale-down (4 -> 3 -> 2)
python test/manual/ascend/test_fault_tolerance_suite.py --test-case cascading_scale_down --cascading-ranks 3 2
```

---

### 2. Continue Strategy (for EXP-3)

**[Terminal 1] Launch Command** (using `--fault-tolerance-on-error-strategy continue`):
```bash
DEEP_USE_MODE=ops python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port 30000 \
  --device npu \
  --tp-size 4 --dp-size 4 --ep-size 4 \
  --moe-dense-tp-size 1 --moe-dp-size 1 --attn-cp-size 1 \
  --enable-dp-attention --enable-dp-lm-head \
  --moe-a2a-backend deepep --deepep-mode low_latency \
  --enable-eplb --eplb-algorithm elasticity_aware --ep-dispatch-algorithm dynamic \
  --ep-num-redundant-experts 128 \
  --cuda-graph-bs-decode 1 2 4 8 \
  --disable-radix-cache \
  --mem-fraction-static 0.5 \
  --max-running-requests 8 \
  --max-total-tokens 4096 \
  --context-length 1024 \
  --watchdog-timeout 30 \
  --nnodes 1 \
  --node-rank 0 \
  --trust-remote-code \
  --elastic-ep-backend mc2 \
  --enable-fault-tolerance \
  --fault-tolerance-on-error-strategy continue \
  --fault-tolerance-timeout 600 \
  2>&1 | tee artifacts/npu_ft/sglang-npu-ft.log
```

**[Terminal 2] Test Execution**:
```bash
python test/manual/ascend/test_fault_tolerance_suite.py --test-case strategy_continue_isolation --victim-rank 3
```

---

### 3. Tensor Parallelism TP > 1 (for EXP-5: DP=2, TP=2)

**[Terminal 1] Launch Command** (4 cards: DP=2, TP=2, EP=2):
```bash
DEEP_USE_MODE=ops python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port 30000 \
  --device npu \
  --tp-size 4 --dp-size 2 --ep-size 2 \
  --moe-dense-tp-size 2 --moe-dp-size 1 --attn-cp-size 1 \
  --enable-dp-attention --enable-dp-lm-head \
  --moe-a2a-backend deepep --deepep-mode low_latency \
  --enable-eplb --eplb-algorithm elasticity_aware --ep-dispatch-algorithm dynamic \
  --ep-num-redundant-experts 128 \
  --cuda-graph-bs-decode 1 2 4 8 \
  --disable-radix-cache \
  --mem-fraction-static 0.5 \
  --max-running-requests 8 \
  --max-total-tokens 4096 \
  --context-length 1024 \
  --watchdog-timeout 30 \
  --nnodes 1 \
  --node-rank 0 \
  --trust-remote-code \
  --elastic-ep-backend mc2 \
  --enable-fault-tolerance \
  --fault-tolerance-on-error-strategy pause \
  --fault-tolerance-timeout 600 \
  2>&1 | tee artifacts/npu_ft/sglang-npu-ft.log
```

**[Terminal 2] Test Execution**:
```bash
python test/manual/ascend/test_fault_tolerance_suite.py --test-case tp_parallel_scale_down --victim-rank 1
```


