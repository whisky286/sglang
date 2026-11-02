# NPU ops/dynamic 手动长请求故障与原 rank 1 恢复验收

用户已回传 ops/dynamic、128 冗余、关闭执行图与关闭调度重叠下原缩容正常；完整 4→3→4 尚未实机通过。此轮在这个组合上启用恢复开关，并满足恢复所需的标准通信组设置。保持长请求中杀 rank 1、等待幸存者 unhealthy 后再缩容的步骤，不使用默认脚本的空闲杀 rank 2 流程。

在 NPU 服务器原 sglang-ft 环境、仓库根目录执行。先在旧服务终端按 Ctrl+C 停止旧服务；下面的终端 1、2、3 都使用同一仓库和环境。沿用已设置的 MODEL_PATH、CANN 与自定义算子库路径，不升级包。

## 1. 更新代码并启动带恢复开关的新四卡服务（终端 1）

先执行更新；若 Git 提示本地修改冲突，保留修改，不强制覆盖。确认更新成功再执行启动块。

```bash
git fetch https://github.com/whisky286/sglang.git codex/npu-ft-recover
git switch --detach FETCH_HEAD
git rev-parse HEAD
```

以下命令将服务放到后台，保存的是 Python 主进程 PID；tail 用于看日志，之后 Ctrl+C 退出 tail 不会停止后台服务。每轮使用新目录，后续终端从 current-recovery-dir.txt 读取该目录。

```bash
RUN_DIR="artifacts/npu_ft/recover-rank1-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > artifacts/npu_ft/current-recovery-dir.txt
git rev-parse HEAD > "$RUN_DIR/commit.txt"

DEEP_USE_MODE=ops \
SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=0 \
python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --device npu \
  --attention-backend ascend \
  --dtype bfloat16 \
  --tp-size 4 \
  --dp-size 4 \
  --ep-size 4 \
  --attn-cp-size 1 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-dense-tp-size 1 \
  --moe-dp-size 1 \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --enable-eplb \
  --eplb-algorithm elasticity_aware \
  --ep-dispatch-algorithm dynamic \
  --ep-num-redundant-experts 128 \
  --elastic-ep-backend mc2 \
  --enable-fault-tolerance \
  --enable-npu-ft-recovery \
  --fault-tolerance-on-error-strategy pause \
  --fault-tolerance-timeout 600 \
  --fault-tolerance-pause-timeout 600 \
  --cuda-graph-bs-decode 1 2 4 8 \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --disable-radix-cache \
  --mem-fraction-static 0.5 \
  --max-running-requests 8 \
  --max-total-tokens 4096 \
  --context-length 2048 \
  --watchdog-timeout 300 \
  --nnodes 1 \
  --node-rank 0 \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 31000 \
  > "$RUN_DIR/server.log" 2>&1 &
printf '%s\n' "$!" > "$RUN_DIR/server.pid"
tail -f "$RUN_DIR/server.log"
```

两个进程组环境变量是恢复实现所需设置：禁用旧消息队列广播，统一使用可重建的标准组；不表示其与恢复开关的影响已经独立实测。若启动失败，不继续故障注入，保留 server.log 和 commit.txt。

## 2. 启动原来的长流式请求（终端 2）

服务就绪后执行，看到持续输出时在终端 3 执行下一步。

```bash
curl --noproxy '*' -N http://127.0.0.1:31000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Write a detailed numbered explanation of how distributed inference works. Keep generating until you reach the token limit."}],
    "max_tokens": 1536,
    "temperature": 0,
    "stream": true
  }'
```

## 3. 保存四卡进程身份并结束本次服务的 rank 1（终端 3）

此命令复用仓库已有的 Scheduler 子进程识别方法，只搜索本次主进程的子树。四卡不齐时直接停止，不复用历史 PID。

```bash
python - <<'PY'
import json
import runpy
from pathlib import Path

run_dir = Path(Path("artifacts/npu_ft/current-recovery-dir.txt").read_text().strip())
server_pid = int((run_dir / "server.pid").read_text())
find = runpy.run_path("test/manual/ep/verify_npu_recovery_serving.py")["schedulers"]
workers = find(server_pid)
if set(workers) != {0, 1, 2, 3}:
    raise RuntimeError(f"Expected four schedulers before injection: {sorted(workers)}")
before = {str(rank): [proc.pid, proc.create_time()] for rank, proc in workers.items()}
(run_dir / "pids-before.json").write_text(json.dumps(before, indent=2))
print(f"Killing this server's rank 1: pid={workers[1].pid}", flush=True)
workers[1].kill()
PY
```

查询状态；等 rank 1 dead 且 0/2/3 unhealthy 后，再执行缩容块。状态未达到时可重复这条只读查询，不重复杀进程。

```bash
curl --noproxy '*' --connect-timeout 2 --max-time 5 -sS \
  http://127.0.0.1:31000/fault_tolerance/status | jq .
```

## 4. 缩容并验证三卡真实生成（终端 3）

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
curl --noproxy '*' --connect-timeout 5 --max-time 2430 -sS \
  http://127.0.0.1:31000/fault_tolerance/apply \
  -H 'Content-Type: application/json' \
  -d '{"instruction":"scale_down","params":{"ranks":[1],"timeout":600}}' \
  | tee "$RUN_DIR/scale-down.json" | jq .
```

缩容返回 success=true 后，执行指定幸存 rank 0 的新请求：

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
curl --noproxy '*' --connect-timeout 5 --max-time 600 -sS \
  http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","routed_dp_rank":0,"sampling_params":{"temperature":0,"max_new_tokens":16}}' \
  | tee "$RUN_DIR/survivor-generation.json" | jq .
```

若终端 2 的旧连接仍挂住，可 Ctrl+C 结束该 curl；这不作为旧请求自动终止或自动续推通过的证据。新请求失败时不继续 recover。

## 5. 补回 rank 1（终端 3）

下列块先检查缩容成功与幸存 rank 生成有效 token，之后才发送 recover，不重试已提交的恢复操作。恢复可能需几分钟，等待这一条请求返回，不重复发送。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
if jq -e '.success == true' "$RUN_DIR/scale-down.json" >/dev/null && \
   jq -e '.meta_info.completion_tokens > 0 and (.text | length > 0) and ((.meta_info.finish_reason.type? // "") != "abort")' \
     "$RUN_DIR/survivor-generation.json" >/dev/null; then
  curl --noproxy '*' --connect-timeout 5 --max-time 2430 -sS \
    http://127.0.0.1:31000/fault_tolerance/apply \
    -H 'Content-Type: application/json' \
    -d '{"instruction":"recover","params":{"ranks":[1],"timeout":600}}' \
    | tee "$RUN_DIR/recover.json" | jq .
else
  printf '%s\n' 'STOP: scale-down or survivor generation did not pass; recover was not sent.' >&2
fi
```

## 6. 核对只有 rank 1 换进程，并指定它生成（终端 3）

```bash
python - <<'PY'
import json
import runpy
from pathlib import Path
import requests

run_dir = Path(Path("artifacts/npu_ft/current-recovery-dir.txt").read_text().strip())
recovered = json.loads((run_dir / "recover.json").read_text())
if not recovered.get("success") or any(r["state"] != "healthy" for r in recovered["ranks"]):
    raise RuntimeError(f"Recovery did not pass: {recovered}")
find = runpy.run_path("test/manual/ep/verify_npu_recovery_serving.py")["schedulers"]
workers = find(int((run_dir / "server.pid").read_text()))
before = json.loads((run_dir / "pids-before.json").read_text())
after = {str(rank): [proc.pid, proc.create_time()] for rank, proc in workers.items()}
if set(after) != set(before) or [rank for rank in before if after.get(rank) != before[rank]] != ["1"]:
    raise RuntimeError(f"Unexpected process replacement: before={before}, after={after}")
(run_dir / "pids-after.json").write_text(json.dumps(after, indent=2))
print("PID check passed: only rank 1 was replaced", flush=True)
with requests.Session() as session:
    session.trust_env = False
    result = session.post("http://127.0.0.1:31000/generate", json={
        "text": "The capital of France is", "routed_dp_rank": 1,
        "sampling_params": {"temperature": 0, "max_new_tokens": 16},
    }, timeout=(5, 600))
    (run_dir / "recovered-rank-generation.json").write_text(result.text)
    result.raise_for_status()
    output = result.json()
print(json.dumps(output, ensure_ascii=False, indent=2))
if not output.get("text") or output.get("meta_info", {}).get("completion_tokens", 0) <= 0:
    raise RuntimeError("Recovered rank returned no completion")
reason = output.get("meta_info", {}).get("finish_reason", {})
if isinstance(reason, dict) and reason.get("type") == "abort":
    raise RuntimeError(f"Recovered rank generation aborted: {output}")
print(f"PASS: one manual 4 -> 3 -> 4 cycle; artifacts: {run_dir}")
PY
```

成功只覆盖一轮手动故障与补回原 rank，非完整精度评测，也未覆盖第二次故障、图恢复或旧请求自动收尾。失败后保留本轮目录，不重发 recover；回传 server.log 最早完整 traceback、scale-down.json、survivor-generation.json、recover.json 与 commit.txt。若无法读取恢复响应，应先检查状态和服务日志，不把客户端超时当作恢复未执行。

本机只对以上 Bash/Python 片段做语法检查，未执行这些故障或恢复命令。当前文档不是 NPU 通过结果。
