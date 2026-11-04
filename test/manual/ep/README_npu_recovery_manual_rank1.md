# NPU ops/dynamic 手动长请求故障与原 rank 1 恢复验收

**最新结果（2026-11-03 用户回传；服务器 ps 显示 11 月 4 日）：手动单轮原 rank 1 恢复 4→3→4 已实机通过。** 缩容后三卡真实生成 16 tokens，recover 返回 `fault_tolerance_recover_committed`、membership_epoch=2，独立 status 查询四卡 healthy，指定回归 rank 1 的请求也生成 16 tokens、finish_reason=length。

恢复前后 rank 0/2/3 的 PID 与创建时间均不变，只有 rank 1 从 3109763 更换为 3237419；共同 PPID 为 3103414。两个相同短输入的 output_ids 一致，仅覆盖本输入，不作为完整精度验收。这里的 PID 仅用于记录，后续实验必须重新查询。

用户随后补充本轮 commit.txt 内容，实机成功基线已确认是 **`3060ee795527680b98752fb2cb719ab59dcf2fe5`**（启动种子广播互等修复）。已知模型为 `/mnt/share/weights/DeepSeek-V2-Lite-Chat`。完整生效参数、包版本和恢复墙钟耗时未随本轮记录打印，不自动复用历史实验数据。此前修复经过保留在本文末尾。

当前成功服务可保留运行。以下是从头复现流程，只在准备重新开始独立实验时执行停止/启动段；保留 ops/dynamic、128 冗余、关闭图/调度重叠，以及长请求中杀 rank 1、等 unhealthy 再缩容的步骤。重复恢复、执行图、旧请求自动收尾、PD 和超初始容量扩容仍待验证。

在 NPU 服务器原 sglang-ft 环境、仓库根目录执行。下面的终端 1、2、3 都使用同一仓库和环境。沿用已设置的 MODEL_PATH、CANN 与自定义算子库路径，不升级包。旧服务由后台启动时，Ctrl+C 只会停止 tail；更新后必须终止本轮旧服务并重新启动，运行中的进程不会自动加载代码修复。

先保存旧日志末尾、进程与提交，再向本轮保存的主进程发 TERM；不使用旧聊天中的 PID。若主进程已退出，跳过 TERM：

```bash
OLD_RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
tail -n 250 "$OLD_RUN_DIR/server.log" > "$OLD_RUN_DIR/server-tail-before-restart.txt"
ps -eo pid,ppid,lstart,args | grep '[s]glang::scheduler_' | tee "$OLD_RUN_DIR/pids-at-failure.txt"
cat "$OLD_RUN_DIR/commit.txt"
OLD_SERVER_PID=$(cat "$OLD_RUN_DIR/server.pid")
if ps -p "$OLD_SERVER_PID" -o args= | grep -q '[s]glang.launch_server'; then
  kill -TERM "$OLD_SERVER_PID"
fi
```

单独检查旧服务退出、31000 端口释放后，再执行第 1 步。若还在监听，先处理旧服务，不并行启动第二份：

```bash
ss -ltnp 'sport = :31000'
ps -eo pid,ppid,lstart,args | grep '[s]glang::scheduler_'
```

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

## 3. 手动核对四卡 PID，再结束本次服务的 rank 1（终端 3）

先看本轮主进程和 Scheduler 列表，保存用于恢复后人工对照。根据本轮启动日志与父进程关系确认 rank 1 的 PID，不复用旧实验的数字；同机其他服务不属于本轮。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ps -p "$(cat "$RUN_DIR/server.pid")" -o pid,ppid,stat,args
ps -eo pid,ppid,lstart,args | grep '[s]glang::scheduler_' | tee "$RUN_DIR/pids-before.txt"
```

看到长请求持续输出时，输入刚确认的本轮 rank 1 PID 后按回车：

```bash
printf '输入本轮 rank 1 的 PID：'
read -r RANK1_PID
kill -9 "$RANK1_PID"
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

先确认恢复返回 success=true、四 rank 均 healthy，再人工对照两个 PID 列表：本轮 rank 0/2/3 不变，只有 rank 1 换进程。此处不运行 Python 自动校验。

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
cat "$RUN_DIR/recover.json" | jq .
cat "$RUN_DIR/pids-before.txt"
ps -eo pid,ppid,lstart,args | grep '[s]glang::scheduler_' | tee "$RUN_DIR/pids-after.txt"
```

然后明确把新请求交给 rank 1：

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
curl --noproxy '*' --connect-timeout 5 --max-time 600 -sS \
  http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","routed_dp_rank":1,"sampling_params":{"temperature":0,"max_new_tokens":16}}' \
  | tee "$RUN_DIR/recovered-rank-generation.json" | jq .
```

人工确认 text 非空、completion_tokens 大于 0、finish_reason 不是 abort，才将这轮标为通过。若未保存 pids-before.txt，不补造进程未重启的证据。

成功只覆盖一轮手动故障与补回原 rank，非完整精度评测，也未覆盖第二次故障、图恢复或旧请求自动收尾。失败后保留本轮目录，不重发 recover；回传 server.log 最早完整 traceback、scale-down.json、survivor-generation.json、recover.json 与 commit.txt。若无法读取恢复响应，应先检查状态和服务日志，不把客户端超时当作恢复未执行。

本机只对以上 Shell 与 JSON 请求体做语法检查，没有在本机执行 NPU 故障或恢复；文档顶部的单轮成功来自用户回传的实机记录。此前嵌入 Python 的辅助校验已按用户偏好改为手动命令；这不会改变服务端的恢复实现。

## 历史修复检查点：权重加载后启动广播（2026-11-03 本机分析）

用户日志时间为 2026-11-04 01:59:43–44。新 rank 1 完成 DeepseekV2ForCausalLM 权重加载、再次初始化 tokenizer 后，没有后续进度。源码显示 NPU FT 替代进程带 `_npu_ft_recovery`，但 `is_ep_joiner=False`，因此旧代码仍进入 `TpModelWorker` 的 WORLD 随机种子广播；幸存者此时已经在 `prepare_survivor` 等新进程的 loaded ACK，不会参与这个启动广播。

修复让 NPU FT 替代进程直接使用启动器已经确定、DPC 重启时复制的随机种子，保留普通启动的广播以及已有 ElasticEP joiner 行为。更新重启后，recover 的新 rank 1 应在 tokenizer 信息之后出现：

```text
NPU FT joiner reused launcher random seed; skipping startup WORLD broadcast
```

这条日志只证明跨过此前的互等点；仍须完成第 6 步的四卡 healthy、进程身份与指定 rank 1 真实生成，才能记录整轮通过。日志未提供最终 timeout/退出栈或实机 SHA，不能将 curl 52 的最终退出原因视为已实测确认。

修复时本机隔离 CPU 回归 109 tests、79 subtests passed，包含原构造函数会调用广播、修复后不调用的对照；NPU 服务未在本机执行。临时测试环境只有 asyncio_mode 未识别配置警告。后续用户单轮 NPU 恢复通过结果见本文顶部；此次记录更新未修改生产代码或重跑不变的单测。

## recover 返回 curl (52) Empty reply from server

2026-11-03 用户在第 5 步回传 curl (52)，未收到恢复 HTTP 正文。恢复进度、主进程是否仍存活、实际提交与底层原因等待日志确认。当前 FT manager 在 apply 异常时会记录错误并结束服务进程树，这是候选解释，不是本轮已证实根因。不要重发 recover 或重启后覆盖现场；先逐组执行只读检查：

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
tail -n 250 "$RUN_DIR/server.log"
```

```bash
curl --noproxy '*' --connect-timeout 2 --max-time 5 -i \
  http://127.0.0.1:31000/fault_tolerance/status
```

```bash
RUN_DIR=$(cat artifacts/npu_ft/current-recovery-dir.txt)
ps -p "$(cat "$RUN_DIR/server.pid")" -o pid,ppid,stat,args
cat "$RUN_DIR/commit.txt"
```
