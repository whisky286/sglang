# NPU 原 rank 回归：真实模型服务验收

这轮验证单机 eager 服务的 `4→3→4`：原 rank 2 退出后，三卡先恢复生成，再由控制器拉起同编号新进程、重建通信和 DeepEP buffer、加载当前专家布局，最后重新开放四卡请求路由。需要重新启动带 `--enable-npu-ft-recovery` 的服务。

**状态（2026-10-30）：代码和验收脚本已实现，本机检查通过；真实 NPU 模型服务尚未验证。** [上一阶段通信探针](README_npu_rank_recovery.md) 的 HCCL/新 buffer 通过结果来自用户回传，只覆盖其声明的范围。

## 1. 更新代码并保留已跑通的环境

在 NPU 服务器的仓库根目录操作。下面切到远端分支的确定快照；Git 若提示本地修改冲突，先保留修改，勿强制覆盖。

```bash
conda activate sglang-ft
git fetch https://github.com/whisky286/sglang.git codex/npu-ft-recover
git switch --detach FETCH_HEAD
git rev-parse HEAD
git diff --stat
```

沿用上一轮通过的 CANN/Python/Torch/torch_npu/DeepEP 环境，不需要为本次改动升级它们。确认已在当前 shell 加载相同 CANN 环境，再导出之前修正过的自定义算子路径；这些变量必须在启动 Python **之前**设置。

```bash
DEEPEP_DIR=$(python -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("deep_ep").origin).parent)')
DEEPEP_OPP="$DEEPEP_DIR/vendors/hwcomputing"
export LD_LIBRARY_PATH="$DEEPEP_OPP/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ASCEND_CUSTOM_OPP_PATH="$DEEPEP_OPP${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export DEEP_USE_MODE=default
export SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

`default` 是上一轮四卡探针跑通的 DeepEP 自定义算子策略。关闭消息队列广播，是为了让回归进程使用新建的 Gloo 控制组；旧广播队列没有重新入组协议。

可以先再次运行原通信探针，确认服务器环境仍正常；已经确认环境没有变化时可直接进入模型验收：

```bash
python test/manual/ep/probe_npu_rank_recovery.py \
  --backend hccl --world-size 4 --fault-rank 2 \
  --mc2 --mc2-strategy default --timeout 120
```

## 2. 启动一套专用测试服务

使用之前已能启动的**非量化** Qwen3-30B-A3B 模型目录，替换下面的 `MODEL_PATH`。首次验收先固定模型和软件栈；不要同时打开 PD、执行图或调度重叠。测试会自动结束这套服务下的 rank 2 进程，请使用专用空闲卡。

```bash
MODEL_PATH=/absolute/path/to/Qwen3-30B-A3B
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --device npu --attention-backend ascend --dtype bfloat16 \
  --tp-size 4 --dp-size 4 --ep-size 4 \
  --moe-dense-tp-size 1 --moe-dp-size 1 --attn-cp-size 1 \
  --enable-dp-attention --enable-dp-lm-head \
  --moe-a2a-backend deepep --deepep-mode low_latency \
  --enable-eplb --eplb-algorithm elasticity_aware \
  --ep-num-redundant-experts 44 \
  --elastic-ep-backend mc2 --enable-fault-tolerance \
  --enable-npu-ft-recovery \
  --fault-tolerance-timeout 600 --fault-tolerance-pause-timeout 900 \
  --disable-cuda-graph --disable-overlap-schedule \
  --mem-fraction-static 0.65 --max-total-tokens 8192 \
  --host 127.0.0.1 --port 30000 \
  > /tmp/npu-recovery-server.log 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > /tmp/npu-recovery-server.pid
```

Qwen3-30B-A3B 有 128 个逻辑专家；44 个冗余槽位使物理槽位总数为 172，每卡 43 个，三卡有 129 个槽位，可以覆盖 128 个逻辑专家。此配置只对应这个专家数，换模型时要重新计算。`--max-total-tokens 8192` 固定首轮 KV 容量；可沿用现有环境能启动的内存参数。

查看启动日志：

```bash
tail -f /tmp/npu-recovery-server.log
```

按 Ctrl+C 只退出 `tail`，后台服务继续运行。也可在启动后直接运行下一步，脚本会等待健康接口就绪。

## 3. 自动完成两轮故障和回归

在同一环境、仓库根目录执行：

```bash
SERVER_PID=$(cat /tmp/npu-recovery-server.pid)
python test/manual/ep/verify_npu_recovery_serving.py \
  --server-pid "$SERVER_PID" --base-url http://127.0.0.1:30000 \
  --fault-rank 2 --cycles 2 --timeout 600 \
  --expected-text Paris \
  --output /tmp/npu-recovery-serving.json \
  > /tmp/npu-recovery-test.log 2>&1
TEST_EXIT=$?
cat /tmp/npu-recovery-test.log
echo "test exit code: $TEST_EXIT"
```

脚本只会结束明确指定的服务器 PID 下、匹配 rank 的 Scheduler 子进程。HTTP 地址必须是本机地址。若服务配置了 API key，在运行前导出 `SGLANG_API_KEY`。

每轮都会检查：

1. 初始四卡中，指定 rank 2 的请求能生成文本。
2. 结束 rank 2，等控制器确认进程退出，然后调用 `scale_down`。
3. 缩容完成后，指定幸存 rank 生成文本。
4. 调用 `recover`，等待重新加载权重、新通信组和新 buffer 的全员模型试跑完成。
5. 只有 rank 2 的 PID 改变，其他三卡 PID 不变；控制器报告四卡 healthy。
6. 再将请求指定给 rank 2，确认实际生成文本。
7. 对同一个 rank 再执行一轮，检查重建后的故障监测仍有效。

`--expected-text Paris` 给默认提示词 `The capital of France is` 增加一个简单语义检查。更换提示词时应同时调整预期文本；这不是完整的模型精度评测。

成功输出最后一行类似：

```json
{"success": true, "report": "/tmp/npu-recovery-serving.json"}
```

报告记录实际 Git SHA、软件包版本、每阶段的响应及 PID。测试进程退出码必须为 0；异常时 `success` 为 false，并保存已完成阶段和错误信息。失败后脚本保留服务现场；恢复事务一旦进入通信切换阶段失败，服务会按现有 FT 策略整体退出，不尝试伪回滚。

### 启动验收后没有输出，或状态接口正常但 `/server_info` 超时

2026-10-30 用户在 `12b479feb8` 对应流程报告验收没有输出：`/fault_tolerance/status` 返回 HTTP 200、四个 rank healthy，但单独请求 `/server_info` 在 8 秒内收到 0 字节并超时。精确实机 SHA 未另行回报。

本机代码核对发现：Scheduler 在开启 FT 后已改用各 DP 内的控制消息广播，而 DPC 仍只给第一个 DP leader 发送普通控制消息；`/server_info` 等待所有 Scheduler 的内部状态回复，因而可能一直等不到其余 rank。修正使 DPC 对 FT、scale joiner 与显式 local-control 配置都给每个活跃 DP leader 发送控制消息。修复后的 NPU 结果仍待重测。

旧验收脚本的第一条日志在 `/health` 和 `/server_info` 之后，且状态查询错误继承了完整恢复事务的等待预算（`--timeout 600` 时为 2430 秒）。新版在加载测试依赖前打印 `test_started`，为每次请求打印 `http_request_started/finished/failed`；只读状态查询使用 30 秒读取超时，生成使用 `--timeout`，FT apply 保留多阶段预算。JSON 报告随事件原子更新，异常或中断也保留阶段；不会自动重试可能已经接受的恢复操作。

遇到此类现象时，保留现场并在另一终端逐条执行下面命令。每个代码块是一条独立命令，末尾不要加续行反斜杠，也不要直接打印完整 `/server_info` 配置：

```bash
tail -n 30 artifacts/npu_ft/recovery-test.log
```

```bash
tail -n 100 artifacts/npu_ft/sglang-npu-ft.log
```

```bash
curl --noproxy '*' --connect-timeout 2 --max-time 5 -i http://127.0.0.1:31000/fault_tolerance/status
```

```bash
curl --noproxy '*' --connect-timeout 2 --max-time 8 -sS -o /tmp/npu-server-info-check.json -w 'server_info HTTP=%{http_code} elapsed=%{time_total}s\n' http://127.0.0.1:31000/server_info
```

端口 31000、`artifacts/npu_ft` 对应这次用户实际启动命令；若采用本页最初的示例，应使用 30000 和对应 `/tmp` 日志路径。`status` 的 healthy 仅代表控制器当前记录，不代表所有 Scheduler 都响应控制请求。若验收已经出现 `scheduler_killed`，先检查服务/报告当前成员状态，再决定如何继续，避免重复故障注入。

## 4. 失败或完成后回传的材料

```bash
git rev-parse HEAD > /tmp/npu-recovery-commit.txt
git diff > /tmp/npu-recovery-local.patch
python --version > /tmp/npu-recovery-python.txt
python -m pip show torch torch-npu sgl-kernel-npu deep-ep > /tmp/npu-recovery-packages.txt
npu-smi info > /tmp/npu-recovery-devices.txt
```

请带回 `/tmp/npu-recovery-server.log`、`/tmp/npu-recovery-test.log`、`/tmp/npu-recovery-serving.json` 和上面五个记录文件。失败时保留完整 traceback 及失败前的阶段，不只截最后一条 HCCL 错误。若卡住超过配置超时，也保留日志和进程/设备状态；不要反复重跑覆盖第一轮证据。

## 实现范围与本机证据

- 新增显式 `POST /fault_tolerance/apply` 请求：`{"instruction":"recover","params":{"ranks":[2],"timeout":600}}`。必须先完成缩容，且一次恢复全部已移除的原始 ranks；控制器自行启动新进程，不需要手动启动 joiner。
- 幸存者暂停并清理在途请求；以成员版本隔离新 WORLD/TP/EP HCCL、Gloo 组；回归进程按当前专家布局加载 checkpoint。DeepEP 缓存和旧 dispatch handle 在暂停边界被替换；全员通信与模型 logits 检查通过后才开放路由。
- 本机用隔离导入/AST 提取的实际模块运行 9 个相关测试文件：101 tests、57 subtests 通过。检查涵盖控制事务、失败门禁、陈旧事件、进程替换、buffer 清理、专家布局、既有缩容。语法编译通过，针对基线没有新增 F/I lint 问题。本机不是完整 SGLang/NPU 环境。
- 本机另以真实多进程 Gloo 验证新的 `RecoveryGroups` 默认组替换和原 rank 映射：`4→3→4` 求和 `10→7→10`，由新进程补回 rank 2，由原 rank 2 发起的对象广播正确。它不能替代目标 torch-npu/HCCL 私有接口兼容性验证。
- 尚未验证：本页真实模型流程、多轮 NPU 显存占用、设备真实硬故障后的重新分配、并发压力和完整输出精度。旧故障组和 buffer 保留到进程退出，避免未经验证的销毁操作；这会增加多轮恢复的内存占用，不保证无限次恢复。
- 本轮只支持单机、`TP=DP=EP`、Attention TP=1、eager、无调度重叠、DeepEP low_latency/default、非量化模型。PD、多机、执行图和增加原始容量之外的 `4→6` 不在本轮验收范围。
