# NPU 原 rank 恢复：第一阶段实现与通信探针

## 基线选择

本阶段从 `46147ef9bc`（`fix(npu): use minimal EPLB recovery layout`）开始，开发分支为 `codex/npu-ft-recover`。

| 基线 | 包含的能力 | 选择 |
|---|---|---|
| `c26679541a` | NPU EPLB 迁移与缺失专家重载，PR #17 | 缺少完整 NPU 缩容编排 |
| `07fccdaf8f` | NPU 故障缩容编排 | 还没有后续最小布局修复 |
| `46147ef9bc` | 上述能力与最小专家恢复布局，PR #18 的 HEAD | 本次开发基线 |
| 上游 PR #40078（查证时 HEAD `e47ce63280`） | 后续 FT 控制面与 API 调整 | 尚未合入，且没有包含本分支的 NPU/MC2 恢复链路；后续单独移植 |

PR 链接：[NPU EPLB #17](https://github.com/gaidandawang-afk/sglang/pull/17)、[NPU FT #18](https://github.com/gaidandawang-afk/sglang/pull/18)、[上游 FT #40078](https://github.com/sgl-project/sglang/pull/40078)。

## 本阶段做了什么

目标是为 `4→3→4` 补回原 rank 建立可验证的基础。这里恢复的是原有 rank 编号，例如 `[0,1,3]` 补回 rank 2；不是在初始容量之外增加 rank 4、5。

1. **统一成员版本**：`FaultToleranceManager` 为成员变化分配 `membership_epoch`，通过控制命令、Scheduler、ModelRunner 传给 `NpuFTCommunication`。通信组集合点使用这个版本，新进程不需要猜测幸存者本地已经经历了多少次缩容。旧版本在创建新控制组前被拒绝。
2. **不可变恢复计划**：`fault_tolerance/recovery.py` 的 `RankRecoveryPlan` 校验原始容量、活跃 mask、恢复成员和完整 Attention DP 组。它只描述候选成员，不表示进程、权重或数据面已经就绪。
3. **回归专家布局**：`ExpertLocationMetadata.init_for_rank_recovery()` 检查旧幸存者已经完整覆盖逻辑专家，保持物理槽位不变，仅构造恢复后的候选路由。回归进程必须按当前布局加载全部本地槽位，不能把 checkpoint 初始布局当成已验证的 serving 布局。
4. **通信探针**：真实启动多个进程，杀掉一个中间 rank，在幸存进程中创建候选通信组，再拉起同编号新进程并创建恢复后的通信组，检查 all-gather、all-reduce、P2P 和进程身份。

## 当前明确没有开放的能力

- 没有新增可接受 NPU `recover` 的 HTTP 操作，也没有移除运行时扩容或 PD 的配置门禁。
- 没有让新进程直接走 Mooncake `join_group`，也没有将 `reinit_process_group(..., False)` 当作新进程入组接口。
- 探针创建新的候选设备组，不验证旧 MC2 buffer、HCCL handle 或 NPU graph 是否能绑定新组。
- 专家恢复方法只构造候选元数据；不会自行加载回归进程权重、修改 live routing 或宣布恢复完成。
- 探针是空闲边界下的通信能力验证，不模拟运行中的 NPU 算子报错，也不承诺故障设备能够在未经修复的情况下重新分配。

因此，本阶段是恢复功能的基础代码，**不是已经实现完整在线扩容恢复**。

## 动作链和关键文件

现有缩容命令新增一条版本传递链：

```text
FaultToleranceManager._apply_scale_down
  → FaultToleranceCommandReqInput.membership_epoch
  → Scheduler._recover_npu_fault_tolerance_scale_down
  → ModelRunner.apply_fault_tolerance_scale_down
  → NpuFTCommunication.rebuild_survivor_groups
  → npu-ft/<target-mask>/<membership_epoch>/mlp-sync
```

`rebuild_survivor_groups()` 仍然只替换 Gloo 控制组，EPLB 设备通信组继续引用原组。这一行为用于保留已验证的缩容路径，不能据此认定恢复数据面已经完成。

后续回归流程将消费：

```text
控制面生成 RankRecoveryPlan
  → 幸存者和回归进程使用同一版本与原 rank 编号
  → 构造回归专家布局并加载权重
  → 验证 HCCL/MC2 与执行图
  → 全员试跑并确认
  → 发布新成员路由
```

## 如何运行通信探针

从仓库根目录，在已安装 SGLang 依赖的环境运行。

### CPU：先验证进程与集合点流程

```bash
PYTHONPATH=python python test/manual/ep/probe_npu_rank_recovery.py \
  --backend gloo --world-size 4 --fault-rank 2 --timeout 30
```

这只需要本机回环 TCP 通信权限，不需要模型或 NPU。2026-10-27 在 macOS、Python 3.12、CPU Torch 2.14.1 上，通过隔离 SGLang 非相关导入执行了这条流程：

- phase 0：`[0,1,2,3]`，all-reduce 结果 `10`。
- phase 1：杀掉 rank 2，`[0,1,3]`，结果 `7`。
- phase 2：新进程以 rank 2 回归，`[0,1,2,3]`，结果 `10`。
- 幸存者 PID 保持不变，回归 rank 的 PID 改变；各阶段 P2P 校验通过。

### NPU：验证候选 HCCL 组

在专用测试卡上使用当前已验证缩容的软件环境，先设置 CANN 和 HCCL 环境，再运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=python \
  python test/manual/ep/probe_npu_rank_recovery.py \
  --backend hccl --world-size 4 --fault-rank 2 --timeout 30
```

rank 按可见设备的本地编号选择 NPU。探针在父进程保留独立 TCPStore，各 worker 用 Gloo 传控制元数据，用 HCCL 执行候选设备通信。

探针使用与 `init_custom_process_group()` 一致的成员集合点构造方式，通过 PyTorch 私有 `_new_process_group_helper` 创建独立候选组，不要求故障进程重新参加旧 WORLD。私有接口和 HCCL 实现有版本差异，所以必须在目标 torch/torch-npu/CANN 组合上运行；CPU Torch 通过不能替代目标 NPU 版本通过。

成功时每个阶段输出 JSON，包含原 rank、PID、成员版本、原 rank 列表与求和结果。最后输出 `success: true`，并明确 scope 仅为候选通信组。

失败时会抛出错误；父进程按超时收集结果并清理自己启动的 worker。旧组保留到 worker 退出，探针没有验证旧故障组的单独销毁契约。

### 目标 NPU 环境首次结果（2026-10-27）

用户按上一轮 HCCL 验证流程运行 `e1cbd28f08` 的探针，报告末尾结果为：

```json
{"success": true, "scope": "candidate process groups only; MC2 and graph rebind remain unverified"}
```

这表示探针内的检查已通过：新建候选组的 `4→3→4`、Gloo 原 rank 同步、HCCL all-reduce/P2P，以及幸存进程不重启、同编号 rank 由新进程回归。该结论来自用户反馈，尚未提供完整阶段日志、硬件型号及精确软件版本。它不证明旧 MC2 buffer 或执行图可以复用。

### 下一步：新建低延迟 buffer 的数值验证

当前 `DeepEPBuffer.get_deepep_buffer()` 一旦有缓存就直接返回，不检查新传入的通信组。因此先验证“用新组创建新 buffer”这条路径；不能直接替换 dispatcher 的 group 后继续拿旧 buffer 运行。

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=python \
  python test/manual/ep/probe_npu_rank_recovery.py \
  --backend hccl --world-size 4 --fault-rank 2 \
  --mc2 --mc2-strategy default --timeout 120
```

新增 `--mc2` 模式会在每个阶段创建新的 DeepEP 低延迟 buffer，执行 BF16 dispatch → 恒等专家计算 → combine，并要求输出与输入乘以 top-k 完全相等。每个 rank 的专家槽位数固定（默认 32），候选组内总物理槽位数随成员数变化（默认 128→96→128）；输入路由使用候选组内的连续编号。

首次执行会输出 torch/torch-npu/已安装算子包版本、wrapper 路径、dispatch/combine 签名和继承的 `DEEP_USE_MODE`。`--mc2-strategy` 默认明确选择 `default`，仅在创建当前 buffer 时临时设置该环境变量，随后还原；不会修改服务环境。该 wrapper 在构造函数内重新读取环境变量，不能只依靠构造函数的同名策略参数。

每阶段首次 dispatch 前还会输出实际策略、`comm_alg`（`default` 策略没有此属性，显示 `null`）与通信域名。如果 wrapper 没有采用请求的策略，探针立即拒绝执行。各 rank 的成功报告包含上述信息及数值误差、接收计数。`default` 仍然执行 DeepEP 自定义 C++/NPU 通信算子；它不是 CPU 或 Gloo 替代路径。

可显式使用 `--mc2-strategy ops` 诊断 torch-npu 路径；不会在失败后自动回退。下面记录的 CANN 9.0 环境中，该路径的 EP=4 基线已经失败，不能据此假设 EP=3 或 `16→15→16` 可用，也不能把切换 fullmesh 当作已验证的卡数限制修复。

可按已验证模型调整 `--mc2-hidden`（默认 2048）、`--mc2-local-experts`（32）、`--mc2-tokens`（16）与 `--mc2-topk`（8）。这些参数是否被当前算子支持必须由目标环境验证；`--mc2` 不支持 Gloo 模式。

**本模式仍不验证原 rank 的 `elastic_info`、旧 buffer 复用、执行图重放、专家权重加载或完整模型服务。** 成功只表明新建候选设备组上的全新低延迟 buffer 能完成数值正确的往返。旧 buffer 保留到进程退出，其安全退休方式仍需后续验证。

### MC2 首次实机失败与策略修正（日志日期 2026-10-28）

用户运行 `36fab44207` 的 `--mc2` 探针时，继承了 `DEEP_USE_MODE=ops`，报告的软件版本为：

| 组件 | 用户日志版本 |
|---|---|
| torch | `2.10.0+cpu` |
| torch-npu | `2.10.0.post2` |
| sgl-kernel-npu | `2026.6.1` |
| deep-ep | `1.0.0+7a396def.cann.9.0.0.b237` |

wrapper 的 dispatch/combine 签名包含可选 `elastic_info`。本次没有传入该参数，错误发生于 **epoch 0、四卡初始基线的第一次 dispatch**，尚未执行杀进程、缩容或回归。调用链为 `OpsLowLatencyCommStrategy → torch_npu.npu_moe_distribute_dispatch_v2 → aclnnMoeDistributeDispatchV4`，tiling 拒绝 `epWorldSize=4`，报 `epWorldSize should be 16 Aligned, but got 4`。前面的内部格式警告不是终止原因。

已有提交 `0489321022` 的 `test/manual/ascend/README_fault_tolerance_mc2.md` 记录过同一栈的四卡限制，并要求 EP=4 使用 `DEEP_USE_MODE=default`。该提交中的 fullmesh 修正针对原 EP=16 且传入 `elastic_info` 的 ops 路径，不是本探针的无 `elastic_info` 四卡基线。因此本轮修正明确选择 `default` 并增加生效检查与调用前诊断，保留 `ops` 作为显式选项。

**修正后的 `default` 路径仍待用户在目标 NPU 上重测。** 本次失败不推翻此前候选 HCCL 组验证通过的结果，也不证明 MC2 能够完成 `4→3→4`。

## 单元验证

标准依赖环境中可运行：

```bash
PYTHONPATH=python python -m pytest \
  test/registered/unit/fault_tolerance/test_recovery.py \
  test/registered/unit/fault_tolerance/test_mc2_recovery_probe.py \
  test/registered/unit/fault_tolerance/test_controller.py \
  test/registered/unit/fault_tolerance/test_manager.py \
  test/registered/unit/fault_tolerance/test_scheduler_control.py \
  test/registered/unit/fault_tolerance/test_npu_mlp_sync_timeout.py \
  test/registered/unit/eplb/test_npu_fault_recovery.py \
  test/registered/unit/elastic_ep/test_npu_mc2.py -q
```

测试覆盖：中间 rank 回归、部分恢复、完整 DP 组、非法成员请求、恢复版本同步、旧版本拒绝、控制组候选失败不安装、专家槽位不搬动、候选路由不污染 live metadata，以及 MC2 元数据缩容/恢复/再次缩容时地址和映射稳定。

本机缺少完整 CUDA/NPU 与 SGLang 依赖，逻辑测试使用隔离加载：实际执行恢复/控制器/manager/专家布局/MC2 模块，替代非相关服务框架导入，并按既有 AST 单元测试方式执行 Scheduler/MLP-sync 逻辑。这是 CPU 逻辑验证，不是完整服务启动或 NPU 集成测试。

初始提交的隔离 CPU 验证结果为 **75 passed，20 subtests passed**；另通过真实 Gloo 多进程 `4→3→4` 探针。用户已报告目标 NPU 环境的 HCCL 探针通过；MC2 首轮实机失败与待重测状态见上文。

加入 MC2 数值门禁的 CPU mock 测试后，当时隔离验证范围为 **80 passed，24 subtests passed**，并再次通过原有真实 Gloo 多进程流程。新增测试检查每次创建新 buffer、候选组内路由有效且无重复、BF16 模式、数值/NaN/形状/dtype 错误拒绝、空通信域名拒绝，以及 `--mc2` 不得以 Gloo 代替 NPU 执行。

本轮策略修正后，隔离 CPU 回归结果为 **84 passed，26 subtests passed**，Ruff 检查通过。新增覆盖默认策略覆盖继承的 `ops`、构造成功/失败后还原环境、显式 `ops` 不自动回退，以及 wrapper 未采用请求策略时在 dispatch 前拒绝执行。尚未执行本轮修正的 NPU 集成验证。

## 继续实现前的验收门槛

1. HCCL 探针已由用户报告通过，后续补齐精确软件版本和完整阶段日志。
2. 证明 MC2 buffer/handle 如何绑定恢复后的设备通信域；若需要重建或重捕图，明确顺序并实测。
3. 实现回归进程启动路径，避免默认 WORLD 初始化、加载后 barrier 和图捕获提前等待缺失成员。
4. 将当前专家布局发送到回归进程，完成全槽位权重加载、格式和覆盖校验。
5. 接入显式恢复事务：准备、暂停、数据面安装、全员试跑、确认、路由提交。确认必须绑定请求与成员版本；新进程心跳不能提前打开路由。
6. 实测 `4→3→4`、部分回归与再次故障，再考虑 Attention TP>1、执行图、overlap 和 PD。
