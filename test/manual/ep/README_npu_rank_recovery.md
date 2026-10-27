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

## 单元验证

标准依赖环境中可运行：

```bash
PYTHONPATH=python python -m pytest \
  test/registered/unit/fault_tolerance/test_recovery.py \
  test/registered/unit/fault_tolerance/test_controller.py \
  test/registered/unit/fault_tolerance/test_manager.py \
  test/registered/unit/fault_tolerance/test_scheduler_control.py \
  test/registered/unit/fault_tolerance/test_npu_mlp_sync_timeout.py \
  test/registered/unit/eplb/test_npu_fault_recovery.py \
  test/registered/unit/elastic_ep/test_npu_mc2.py -q
```

测试覆盖：中间 rank 回归、部分恢复、完整 DP 组、非法成员请求、恢复版本同步、旧版本拒绝、控制组候选失败不安装、专家槽位不搬动、候选路由不污染 live metadata，以及 MC2 元数据缩容/恢复/再次缩容时地址和映射稳定。

本机缺少完整 CUDA/NPU 与 SGLang 依赖，逻辑测试使用隔离加载：实际执行恢复/控制器/manager/专家布局/MC2 模块，替代非相关服务框架导入，并按既有 AST 单元测试方式执行 Scheduler/MLP-sync 逻辑。这是 CPU 逻辑验证，不是完整服务启动或 NPU 集成测试。

本轮该范围的隔离 CPU 验证结果为 **75 passed，20 subtests passed**；另通过真实 Gloo 多进程 `4→3→4` 探针。目标 NPU 软件栈上的 HCCL/MC2 测试尚未运行。

## 继续实现前的验收门槛

1. 在目标 NPU 环境运行 HCCL 探针，记录精确软件版本和结果。
2. 证明 MC2 buffer/handle 如何绑定恢复后的设备通信域；若需要重建或重捕图，明确顺序并实测。
3. 实现回归进程启动路径，避免默认 WORLD 初始化、加载后 barrier 和图捕获提前等待缺失成员。
4. 将当前专家布局发送到回归进程，完成全槽位权重加载、格式和覆盖校验。
5. 接入显式恢复事务：准备、暂停、数据面安装、全员试跑、确认、路由提交。确认必须绑定请求与成员版本；新进程心跳不能提前打开路由。
6. 实测 `4→3→4`、部分回归与再次故障，再考虑 Attention TP>1、执行图、overlap 和 PD。
