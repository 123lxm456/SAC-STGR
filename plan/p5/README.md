# P5：多种子风险识别及闭环对照

> 进一步清理（2026-11-23）：两次已被成功 smoke 替代的失败试跑 `20261122T123630783825Z`、`20261122T132941429168Z` 的模型/预测文件已删除，共 36 个、41,633,466 字节；日志、元数据、源码归档、成功 smoke 及正式训练完整保留。`scripts/clean_obsolete_p5_results.py --apply` 可在服务器执行同样的受限清理，默认无参数只预览；清理不会修改 Git 历史，也不释放运行中进程的 RAM/显存。

> 2026-11-23 复核：工程 smoke 通过；正式运行已停止，种子 11 有六组完整训练凭据及一组 S0 结果待补命令凭据，尚无正式离线/闭环结果，P5 未验收。见 [本轮评估及清理](review-20261123.md)。

2026-11-22 用户已正式授权 P5。当前是**工程实现，待目标实验验收**；P0–P4 历史验收保持原有范围。主入口为 `scripts/accept_p5.py`，不会连接 SSH 或覆盖旧运行。实验定义锁定在 [protocol.json](protocol.json)。

## 任务与执行顺序

1. 在原 Linux/RTX 4090、Python 3.7.12、PyTorch 1.13.1+cu117、SUMO 1.12.0 环境执行新旧全部测试（CUDA 无跳过），记录当前源码、源码归档、环境和协议。
2. 复用 P4 的路线/场景划分及 H=10 标签协议；严格重审已经接受的 **B0 全部 train 扩展和冻结 validation/test**，作为六种风险器共用的离线数据。它们由同一冻结行为 actor 产生，不再让 B0/M1 分别用自己的轨迹比较架构。删失不进入分类或校准。保留 test 此前已用于 P4 工程诊断的事实，不用它搜索超参数。
3. B0、M1、A-B、C1、C2、C3 分别以种子 11/23/37 从头离线训练，每次 5,000 个 AdamW 更新、batch=128、正负 1:1、lr=1e-5、weight_decay=1e-5。固定最后更新的权重，不选择测试集最佳点。自然比例 validation 拟合 Platt；test 只评价。离线校准不自动改动在线奖励。
4. 七组共享 SAC 从头训练（增加 S0 零风险分支），每组每种子 100,000 个真实决策步，每轮 1,000 决策；每 50 决策在 replay 足够时执行原 SAC 的 5 个更新。风险器每触发最多尝试 3 个平衡 batch=128 更新，需各有 64 个自然正负窗口；记录尝试与实际更新，禁止人工制造碰撞。各组初始 actor 按相同种子重新设定，避免风险网络初始化消耗的随机数改变 actor 初值。
5. 七组 × 三训练种子 × 三测试种子（3000/3001/3002）× 三密度，合计 189 次闭环评价。每次固定 **3,600 个仿真步**，包括 reset 和等待观测的空闲步；不按控制动作数截断。密度为 P4 test 路线的 flow vehsPerHour × 0.5/1/1.5，显式背景车辆不变，不宣称整个网络需求精确翻倍。训练仅用 train 场景。
6. 重算保存预测的 AUPRC/AP、AUROC、漏报及正例分母、Brier、ECE；重新读取闭环物理日志，检查权重、数据、命令退出码、完整矩阵及冻结约束。先对同一训练种子的测试重复求均值，再以训练种子为单位报告均值、样本标准差及配对差值的描述性 bootstrap 区间。三种子是最低重复数，不代表统计功效充分；零事件及退化区间不能证明零风险。
7. PPO、FCFS、混合交通仍是原方案要求的系统参照。仓库无完整原方法实现/训练配置，不临时编造替代品。保留其明确证据入口及完整验收门槛，见下文；缺少时即使主矩阵完成，P5 仍不通过完整验收。

## 模型和兼容边界

| 名称 / CLI | 实现与限制 |
|---|---|
| B0 / `b0` | 复用既有时间风险器 |
| M1 / `m1` | 复用既有九 token 时空风险器 |
| A-B / `a_b` | M1 的时间近期偏置为零 |
| C1 / `c1` | 以有序 576→29→576 MLP 替换空间注意力；完整参数量与 M1 相差约 0.43%，不是置换不变集合模型 |
| C2 / `c2` | 空间层后八组均值汇总，与自车拼接投影成单个时间 token，取消末步交叉读出 |
| C3 / `c3` | 去末步交叉注意力，仅用末步自车表征；动作仍通过自车投影进入网络，不能称完全去动作条件 |
| S0 / `s0` | 原共享 SAC、原其他奖励，风险分数严格为零；不训练风险器 |

P5 使用独立工厂及 worker，不扩展旧 `train_sac.py` 的参数选项。P0–P4 默认行为不变；`Env` 仅增加未配置时不启用的物理观测回调。P4 缓冲、真实动作确认、有效步掩码、SAC replay 和优化器实现均复用。离线风险器与闭环在线风险器分别从头训练，权重不混用，不把离线分类优势直接当作闭环收益。

闭环新增统计：SUMO 碰撞车辆对连续重复报告只在起点计一次，间断后再次报告是新事件；同时保留唯一碰撞车辆数。报告每千出发车辆的碰撞事件数、碰撞车辆比例、无碰撞到达数/出发数、未完行程、AQL、AWT、平均速度及所有分母。AQL/AWT 是原环境各路口车流组的时间均值，速度是车辆时间均值，不是完整行程旅行时间。读取接口依据 [SUMO 官方 Simulation Value Retrieval](https://sumo.dlr.de/docs/TraCI/Simulation_Value_Retrieval.html#command_0x23_get_collisions)；目标版本不支持事件详情时明确失败，不退化为旧 conflict_rate。

## 上传和执行（由用户完成 SSH）

历史交付包已于 2026-11-23 清理，清单见 [退役交付及缓存清理](cleanup-retired-deliveries-20261123.json)。仅评估当前进度时，直接将 `scripts/review_p5_progress.py` 上传服务器同路径，执行 `python -B scripts/review_p5_progress.py plan/p5/runs/20261122T141242292188Z plan/p5/runs/20261122T144249871603Z --output 新命名的报告.json`；该命令不启动训练，也不授予完整 P5 验收。

如后续需要同步完整工程，可按下例重新生成新命名包。包包含活跃代码/测试及文档，不包含 P4 原始 JSONL、旧运行权重或 references。必须在**原有服务器项目目录**更新，以保留 P4 原始数据和历史验收；不要解压到缺少这些证据的新空项目：

```powershell
python -B scripts/package_p5.py --output plan/p5/transfer/p5-current-source.tar.gz
scp -P 5000 plan/p5/transfer/p5-current-source.tar.gz plan/p5/transfer/p5-current-source.tar.gz.sha256 lab2439@121.36.39.81:~/Work/LXM/SAC-STGR/
```

服务器执行：

```bash
ssh -p 5000 lab2439@121.36.39.81
cd ~/Work/LXM/SAC-STGR
conda activate SAC_RWB
sha256sum -c p5-current-source.tar.gz.sha256
tar -xzf p5-current-source.tar.gz
python - <<'PY'
import hashlib, json
from pathlib import Path
for entry in json.loads(Path('p5-delivery-manifest.json').read_text()):
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
print('P5 delivery hashes OK')
PY
df -h .
python -B scripts/accept_p5.py --smoke
```

预期：交付校验 OK；新增 UTC 运行目录，全部回归无 CUDA 跳过；3 组 × 1 种子各训练 **1,000 决策**、B0/M1 各 2 次离线优化，3 次 nominal/100 仿真步冻结评价。初版 100 决策在稀疏 train 场景仅产生约 160 条经验，未填满 SAC 的 256 批次，已延长 smoke 预算；研究协议、SAC batch 和“必须实际 SAC 更新”的门槛均不改。成功的 smoke **退出码为 2**，`engineering_passed=true`、`passed=false`、`status=smoke_passed_not_research`；risk_updates=0 在 smoke 可接受。退出码 1 表示错误，应回传该目录，不降低门槛。

可先回传 smoke 证据确认目标兼容性；正式研究命令如下（已获授权，无需再次授权，运行耗时取决于服务器）：

```bash
python -B scripts/accept_p5.py
```

完整任务含 21 次从头在线训练、18 次离线训练及 189 次闭环评价，严格串行使用同一 GPU。应在持久会话运行并保留终端打印的 Evidence 路径。大量在线 JSONL 逐窗口保存，先通过 smoke 检查实际日志增长和剩余磁盘；不要删除运行中的日志或已冻结 P4 数据。没有完整 optimizer/replay 续训，不支持从中途权重冒充完整预算重训；失败目录保留供定位。

正式运行预期：命令/源码/权重/原始统计一致；若自然风险更新为零或系统参照仍缺少，退出码 2、`needs_research_evidence`，不授予完整 P5。全部门槛满足才退出 0、`passed=true`。性能更差也是有效研究结果，不得因此重选种子或删除运行。

## 回传、复审与完整验收

将下面路径换成终端打印的具体 UTC 目录（不要把 smoke 当正式运行）：

```bash
python -B scripts/export_p5.py plan/p5/runs/具体UTC目录
```

脚本打印回传压缩包路径，附 `.sha256`。包保留环境、源码快照、日志/退出码、协议、预测、模型、物理事件流和审计；庞大的在线 samples/events JSONL 及共用 P4 原始流继续留服务器。请上传该包及校验文件，再进行目标结果复核。

原系统参照补齐后，在相同 run 下添加 `system-references.json` 并单独复审（不覆盖初次审计）：

```bash
python -B scripts/audit_p5.py plan/p5/runs/具体UTC目录 --output plan/p5/runs/具体UTC目录/audit-with-system-references.json
```

系统参照文件须含同一 `protocol_sha256`、`methods` 下 `ppo/fcfs/mixed` 的 source_archive 和 method_review（每项 path/sha256，路径必须在该 run 内）、如实说明原方法观测/动作/环境映射的非空 `interface`、`comparison_scope="system_reference_only"`、training_decisions；学习方法训练预算同协议。`evaluations` 必须覆盖每个方法、三训练种子、三密度、三评价种子，每条包含 method/training_seed/density/evaluation_seed/scenario_id/physical_path/physical_sha256/metrics。物理日志使用同一逐步 schema，必须经重算一致；FCFS 不需要学习预算，其三个训练种子位置仅用于配对索引，不能当作三个独立训练重复。原方法身份仍由归档技术评审确认，哈希一致不代替方法正确性审查。允许原方法观测/动作不同，但在相同 SUMO 流和物理暴露下仅作系统参照，不用于单模块归因；其他论文的非共同场景数字不能充数。

| 门槛 | 通过标准 |
|---|---|
| 工程 | 目标锁定环境；新旧测试全部通过且无跳过；训练/离线/评价命令成功；P0–P4 保护检查通过 |
| 公平性 | 固定协议/预算/配对种子；SAC 初值配对；共同离线数据哈希；训练/测试场景隔离；评价期间零更新 |
| 自然学习 | 所有非 S0 在线组每种子均有真实风险优化；未发生时保留证据并标记不足，不能伪造正例 |
| 评价 | 全部 18 个离线及 189 个闭环结果完整；标签、预测、物理日志、检查点、计数可重算；三密度及配对统计齐全 |
| 系统参照 | PPO/FCFS/混合交通原方法来源、配置、评审及共同评价证据完整 |
| 研究结论 | 不要求正向结果；主张必须受消融、事件支持量和不确定性限制，代码实现或短运行不等于研究验收 |

本机仅可运行 `MKL_THREADING_LAYER=SEQUENTIAL`、配置 `.local-test-deps` 后的 `python -B scripts/accept_p5.py --local`；结果永远 `passed=false`，不能替代上述目标门槛。
