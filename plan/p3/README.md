# P3 在线接入与工程验收

P0/P1/P2 已于 2026-11-20 在 Linux/RTX4090 通过统一验收，原始证据已回传。P3 在确认后开始；当前本地实现完成、目标环境工程验收待执行。P4/P5 没有开始。

## 实现及复用

- `train_sac.py --risk_variant b0|m1`：默认保留 B0，M1 使用已验收 P2 的 64 维、一层空间和两层时间编码器。
- `algos/online_risk.py`：M1 适配器继承原 P2 网络，复用 P1 `RiskExperienceBuffer`。评分前按真实车 ID、控制片段、路口、组别及仿真步维护历史；当前步输入与发送 SUMO 的物理加速度一致，按虚拟 ID 将批量分数返回奖励。
- `algos/risk_training.py`：原 B0 的 BCE 优化方法原样抽出，两种模型共用，优化器不在调用间重建。触发、平衡抽样、学习率、epoch 和有效优化次数计数沿用 P1。
- M1 参数名与独立 P2 一致，可互相加载纯参数文件；风险文件名包含模型种类。参数文件不包含 replay、优化器、随机状态，不称完整续训检查点。
- `risk_runtime_<run>.json` 记录实际模型配置、参数量、SAC/风险更新次数、样本池数量、批量评分次数和时延。训练成功或异常退出均关闭环境与 TensorBoard。

P3 仍使用原终局结局监督。控制片段被替换时不捏造安全标签，固定未来 H、删失和独立数据划分留给 P4。短运行以及随机输入时延不用于比较安全或效率收益。

## 时延定义

在线 `end_to_end_ms` 使用同步墙钟，包含历史及张量组装、设备传输、网络前向和 CPU 车辆映射；首调用成本计入。保存最近 10,000 次调用的分位数，累计均值与调用/车辆数覆盖全程。`amortized_ms_per_vehicle` 是批量摊销成本，不是单车时延。

`scripts/benchmark_risk.py` 另测已经放到设备上的张量前向，B0/M1 各使用批量 1、8、32 和有效长度 1、10；暖身 5 次后测 20 次，输出 12 组参数量、均值、P50/P95 和原始样本。输入为随机数，没有训练权重，不代表模型识别能力。两种时延口径不能混用。

## 用户在服务器执行

先将本轮代码、测试及 `plan/acceptance/p012-accepted-20261120.json` 同步到服务器。保持既有 `references/`、旧验收目录和环境原样；不要解压旧修复包覆盖本轮新代码。

```bash
cd ~/Work/LXM/SAC-STGR
conda activate SAC_RWB
python -B scripts/accept_p3.py
```

脚本检查已通过的 P0/P1/P2 记录和锁定环境，运行全部单元测试、静态检查和时延测试，再分别从 B0/M1 的新副本进行 100 步训练及测试入口加载。运行副本只移除 `netstate-dump` 输出，避免产生大型 rawdump.xml；不改变场景行为。所有原始输入、工作目录、配置差异和日志独立归档。

预期打印 `Evidence: .../plan/p3/runs/<UTC时间>/`。当前共 45 项测试，目标环境全部通过、无跳过；B0/M1 各有 100 次在线风险评分、SAC 更新大于 0、训练与测试退出码为 0，`latency.json` 有 12 组数据。`summary.json` 的 `mode` 为 target、`passed` 为 true、`p4_started` 为 false。

请回传该新目录，或至少回传：

```bash
RUN=/实际打印的项目路径/plan/p3/runs/实际时间目录
cat "$RUN/summary.json"
cat "$RUN/unit-results.json"
```

复核需要 `latency.json`、`b0-runtime.json`、`m1-runtime.json`、`environment.json`、`source-manifest.json` 和训练/测试日志。服务器原目录及检查点先保留。

### 自然结局风险更新的证据边界

100 步短运行可能 risk=0。此时只通过接入工程门槛；M1 CUDA 优化路径由合成标注样本测试验证，`natural_risk_learning_observed` 必须为 false，不能声称自然在线风险学习已经验证。

需要观察真实碰撞池积累及自然更新时，在复核短运行之后使用独立新运行：

```bash
python -B scripts/accept_p3.py --episodes 10 --steps 1000 --require-online-risk-updates
```

此命令可能耗时较长，单组训练上限四小时。若仍没有足够碰撞样本，会明确失败并保留日志，不降低样本门槛、不注入伪样本。该诊断仍不构成 P5 性能结论。

## 本机辅助验证

```powershell
$env:PYTHONPATH = Join-Path ((Get-Location).ProviderPath) '.local-test-deps'
python -B scripts/accept_p3.py --local
```

本机运行全部回归测试和 CPU 时延口径检查，允许三个 CUDA 项跳过，不运行真实 SUMO，不授予目标 P3 验收。
