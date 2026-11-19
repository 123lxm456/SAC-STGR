# P2 九 token 时空风险器

状态：P2 阶段验收通过。独立模型已实现，本机 CPU 检查和 Linux + RTX 4090 目标环境测试均通过。没有把模型接入 SAC，也没有训练权重或性能结果。

## 实现范围

- `src/sac_rwb/algos/spatiotemporal_risk_model.py` 提供 `SpatioTemporalRiskPredictor`，接口为 `forward(state_seq:[B,10,98], action_seq:[B,10,1], valid_mask:[B,10]) -> score:[B]`。
- 字段拆分为自车 `[B,10,3]` 和八组聚合车流 `[B,10,8,12]`；自车动作只在自车 token 中输入。九个 token 使用相对组别嵌入。
- 每步九 token 经一层四头空间 Encoder；九个位置分别经过共享参数的两层四头时间 Encoder，且由有效步掩码排除左侧填充键。时间层可开启或关闭固定近期偏置。
- 末步自车查询八组，拼接后经 `128→64→1`、ReLU 和 Sigmoid 输出风险分数。`save_model` 与 `load_model` 保存和加载模型参数。
- P1 的 `TransformerRiskPredictor` 保留为 B0，当前新模型没有替换训练入口。标签生成、优化预算及在线接入分别由 P3、P4 处理。

## 本机已执行的检查

在项目根目录执行：

```bash
python -B -m unittest discover -s tests -p 'test_p2.py' -v
```

本机 Windows CPU 环境运行 8 项通过，CUDA 项跳过。检查覆盖默认架构、字段索引和八组顺序、各层形状、左填充扰动不影响末步评分、当前动作及所有编码／读出模块的梯度、保存加载一致性、近期偏置开关和非法输入拒绝。本机结果不能代替目标服务器验证。

## Linux + RTX 4090 阶段验收

在 P0 锁定的服务器环境和当前项目根目录执行：

```bash
python -B -c 'import torch; assert torch.cuda.is_available(); assert "RTX 4090" in torch.cuda.get_device_name(0); print(torch.__version__, torch.cuda.get_device_name(0))'
python -B -m unittest discover -s tests -p 'test_p2.py' -v
```

首次服务器验证使用 PyTorch 1.13.1+cu117 和 RTX 4090，6 项通过、1 项失败、1 项报错：浮点键填充掩码导致 CUDA 报错，并使填充隔离失效。改用布尔键填充掩码并新增回归测试后，用户于 2026-11-19 在同一 `SAC_RWB` 环境回传完整终端输出：9 项测试全部通过、无跳过，退出码 0。CUDA 前向、反向、保存加载及填充隔离均通过；完整日志由用户保存在服务器，当前仓库仅记录该次回传结果。

验收标准为全部 9 项通过、CUDA 项不得跳过；本次已达到。模型与 B0 基线代码分离，`references/` 保持原样。P3 再进行 SAC／SUMO 在线接入与时延测量。
