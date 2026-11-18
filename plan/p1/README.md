# P1 共同修复：本地实现与远端验收

状态：工作副本中的共同修复已写入；本机无法运行 PyTorch，Linux + RTX 4090 验收待用户手动执行。不得将 P1 记为通过，也不得在 P1 验收前开始 P2。

## 修复范围

- `src/sac_rwb/Env.py`：八组轮换均按 8 取模，并把 SUMO 种子传给包装器。
- `src/sac_rwb/core/sumo_interface.py`：SUMO 种子从配置传入，缺省仍为原值 1。
- `src/sac_rwb/sac/SAC.py`：一次训练调用完成 5 次更新并返回计数。
- `src/sac_rwb/train_sac.py`：使用真实终止标志；按 `update_every` 与 replay 容量触发 SAC；风险训练要求两类样本均足额；记录优化次数；推理用 `eval()`；每个 episode 清除未完成历史；统一随机种子。
- `src/sac_rwb/algos/transformer_risk_model.py`：在线与训练窗口统一左侧零填充及有效步掩码；掩码进入原时间 Transformer；移除未执行动作的伪安全样本；风险优化器跨调用保留。
- `src/sac_rwb/test_sac.py`：初始化 SAC 接口维度与动作界，明确指定检查点，修复输出路径参数。
- `tests/test_p1.py`：八组轮换、填充与掩码、伪安全样本、SAC 与风险器优化步数的聚焦测试。

这些是所有后续对照共同使用的修复，不能把其收益记为新时空风险器的贡献。P4 的固定未来预测窗、删失与数据划分尚未实现。

## 在服务器上手动验收

用户自行将**当前项目**传至原 Linux + RTX 4090 服务器的新目录，激活 P0 使用的 `SAC_RWB` 环境。以下命令从项目根目录开始；不要在 `references/` 或 P0 的 `remote-run/work/` 中运行。

```bash
python -B -m unittest discover -s tests -p 'test_p1.py' -v
mkdir -p plan/p1/remote-run
test ! -e plan/p1/remote-run/work
cp -a src/sac_rwb plan/p1/remote-run/work
cd plan/p1/remote-run/work
timeout 20m python -B train_sac.py --epochs 1 --max_e_steps 100 --update_every 50 --save_interval 100 --dvc cuda --seed 0 > ../train.log 2>&1
echo $? > ../train.exitcode
python -B test_sac.py --run_num 0 --timestep 0 --max_e_steps 100 --test_logs_path ../test_results.pkl --dvc cuda --seed 2 > ../test.log 2>&1
echo $? > ../test.exitcode
```

`run_num=0` 仅适用于新建的空 `work/`。若工作目录已有训练日志，查看训练日志中的 `current logging run number`，据此指定测试的 `--run_num`。测试流种子 2 与训练种子 0 分开；这只是入口检查，不是 P5 的独立性能实验。

验收应确认：单元测试全部通过；训练与测试退出码均为 0；训练日志含 `Optimizer updates` 且 SAC 更新次数大于 0；日志中 SUMO `--seed 0` 与测试 `--seed 2` 对应；检查点及测试 pickle 存在。100 步短运行中的风险器更新次数可能为 0，因为真实碰撞样本尚不足 64；风险器更新路径由合成正负样本单元测试验证，不能以该短运行声称风险学习有效。保留完整日志、退出码和实际版本记录，回传后再决定 P1 是否通过。
