# P1 共同修复：本地实现与远端验收

状态：P1 最低验收通过。用户在 Linux + RTX 4090 上手动完成单元测试、训练和测试短运行；助手没有操作 SSH。100 步运行不构成风险器在线学习或性能证据。

## 修复范围

- `src/sac_rwb/Env.py`：八组轮换均按 8 取模，并把 SUMO 种子传给包装器。
- `src/sac_rwb/core/sumo_interface.py`：SUMO 种子从配置传入，缺省仍为原值 1。
- `src/sac_rwb/sac/SAC.py`：一次训练调用完成 5 次更新并返回计数。
- `src/sac_rwb/train_sac.py`：使用真实终止标志；按 `update_every` 与 replay 容量触发 SAC；风险训练要求两类样本均足额；记录优化次数；推理用 `eval()`；每个 episode 清除未完成历史；统一随机种子。
- `src/sac_rwb/algos/transformer_risk_model.py`：在线与训练窗口统一左侧零填充及有效步掩码；掩码进入原时间 Transformer；移除未执行动作的伪安全样本；风险优化器跨调用保留。
- `src/sac_rwb/test_sac.py`：初始化 SAC 接口维度与动作界，明确指定检查点，修复输出路径参数。
- `tests/test_p1.py`：八组轮换、填充与掩码、伪安全样本、SAC 与风险器优化步数的聚焦测试。

这些是所有后续对照共同使用的修复，不能把其收益记为新时空风险器的贡献。P4 的固定未来预测窗、删失与数据划分尚未实现。

## 2026-11-18 验收结果

- 用户回传的终端输出显示 `tests/test_p1.py` 共 5 项全部通过，其中合成正负样本检查覆盖风险器优化路径。
- `remote-run/train.exitcode` 与 `remote-run/test.exitcode` 均为 0；`test_results.pkl` 存在。回传运行副本中的六个改动代码文件与当前工作副本逐行一致，文件哈希差异仅来自行尾格式。
- 用户提供的训练日志摘录：`Episode: 0 | Timestep: 100`，`Optimizer updates: SAC=5, risk=0; risk pools: safe=34, collision=0`。风险器没有在线更新，符合正例池不足的门槛；不能据此推断长期风险学习有效。
- 用户提供的日志摘录显示训练 SUMO `--seed 0`、测试 SUMO `--seed 2`，并列出 `model/SAC_model_0/` 下 actor 与 critic 两份检查点。测试成功退出说明指定检查点可以由测试入口加载。
- 完整 `train.log`、`test.log` 和短运行检查点已在 2026-11-19 回传本地并核对；100 episode 诊断另有 40 episode 日志与检查点，退出码 143。P1 短运行的风险器更新数仍为 0，诊断中的有效更新不能当作独立性能结果。

P1 通过范围限于共同输入与训练缺陷修复的功能检查及有界短运行。P2 独立时空风险器已有本机实现，目标服务器验收待完成；P4 固定未来标签及 P5 独立性能对照仍待实施。

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

## 100 个 episode 诊断：tmux 与 TensorBoard

此运行只检查 P1 共同修复后的原时间风险器 B0 能否持续训练、碰撞样本能否积累以及风险器是否实际更新。它不是 P2 新模型实验，也不能用于比较方法效果。用户自行操作服务器；助手不发起 SSH。

先在服务器终端进入当前项目并创建 tmux 会话：

```bash
cd ~/Work/LXM/SAC-STGR
tmux new -s sacrwb_train
```

在 **tmux 会话内**执行以下命令。运行副本与 P1 短运行分开；仅从该副本的 SUMO 配置中移除净状态 XML 导出，防止 100 个 episode 产生大量 `rawdump.xml`。

```bash
conda activate SAC_RWB
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
cd ~/Work/LXM/SAC-STGR
mkdir -p plan/p1/diagnostic-100
if [ -e plan/p1/diagnostic-100/work ]; then echo 'Diagnostic work directory already exists'; exit 1; fi
cp -a src/sac_rwb plan/p1/diagnostic-100/work
cd plan/p1/diagnostic-100/work
sed -i '/<netstate-dump value="rawdump.xml"/d' real_data/osm.sumocfg
set -o pipefail
timeout 4h python -B train_sac.py --epochs 100 --max_e_steps 1000 --save_interval 10000 --dvc cuda --seed 0 2>&1 | tee ../train.log
echo $? > ../train.exitcode
```

训练运行时按 `Ctrl+b`、再按 `d` 从 tmux 分离，训练会继续；之后可以断开自己的 SSH。重新登录服务器后使用：

```bash
tmux ls
tmux attach -t sacrwb_train
```

需要另开一个服务器终端查看曲线时，在该终端执行：

```bash
conda activate SAC_RWB
cd ~/Work/LXM/SAC-STGR/plan/p1/diagnostic-100/work
tensorboard --logdir tensorboard_logs/unsignalizedIntersections/ --port 6006 --host 127.0.0.1
```

若浏览器运行在自己的电脑上，先由用户通过 SSH 本地端口转发或已有的 IDE 端口转发功能连接服务器的 6006 端口，再打开 `http://localhost:6006`。不做端口转发时，自己电脑上的 `localhost` 不指向服务器。

训练结束后在服务器 `work/` 目录检查：

```bash
cat ../train.exitcode
grep 'Optimizer updates' ../train.log | tail -n 10
tail -n 20 ../train.log
```

退出码 0 表示完成 100 个 episode；124 表示达到四小时上限。重点记录 `risk` 优化次数、`safe` 与 `collision` 池数量，不把单次奖励曲线视为最终性能结论。
