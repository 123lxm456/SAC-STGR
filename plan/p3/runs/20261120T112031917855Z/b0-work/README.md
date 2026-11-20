# SAC-RWB 工作副本

P0/P1/P2 已完成目标环境验收。训练入口默认 B0；使用 `--risk_variant m1` 选择 P3 新风险器适配器，默认 M1 为 P2 的 64 维架构。P3 目标环境接入验证待完成。只读原始版本位于项目根目录 `references/SAC-RWB-main/`。

目标环境：Ubuntu 22.04.4、RTX4090、驱动 555.42.06、Python 3.7.12、PyTorch 1.13.1+cu117、SUMO 1.12.0。`requirements.txt` 按实际运行依赖统一，完整历史 freeze 在根目录 `plan/p0/environment-lock.txt`。未验证从空环境安装，优先使用原 `SAC_RWB` 环境。

从项目根目录执行 `python -B scripts/accept_p012.py`，完整步骤见 `plan/acceptance/README.md`。单独短运行应先复制本目录到新的运行目录，从副本执行：

```bash
python -B train_sac.py --epochs 1 --max_e_steps 100 --save_interval 100 --dvc cuda --seed 0
python -B test_sac.py --run_num 0 --timestep 100 --max_e_steps 100 --dvc cuda --seed 2
```

`run_num=0` 仅适用于新副本。新检查点后缀为实际决策步数；旧文件仍按原后缀加载。训练预算由 `--epochs` 与 `--max_e_steps` 控制。SAC 参数文件不提供完整断点续训，`--Loadmodel True` 会明确拒绝。风险器加载失败会停止，避免误用随机初始化模型。

风险窗口在控制片段、路口、组别顺序或仿真步连续性变化时重建，被替换的未判定窗口不生成标签。当前仍用原终局结局监督，P4 的固定未来窗与删失协议尚未实现。历史检查点只作诊断证据。

P3 请从根目录执行 `python -B scripts/accept_p3.py`，使用新的独立运行副本；步骤见 `plan/p3/README.md`。`--d_model`、`--nhead` 和 `--num_layers` 是 B0 参数，M1 使用固定的 P2 默认配置并将实际配置写入运行报告。
