# P0/P1/P2 修订版统一验收

> 2026-11-20 修订版统一验收已通过，原始证据目录 `20261120T104147769112Z/` 已回传。该次 35 项全部通过，P0/P1 运行退出码为 0。下方为重现操作；P3 已开始，见 `plan/p3/README.md`，后续新增测试数量以实际报告为准。

2026-11-20 本轮只完成共同修复、独立模型验证和清理；P3 未开始。助手不连接 SSH，以下服务器命令由用户执行。不得用旧日志替代本轮源码的验收。

## 服务器执行

1. 将当前项目传至服务器新目录，`references/` 如已存在保持原样；如缺失，另行提供经校验的输入包，P0 只在独立运行目录重建工作副本。保留字节及脚本 LF 行尾。不需要传 `.local-test-deps/`，不要覆盖历史运行目录。
2. 进入新项目根目录，激活既有环境，不安装或升级依赖：

```bash
cd ~/Work/LXM/SAC-STGR
conda activate SAC_RWB
python -B scripts/accept_p012.py
```

若实际上传目录不同，修改 `cd`。脚本不会进行 SSH、安装依赖或实现 P3。

3. 终端打印 `Evidence: .../plan/acceptance/<UTC时间>/`。使用打印出的实际路径查看：

```bash
RUN=/实际项目路径/plan/acceptance/实际时间目录
cat "$RUN/summary.json"
cat "$RUN/unit-results.json"
```

## 预期结果

- 程序退出码为 0；`summary.json` 的 `mode` 为 `target`、`passed` 为 true，P0/P1/P2 均为 passed，`p3_started` 为 false。
- 当前共 35 项测试全部通过，`skipped`、`errors`、`failures` 均为空。包含默认 B0 CUDA 优化，以及默认 64 维 P2 CUDA 前后向、掩码与保存加载。
- P0 环境、输入及训练退出码均为 0，原系统完成 100 个决策步。
- P1 修订版训练/测试退出码均为 0，SAC 更新大于 0；训练种子 0、测试种子 2。新的 SAC 检查点后缀为 100。短运行 risk=0 可以接受，其 CUDA 更新路径由合成标注样本测试验证，不代表在线学习效果。
- `source-manifest.json` 保存源码与输入哈希；环境、命令、原始日志、退出码独立归档，结束时再次核对源码未变化。

环境按历史实际版本检查：Python 3.7.12、PyTorch 1.13.1+cu117、CUDA 11.7、RTX4090、驱动 555.42.06、SUMO 1.12.0 及运行依赖。版本不同会停止并留下错误摘要；请回传错误及环境信息，不要直接改版本断言。

## 回传证据

回传整个新验收目录最便于复核。若过大，服务器保留完整目录，再生成精简证据包：

```bash
tar -czf "${RUN}-evidence.tgz" -C "$RUN" \
  summary.json environment.json source-manifest.json \
  static.json static.log static.exitcode \
  unit-results.json unit.log unit.exitcode \
  p0.log p0.exitcode p1-train.log p1-train.exitcode p1-test.log p1-test.exitcode \
  p0/environment.log p0/environment.exitcode p0/inputs.log p0/inputs.exitcode \
  p0/snapshot.json p0/train.log p0/train.exitcode p0/command.txt p0/work/sumolog
```

打包命令适用于完整成功运行。失败时部分文件尚未生成，请直接回传已有目录、`summary.json` 及失败步骤日志。原始运行副本、检查点和 `test_results.pkl` 继续保留在服务器，复核后再决定归档。

## 本机辅助检查

本机 `.local-test-deps/` 中的 Gym 0.26.2、sumolib/traci 1.23.1 仅用于辅助导入与逻辑检查，未改系统环境。Windows CPU 结果不代表目标环境验收。

```powershell
$env:PYTHONPATH = Join-Path ((Get-Location).ProviderPath) '.local-test-deps'
python -B scripts/accept_p012.py --local
```

本机允许两项 CUDA 测试跳过，`summary.json` 中 `passed` 保持 false；已有目录会拒绝覆盖。


## 参考输入与过时修复包

旧的 `p012-archive-fix-20261120.tar.gz` 及其校验文件已在 P0–P3 复核清理中删除：它们只是交付载体，包含已过时的脚本，不应再解压覆盖当前代码。历史包哈希保留于 `plan/review/cleanup-p0123-20261120.json`。

服务器有 `references/` 时只读校验，不修改。缺少该目录时，仍保留并使用 `plan/p0/transfer/reference-inputs-20261120.tar.gz`，由 `scripts/p0_snapshot.py` 在内存核对保护清单后只写独立运行副本。该原始输入包及其校验文件没有删除；不要将修订版 src/ 冒充原始输入。

## 历史输出压缩

`20261120T104147769112Z` 下 P0 和 P1 工作目录的两份 `real_data/rawdump.xml` 是 SUMO 输出，不是输入，已先逐字节验证再无损压缩成 `rawdump.xml.gz`，原 XML 副本删除。日志、源码快照、场景、检查点和 pickle 均保留。需要轨迹输出时可用 `gzip -dk rawdump.xml.gz` 恢复，原内容 SHA-256 见清理清单；不要在 references/ 中执行恢复。
