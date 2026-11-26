# P5 按任务继续与退役产物清理（2026-11-23）

> 此页保留旧协议执行历史，不是当前执行指令。2026-11-26 已改为四组协议；当前审查/复用只接受已核实的固定哈希兼容映射；已有有效权重和原始日志必须保留，不要重新启动此页的历史命令。当前操作见 [范围调整与清理](scope-cleanup-20261126.md)。

当前 P5 仍未验收。用户服务器已确认没有实验进程，正式原目录为 `plan/p5/runs/20261122T144249871603Z`，仅有六个训练命令的退出凭据和七组结果。此更新补齐**任务级继续**，不实现 optimizer/replay 的半程恢复，不改研究预算、种子、模型、标签或评价指标。

## 继续规则

- `scripts/p5_reuse.py` 默认只检查和打印提案，不训练、不写旧目录。它复用现有只读审查，验证原模型/worker 源码、协议、共同语料、场景、权重、初始化配对、日志及退出码。
- 只有带实际退出 0 及完整产物的任务可复制到新目录，并保留原始 argv、日志、退出码、来源和逐文件哈希。训练数据大文件通过 Linux 符号链接引用原目录，避免再复制数十 GB。没有原始训练流的本地 Git 回传副本不能用于真实继续执行。
- S0 的旧结果缺少原始退出码，不补写 0，也不冒充该命令已被确认成功；它在新目录按 seed=11、100,000 决策独立重跑。六组有凭据的风险器不重新训练。
- 已完成离线/闭环任务也可复用；缺执行凭据的任务重新执行。闭环仅在对应原 actor 训练同样被复用时复用，防止把旧 actor 的评价混入新训练。
- 新入口为 `scripts/accept_p5.py --reuse-from 原目录`，创建新的 UTC 运行目录，重新执行目标环境/全部单测/静态检查和共同语料审计。它先复用或执行 seed=11 的任务，再完成其他种子和评价。
- `reuse-manifest.json` 记录来源和哈希；`audit_p5.py` 增加来源链重审，不将新目录中的继承日志描述成再次发生的训练。缺来源、权重改变、不同模型/worker/数据/预算或循环来源均拒绝验收。
- 增加 Linux 文件锁及孤立 worker 检查，避免同时启动多个主实验。顶层摘要在开始及每个任务前后写入进度，Ctrl-C 留下 interrupted 状态。不可捕捉的系统终止仍须结合进程状态判断，不把 running 字段单独当作进程存活证据。

本次提案：复用 **6** 个训练任务；其余 **15** 个在线任务（包括重跑 S0 seed=11）、**18** 个离线任务和 **189** 个闭环任务待执行。本地只读提案见 [reuse-plan-20261123.json](reuse-plan-20261123.json)；本地原始训练流缺失是预期，实际继续仅在原服务器。

## 清理结果

在先前删除 36 个失败试跑模型/预测文件（41,633,466 字节）的基础上，本轮已删除另 **12 个退役中间文件，共 13,542,574 字节**：四次旧本机验证、两次失败 smoke 的六份重复源码归档，以及两次失败 smoke 的六份旧物理流。每个文件均先核对 Git 原版本可恢复；摘要、日志、元数据、源码指纹、场景等仍保留，历史失败不会被伪装为通过。

清单及恢复提交见 [retired-run-artifacts-20261123.json](retired-run-artifacts-20261123.json)。最近成功 smoke `20261122T141242292188Z`、正式原目录 `20261122T144249871603Z` 及 P0–P4 必要资料未删除。更新后的 `clean_obsolete_p5_results.py --apply` 可在服务器做同样的固定名单清理，拒绝被重新用于研究/成功运行的旧目录，不递归删除目录。

**正式原目录仍被继续后的数据链接和来源链使用，不能删除。** 此清理减少工作区磁盘内容，不改写 Git 历史，不声称释放 RAM/显存。

## 用户执行命令

小型更新包只包含脚本、测试、属性和本说明，不包含研究运行数据、模型或新的源数据，不会覆盖旧 run。先从本机项目根目录 PowerShell 上传：

```powershell
scp -P 5000 plan/p5/transfer/p5-continuation-20261123.tar.gz plan/p5/transfer/p5-continuation-20261123.tar.gz.sha256 lab2439@121.36.39.81:~/Work/LXM/SAC-STGR/
```

服务器先确认没有相关实验进程；若有输出，不覆盖正在使用的代码：

```bash
cd ~/Work/LXM/SAC-STGR
conda activate SAC_RWB
pgrep -af 'scripts/(accept_p5|p5_worker)\.py'
```

没有活动进程时，校验和更新：

```bash
sha256sum -c p5-continuation-20261123.tar.gz.sha256 && tar -xzf p5-continuation-20261123.tar.gz
python -B scripts/clean_obsolete_p5_results.py --apply
python -B scripts/p5_reuse.py plan/p5/runs/20261122T144249871603Z
```

预期 `reused_online=6`、`remaining_online=15`、`remaining_offline=18`、`remaining_closed_loop=189`，并且 **raw_journals_present_for_reused_online=true**。若数据缺失、源码或协议不匹配，先回传错误，不改门槛或覆盖旧结果。

然后启动已授权的正式继续，使用 nohup 保留顶层日志并避免 SSH 断开终止实验：

```bash
P5_LOG="plan/p5/continuation-$(date -u +%Y%m%dT%H%M%SZ).log"
nohup python -u -B scripts/accept_p5.py --reuse-from plan/p5/runs/20261122T144249871603Z > "$P5_LOG" 2>&1 < /dev/null &
echo "PID=$! LOG=$P5_LOG"
tail -n 30 "$P5_LOG"
```

目标测试必须无 CUDA 跳过；日志会打印新 `Evidence` 目录、`Validating reusable task`、`Reused verified task`，之后才打印需要重新执行的任务。读取并哈希大原始日志需要时间，不是再次训练。执行期间不要在该服务器项目内拉取或覆盖源码，否则会破坏源码冻结条件。

结束后回传新目录摘要和审计；原系统参照未补齐时，即使主矩阵完成，完整 P5 仍是 `needs_research_evidence`。不要将继续入口可运行、通过预检或模型更新次数当作性能结论。

## 实际验证范围

八项继续机制专项已通过，覆盖真实复制/原始命令链、原始日志链接分支、预算/数据拒绝、丢失原始流、无退出记录不得复用、权重改动、源码改变、来源缺失及孤立 worker 检查。三个清理专项通过。最新完整本机回归见 [continuation-local-unit-20261123.json](continuation-local-unit-20261123.json)：122 项中 118 通过、4 CUDA 跳过，0 失败/错误。原实际 worker 的替代环境集成测试同样通过，不将它当作新的 SUMO 结果。

真实 RTX 4090 继续执行、原始日志链接及完整研究矩阵仍待服务器验证；没有新模型优越性结论。本轮遵循 verification 的实际命令/日志门控，没有修改论文或正式实验协议。
