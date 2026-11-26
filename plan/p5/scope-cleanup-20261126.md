# P5 范围调整与磁盘清理（2026-11-26）

用户已要求停止三项结构消融，并确认保留同时服务于其他实验的混合历史凭据。当前协议为 p5-v2；唯一可训练组为 B0、M1、A-B、S0，风险器为前三组。原预算、标签、训练及评价种子不变：12 次在线训练、9 次离线训练、108 次闭环评价。现行实现/配置/测试/研究主张不再要求退役组；没有独立模块对照时，不单独归因于空间注意力、逐组历史或读出模块。

## 依赖与保留原则

- P5 模型工厂只删去退役分支；B0/M1 工厂、共享 SAC、P4 标签和环境接口未改。
- 保留组的模型初始化和推理输出在相同本机 Torch 下逐项相等，见 [等价验证](retained-model-equivalence-20261126.json)。这不替代目标 CUDA 验证。
- 原始 `source.tar.gz`、plan/commands/summary、源码及复用清单、历史评估报告不改写；其中的旧组名是历史凭据，已获用户确认保留。Git 历史和原始参考资料不重写。
- 独立目录、任务日志及 work 目录由专门清理策略识别；清理脚本中的退役标识仅用于删除，不是可执行实验选项。保留组的 final/initial 权重、校准、预测、物理日志和原始训练流均保留。当前协议选择固定预算末点，并无可按文件名随意淘汰的“非最佳”权重。
- 新旧运行中同名 actor 或风险权重可能是配对初始化或带来源链的复用副本；它们的路径和哈希属于审计契约，不按“内容重复”擅自删除。大型 journals 的已有符号链接不会被展开复制；删除退役链接不会跟随目标。
- 旧协议只有固定协议哈希、固定模型源码哈希才能投影为当前四组。其他 runtime 文件必须一致，数据/场景/预算仍严格核验；兼容记录见 [范围兼容白名单](scope-compatibility-20261126.json)。未知旧协议或源码差异仍拒绝复用，不伪造命令退出码。

## 已知占用及本机处理

服务器只读反馈：`plan/p5` 为 86G；旧正式运行 online 为 28G，继续运行 online 为 58G；后者 offline 为 34M、closed-loop 为 221M。未发现 accept/worker 进程。根文件系统约余 94G。原始训练流是优先盘点对象，不能把全部 86G 当成废弃权重。

本地已删除 495 个专用产物，逻辑大小 146,475,775 字节（139.69 MiB），包含两次正式运行中的退役 online/offline/closed-loop、任务日志等，见 [删除前盘点](local-retirement-scan-20261126.json) 和 [删除回执](local-retirement-receipt-20261126.json)。本地结果不代表远端释放量；Windows/WSL 及 Git 对象空间也不等于这些文件的逻辑大小。

保留四组的本地快照已重审：12 online、9 offline、108 closed-loop，均有执行凭据，见 [当前范围进度](retained-progress-20261126.json)。这里只读快照，不重审缺失的服务器 P4/online 原始流，不授予完整 P5 验收。完整验收仍须目标环境回归、自然学习门槛及原系统参照等全部通过。

## 服务器操作

先同步本轮源包到原项目；不要解压到缺少 P4 数据和旧来源链的空目录。包不包含模型、训练原始流或历史 runs。当前没有训练在运行，仍须以服务器执行时进程及全局锁检查为准。所有路径均为小写 `plan/p5`；若另有 `plan/P5`，先单独盘点，不能直接套用删除。

本机 PowerShell（项目目录）：

```powershell
scp -P 5000 plan/p5/transfer/p5-four-group-20261126.tar.gz plan/p5/transfer/p5-four-group-20261126.tar.gz.sha256 lab2439@121.36.39.81:~/Work/LXM/SAC-STGR/
```

SSH 服务器：

```bash
cd ~/Work/LXM/SAC-STGR
conda activate SAC_RWB
pgrep -af '[a]ccept_p5.py|[p]5_worker.py'
```

以上无进程输出后继续；有进程时先回传，不覆盖运行中的源代码。

```bash
sha256sum -c p5-four-group-20261126.tar.gz.sha256 && tar -xzf p5-four-group-20261126.tar.gz
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SCAN="plan/p5/server-retired-scan-$STAMP.json"
RECEIPT="plan/p5/server-retired-receipt-$STAMP.json"
df -h .
python -B scripts/clean_retired_p5.py --scan "$SCAN"
```

盘点会流式计算拟删除文件的 SHA-256，可能需数分钟；不载入全部大文件到内存。JSON 中 totals 按 run/phase/variant/seed 汇总逻辑及分配空间，largest_preserved 列出最大保留文件。若 active_processes 或 blocked_links 非空，停止并回传报告；不要手工绕过。以下应用只删除已授权退役专用产物，保存回执；有新文件、改动文件、保留链接依赖或运行锁冲突时拒绝。

```bash
python -B scripts/clean_retired_p5.py --apply "$SCAN" --receipt "$RECEIPT"
du -sh plan/p5
df -h .
python -B scripts/run_unit_checks.py --output "plan/p5/server-four-group-unit-$STAMP.json" --require-cuda
python -B scripts/verify_plan.py --report "plan/p5/server-four-group-static-$STAMP.json"
python -B scripts/review_p5_progress.py plan/p5/runs/20261123T094822803672Z --output "plan/p5/server-four-group-review-$STAMP.json"
python -B scripts/audit_p5.py plan/p5/runs/20261123T094822803672Z --output "plan/p5/server-four-group-audit-$STAMP.json"
```

预期：清理 completed、回执给出删除逻辑字节与实际文件系统空闲差值；测试零失败且 CUDA 无跳过；静态通过；快照检查 12/9/108 且 pending 空。audit 缺少自然学习或原系统参照时会退出 2，必须如实报告不足，不能把工程通过当成完整研究通过。不会重启训练。请回传盘点摘要、回执摘要、清理后 du/df、单测与 audit 输出。未确定用途的其他大文件先依据盘点报告逐项评审，默认不删。

## 验证与限制

本机最终回归 129 项：122 通过、7 跳过、0 失败/错误，见 [单测报告](removal-final-unit-20261126.json)。跳过为 4 个 CUDA、2 个 Linux 删除专项及 1 个 Windows 符号链接权限测试；Linux 清理保护在 WSL 独立运行 5 项全部通过。静态保护检查通过，64 个活跃 Python 文件通过 Python 3.7 语法解析，章节 docx 没有退役实验引用。另已核实并移除旧 source-only 交付包及校验文件 2 个，共 80,912 字节；本地累计删除 497 文件、146,556,687 字节（139.77 MiB），见 [旧交付清理](obsolete-patch-cleanup-20261126.json)。远端未执行清理前，不报告已释放服务器磁盘空间。Windows 跳过 CUDA/Linux 专项不等于目标通过；Linux 清理保护在本机 WSL 的临时目录独立验证。


## 服务器最终回传确认

用户回传的清理回执为 completed：删除逻辑大小 36,029,285,960 字节，文件系统空闲差值增加 36,029,534,208 字节（约 33.55 GiB；差值可能包含同期文件系统活动）。`du -sh plan/p5` 为 52G，根文件系统可用 172G、使用率 91%。盘点中的退役专用文件共 24 个，已由受限策略清理；其他组训练流、权重及混合历史凭据保留。

目标环境 129 项测试全部通过且无跳过，13 项静态保护检查通过；保留矩阵为 12 online / 9 offline / 108 closed-loop，pending 为空。服务器 `server-audit-20261126T052923Z.json` 确认 `primary_matrix_integrity_passed=true`、`missing_natural_risk_updates=[]`。完整研究验收唯一报告缺项为 PPO、FCFS、mixed 原系统参照；因此仍为 `passed=false / needs_research_evidence`。无需因本轮删组重新训练现有四组。工程删组与清理验证完成，不把完整主矩阵等同于完整 P5 通过。详见 [服务器结果记录](server-verification-review-20261126.json)。
