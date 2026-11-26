# P5 无损存储迁移

2026-11-26 服务器迁移已完成，最终只保留 `20261123T094822803672Z` 一个运行目录。删除回执记录 336 个清单路径移除，迁移范围的 inode 去重占用从 53,052,616,704 降至 2,566,201,344 字节，实际减少 50,486,415,360 字节。删除后完整 P5 审计与迁移前一致；删除后 P0—P4 目标复验 133 项单测无跳过、原始数据复审及结构检查均通过。P5 仍缺 PPO、FCFS、mixed 系统参照，状态为 `needs_research_evidence`，不授予研究验收。

## 目标结构

```text
plan/p5/runs/20261123T094822803672Z/
├── online/、offline/、closed-loop/  # 正式四组权重、预测与物理结果
├── scenarios/                     # 正式配置
├── data/objects/<SHA256>.jsonl.gz   # 30 份去重后的原始流，无损存储
├── provenance/
│   ├── 20261122T141242292188Z/      # smoke 原始凭据和结果
│   ├── 20261122T144249871603Z/      # 原正式任务来源凭据和结果
│   └── storage-stage/              # 原压缩回执
├── storage-index.json              # 原路径→新位置及原始内容哈希
└── migration/                      # 迁移前后审计、逐文件清单及删除回执
```

新目录保留两种旧运行的有价值凭据，根目录只剩一个正式实验。仅含 `sumolog` 的 `work-*`、`eval-work-*` 删除；其他未知工作内容会阻止迁移。复制旧模型/场景时按实际 SHA-256 去重，Linux 硬链接保留必要原路径，避免丢失审计所需的权重。原始 `plan/commands/reuse-manifest/source` 均保持原字节，执行历史与正式结果不混并。

压缩对象不提交 Git；压缩对象与索引需一起备份，Git 中的清单不能替代实验数据。回传导出入口排除 `data/objects/`，保留索引、模型和证据。

## 门槛和操作

1. `python -B scripts/consolidate_p5.py --prepare plan/p5/storage/primary-20261123T094822803672Z/stage-20261126T121550709575Z.json`：先计算旧布局完整审计，再完整解压校验压缩流并与原文件 SHA-256 比较。仅创建硬链接、来源凭据副本和索引，不删除原文件。
2. `python -B scripts/consolidate_p5.py --verify`：按压缩索引读取并核验全部对象，重算完整 P5 审计，与步骤 1 返回的全部指标/状态严格相等；执行目标全量测试、静态保护及 P0—P4 目标原始数据复验。通过后写独立凭据。
3. `python -B scripts/consolidate_p5.py --finalize`：验证凭据与当前代码、索引、清单一致，再核验全部保留文件和压缩对象；检查所有待删除源路径的 inode/大小/修改时间、原始内容 SHA-256 和目录覆盖，拒绝未知文件。逐文件删除清单覆盖的旧副本及原始流，删除空目录，重新审计并记录实际占用与文件系统空闲差值。

也可通过 `bash scripts/migrate_p5_storage.sh` 串行执行三步，任一步失败则不执行下一步。脚本单独启动 shell，错误不会关闭 SSH 登录终端。迁移期间不得改动源码、运行 Git 更新或启动实验。迁移和训练共用互斥锁。

脚本报错时保留日志并停止，不运行手动 `rm -rf`。prepare 在索引尚未发布时允许核对已有副本后重试；索引已经发布后使用 verify。finalize 若中途因系统故障中断，应按删除回执盘点剩余文件后恢复，不能把部分删除当作完成。

## 后续读取与实验复现

`audit_p5.py`、`review_p5_progress.py` 和 `p5_reuse.py` 通过索引读取来源凭据及压缩原始流；旧运行路径传给审查入口时会解析到 `provenance/`。`verify_p01234.py` 和旧结构审计也会解析 SUMO 历史配置内的旧输入路径，不改写原始 XML。

流式分析示例（项目根目录）：

```python
import sys
sys.path[:0] = ['scripts', 'src/sac_rwb']
from p5_storage import iter_journal_rows, read_online_dataset, relocated
path = 'plan/p5/runs/20261123T094822803672Z/online/m1/23/online-data/samples.jsonl'
count = sum(1 for row in iter_journal_rows(path))
```

完整消费或正常关闭 `open_journal` 会验证解压原字节 SHA-256 和大小。`read_online_dataset(directory)` 兼容既有严格 P4 标签审计：临时展开单份数据后调用原 reader，返回后删除临时副本；需要单份原始流大小的临时空间及原 reader 的内存。只做统计时优先流式读取。不得把 `.jsonl.gz` 直接传给旧的纯文本 reader。

历史原始命令是执行凭据；再次运行时应使用新输出路径，并通过 `relocated()` 解析保存的旧配置路径。原始配置 XML 的绝对输入路径也需在独立的复现工作副本中解析；不能直接执行已归档的旧命令来覆盖冻结结果。

P5 系统参照缺失不会由存储迁移补齐。迁移前后均应为主矩阵完整、自然风险更新无缺项、完整研究仍为 `needs_research_evidence`，直至 PPO/FCFS/mixed 证据真正补齐。
