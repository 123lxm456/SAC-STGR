# P0 原系统复核记录

> 2026-11-20 修订版以 [统一验收步骤](../acceptance/README.md) 为准；下方旧测试记录保留为历史证据。本轮目标环境验证尚未完成。

状态：历史 P0 最低验收已通过。2026-11-20 已修正复现流程与依赖清单；本轮统一目标环境复验待用户执行，不能据历史通过启动 P3。

## 本地已完成

- `requirements-server.txt` 现按历史实际环境统一为当前入口所需的运行依赖；完整历史 pip freeze 另存 `environment-lock.txt`。`src/sac_rwb/requirements.txt` 与运行依赖清单一致；参考源码中的清单原样保留。未验证全新环境安装。
- `snapshot.json` 列出 23 个参考源码及场景文件与工作副本的 SHA-256、大小；复制后逐个重算一致。
- 原论文 PDF 由 `plan/review/protected-integrity.json` 的保护基线记录，没有复制进可运行目录。
- 车流文件含 6 条 route、6 条 flow、30,502 条显式 vehicle；目标路口 229、499、332、334 均存在于路网。
- 当前本机为 Windows 可见的 MX330；运行证据来自用户在 Linux + RTX 4090 服务器执行后复制回的日志。

## 远端验收结果

- `remote-run/environment.log`：2026-11-17 16:54:19 +08:00；Ubuntu 22.04.4 LTS、NVIDIA GeForce RTX 4090（24564 MiB）、驱动 555.42.06、SUMO 1.12.0、Python 3.7.12、PyTorch 1.13.1+cu117、PyTorch CUDA 11.7，CUDA 可用；`pip freeze` 记录了实际安装依赖。
- `remote-run/inputs.log`：23 对参考文件与工作副本的 SHA-256 校验通过。
- `remote-run/environment.exitcode`、`inputs.exitcode`、`train.exitcode`：均为 0。
- `remote-run/command.txt`：原入口执行 `--epochs 1 --max_e_steps 100 --save_interval 10000 --dvc cuda --seed 0`，有 20 分钟超时上限。SUMO 包装器的命令行显示固定 `--seed 1`。
- `remote-run/train.log`：Episode 0 完成，Timestep 100，Average Reward -251.27，Episode Time 4.04 秒；未见 traceback 或错误标记。启动时 SUMO 的 `TraCI issued load command` 是环境 reset 时的重载记录；这份日志不能证明长期训练收敛或风险器有效更新。
- `remote-run/work/sumolog`：SUMO 在仿真时间 133.00 秒因 TraCI 请求终止；记录插入车辆 470 辆、碰撞引起的 teleport 3 次。这些只是该次短运行的原始事件，不作为安全性或效率性能结论。仿真时间与训练入口的 100 个决策步不是同一计数。

### 依赖清单与实际环境的差异

| 包 | 用户给定清单 | 实际 `pip freeze` |
|---|---:|---:|
| gym | 0.18.3 | 0.21.0 |
| imageio | 2.31.2 | 2.9.0 |
| Pillow | 7.0.0 | 8.2.0 |

其余已列版本与清单一致。当前可复现的短运行环境应以 `remote-run/environment.log` 的实际版本为准；后续实验开始前需统一决定采用实际环境还是重建到给定清单，并对所有对照使用同一环境。此事不改变 P0 原系统短运行的通过结论。

`remote-run/work/` 已重新完整复制回；其中 23 个原项目文件全部存在；2026-11-20 重算发现 20 个文件因 LF→CRLF 转换而字节哈希不同，规范化换行后 23 个文件内容全部一致，`sumolog` 亦可核对。先前两份大型 XML 回传不完整的问题已消除。P0 未进行训练曲线、模型权重或安全效率性能验证。

## 远端复现命令

把当前项目完整复制到服务器，激活以前用于该项目的 Python 环境，在项目根目录运行：

```bash
bash scripts/p0_remote.sh
```

脚本只检查环境和输入，随后在 `plan/p0/remote-run/work/` 中复制原项目并执行一次最多 100 仿真步、最长 20 分钟的原训练入口。结果写在 `plan/p0/remote-run/`；脚本不安装依赖、不改参考源码，也不执行 P1 修复。命令行随机种子为 0；原 SUMO 包装器另将 SUMO 种子硬编码为 1，两者在 P0 仅如实记录。

本次环境、输入、训练日志和完整工作目录已复制回本地。再次复现时脚本在任何日志写入前独占创建新的 `verification-*` 目录，从 `references/` 重建原系统，不读取已经修订的 `src/`；传入已有目录会立即拒绝。历史 `remote-run/` 不会覆盖。
