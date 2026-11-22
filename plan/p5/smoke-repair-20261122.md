# Git 拉取恢复及 smoke 短预算修复

用户回传 Git `curl 18 / early EOF` 及服务器 smoke 报错。本次以单请求 HTTP/1.1 执行 fetch 成功，再 `git merge --ff-only origin/main`，本地从 `2d60248` 快进到 `446999e`。未重置分支、未覆盖用户未提交改动，未改永久 Git 配置。拉取恢复后工作区原先干净，之后产生本修复的代码/文档改动。

服务器 `20261122T123630783825Z` 和 `20261122T132941429168Z` 两次记录均为：目标 106 项回归无跳过通过；各执行命令退出 0；三组 `completed=true`、`decision_steps=100`、`sac_updates=0`，最终审计拒绝 `Incomplete online training`。B0 首轮日志有 160 个已执行窗口，低于 SAC 批次 256。故是 smoke 暴露预算不足，不能解释成已经优化或验收通过。

修复 `scripts/accept_p5.py`：独立构造 smoke 协议并将在线预算延长至 1,000 决策；深拷贝协议，研究预算不变。保留 SAC batch=256、实际 SAC 更新及自然风险学习门槛。`scripts/audit_p5.py` 的报错补充组别、种子、步数和实际 SAC 更新数。新增 smoke 预算/协议不变回归，并在集成测试中注入 SAC=0，验证审计依旧拒绝通过。

新版本仍需服务器重新执行，旧失败运行保持原样，不改写旧结果。压缩包/校验文件缺失说明当时没有上传交付载体；服务器已有代码能够执行，不是导致计数为零的原因。

本机执行修复后的 `python -B scripts/accept_p5.py --local`，新证据目录为 `plan/p5/runs/20261122T140654903738Z/`；最终计数以该目录 `unit-results.json`、`summary.json` 为准。完整 P5 仍未验收，系统参照及正式研究实验门槛保持不变。

本轮修复尚未提交/推送，不能期待服务器再次 git pull 自动拿到它。新交付包使用独立文件名；先在本机项目根目录 PowerShell 上传：

```powershell
scp -P 5000 plan/p5/transfer/p5-smoke-fix-20261122.tar.gz plan/p5/transfer/p5-smoke-fix-20261122.tar.gz.sha256 lab2439@121.36.39.81:~/Work/LXM/SAC-STGR/
```

然后在服务器执行：

```bash
cd ~/Work/LXM/SAC-STGR
conda activate SAC_RWB
sha256sum -c p5-smoke-fix-20261122.tar.gz.sha256
tar -xzf p5-smoke-fix-20261122.tar.gz
python -B scripts/accept_p5.py --smoke
```

预期三组各 1,000 决策且 SAC 更新大于零；最终退出码 2、`smoke_passed_not_research`。若退出 1，保留并回传新目录；不要删除先前两个失败运行。若 Git 下载再次中断，可单次使用 `git -c http.version=HTTP/1.1 -c http.maxRequests=1 pull --ff-only origin main`，无需修改全局配置。
