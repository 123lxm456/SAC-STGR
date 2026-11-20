# 历史工作副本归档

`cold-working-copies-20261120.tar.gz` 完整保存三个早期运行的 `work/` 副本：

- `plan/p0/remote-run/work/`：原系统短运行。
- `plan/p1/remote-run/work/`：共同修复后的早期短运行。
- `plan/p1/diagnostic-100/work/`：中止的长期诊断。

这些副本已由后续正式验收覆盖；归档前逐文件记录 SHA-256，归档后逐文件解压读取校验，再删除展开副本。原目录下 `work/` 外的日志与退出码保留。完整清单和归档校验值见 `../review/cold-snapshot-cleanup-20261120.json`。当前源码、references、原始输入包及正式验收工作目录不属于此归档范围。

需要检查旧快照时，从项目根目录在 Linux 上执行：

```bash
(cd plan/archive && sha256sum -c cold-working-copies-20261120.tar.gz.sha256)
HISTORY_DIR=$(mktemp -d /tmp/sac-stgr-history.XXXXXX)
tar -xzf plan/archive/cold-working-copies-20261120.tar.gz -C "$HISTORY_DIR"
printf '%s\n' "$HISTORY_DIR"
```

归档使用原项目相对路径，恢复后的旧工作目录位于上述临时目录下的 `plan/.../work/`。复查历史文件使用临时副本；当前代码验收继续使用 `scripts/accept_p012.py` 和 `scripts/accept_p3.py`。
