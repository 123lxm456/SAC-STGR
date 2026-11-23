# 服务器 smoke 反馈与校验清单行尾修复

用户回传 `plan/p5/runs/20261122T141242292188Z` 的终端结果：unit/static、三组在线训练、两组离线拟合、三组冻结评价全部退出 0；最终 `engineering_passed=true`、`status=smoke_passed_not_research`、`research_matrix_completed=false`、`passed=false`。这是符合设计的工程 smoke 通过输出，尚非完整 P5 研究验收。此处依据用户提供的日志，原始结果包尚待回传独立核对。

同次 SHA-256 检查失败，报文件名末尾 `$'\r'`。本机逐字节检查确认两份已有 `.sha256` 均为 CRLF；是 Windows `write_text` 自动换行造成，不能据此推断压缩包已损坏。已修复 `scripts/package_p5.py`、`scripts/export_p5.py` 改用 UTF-8 字节输出固定 LF，并在 `.gitattributes` 增加 `*.sha256 text eol=lf`；交付包后续同时包含该属性文件。

新增 `tests/test_delivery_checksums.py`，实际执行 Windows 平台两个专项测试均通过：生成交付及回传包，核对校验内容无 CR、结尾 LF、文件名与归档实际 SHA-256 一致。两份现有交付清单在先核对归档实际哈希后已转为 LF，压缩包内容及 SHA-256 未改变。

服务器无需重新训练或重跑本次 smoke，仅修复清单末尾 CR 并补做原包校验：

```bash
sed -i 's/\r$//' p5-smoke-fix-20261122.tar.gz.sha256
sha256sum -c p5-smoke-fix-20261122.tar.gz.sha256
python -B scripts/export_p5.py plan/p5/runs/20261122T141242292188Z
```

校验预期 `p5-smoke-fix-20261122.tar.gz: OK`；回传 exporter 打印的包及校验文件供独立审查。已授权的完整主实验可在原项目、原环境运行 `python -B scripts/accept_p5.py`，仍保留原系统参照缺失、自然优化及完整矩阵的研究验收门槛。
