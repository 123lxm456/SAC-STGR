# P5实验分析报告

- report.pdf：中文正式报告。
- report.html：图表完整的浏览器版本，与PDF同源。
- figures/：7组450dpi PNG及SVG。
- tables/：三线表booktabs片段、60行配对变化和暴露计数；LaTeX片段需booktabs及支持中文的XeLaTeX/ctex环境。
- data/：全部108次测试指标、60s曲线、训练末点、种子均值、离线指标和事件时序。
- audit/：输入字节哈希、指标重算、日志扫描及排版验证。

复现顺序：python -B analyze.py；python -B test_dynamics.py；捆绑Python -B render_report.py。原始文件保持只读。训练五指标时序未记录，不能恢复，报告没有虚构训练曲线。
