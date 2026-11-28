from pathlib import Path
import json
p=Path(__file__).resolve().parent
f=p/'render_report.py'
s=f.read_text(encoding='utf-8-sig')
s=s.replace('正式测试的实际到达/出发比例低于50%，不能忽视行程截尾。','106/108次正式测试的到达/出发比例低于50%，全部运行范围为35.89%至52.02%，不能忽视行程截尾。')
s=s.replace('字符串扫描结果另存audit/log_scan.csv；命中只用于定位，不自动判定仿真失败。','字符串扫描结果另存audit/log_scan.csv。唯一Traceback来自迁移历史中的一次存储回归失败（test_snapshot_detects_source_replacement_before_deletion），其后迁移复验日志记录133项通过；不是正式策略评价失败。Error字符串也包括预期异常测试，teleport字符串主要命中启动参数time-to-teleport=-1，不能当作实际传送事件。启动参数关闭SUMO警告，因此Warning计数为零不证明没有异常。')
s=s.replace('累计曲线起步瞬态不能解读为训练收敛。','累计曲线起步瞬态不能解读为训练收敛；无车辆暴露时速度缺失保留为空，不填零或插值。')
s=s.replace('不同需求档位的排序也会变化。','M1相对B0的总体五指标配对区间均跨零，三个种子的差值还出现方向反转，平均改善或恶化都不能被表述为稳定、已确认的普遍效应。不同需求档位的排序也会变化。')
f.write_text(s,encoding='utf-8')
(p/'plan/project-overview.md').write_text('# P5实验分析任务\n\n范围：既有P5全部四组正式结果；中文学术实验分析报告；用户已明确五指标、曲线、三线表及客观结论要求，无需另行选题。\n原始数据、代码和references只读；仅写本任务reports目录。阶段S3实验分析及S5复核。',encoding='utf-8')
(p/'plan/outline.md').write_text('摘要；数据与方法；指标统计口径；对比实验；三密度测试轨迹；消融；变化及不确定性；训练记录缺口；离线与开销；异常；建议结论；原始计数与复现。',encoding='utf-8')
(p/'plan/task-packets/p5-report.md').write_text('# Task packet\n\nScope: all retained P5 runs; five metrics and evidence-bound conclusions.\nInputs: project method/protocol/code; all 108 physical streams, 21 training results, 18 prediction arrays, all logs, provenance.\nAllowed writes: reports/p5_analysis_20261128 only.\nRequired artifacts: Chinese PDF/HTML, 450dpi PNG/SVG, comparison/ablation booktabs, raw-precision CSV, verification/audit.\nReject: fabricated training curves; selective seeds; treating windows as independent; unsupported safety or component claims; writes to inputs.\nValidation: analyze.py, check_claims.py, PDF full rendering and review, original hash recheck; independent agent /root/audit.\n',encoding='utf-8')
(p/'plan/experiment-protocol.md').write_text('采用现行plan/p5/protocol.json固定结果，不重新训练、不选测试最佳点。3训练种子×3测试种子×3密度×4方案；先对测试种子平均，再以训练种子n=3。整体密度等权。碰撞事件起点/出发×1000；AQL/AWT按组时间，速度按车辆时间；完成数arrived并附安全完成。三种子全部27个有放回配对重采样仅作描述。训练五指标时序缺失，不生成虚构曲线。',encoding='utf-8')
(p/'plan/review/method-experiment-traceability.md').write_text('|主张|对照|证据边界|\n|---|---|---|\n|整体时空架构改善|M1-B0|三密度碰撞均值更高；效率部分较好；综合优势未证实|\n|近期偏置必要|M1-A-B|效果随密度变化，总体A-B五项均值更好|\n|风险反馈有用|M1-S0|无一致闭环收益，不等同无所有安全机制|\n|独立空间/历史/读出创新|无单项消融|证据不足|\n|更快收敛/稳定训练|无保存训练时序|无法判断|\n',encoding='utf-8')
(p/'tables/table-schema.md').write_text('对比：B0/M1/S0；消融：M1/A-B/S0；分密度：全部四组。五指标均值±训练种子样本SD，先测试重复平均；粗体最优，下划线次优，仅描述。性能变化(M1-reference)/reference；另附绝对差及配对探索区间。原始计数求和另列。数据data/test_runs.csv、seed_density_means.csv、seed_equal_density_means.csv。',encoding='utf-8')
(p/'figures/data-manifest.md').write_text('# Figure data manifest\n\n全部真实数据，不插补、不合成。\n\n|Figure|Data|Aggregation|Script|Outputs|\n|---|---|---|---|---|\n|fig_test_low/nominal/high|data/test_time_curves.csv|60秒非重叠块；碰撞率与完成数累计；先测试重复后训练种子均值±SD|analyze.py|450dpi PNG + SVG|\n|fig_density_response|data/seed_density_means.csv|训练种子均值±SD|analyze.py|450dpi PNG + SVG|\n|fig_safety_efficiency|data/seed_density_means.csv|各种子及均值|analyze.py|450dpi PNG + SVG|\n|fig_all_seed_results|data/seed_equal_density_means.csv|密度等权的全部三种子|analyze.py|450dpi PNG + SVG|\n|fig_training_endpoints|data/online_endpoints.csv; offline_final_loss.csv|实际保存末点，不是训练曲线|analyze.py|450dpi PNG + SVG|\n',encoding='utf-8')
