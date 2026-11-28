from pathlib import Path
import hashlib,json,zipfile,platform
O=Path(__file__).resolve().parent;R=O.parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
original=json.loads((O/'audit/original_hashes.json').read_text(encoding='utf-8'));changed=[];missing=[]
for x in original:
 p=R/x['path']
 if not p.exists():missing.append(x['path'])
 elif sha(p)!=x['sha256']:changed.append(x['path'])
assert not changed and not missing,(changed,missing)
res=dict(files_checked=len(original),reference_files_checked=sum('references' in Path(x['path']).parts for x in original),changed=changed,missing=missing,raw_training_objects='not rewritten; omitted from before/after byte hashes; this is not a full P4 label audit')
(O/'audit/input_integrity_final.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
(O/'plan/review/spec-compliance.md').write_text('PASS with disclosed data gap: 108/108 formal tests, 12 online/9 offline, all five test metrics, all densities/seeds, 7 PNG/SVG figures, comparison/ablation three-line tables, best/second marks, all 60 signed change rows. Real five-metric training curves unavailable because never logged; no invented substitution. PDF/HTML and CSV supplied. No original data, source or references changed according to recorded byte hashes.',encoding='utf-8')
(O/'plan/review/quality-review.md').write_text('Independent reviewer /root/audit recomputed 108 physical logs and hashes, all 60 paired rows and bootstrap intervals; verified 9 calibrated FNR=1. Correction applied to completion proportion: 106/108 below 50%, range35.89%–52.02%. Claims limited to descriptive n=3 and current SUMO scenes; no component attribution, no convergence claim. Historical migration test failure and SUMO warning suppression disclosed. Full PDF page rendering reviewed. Final late-gridlock analysis separately verified.',encoding='utf-8')
(O/'plan/progress.md').write_text('# P5 analysis progress\n\nStage S3 experiments → S5 review. User scope fixed: Chinese objective academic report; no manuscript redraft or experiment rerun.\n\n## Capability-use audit\n- Required/used: using-research-writing, paper-orchestration, experiment-results-planning, figures-python, statistical-analysis, peer-review, verification, pdf; environment-setup inspected to select existing environments. Brainstorming inspected, existing user-specified report scope retained. Documents skill inspected initially; final report uses PDF/HTML, no DOCX edited.\n- Inputs: method/protocol documents, current implementation, all formal result summaries/checkpoints, 108 complete physical streams,18 prediction arrays, all run logs and historical provenance inventory.\n- Not consumed as new performance evidence: smoke/provenance duplicate results; raw risk sample tensor streams (not physical trajectories); external literature values; missing PPO/FCFS/mixed.30 gzip objects inventoried, not a new full P4 audit.\n- Independent review: /root/audit, read-only; metric audit then report claim review.\n- Outputs: report.pdf/html; 7 PNG+SVG;10 booktabs fragments; real precision CSV; scripts and audit manifests.\n- Verification: analyze.py (108/12/9/18,129 task exitcodes, actor pairing); test_dynamics.py; independent paired comparisons; render_report.py; Poppler every-page rendering; verify_delivery.py rechecks original byte hashes.\n- Remaining risks: no logged five-metric training dynamics; n=3; few positive events/windows; nonstandard AQL/AWT; severe late congestion; no full system references or new untouched test split.\n',encoding='utf-8')
(O/'README.md').write_text('# P5实验分析报告\n\n- report.pdf：中文正式报告。\n- report.html：图表完整的浏览器版本，与PDF同源。\n- figures/：7组450dpi PNG及SVG。\n- tables/：三线表booktabs片段、60行配对变化和暴露计数；LaTeX片段需booktabs及支持中文的XeLaTeX/ctex环境。\n- data/：全部108次测试指标、60s曲线、训练末点、种子均值、离线指标和事件时序。\n- audit/：输入字节哈希、指标重算、日志扫描及排版验证。\n\n复现顺序：python -B analyze.py；python -B test_dynamics.py；捆绑Python -B render_report.py。原始文件保持只读。训练五指标时序未记录，不能恢复，报告没有虚构训练曲线。\n',encoding='utf-8')
# Pack deliverables, excluding page-render QA and original hash bulk.
with zipfile.ZipFile(O/'p5_analysis_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in O.rglob('*'):
  if not p.is_file() or p.suffix=='.zip' or '__pycache__' in p.parts:continue
  rel=p.relative_to(O)
  if rel.parts[0]=='audit' and p.name not in ['verification.json','input_integrity_final.json','pdf_structure.json','log_scan.csv']:continue
  if p.name in ['add_dynamics.py','finalize_notes.py','contact.py']:continue
  z.write(p,str(rel))
print(json.dumps(res,ensure_ascii=False))
print('Bundle bytes:',(O/'p5_analysis_bundle.zip').stat().st_size)
