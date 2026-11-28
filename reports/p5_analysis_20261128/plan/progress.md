# P5 analysis progress

Stage S3 experiments → S5 review. User scope fixed: Chinese objective academic report; no manuscript redraft or experiment rerun.

## Capability-use audit
- Required/used: using-research-writing, paper-orchestration, experiment-results-planning, figures-python, statistical-analysis, peer-review, verification, pdf; environment-setup inspected to select existing environments. Brainstorming inspected, existing user-specified report scope retained. Documents skill inspected initially; final report uses PDF/HTML, no DOCX edited.
- Inputs: method/protocol documents, current implementation, all formal result summaries/checkpoints, 108 complete physical streams,18 prediction arrays, all run logs and historical provenance inventory.
- Not consumed as new performance evidence: smoke/provenance duplicate results; raw risk sample tensor streams (not physical trajectories); external literature values; missing PPO/FCFS/mixed.30 gzip objects inventoried, not a new full P4 audit.
- Independent review: /root/audit, read-only; metric audit then report claim review.
- Outputs: report.pdf/html; 7 PNG+SVG;10 booktabs fragments; real precision CSV; scripts and audit manifests.
- Verification: analyze.py (108/12/9/18,129 task exitcodes, actor pairing); test_dynamics.py; independent paired comparisons; render_report.py; Poppler every-page rendering; verify_delivery.py rechecks original byte hashes.
- Remaining risks: no logged five-metric training dynamics; n=3; few positive events/windows; nonstandard AQL/AWT; severe late congestion; no full system references or new untouched test split.
