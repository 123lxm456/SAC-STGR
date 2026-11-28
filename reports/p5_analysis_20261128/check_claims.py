from pathlib import Path
import pandas as pd
p=Path(__file__).resolve().parent
d=pd.read_csv(p/'data/test_runs.csv');f=pd.read_csv(p/'data/offline_metrics.csv');c=pd.read_csv(p/'tables/paired_changes.csv')
print('arrived/departed min max', (d.arrived/d.departed).min(), (d.arrived/d.departed).max())
print(f.groupby(['variant','kind'])[['auprc_average_precision','brier','false_negative_rate']].mean().to_string())
print(c[c.density=='equal_density'].to_string(index=False))
print(pd.read_csv(p/'audit/log_scan.csv')[['Traceback','Error:','Warning:','teleport','collision']].sum())
