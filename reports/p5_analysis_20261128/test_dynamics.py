from pathlib import Path
import pandas as pd,json
p=Path(__file__).resolve().parent
c=pd.read_csv(p/'data/test_time_curves.csv')
t=c[c.time==3600].groupby(['density','variant'])[['aql','awt','mean_speed','arrived']].mean()
print(t.to_string())
events=[]
for f in (p.parents[1]/'plan/p5/runs/20261123T094822803672Z').glob('closed-loop/*/*/*/*/physical.jsonl'):
 v,s,d,e=f.parts[-5:-1];active=set();early=late=0
 with f.open() as src:
  for line in src:
   x=json.loads(line);pairs={tuple(sorted(z)) for z in x['pairs']};n=len(pairs-active);active=pairs
   if x['step']<=1800:early+=n
   else:late+=n
 events.append(dict(variant=v,seed=int(s),density=d,evaluation_seed=int(e),first_half_events=early,last_half_events=late))
pd.DataFrame(events).to_csv(p/'data/test_event_timing.csv',index=False)
t.to_csv(p/'tables/test_final_60s.csv')
print(pd.DataFrame(events).groupby('variant')[['first_half_events','last_half_events']].sum().to_string())
