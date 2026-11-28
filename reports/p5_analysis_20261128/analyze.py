from pathlib import Path
import json,sys,hashlib,itertools,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score,roc_auc_score
O=Path(__file__).resolve().parent; R=O.parents[1]; U=R/'plan/p5/runs/20261123T094822803672Z'
sys.path.insert(0,str(R/'src/sac_rwb'))
from algos.p5_metrics import PhysicalMetrics
for x in ['figures','tables','data','audit','plan/task-packets','plan/review']:(O/x).mkdir(parents=True,exist_ok=True)
V=['b0','m1','a_b','s0']; L=dict(zip(V,['B0','M1','A-B','S0'])); D=['low','nominal','high']
M=['collision_events_per_1000_departed','aql','awt','mean_speed','arrived']
CN=['碰撞/千出发 ↓','AQL ↓','AWT(s) ↓','速度(m/s) ↑','完成车辆 ↑']
EN=['Collision events / 1000 departures','AQL (veh / group)','AWT (s)','Mean speed (m/s)','Completed vehicles']
C=['#0077BB','#CC3311','#009988','#EE7733']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
# Manifest only pre-existing inputs; all writes confined to O.
manifest=[]; consumed=[]
paths=[R/'README.md',R/'research_implementation_plan.md']+list((R/'src').rglob('*.py'))+list((R/'scripts').glob('*.py'))+list((R/'chapters').glob('*.md'))+list((R/'plan').glob('*.md'))+list((R/'plan/p5').glob('*.md'))+[R/'plan/review/method-experiment-traceability.md']
for p in paths:
 text=p.read_text(encoding='utf-8',errors='replace');consumed.append(dict(path=str(p.relative_to(R)),characters=len(text),sha256=sha(p)))
dump(O/'audit/consumed_text.json',consumed)
for p in R.rglob('*'):
 if not p.is_file() or O in p.parents or any(t in p.parts for t in ['.git','.local-test-deps','NUL','__pycache__']):continue
 if 'objects' in p.parts:continue
 manifest.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p)))
dump(O/'audit/original_hashes.json',manifest)
a=[];curves=[];checks=[]
for p in sorted(U.glob('closed-loop/*/*/*/*/result.json')):
 v,s,d,e=p.relative_to(U/'closed-loop').parts[:4];s=int(s);e=int(e);q=json.loads(p.read_text())
 assert v in V and s in [11,23,37] and d in D and e in [3000,3001,3002]
 pm=PhysicalMetrics();block=[]
 with p.with_name('physical.jsonl').open() as f:
  for line in f:
   x=json.loads(line);pm.add(x);block.append(x)
   if x['step']%60==0:
    groups=sum(z['groups'] for z in block);vehicles=sum(z['vehicles'] for z in block)
    curves.append(dict(variant=v,seed=s,density=d,evaluation_seed=e,time=x['step'],collision_events_per_1000_departed=1000*pm.pair_events/len(pm.departed) if pm.departed else np.nan,aql=sum(z['queue_sum'] for z in block)/groups,awt=sum(z['wait_sum'] for z in block)/groups,mean_speed=sum(z['speed_sum'] for z in block)/vehicles if vehicles else np.nan,arrived=len(pm.arrived)))
    block=[]
 assert pm.summary()==q['metrics'] and pm.steps==3600 and q['sac_updates']==q['risk_updates']==0
 assert sha(p.with_name('physical.jsonl'))==q['physical_sha256']
 train=json.loads((U/f'online/{v}/{s}/result.json').read_text())
 assert train['actor_sha256']==q['actor_sha256'] and train['risk_sha256']==q['risk_sha256']
 checks.append(str(p.relative_to(R)))
 a.append(dict(variant=v,seed=s,density=d,evaluation_seed=e,**pm.summary()))
assert len(a)==108
df=pd.DataFrame(a);cv=pd.DataFrame(curves)
df.to_csv(O/'data/test_runs.csv',index=False);cv.to_csv(O/'data/test_time_curves.csv',index=False)
sd=df.groupby(['variant','seed','density'])[M].mean().reset_index()
overall=sd.groupby(['variant','seed'])[M].mean().reset_index()
sd.to_csv(O/'data/seed_density_means.csv',index=False);overall.to_csv(O/'data/seed_equal_density_means.csv',index=False)
od=[];off=[];loss=[]
for p in U.glob('online/*/*/result.json'):
 q=json.loads(p.read_text());v=p.parts[-3];s=int(p.parts[-2])
 assert q['completed'] and q['decision_steps']==100000 and q['sac_updates']>0
 for k,f in [('actor_sha256','actor.pth'),('risk_sha256','risk.pth'),('initial_actor_sha256','actor-initial.pth')]:assert sha(p.with_name(f))==q[k]
 od.append(dict(variant=v,seed=s,decision_steps=q['decision_steps'],sac_updates=q['sac_updates'],risk_updates=q['risk_updates'],risk_attempts=q['risk_update_attempts'],initial_actor=q['initial_actor_sha256'],mean_batch_ms=q['inference']['mean_end_to_end_ms_per_batch']))
for s in [11,23,37]:assert len({x['initial_actor'] for x in od if x['seed']==s})==1
for p in U.glob('offline/*/*/result.json'):
 q=json.loads(p.read_text());v=p.parts[-3];s=int(p.parts[-2]);assert q['updates']==5000
 assert sha(p.with_name('risk.pth'))==q['checkpoint_sha256']
 for split in ['validation','test']:
  z=np.load(p.with_name(split+'-predictions.npz'));y=z['labels'];pr=z['scores'];t=q['splits'][split]['raw']
  assert len(y)==t['samples'] and y.sum()==t['positive_windows']
  assert np.isclose(average_precision_score(y,pr),t['auprc_average_precision']) and np.isclose(roc_auc_score(y,pr),t['auroc'])
  assert np.isclose(np.mean((pr-y)**2),t['brier']) and ((pr<.5)&(y==1)).sum()==t['false_negatives']
 for kind in ['raw','calibrated']:
  t=q['splits']['test'][kind]
  off.append(dict(variant=v,seed=s,kind=kind,**{k:t[k] for k in ['samples','positive_windows','auprc_average_precision','auroc','brier','ece','false_negative_rate','false_negatives']},parameters=q['engineering']['parameter_count'],batch1_p50_ms=q['engineering']['cases'][0]['p50_ms']))
 loss.append(dict(variant=v,seed=s,final_loss=q['final_loss']))
assert len(od)==12 and len(loss)==9
od=pd.DataFrame(od);off=pd.DataFrame(off);loss=pd.DataFrame(loss)
od.to_csv(O/'data/online_endpoints.csv',index=False);off.to_csv(O/'data/offline_metrics.csv',index=False);loss.to_csv(O/'data/offline_final_loss.csv',index=False)
logs=[]
for p in U.rglob('*.log'):
 text=p.read_text(encoding='utf-8',errors='replace');logs.append(dict(path=str(p.relative_to(R)),**{k:len(re.findall(k,text,re.I)) for k in ['Traceback','Error:','Warning:','teleport','collision']}))
pd.DataFrame(logs).to_csv(O/'audit/log_scan.csv',index=False)
exitcodes=[]
for p in U.glob('*.exitcode'):
 if p.name.startswith(('eval-','train-','offline-')):
  assert p.read_text().strip()=='0',p
  exitcodes.append(p.name)
changes=[]
for d in D+['equal_density']:
 src=sd[sd.density==d] if d!='equal_density' else overall
 for ref in ['b0','a_b','s0']:
  for m in M:
   x=src[src.variant=='m1'].set_index('seed')[m].sort_index().values;y=src[src.variant==ref].set_index('seed')[m].sort_index().values
   delta=x-y;boot=[np.mean(delta[list(i)]) for i in itertools.product(range(3),repeat=3)]
   changes.append(dict(density=d,reference=ref,metric=m,m1_mean=x.mean(),reference_mean=y.mean(),absolute_delta=delta.mean(),relative_change_percent=100*(x.mean()-y.mean())/y.mean() if y.mean() else np.nan,ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),seed_deltas=json.dumps(delta.tolist())))
changes=pd.DataFrame(changes);changes.to_csv(O/'tables/paired_changes.csv',index=False)
ex=df.groupby(['density','variant'])[['collision_pair_events','collision_vehicles','departed','arrived','successful_arrivals','unfinished_departed_vehicles']].sum().reset_index();ex.to_csv(O/'tables/exposure_counts.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'legend.frameon':False,'svg.fonttype':'none'})
def save(f,n):
 f.savefig(O/f'figures/{n}.png',dpi=450,bbox_inches='tight');f.savefig(O/f'figures/{n}.svg',bbox_inches='tight');plt.close(f)
for d in D:
 f,axs=plt.subplots(3,2,figsize=(7.3,8.1));axs=axs.ravel()
 for i,m in enumerate(M):
  for j,v in enumerate(V):
   t=cv[(cv.density==d)&(cv.variant==v)].groupby(['seed','time'])[m].mean().unstack('seed');mu=t.mean(axis=1);st=t.std(axis=1,ddof=1)
   axs[i].plot(t.index,mu,c=C[j],label=L[v],lw=1.2);axs[i].fill_between(t.index,mu-st,mu+st,color=C[j],alpha=.10,lw=0)
  axs[i].set_title('('+chr(97+i)+') '+EN[i],fontsize=9);axs[i].set_xlabel('Simulation time (s)');axs[i].set_xlim(0,3600);axs[i].set_ylim(bottom=0)
 axs[5].axis('off');axs[5].legend(*axs[0].get_legend_handles_labels(),loc='upper left',ncol=2)
 axs[5].text(0,.55,'Final frozen policies; no learning.\nAQL/AWT/speed: 60 s blocks.\nCollisions: cumulative rate.\nCompletions: cumulative count.\nBand: SD across 3 training seeds\nafter averaging 3 test seeds.',va='top')
 f.suptitle('Test trajectories: '+d+' demand',fontsize=12);f.tight_layout(rect=[0,0,1,.97]);save(f,'fig_test_'+d)
stats=sd.groupby(['density','variant'])[M].agg(['mean','std']);stats.to_csv(O/'tables/density_mean_sd.csv')
f,axs=plt.subplots(3,2,figsize=(7.3,8));axs=axs.ravel()
for i,m in enumerate(M):
 for j,v in enumerate(V):
  axs[i].errorbar(np.arange(3)+(j-1.5)*.045,[stats.loc[(d,v),(m,'mean')] for d in D],yerr=[stats.loc[(d,v),(m,'std')] for d in D],color=C[j],marker=['o','s','^','D'][j],capsize=2,label=L[v],lw=1)
 axs[i].set_xticks(range(3),['Low','Nominal','High']);axs[i].set_title(EN[i],fontsize=9);axs[i].set_ylim(bottom=0)
axs[5].axis('off');axs[5].legend(*axs[0].get_legend_handles_labels(),loc='upper left');axs[5].text(0,.4,'Mean +/- sample SD (n=3)\nTest repetitions averaged first.\nLines connect demand levels;\nthese are not learning curves.',va='top');f.tight_layout();save(f,'fig_density_response')
f,axs=plt.subplots(1,3,figsize=(7.3,2.8))
for k,d in enumerate(D):
 for j,v in enumerate(V):
  t=sd[(sd.density==d)&(sd.variant==v)];axs[k].scatter(t[M[0]],t.arrived,s=22,color=C[j],marker=['o','s','^','D'][j],alpha=.6)
  axs[k].scatter(t[M[0]].mean(),t.arrived.mean(),s=70,color=C[j],marker=['o','s','^','D'][j],edgecolors='black',linewidths=.5,label=L[v])
 axs[k].set_title(d);axs[k].set_xlabel('Events / 1000 departures');axs[k].set_ylabel('Completed vehicles')
axs[2].legend(fontsize=7);f.tight_layout();save(f,'fig_safety_efficiency')
f,axs=plt.subplots(1,2,figsize=(7.3,2.8))
for j,v in enumerate(V[:3]):
 axs[0].scatter([j]*3,loss[loss.variant==v].final_loss,c=C[j],s=25)
 axs[1].scatter([j]*3,od[od.variant==v].risk_updates,c=C[j],s=25)
for ax in axs:ax.set_xticks(range(3),[L[v] for v in V[:3]])
axs[0].set_ylabel('Final mini-batch BCE (5000 updates)');axs[1].set_ylabel('Actual online risk updates')
f.suptitle('Recorded training endpoints only - learning trajectories unavailable',fontsize=10);f.tight_layout();save(f,'fig_training_endpoints')
f,axs=plt.subplots(3,2,figsize=(7.3,8));axs=axs.ravel()
for i,m in enumerate(M):
 for j,v in enumerate(V):
  t=overall[overall.variant==v].sort_values('seed');axs[i].scatter([j-.1,j,j+.1],t[m],c=C[j],s=28);axs[i].plot([j-.22,j+.22],[t[m].mean()]*2,c=C[j],lw=2)
 axs[i].set_xticks(range(4),[L[v] for v in V]);axs[i].set_title(EN[i],fontsize=9);axs[i].set_ylim(bottom=0)
axs[5].axis('off');axs[5].text(0,.9,'Each dot is one training seed.\nAll densities and test seeds\nare equally averaged.\nHorizontal bar: seed mean.',va='top');f.tight_layout();save(f,'fig_all_seed_results')
dump(O/'audit/verification.json',dict(closed_loop_verified=108,online_verified=12,offline_verified=9,prediction_arrays_recomputed=18,initial_actors_paired=True,exitcodes_verified=len(exitcodes),physical_curves_training_available=False,object_files_present=len(list((U/'data/objects').glob('*')))))
dump(O/'data/analysis_summary.json',dict(overall=overall.groupby('variant')[M].mean().to_dict('index'),offline=off.groupby(['variant','kind'])[['auprc_average_precision','auroc','brier','false_negative_rate']].mean().reset_index().to_dict('records')))
print('Verified 108 physical logs, 12 online checkpoints, 9 offline checkpoints, 18 prediction arrays.')
print(overall.groupby('variant')[M].mean().to_string())
