from pathlib import Path
import json,re,html,sys
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_JUSTIFY
from pypdf import PdfReader
O=Path(__file__).resolve().parent
V=['b0','m1','a_b','s0'];L=dict(zip(V,['B0','M1','A-B','S0']));D=['low','nominal','high']
M=['collision_events_per_1000_departed','aql','awt','mean_speed','arrived']
CN=['物理碰撞/千出发 ↓','AQL(辆/组) ↓','AWT(s) ↓','速度(m/s) ↑','完成车辆(辆) ↑']
sd=pd.read_csv(O/'data/seed_density_means.csv');overall=pd.read_csv(O/'data/seed_equal_density_means.csv');ch=pd.read_csv(O/'tables/paired_changes.csv');off=pd.read_csv(O/'data/offline_metrics.csv');runs=pd.read_csv(O/'data/test_runs.csv');online=pd.read_csv(O/'data/online_endpoints.csv')
S=[]
def add(title,*items):S.append(dict(title=title,items=list(items)))
def tab(caption,headers,rows):return dict(type='table',caption=caption,headers=headers,rows=rows)
def fig(name,caption):return dict(type='figure',name=name,caption=caption)
def tbl(src,vs,caption):
 st=src.groupby('variant')[M].agg(['mean','std']);rows=[]
 for v in vs:
  cells=[L[v]]
  for i,m in enumerate(M):
   means=st.loc[vs,(m,'mean')];order=sorted(set(round(x,12) for x in means),reverse=i>=3);a=st.loc[v,(m,'mean')];b=st.loc[v,(m,'std')]
   txt=f'{a:.3f} ± {b:.3f}' if i<4 else f'{a:.1f} ± {b:.1f}'
   if abs(a-order[0])<1e-10:txt='<b>'+txt+'</b>'
   elif len(order)>1 and abs(a-order[1])<1e-10:txt='<u>'+txt+'</u>'
   cells.append(txt)
  rows.append(cells)
 return tab(caption,['方案']+CN,rows)
def dt(ref,d='equal_density'):
 t=ch[(ch.reference==ref)&(ch.density==d)].set_index('metric')
 return '；'.join(f'{CN[i].split("(")[0].replace(" ↓","").replace(" ↑","")} {t.loc[m,"relative_change_percent"]:+.2f}%' for i,m in enumerate(M))
add('摘要',
'对 P5 正式四组实验的12次在线训练、9次离线训练和108次冻结闭环测试进行全量分析。所有正式测试物理日志逐行重算，结果与保存摘要及SHA-256完全一致。M1相对B0的物理碰撞事件率在三种密度均更高；排队、等待和通行指标的部分改善伴随安全代价，不能支持“降低碰撞同时保持效率”或“具有稳定综合优势”的结论。近期偏置消融效果随密度和种子变化，不能确认其普遍必要性。',
'训练阶段没有保存五项物理指标时序，也没有完整损失轨迹或中间检查点。真实五指标训练曲线、收敛速度及训练动态稳定性无法追溯。本报告保留这一缺口，以全部真实测试曲线及训练末点描述能够观察到的结果，不以测试轨迹代替训练曲线。')
add('1 数据范围与方法',
'项目中实际目录为 plan/p5/runs/20261123T094822803672Z；用户所述 plan/run/p5 不存在。统计覆盖正式 online、offline 和 closed-loop 中的全部有效四组。provenance 下3份历史smoke物理日志、旧运行复用凭据以及已退役结构组不作为新增重复。旧summary中的历史任务名称不用于推断当前有效矩阵。',
'M1采用自车与八组聚合车流共九个token，先空间Transformer，再沿各组的历史进行共享时间Transformer编码，最后自车条件读出风险。B0为共同修复环境及标签后的原单序列时间风险器；A-B仅将M1的时间近期偏置置零；S0取消风险分数和风险奖励，保留原其他奖励与死锁干预。因此S0不等于完全没有安全机制，B0也不是仅移除一个空间层的严格单模块消融。',
'训练种子11、23、37各自从头训练100,000决策步；相同种子的初始actor哈希在四组间一致。风险器另在共同P4语料上离线训练5,000次更新，batch=128、学习率与weight decay均为1e-5。离线和在线风险器分别从头训练，离线识别表现不直接对应闭环策略收益。均使用固定预算末点权重，不按测试表现选优。',
'每个训练种子在low/nominal/high三档需求及3000/3001/3002三测试种子下评价，合计108次。每次3,600仿真步，步长1s，包含reset和等待观测的空闲时间；SAC及风险器评价期更新均为零。密度仅将测试路线flow的vehsPerHour缩放为0.5/1/1.5，显式背景车辆不变，不能视为全网实际出发量同比例变化。',
'目标实验环境为Linux、RTX 4090、Python 3.7.12、PyTorch 1.13.1+cu117和SUMO 1.12.0。本次只进行本地统计绘图，不训练、不运行仿真。PPO、FCFS、mixed的共同场景结果仍缺失，完整P5研究验收未通过；其他论文非共同场景的数字不能填补该缺口。')
add('2 指标定义与统计方法',
'物理碰撞率=1,000×车辆对碰撞事件起点数/唯一出发车辆数，单位为事件/千出发。连续重复报告的同一车辆对去重，消失后再出现记新事件；不是百分比，不是唯一碰撞车辆占比，也不是旧conflict_rate。附录同时报告真实事件数、碰撞车辆数及出发分母。',
'AQL=各步各组区域车辆数总和/组时间样本数。四路口×八车流组，共32组。Env按距路口≤30m收集所有进口车辆，没有停车速度阈值，故项目AQL实为区域占用车辆数，不能等同标准停车队列长度。AWT先求每组内车辆已有路口等待值均数，再对组和时间等权平均；空组取零。该值不是完成车辆的平均全程延误，也不是全部车辆等权平均等待。',
'平均速度=全网各步车辆速度总和/车辆时间样本数，单位m/s；不能先求各步平均速度再简单平均。完成车辆数取唯一到达数arrived；另附successful_arrivals即未发生碰撞的到达数。截止3,600步仍未到达的行程被截尾，完成数增加不能自动说明所有行程用时均下降。',
'每指标先对同一训练种子、同一密度的三个测试种子求平均，再报告三个训练种子的均值与样本标准差。跨密度总体按三档等权平均。表中粗体最优、下划线次优，排序仅代表表内均值，不代表显著性。变化幅度统一为100×(M1−参照)/参照，前三指标负值为改善，后两指标正值为改善。',
'配对比较以训练种子为单元，n=3。枚举全部27种有放回重采样的配对差值均值，给出探索性95% percentile bootstrap区间。区间对应绝对差，不能视为大样本推断或非劣效证据。不把108次测试、相邻时间步或风险窗口当成独立训练重复，也不通过多个检验挑选显著结果。')
add('3 对比实验：安全与通行效率',
tbl(overall,['b0','m1','s0'],'表1 对比实验，三密度等权；均值±训练种子间样本标准差，n=3。'),
'M1相对B0：'+dt('b0')+'。碰撞率上升约70.7%，同时AQL下降约4.7%、AWT下降约12.4%、速度提高约3.9%、完成数提高约3.2%。这是安全与效率权衡，不能解释为双重提升。',
'M1相对S0：'+dt('s0')+'。跨密度平均并未体现更低碰撞事件率，排队、等待和完成数反而较差；平均速度仅略高。风险分支的存在和自然更新完成不等于闭环安全收益得到验证。',
fig('fig_density_response','图1 三密度五指标测试响应。均值±样本标准差，n=3训练种子；三个测试种子已先平均。连接需求档位的线不是学习曲线。'))
for i,d in enumerate(D,2):
 add(f'3.{i-1} {d}需求分层',
 tbl(sd[sd.density==d],V,f'表{i} {d}需求下全部四组，均值±样本标准差。'),
 'M1相对B0：'+dt('b0',d)+'。M1相对A-B：'+dt('a_b',d)+'。M1相对S0：'+dt('s0',d)+'。',
 fig('fig_test_'+d,f'图{i} {d}测试时域曲线，全部九次测试纳入。阴影为三个训练种子的样本标准差；每种子先平均三个测试流。碰撞为累计事件率、完成数为累计到达，AQL/AWT/速度按不重叠60s块汇总，无额外平滑。累计曲线起步瞬态不能解读为训练收敛；无车辆暴露时速度缺失保留为空，不填零或插值。'))
tail=pd.read_csv(O/'tables/test_final_60s.csv')
tailrows=[]
for d in D:
 for v in V:
  r=tail[(tail.density==d)&(tail.variant==v)].iloc[0]
  tailrows.append([d,L[v],f'{r.aql:.3f}',f'{r.awt:.1f}',f'{r.mean_speed:.3f}'])
add('3.4 测试时域暴露的持续拥堵',
tab('表4a 测试最后60s的真实分块均值；每方案每密度九次测试全部纳入。',['需求','方案','末段AQL','末段AWT(s)','末段速度(m/s)'],tailrows),
'各方法从测试初期较高速度逐步下降，到最后60s的各组均值仅约0.09至0.16m/s；末段AWT约495至655s，AQL末段高于全时段均值，累计到达数增长趋缓。M1 nominal末段速度仅0.099m/s、AWT约637.1s，说明当前策略没有维持稳定畅通。时间平均速度约1.1至1.4m/s掩盖了更严重的末段退化，不能单以全时段均值宣称持续通行能力。',
'事件时序重算显示：M1的174起碰撞中167起发生于前1800s，B0为98/98，A-B为151/155，S0为176/181。后半程事故较少与低速滞留同时出现，但这些记录不能单独证明因果。累计碰撞率随时间下降也受累计出发分母扩大影响，不是学习到更安全策略的证据。需要同时控制车辆暴露、滞留与完成情况，才能区分真实风险降低和缺少运动机会。',
'三密度时域曲线未显示持续畅通的稳态运行，当前实验尤其不能支持拥堵环境下已具备稳定运行能力。建议增加路口分项与未插入需求统计，并检查死锁干预及长期饱和机制；这些是后续验证方向，本报告不改写源代码或现有实验。')
add('4 消融实验与创新点证据',
tbl(overall,['m1','a_b','s0'],'表5 消融实验，三密度等权均值±样本标准差；粗体和下划线只在表内三组排序。'),
'M1相对A-B：'+dt('a_b')+'。总体上，去除近期偏置的A-B拥有更低碰撞率、更低AQL和AWT、更高速度及更多完成车辆。nominal档A-B五项均值均优于M1。该均值层面的全面优势不等同经过充分功效验证的统计支配，但已足以否定“偏置在全部条件都有效”的描述。',
'low档M1相对A-B的等待、速度和完成数较好，但碰撞率更高且AQL更差；high档M1碰撞率较低，而速度、等待和完成数较差，AQL也略高。偏置作用具有明显场景依赖，没有一致证据证明其必要性。',
'M1与B0同时改变token嵌入、表示宽度、空间及逐组时间组织和末步读出，只能比较整体架构。M1与S0检验整个风险反馈分支；独立空间层、逐组历史保持及读出模块的消融已经退出当前协议，现有数据不能分别支持这些结构创新点。若论文将创新限定为聚合车流观测下的整体适配，当前结果可作为其效果评价，但并未证明安全改善。',
fig('fig_safety_efficiency','图5 安全效率联合分布。小点为各训练种子的测试均值，黑边大点为三种子平均；期望方向为左上。所有需求档位都保留。'),
fig('fig_all_seed_results','图6 全部训练种子的跨密度等权测试结果。每点为一训练种子，横线为均值；不隐藏较差种子。'))
rows=[]
for _,r in ch[ch.density=='equal_density'].iterrows():rows.append([L[r.reference],CN[M.index(r.metric)],f'{r.absolute_delta:+.4f}',f'{r.relative_change_percent:+.2f}%',f'[{r.ci_low:+.4f}, {r.ci_high:+.4f}]'])
add('5 性能变化与重复性',
tab('表6 M1相对各参照的跨密度等权变化及探索性配对区间。',['参照','指标','绝对变化','相对变化','配对95%区间'],rows),
'各密度的全部配对差值、区间和三个种子原值保存在tables/paired_changes.csv。n=3的bootstrap分布只有有限支持点，区间可能较窄或退化，不应据此作“显著优于”“达到等效”或“风险为零”的推断。',
'种子间误差棒和图6显示最终策略的重复性存在波动，但它不是训练迭代稳定性。M1相对B0的总体五指标配对区间均跨零，三个种子的差值还出现方向反转，平均改善或恶化都不能被表述为稳定、已确认的普遍效应。不同需求档位的排序也会变化。增加测试流可减少场景噪声，却不能代替独立训练重复；应预先锁定安全容忍范围、效率非劣效界值及样本量，再做验证。')
add('6 训练日志缺口、收敛速度与稳定性',
'train()未创建或注入PhysicalObserver，因此没有逐步或逐episode的五项物理指标；训练result仅记录预算、优化次数、权重、样本池和推理统计。DataMonitor虽在内存中收集部分量，但worker不调用持久化接口，Env.close只关闭SUMO，正式目录没有相应pkl文件。在线样本/事件流服务标签审计，不包含恢复全网速度、32组队列和到达暴露所需的完整分母。',
'fit_offline()将每次损失追加到内存losses，但最终只写final_loss，没有完整损失轨迹。正式运行也没有中间checkpoint冻结测试。因而当前不能绘制真实五指标训练曲线，不能判定M1收敛更快、训练更稳定或训练末段已经收敛。补画平滑曲线或将测试时间误作训练步都会引入不存在的证据。',
fig('fig_training_endpoints','图7 实际保存的训练末点：最终mini-batch BCE和自然风险更新次数，每点为一个训练种子。没有用连接线将末点伪装成学习轨迹。'),
'四组各训练100,000决策步，SAC实际优化均为9,990次。M1自然风险更新5724/5772/5739次，B0为5802/5730/5742次，A-B为5790/5775/5703次，尝试均5994次，S0为零。不同策略产生的自然事件暴露不同，即使触发预算相同，有效风险优化数也不完全一致。这一差异需要披露，但不能仅凭更新数直接解释安全表现。',
'建议后续新运行统一记录每仿真步物理量、决策步与时间映射、每episode的总奖励与各奖励项、SAC及风险损失、梯度和自然风险池暴露，并按预定间隔保存冻结checkpoint评价。训练探索曲线与确定性策略测试曲线分开标注，不回填现有运行。')
r=[]
for v in V[:3]:
 a=off[(off.variant==v)&(off.kind=='raw')];b=off[(off.variant==v)&(off.kind=='calibrated')]
 r.append([L[v]]+[f'{a[k].mean():.5f} ± {a[k].std():.5f}' for k in ['auprc_average_precision','auroc','brier','false_negative_rate']]+[f'{b.brier.mean():.6f}'])
add('7 离线风险识别与计算开销补充',
tab('表7 共同离线测试集，均值±训练种子间标准差；校准Brier列为均值。',['风险器','AP','AUROC','原始Brier','原始FNR','校准Brier'],r),
'三种风险器共用33,594个测试窗口，仅24个正窗口，正例率约0.000714。24个正窗口不等同24起独立事故，相邻窗口相关，风险识别证据支持量有限。AP的随机排序基准约为该正例率；较高AUROC不能抵消较低AP或较高漏报。',
'校准只在validation拟合，未用于在线风险奖励。校准后的0.5阈值在九个测试结果中FNR均为1；Brier较低不能单独证明风险可用于可靠报警。常数零预测在这个稀疏测试集上的Brier约为24/33594=0.000714，需要与该基率参照一起判断校准收益。原始分数不应直接解释为已校准碰撞概率。',
'参数量、batch=1同步前向p50、在线平均批推理时延和真实优化次数保存在data/offline_metrics.csv及data/online_endpoints.csv。基准前向使用合成输入，不含历史组装、传输和环境交互；在线摊销每车耗时也不等于单车延迟。计算成本需与实测闭环收益一同评价，不从注意力矩阵数量推断实时优势。')
cost=[]
for v in V[:3]:
 a=off[(off.variant==v)&(off.kind=='raw')];b=online[online.variant==v]
 cost.append([L[v],str(int(a.parameters.iloc[0])),f'{a.batch1_p50_ms.mean():.3f}',f'{b.mean_batch_ms.mean():.3f}',f'{b.risk_updates.mean():.1f}'])
add('7.1 开销记录',
tab('表8 风险器计算记录；时延为同类记录的训练种子均值，单位ms。',['风险器','参数量','batch1前向p50','在线批均时延','有效风险更新均值'],cost),
'时延的输入和批规模不同，列间不作直接速度倍数比较；batch1前向用于同一设备的模型比较，在线批均时延受实际车辆数与交通轨迹影响。')
add('8 异常与证据边界',
'108次正式测试均完成固定3,600步，评价期更新为零，所有物理摘要与日志一致，没有因性能较差剔除运行。早期smoke与复用来源未重复计入。日志中的错误、警告、teleport和collision字符串扫描结果另存audit/log_scan.csv。唯一Traceback来自迁移历史中的一次存储回归失败（test_snapshot_detects_source_replacement_before_deletion），其后迁移复验日志记录133项通过；不是正式策略评价失败。Error字符串也包括预期异常测试，teleport字符串主要命中启动参数time-to-teleport=-1，不能当作实际传送事件。启动参数关闭SUMO警告，因此Warning计数为零不证明没有异常。',
'策略会影响道路阻塞及能实际插入路网的出发数，因此碰撞率必须配合绝对事件数与暴露分母解释。AQL较小也可能来自较少车辆进入30m区域，而不代表全路网所有等待减少。平均速度较低及大量未完成车辆显示通行仍受固定时窗和拥堵制约。106/108次正式测试的到达/出发比例低于50%，全部运行范围为35.89%至52.02%，不能忽视行程截尾。',
'三档需求结果并非严格单调，且实际出发数未按0.5/1/1.5成比例变化。部分flow缩放、背景车、插入约束、策略随机性及固定时窗均可能影响趋势；这些是合理候选解释，现有日志并未分别识别其因果贡献。不能把高密度下某些指标变好自动称为鲁棒性。',
'当前test在P4阶段曾用于工程诊断，不能称为从未接触的最终留出集。三训练种子、少量风险正例和缺失系统参照共同限制外推范围。完整P5研究验收缺失并不否定主矩阵的记录完整性，但不允许据四组结果宣称优于所有系统方案。')
add('9 建议与结论',
'优先补充更多独立训练种子、此前未使用的新路线与需求场景，并统一记录训练物理日志。拟调节风险奖励系数、近期偏置、网络宽度或安全约束时，应先使用验证场景选参，再在新的留出测试上一次性评价；不回看测试后选取最有利种子或时段。',
'补充加载但未插入车辆、路网边界外等待、逐路口分项、完整行程延误、无碰撞完成数与更长测试时段，同时保留事件起点、车辆占比和原始分母。补齐具有原方法身份、配置和接口审查的PPO、FCFS、mixed系统参照。若需要逐模块创新结论，另立新协议做单因素、尽量容量与计算预算匹配的消融；如保持四组范围，则收缩到整体架构及其安全效率权衡。',
'结论是：当前实验未证实主方案能够相对B0降低物理碰撞率，也未证实M1在安全和通行效率上具备跨场景稳定综合优势。近期偏置的效果不一致，单独结构模块贡献证据不足。可以报告整体架构的真实效率变化和安全代价；不能宣称已实现安全效率双赢、训练收敛更快或训练更稳定。')
ex=pd.read_csv(O/'tables/exposure_counts.csv')
er=[]
for d in D:
 for v in V:
  t=ex[(ex.density==d)&(ex.variant==v)].iloc[0]
  er.append([d,L[v]]+[str(int(t[k])) for k in ['collision_pair_events','departed','arrived','successful_arrivals','unfinished_departed_vehicles']])
add('附录A 真实事件与暴露',
tab('表A1 每密度每方案九次测试计数求和。不能将合计比率替代正文运行宏平均。',['需求','方案','事件数','出发数','完成数','无碰撞完成','未完行程'],er),
'唯一碰撞车辆数、碰撞车辆比例、组时间及车辆时间样本数、全部单次运行指标见data/test_runs.csv。碰撞对事件总计：M1为174、B0为98、A-B为155、S0为181。原始事件数和按出发归一化后的均值排序应同时报告，避免单用分母变化造成误判。')
add('附录B 数据来源与复现',
'方案与边界依据：README.md、research_implementation_plan.md、chapters两份方法稿、plan/experiment-protocol.md、plan/p5/protocol.json、plan/p5/README.md及scope-cleanup-20261126.md。指标依据：src/sac_rwb/algos/p5_metrics.py、scripts/p5_worker.py、src/sac_rwb/Env.py与core/sumo_interface.py。',
'数据依据：正式run的online/*/*/result.json、offline/*/*/{result.json,validation-predictions.npz,test-predictions.npz}、closed-loop/*/*/*/*/{physical.jsonl,result.json}。本报告逐行重算108份物理流、复核21组训练权重及配对初值、重算18份预测数组的样本数/AP/AUROC/Brier/漏报，并扫描所有运行日志。原始源码及结果清单和哈希见audit目录。',
'30份原始训练压缩对象已盘点。在线风险样本/事件不是五指标物理日志，未据其重建不存在的训练曲线；本次没有重新执行P4/P5标签因果审计或目标GPU环境验收。早期验收结论仅作为历史来源，不冒充本次重新验证。',
'复现方式：在具有numpy、pandas、matplotlib、scikit-learn的环境运行analyze.py；再以捆绑Python运行render_report.py。全部图输出450dpi PNG和SVG；三线表同时提供HTML/PDF和booktabs LaTeX，原始精度保留于CSV。所有新增产物位于reports/p5_analysis_20261128，原始数据、项目源代码及references未被写入。')
# HTML and reusable table files.
css='body{font-family:"Microsoft YaHei","SimSun",sans-serif;max-width:1100px;margin:40px auto;line-height:1.85;color:#202833;padding:0 22px}h1{font-size:28px}h2{font-size:21px;margin-top:2em}p{text-align:justify}table{border-collapse:collapse;width:100%;font-size:12px;margin:22px 0;border-top:2px solid #222;border-bottom:2px solid #222}thead{border-bottom:1px solid #222}td,th{padding:8px 4px;text-align:center;border:0}figure{margin:28px 0;break-inside:avoid}img{max-width:100%}figcaption{font-size:13px;line-height:1.6;color:#434c59}u{text-underline-offset:3px}@media print{body{margin:0;font-size:10pt}h2{break-after:avoid}table{font-size:8pt}@page{size:A4;margin:18mm}}'
body='<h1>P5 实验结果分析报告</h1><p>空间与时间Transformer风险引导的无信号交叉口决策<br>统计日期：2026-11-28</p>'
table_index=0
for sec in S:
 body+='<h2>'+sec['title']+'</h2>'
 for t in sec['items']:
  if isinstance(t,str):body+='<p>'+t+'</p>';continue
  if t['type']=='figure':
   body+='<figure><img src="figures/'+t['name']+'.png"><figcaption>'+t['caption']+'</figcaption></figure>'
  else:
   table_index+=1
   body+='<table><thead><tr>'+''.join('<th>'+x+'</th>' for x in t['headers'])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+x+'</td>' for x in row)+'</tr>' for row in t['rows'])+'</tbody></table><p>'+t['caption']+'</p>'
   lines=[r'\begin{tabular}{'+'l'*len(t['headers'])+'}',r'\toprule',' & '.join(t['headers'])+r' \\',r'\midrule']
   for row in t['rows']:
    cells=[]
    for x in row:
     x=x.replace('%',r'\%').replace('±',r'$\pm$')
     x=re.sub(r'<b>(.*?)</b>',lambda m:r'\textbf{'+m[1]+'}',x);x=re.sub(r'<u>(.*?)</u>',lambda m:r'\underline{'+m[1]+'}',x)
     cells.append(x)
    lines.append(' & '.join(cells)+r' \\')
   lines += [r'\bottomrule',r'\end{tabular}']
   (O/f'tables/table_{table_index:02}.tex').write_text('\n'.join(lines),encoding='utf-8')
(O/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>P5实验结果分析报告</title><style>'+css+'</style><body>'+body+'</body></html>',encoding='utf-8')
(O/'data/report_sections.json').write_text(json.dumps(S,ensure_ascii=False,indent=2),encoding='utf-8')
# A4 PDF with embedded CJK TrueType; three-rule tables.
pdfmetrics.registerFont(TTFont('CJK','C:/Windows/Fonts/simsun.ttc',subfontIndex=0))
pdfmetrics.registerFont(TTFont('CJK-Bold','C:/Windows/Fonts/simhei.ttf'))
pdfmetrics.registerFontFamily('CJK',normal='CJK',bold='CJK-Bold',italic='CJK',boldItalic='CJK-Bold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCN',fontName='CJK',fontSize=10,leading=16,spaceAfter=9,wordWrap='CJK',alignment=TA_JUSTIFY))
styles.add(ParagraphStyle(name='HeadCN',fontName='CJK-Bold',fontSize=14,leading=21,spaceBefore=15,spaceAfter=10,keepWithNext=True,wordWrap='CJK'))
styles.add(ParagraphStyle(name='TitleCN',fontName='CJK-Bold',fontSize=23,leading=32,spaceAfter=16,wordWrap='CJK'))
styles.add(ParagraphStyle(name='CellCN',fontName='CJK',fontSize=7.3,leading=11,wordWrap='CJK',alignment=1))
styles.add(ParagraphStyle(name='CaptionCN',fontName='CJK',fontSize=8.6,leading=13,spaceAfter=13,wordWrap='CJK'))
story=[Paragraph('P5 实验结果分析报告',styles['TitleCN']),Paragraph('空间与时间Transformer风险引导的无信号交叉口决策<br/>统计日期：2026-11-28',styles['BodyCN']),Spacer(1,8)]
for sec in S:
 if sec['title'].startswith('附录B'):
  styles['BodyCN'].fontSize=9
  styles['BodyCN'].leading=13
 story.append(Paragraph(sec['title'],styles['HeadCN']))
 for t in sec['items']:
  if isinstance(t,str):story.append(Paragraph(t,styles['BodyCN']))
  elif t['type']=='figure':
   im=Image(str(O/f'figures/{t["name"]}.png'));w=475;h=w*im.imageHeight/im.imageWidth
   if h>527:w*=527/h;h=527
   im.drawWidth=w;im.drawHeight=h
   story.append(KeepTogether([im,Spacer(1,5),Paragraph(t['caption'],styles['CaptionCN'])]))
  else:
   cells=[[Paragraph(x,styles['CellCN']) for x in row] for row in [t['headers']]+t['rows']]
   n=len(t['headers']);widths=[42]+[(475-42)/(n-1)]*(n-1)
   tb=Table(cells,colWidths=widths,repeatRows=1,hAlign='CENTER')
   tb.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),1,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,-1),(-1,-1),1,colors.black),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3)]))
   story += [tb,Spacer(1,5),Paragraph(t['caption'],styles['CaptionCN'])]
def footer(canvas,doc):
 canvas.setFont('CJK',8);canvas.setFillColor(colors.HexColor('#667085'));canvas.drawCentredString(297.6,25,str(doc.page))
doc=SimpleDocTemplate(str(O/'report.pdf'),pagesize=(595.28,841.89),rightMargin=60,leftMargin=60,topMargin=48,bottomMargin=48,title='P5 实验结果分析报告',author='SAC-STGR 实验分析')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(O/'report.pdf')
text='\n'.join(p.extract_text() for p in reader.pages)
assert len(reader.pages)>5 and '碰撞' in text and '收敛' in text
print('Report pages:',len(reader.pages),'tables:',table_index,'figures:',sum(isinstance(t,dict) and t['type']=='figure' for s in S for t in s['items']))
(O/'audit/pdf_structure.json').write_text(json.dumps(dict(pages=len(reader.pages),tables=table_index,extracted_characters=len(text)),indent=2),encoding='utf-8')
