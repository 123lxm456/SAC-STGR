from pathlib import Path
p=Path(__file__).resolve().parent
f=p/'render_report.py';s=f.read_text(encoding='utf-8')
needle="add('4 消融实验与创新点证据',"
insert="""tail=pd.read_csv(O/'tables/test_final_60s.csv')
tailrows=[]
for d in D:
 for v in V:
  r=tail[(tail.density==d)&(tail.variant==v)].iloc[0]
  tailrows.append([d,L[v],f'{r.aql:.3f}',f'{r.awt:.1f}',f'{r.mean_speed:.3f}'])
add('3.4 测试时域暴露的持续拥堵',
tab('表4a 测试最后60s的真实分块均值；每方案每密度九次测试全部纳入。',['需求','方案','末段AQL','末段AWT(s)','末段速度(m/s)'],tailrows),
'各方法从测试初期较高速度逐步下降，到最后60s的各组均值仅约0.09至0.16m/s；末段AWT约495至655s，AQL持续累积，累计到达数增长趋缓。M1 nominal末段速度仅0.099m/s、AWT约637.1s，说明当前策略没有维持稳定畅通。时间平均速度约1.1至1.4m/s掩盖了更严重的末段退化，不能单以全时段均值宣称持续通行能力。',
'事件时序重算显示：M1的174起碰撞中167起发生于前1800s，B0为98/98，A-B为151/155，S0为176/181。后半程事故较少与低速滞留同时出现，但这些记录不能单独证明因果。累计碰撞率随时间下降也受累计出发分母扩大影响，不是学习到更安全策略的证据。需要同时控制车辆暴露、滞留与完成情况，才能区分真实风险降低和缺少运动机会。',
'三密度时域曲线没有显示长期稳态通行，当前实验尤其不能支持拥堵环境下已具备稳定运行能力。建议增加路口分项与未插入需求统计，并检查死锁干预及长期饱和机制；这些是后续验证方向，本报告不改写源代码或现有实验。')
"""
assert needle in s
s=s.replace(needle,insert+needle)
s=s.replace("fontSize=10,leading=17","fontSize=10,leading=16")
f.write_text(s,encoding='utf-8')
