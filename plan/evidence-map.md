# 方法设计证据映射

| 来源 | 可核对的论文结论 | 本项目可使用的论点 | 限制 |
|---|---|---|---|
| Ngiam 等，Scene Transformer，ICLR 2022，https://arxiv.org/abs/2106.08417 | 按参与者与时间轴分解注意力，以较低计算成本建模两轴关系 | 空间、时间分轴编码有先例；支持复杂度论证 | 论文输入为逐参与者轨迹，不能证明本项目八组统计的收益 |
| Yuan 等，AgentFormer，ICCV 2021，https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_AgentFormer_Agent-Aware_Transformers_for_Socio-Temporal_Multi-Agent_Forecasting_ICCV_2021_paper.html | 跨参与者与时间的关联可能被过早独立汇总损失 | 本方案保留九个 token 经过时间层的动机 | 其消融是轨迹预测，不能外推碰撞率 |
| Lee 等，Set Transformer，ICML 2019，https://proceedings.mlr.press/v97/lee19d.html | 自注意力可处理集合元素关系，注意力读出可汇聚表示 | 组间自注意力和末步读出的构件依据 | 本项目组别有路口语义，不是无序同质集合 |
| Nayakanti 等，Wayformer，ICRA 2023，https://waymo.com/research/wayformer/ | 比较融合位置和分解注意力以权衡质量与效率 | 容量、融合位置和成本须共同测试 | 不证明串行方案最优 |
| 原项目 `references/SAC-RWB-main/` | 98 维观测、10 步风险历史、共享 SAC 与在线风险奖励 | 本研究接口和对照实现的事实基础 | 原源码的旋转索引与伪安全标签需要审计 |
