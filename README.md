# OPD One-Training-Sample → Useful State Coverage

本仓库记录一个基于 **On-Policy Distillation（OPD）** 的研究想法：从原论文提出的 **State Coverage** 出发，进一步研究 **哪些 state 真正值得学习，以及如何在正式训练前识别这些 state**。

## 研究起点

本项目直接受到以下论文启发：

> Zixuan Fu, Bingxiang He, Yuxin Zuo, Haohuan Huang, Jinqian Zhang, Ruhang Xiao, Cheng Qian, Qinyu Luo, Huan-ang Gao, Yudong Wang, Zhiyuan Liu, Ning Ding, Chaojun Xiao.  
> **Rethinking On-Policy Distillation of Large Language Models II: One Training Example.**  
> arXiv:2609.04172, 2026.  
> https://arxiv.org/abs/2609.04172

原论文的核心发现包括：

- 单个 query 的 OPD 仍可持续优化数百步，并恢复 full-data OPD 的大部分收益；
- 一个 query 的 rollout 可覆盖 full-data OPD 所访问 state 的约 **71.5%**；
- 增加语义多样的 query 后，16 个 query 的 state coverage 可达到约 **98.9%**，性能接近 full-data；
- 因此，OPD 中真正重要的学习单位可能不是 query 本身，而是 query 所诱导出的 rollout states；
- 论文据此提出 OPD 可能是 **data-overfed but algorithm-starved**。

本项目进一步追问：

> **State Coverage 是否足以代表数据质量？如果不是，能否在正式训练前估计 Useful State Coverage，并用它选择真正高价值的少量 query？**

## 核心思路

```text
Query Diversity
      ↓
State Coverage
      ↓
Useful State Coverage
      ↓
Dynamic Student-Dependent Data Selection
```

一个 query 的价值不应只由其文本本身、难度或语义多样性决定，而应由它对当前 Student 所诱导出的 **有学习价值的 state** 决定。

详细方案见：

- [中文研究 Proposal](./PROPOSAL.zh-CN.md)

## 当前研究问题

1. 如何定义 useful state？
2. 如何在 full OPD training 之前估计 useful state？
3. Raw State Coverage 与 downstream gain 的相关性是否足以支持其作为数据质量指标？
4. 能否用少量 pilot rollout 预测 query 的真实训练价值？
5. 能否在相同 query / rollout / teacher-compute budget 下优于 semantic-diversity selection？

## Status

Research idea / early-stage proposal.
