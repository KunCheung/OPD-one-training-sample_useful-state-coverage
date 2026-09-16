# OPD One-Training-Sample → Useful State Coverage

本仓库记录一个基于 **On-Policy Distillation（OPD）** 的研究想法：从原论文提出的 **State Coverage** 出发，进一步研究 **哪些 state 真正值得学习，以及如何在正式训练前识别这些 state**。

## 研究起点

本项目主要受到两篇工作启发：

> Yaxuan Li et al.  
> **Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe.**  
> arXiv:2604.13016, 2026.  
> https://arxiv.org/abs/2604.13016

> Zixuan Fu et al.  
> **Rethinking On-Policy Distillation of Large Language Models II: One Training Example.**  
> arXiv:2609.04172, 2026.  
> https://arxiv.org/abs/2609.04172

两篇工作的核心启发分别是：

- **Paper I**：Teacher 有信息，不等于 Student 能利用这些信息；OPD 是否成功与 Teacher–Student compatibility / exploitability 密切相关。
- **Paper II**：少量 Query 并不等于少量训练状态；一个 Query 可以通过 rollout 诱导出大量 states，State Coverage 比 Query Count 更接近 OPD 的有效数据规模。

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

## 内容

- [中文研究 Proposal](./PROPOSAL.zh-CN.md)
- [中文技术 Blog：从 One Training Example 到 Useful State Coverage](./BLOG.zh-CN.md)

## 当前研究问题

1. 如何定义 useful state？
2. 如何在 full OPD training 之前估计 useful state？
3. Raw State Coverage 与 downstream gain 的相关性是否足以支持其作为数据质量指标？
4. 如何同时建模 state novelty、information gain、exploitability、teacher reliability 和 task relevance？
5. 能否用少量 pilot rollout 预测 query 的真实训练价值？
6. 能否在相同 query / rollout / teacher-compute budget 下优于 semantic-diversity selection？

## Status

Research idea / early-stage proposal.
