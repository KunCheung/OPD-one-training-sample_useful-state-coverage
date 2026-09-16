# Not All States Are Equal：面向 On-Policy Distillation 的 Useful State Coverage 数据选择

## 0. 研究背景与起点

本研究直接受到以下工作的启发：

> Zixuan Fu, Bingxiang He, Yuxin Zuo, Haohuan Huang, Jinqian Zhang, Ruhang Xiao, Cheng Qian, Qinyu Luo, Huan-ang Gao, Yudong Wang, Zhiyuan Liu, Ning Ding, Chaojun Xiao.  
> **Rethinking On-Policy Distillation of Large Language Models II: One Training Example.**  
> arXiv:2609.04172, 2026.  
> https://arxiv.org/abs/2609.04172

该论文研究了一个非常反直觉的问题：**OPD 是否真的需要大规模训练 query？**

论文发现：

- 单个 query 也可以进行数百步 OPD 训练，并恢复 full-data OPD 的大部分收益；
- 单个 query 所诱导出的 rollout states 可以覆盖 full-data OPD state space 的约 **71.5%**；
- 使用 16 个语义多样的 query，state coverage 可以达到约 **98.9%**，性能接近 full-data；
- 因此，OPD 中真正重要的训练单位可能不是 query 本身，而是 query rollout 中不断访问到的 states；
- 论文据此提出 OPD 可能是 **data-overfed but algorithm-starved**。

这项工作带来了一个重要视角：

```text
Query → Trajectory → State
```

也就是说，一个 query 更像是一个 **state generator 的 seed**。Student 从同一个 query 出发，可以通过随机 rollout 产生大量不同 trajectory；trajectory 中的每个中间 prefix 都形成一个 state，并由 Teacher 提供 token-level supervision。

但是，这里还存在一个关键缺口：

> **覆盖更多 state，是否就意味着这些 state 更值得学习？**

我们认为答案是否定的。

---

# 1. 核心问题：State Coverage ≠ Useful State Coverage

原论文中的 State Coverage 主要关注：

> 某一组 query 的 rollout 到达了多少个 state clusters。

但不同 state 的训练价值并不相同。

例如：

### State A：Student 已经会了

```text
2 + 2 =
```

假设：

\[
P_S(4)=0.99
\]

Teacher：

\[
P_T(4)=0.995
\]

虽然这是一个完全正确、有效的数学 state，但 Teacher 与 Student 几乎没有差异，因此新的学习信号非常弱。

---

### State B：Student 明显没有掌握

例如几何证明中的某个中间状态：

```text
已知 AB = AC，需要证明 ∠B = ∠C，下一步应该……
```

Student：

```text
使用勾股定理       0.35
利用等腰三角形性质 0.25
作辅助线           0.20
```

Teacher：

```text
利用等腰三角形性质 0.80
```

这里 Teacher 与 Student 的分布差异明显，因此具有更强的 learning signal。

---

### State C：虽然差异大，但没有学习价值

例如 Student rollout 进入乱码、重复或完全 off-task 的状态：

```text
banana banana @@@ random random ...
```

Teacher 与 Student 的分布可能差异很大，但这种 state 对目标任务未必有价值。

因此：

\[
\boxed{\text{State Coverage} \neq \text{Learning Utility}}
\]

这就是本研究的核心出发点。

---

# 2. 核心研究问题

本文研究：

> **能否在完整 OPD 训练之前，通过少量 pilot rollout 估计 state 的 learning utility，并进一步选择少量但高价值的 query？**

具体拆成三个研究问题：

### RQ1：Raw State Coverage 是否足以预测 downstream performance？

也就是说：

\[
StateCoverage(Q)
\]

与最终性能提升：

\[
\Delta Performance(Q)
\]

之间是否存在稳定、因果意义上的关系？

还是说，需要进一步区分不同 state 的价值？

---

### RQ2：能否在正式训练之前估计 query 的真实学习价值？

我们不希望先完整训练再知道数据好不好，而希望：

```text
Candidate Queries
      ↓
Frozen Student
      ↓
少量 Pilot Rollout
      ↓
估计 State Utility
      ↓
选择 Query
      ↓
正式 OPD
```

即进行一种 **training-before-training**。

---

### RQ3：Useful-State-Aware Selection 是否优于传统 Query Selection？

比较：

- Random Selection
- Difficulty-based Selection
- Semantic Diversity
- Raw State Coverage
- Teacher-Student Gap
- Useful State Coverage
- Adaptive Useful State Coverage

在相同 query budget、rollout token budget 和 teacher inference budget 下，谁能够获得更高 downstream gain？

---

# 3. 核心假设：高质量数据不是静态属性

传统数据工程往往将一条数据的质量理解成：

\[
Quality(q)
\approx
Correctness + Difficulty + Diversity + Coverage
\]

例如数学数据会强调：

- Algebra
- Geometry
- Probability
- Number Theory
- 不同难度

这种思想主要是在 **query space** 中做覆盖。

但对于 OPD，我们认为 query 的质量应该重新定义为：

\[
\boxed{
Quality(q)
=
Quality(q\mid \pi_S,\pi_T,Q_{selected},Task)
}
\]

也就是说，一条 query 是否高质量，与以下因素有关：

- 当前 Student 已经掌握了什么；
- Teacher 在哪些 state 上可以提供有效监督；
- 已选 query 已经覆盖了哪些 state；
- 当前目标任务需要哪些能力。

因此：

> **高质量数据不是 query 本身的静态属性，而是相对于 Student、Teacher、目标任务和当前已选数据动态定义的。**

---

# 4. Useful State 的定义

我们定义一个 state 的学习价值为：

\[
\boxed{
U(s)
=
I(s)\cdot C(s)\cdot R(s)\cdot N(s)
}
\]

其中：

- \(I(s)\)：Informativeness
- \(C(s)\)：Teacher Confidence
- \(R(s)\)：Task Relevance
- \(N(s)\)：Novelty

下面分别解释。

---

## 4.1 Informativeness：Teacher-Student Gap

最重要的问题是：

> Student 在这个 state 上还有多少东西可学？

可以用 Teacher 和 Student 的 distribution difference 作为 proxy：

\[
I(s)
=
D_{JS}(\pi_T(\cdot|s),\pi_S(\cdot|s))
\]

也可以使用更便宜的 top-k teacher-student log-probability gap。

如果：

\[
\pi_T(\cdot|s)\approx \pi_S(\cdot|s)
\]

则：

\[
I(s)\approx0
\]

说明 Student 已经基本掌握这个 state。

因此，哪怕 state 很正确、很典型，也不一定值得继续花训练预算。

---

# 4.2 Teacher Confidence：Teacher 自己是否可靠

仅有 Teacher-Student Gap 不够。

如果 Student 进入了一个非常奇怪的 state：

```text
banana banana xyz @@@
```

Teacher 和 Student 可能 disagreement 很大，但 Teacher 自己也未必确定下一步应该如何生成。

因此，可以加入 Teacher entropy：

\[
C(s)
=
1-\frac{H(\pi_T(\cdot|s))}{H_{max}}
\]

Teacher entropy 越低：

\[
C(s)\uparrow
\]

说明 Teacher 提供的 supervision 越稳定。

---

# 4.3 Task Relevance：这个 state 是否属于目标能力

假设目标是 Math OPD，但开放式 query rollout 生成了：

```text
Write a romantic poem about the moon.
```

Teacher 可能非常确定，Student 也可能与 Teacher 有较大 gap，但这种 state 对数学能力训练未必有价值。

因此定义：

\[
R(s)=sim(h_T(s),H_{target})
\]

其中：

- \(h_T(s)\)：Teacher 对当前 state 的 hidden representation；
- \(H_{target}\)：少量目标任务 anchor states 的 representation。

也可以用：

- lightweight domain classifier；
- embedding similarity；
- Teacher relevance scoring。

---

# 4.4 Novelty：是否已经被大量覆盖

如果某个 state region 已经被之前选中的 queries 反复访问，再继续采样的边际收益应该下降。

定义：

\[
N(s|Q_{selected})
=
1-
\max_{s'\in S(Q_{selected})}
sim(h(s),h(s'))
\]

因此我们真正优化的是：

\[
\boxed{\Delta UsefulStateCoverage}
\]

而不是：

\[
\text{Raw State Count}
\]

---

# 5. Useful State Coverage（USC）

首先，用冻结的初始 Student 对候选 query 做少量 pilot rollout：

\[
q_i
\rightarrow
K\text{ pilot rollouts}
\]

得到 state pool：

\[
S_i=
\{s_{i1},s_{i2},...,s_{in}\}
\]

然后利用 Teacher hidden representation：

\[
h_T(s)
\]

对 state 聚类：

\[
C=\{c_1,c_2,...,c_M\}
\]

对于每个 cluster：

\[
w_c
=
\frac{1}{|c|}
\sum_{s\in c}
I(s)C(s)R(s)
\]

定义 Useful State Coverage：

\[
USC(Q)
=
\sum_c
w_c
\left(
1-e^{-\lambda n_c(Q)}
\right)
\]

其中：

\[
n_c(Q)
\]

表示 selected queries 对 state cluster \(c\) 的访问次数。

这里使用 saturation term：

\[
1-e^{-\lambda n_c}
\]

是为了体现 diminishing return：

```text
第一次覆盖某 state region → 价值高
第二次 → 仍有一定价值
第 100 次 → 新增价值很低
```

---

# 6. 核心算法：Probe → Score → Select → Train

整个方法如下：

```text
Candidate Query Pool
        │
        ▼
Frozen Initial Student
        │
   K Pilot Rollouts
     per Query
        │
        ▼
   Induced States
        │
        ├── Teacher-Student Gap
        ├── Teacher Confidence
        ├── Task Relevance
        └── State Representation
        │
        ▼
Useful State Utility
        │
        ▼
Marginal USC Selection
        │
        ▼
Small Query Set
        │
        ▼
Full OPD Training
```

正式训练前 Student 参数完全冻结，因此这一步只是 probing，不进行完整 OPD。

对 query \(q_i\)，定义：

\[
Score(q_i|Q)
=
USC(Q\cup\{q_i\})-USC(Q)
\]

进一步考虑 rollout cost：

\[
q^*
=
\arg\max_{q_i}
\frac{
USC(Q\cup\{q_i\})-USC(Q)
}{
Cost(q_i)
}
\]

这样可以避免某些 query 通过生成超长 rollout 获得虚高 coverage。

---

# 7. Dynamic Useful State Coverage

Useful state 并不是固定的。

例如训练前：

\[
D_{JS}(\pi_T,\pi_{S_0})=0.8
\]

训练一段时间后：

\[
D_{JS}(\pi_T,\pi_{S_t})=0.05
\]

说明 Student 已经从 Teacher 学会了这个 state。

此时：

\[
U_t(s)\downarrow
\]

因此我们进一步提出 Dynamic USC：

```text
Initial Student
      ↓
USC Selection
      ↓
Train M Steps
      ↓
Re-Probe
      ↓
更新 Teacher-Student Gap
      ↓
重新发现高价值 State Region
      ↓
重新选择 Queries
      ↓
Continue Training
```

形成：

\[
\boxed{
Explore\rightarrow Learn\rightarrow Re-score\rightarrow Explore
}
\]

与传统 active learning 不同的是，这里的数据选择不是只根据 query uncertainty，而是根据：

> **query 所诱导出的 trajectory-state learning utility。**

---

# 8. 关键实验设计

## Experiment 1：证明 State Coverage 不等于 Learning Utility

构造三类 query：

### Easy-Diverse

特点：

\[
Coverage\uparrow,
TeacherStudentGap\downarrow
\]

即 state 很丰富，但 Student 基本已经掌握。

---

### Hard-Redundant

特点：

\[
Coverage\downarrow,
TeacherStudentGap\uparrow
\]

state diversity 不高，但每个 state 对 Student 都有明显 learning signal。

---

### Useful-Diverse

同时满足：

\[
Coverage\uparrow,
Gap\uparrow,
Relevance\uparrow
\]

比较以下指标与 downstream gain 的相关性：

- Query Diversity
- Raw State Coverage
- Gap-only
- Useful State Coverage

核心假设：

\[
Corr(USC,\Delta Performance)
>
Corr(StateCoverage,\Delta Performance)
\]

---

# 9. Experiment 2：Pilot USC 能否预测 Query Value

对每条 query：

1. Frozen Student；
2. rollout 2/4/8 次；
3. 计算 USC；
4. 单独进行 one-query OPD；
5. 得到真实 validation gain：

\[
\Delta Acc(q)
\]

然后比较：

\[
Corr(Score(q),\Delta Acc(q))
\]

Baseline：

- Query Difficulty
- Student Loss
- Semantic Diversity
- Raw State Coverage
- Gap-only
- Teacher Entropy

如果 USC 在训练前对最终 gain 的预测能力显著更强，就能证明它不仅是解释性指标，而是实用的数据选择信号。

---

# 10. Experiment 3：Query Selection

固定 query budget：

\[
1,
4,
16,
64
\]

比较：

| 方法 | Selection Principle |
|---|---|
| Random | 随机选择 |
| Difficulty | 选择最难 query |
| Semantic Diversity | query embedding clustering |
| Raw State Coverage | 最大化 state diversity |
| Gap-only | 最大 Teacher-Student disagreement |
| USC | 最大 Useful State Coverage |
| Adaptive-USC | 动态重新选择 |
| Full Data | 上限参考 |

评价指标：

- Downstream Accuracy
- Fraction of Full-Data Gain Recovered
- Student Rollout Tokens
- Teacher Inference Tokens
- Wall-clock Cost

---

# 11. Compute-Aware Evaluation

原论文虽然只需要极少 query，但 OPD 本身仍然需要大量 rollout 和 Teacher inference。

因此：

> Few Queries ≠ Low Compute

需要同时考虑数据效率和计算效率。

定义：

\[
LearningEfficiency
=
\frac{
\Delta Performance
}{
StudentRolloutTokens
+
\alpha\cdot TeacherTokens
}
\]

这样可以区分：

- Data Efficiency
- State Efficiency
- Compute Efficiency

并避免出现“16 条 query 很高效，但实际上消耗了海量 rollout token”的误导。

---

# 12. 预期结论

本文的核心 hypothesis 是：

\[
\boxed{
Performance
\propto
UsefulStateCoverage
}
\]

而不是：

\[
Performance
\propto
QueryCount
\]

也不是简单：

\[
Performance
\propto
RawStateCoverage
\]

我们预期在预测 downstream gain 上出现：

```text
Semantic Diversity
      <
Raw State Coverage
      <
Useful State Coverage
```

并且 USC-selected 16/64 queries 可以在相同训练预算下：

- 达到更高 validation accuracy；
- 恢复更高比例的 full-data gain；
- 减少冗余 rollout；
- 降低 Teacher inference cost。

---

# 13. 论文贡献

## C1. Conceptual Contribution

提出：

\[
\boxed{
Data Quality
\rightarrow
State Learning Utility
}
\]

OPD 数据质量不是 query 的静态属性，而是：

\[
Quality(q|\pi_S,\pi_T,Q_{selected},Task)
\]

的动态属性。

---

## C2. Metric

提出：

\[
\boxed{Useful State Coverage\ (USC)}
\]

不再把所有 state clusters 等权，而是根据：

- Teacher-Student Gap
- Teacher Confidence
- Task Relevance
- Novelty

对不同 state 进行加权。

---

## C3. Algorithm

提出无需 full-data OPD reference run 的：

\[
\boxed{
Pilot-Rollout State-Aware Data Selection
}
\]

核心流程：

```text
Probe → Score → Select → Train
```

---

## C4. Dynamic Data Quality

提出：

> 数据质量会随着 Student 的学习动态变化。

同一个 query 在训练初期可能很有价值，但随着 Student 对其对应 state 的 Teacher-Student gap 下降，其 learning utility 也应该下降。

---

## C5. Empirical Contribution

计划在多个 domain 上验证：

- Mathematical Reasoning
- Code Generation
- Instruction Following
- Agentic Tool Use

---

# 14. 与原论文的核心差异

原论文回答：

> **Why can very few queries work in OPD?**

其核心解释是：

\[
OneQuery
\rightarrow
ManyStates
\]

原论文通过 State Coverage 说明，少量 query 可以诱导出大量与 full-data 类似的 states。

本研究进一步回答：

> **Given a large candidate pool, which few queries should we actually choose?**

我们的答案是：

\[
Query
\rightarrow
States
\rightarrow
UsefulStates
\rightarrow
MarginalLearningUtility
\]

两者逻辑关系为：

```text
原论文：
少量 query 为什么够？
        ↓
因为一个 query 可以产生很多 states

我们的工作：
既然 states 才是训练单位，
那哪些 states 才真正值得学？
        ↓
Useful State Coverage
        ↓
用 Useful State 反向指导 Query Selection
```

---

# 15. 最核心的论文 Story

整篇论文不应简单讲成：

> “我们提出了一个更好的 data selection score。”

更好的 story 是：

### 第一层：重新定义训练数据单位

已有工作表明：

\[
Query
\rightarrow
State
\]

OPD 的有效学习单位是 rollout state。

---

### 第二层：进一步指出 state 也不是等价的

\[
State
\neq
TrainingValue
\]

一个 state 是否值得学习，取决于 Student 是否仍有东西可以从 Teacher 学到。

---

### 第三层：重新定义 OPD 数据质量

\[
\boxed{
QueryQuality
=
ExpectedMarginalUsefulStateCoverage
}
\]

---

### 第四层：这种质量可以在正式训练前估计

通过少量 pilot rollout：

\[
Probe
\rightarrow
Estimate
\rightarrow
Select
\rightarrow
Train
\]

从而将这一概念真正变成一个可执行的数据选择方法。

---

# 16. Candidate Titles

首选：

> **Not All States Are Equal: Useful State Coverage for Data-Efficient On-Policy Distillation**

备选：

> **Beyond State Coverage: Learning-Utility-Aware Data Selection for On-Policy Distillation**

> **Rethinking Data Quality for On-Policy Distillation: Which States Are Worth Learning?**

> **From Query Diversity to Learning Utility: State-Aware Data Selection for On-Policy Distillation**

---

# 17. 一句话 Pitch

> **OPD 中最好的训练 query，不是输入层面最复杂或最多样的 query，也不是单纯能产生最多 state 的 query，而是能够诱导出“新的、与目标任务相关、Teacher 可靠且 Student 尚未掌握”的 states。**

---

# 18. Reference

Fu, Z., He, B., Zuo, Y., Huang, H., Zhang, J., Xiao, R., Qian, C., Luo, Q., Gao, H., Wang, Y., Liu, Z., Ding, N., & Xiao, C.  
**Rethinking On-Policy Distillation of Large Language Models II: One Training Example.**  
arXiv:2609.04172, 2026.  
https://arxiv.org/abs/2609.04172
