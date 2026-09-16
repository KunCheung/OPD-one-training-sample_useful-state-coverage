# Not All States Are Equal：从 State Coverage 到 Useful State Coverage

## 0. 一句话 Idea

现有 OPD 工作分别回答了两个相邻问题：

- **Paper I（2604.13016）**：到了一个 state 之后，Teacher 的监督 **Student 能不能学进去**？
- **Paper II（2609.04172）**：少量 query 能把 Student **带到哪些 state**？

本文进一步研究：

> **有限训练预算下，哪些 state 值得主动访问和学习？又应该选择哪些 query，才能诱导出这些高价值 state？**

核心观点：

$$
\boxed{\text{Coverage tells us where the student goes; utility tells us where it is worth learning.}}
$$

也就是说：**State Coverage 只描述“去了哪里”，并不等价于“那里值得学、Teacher 能教、Student 学得进去”。**

---

# 1. 研究背景：两篇 Rethinking OPD 工作分别解决了什么？

## 1.1 Paper I：OPD 的关键不是 Teacher 更强，而是监督是否可利用

**Yaxuan Li et al.**  
*Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe*  
arXiv:2604.13016, 2026  
https://arxiv.org/abs/2604.13016

该论文主要研究 **OPD 为什么有时成功、有时失败**。

其核心发现包括：

1. Student 与 Teacher 需要具有较兼容的 **thinking pattern**；
2. Teacher 即使 benchmark 分数更高，也必须提供 Student 尚未见过的 **new capabilities**；
3. 成功 OPD 中，Student 与 Teacher 会逐渐在 Student-visited states 上的高概率 token 集合上对齐；
4. 一个较小的 shared high-probability token set 可以承载约 97%–99% 的概率质量；
5. 对失败 OPD，论文提出 **off-policy cold start** 和 **teacher-aligned prompt selection** 等恢复策略。

因此，这篇论文揭示：

$$
\boxed{\text{Teacher has information} \;\neq\; \text{Student can exploit the information}}
$$

仅仅看到 Teacher-Student gap 很大，并不能说明这个监督一定有训练价值。

---

## 1.2 Paper II：OPD 的数据单位可能不是 Query，而是 State

**Zixuan Fu et al.**  
*Rethinking On-Policy Distillation of Large Language Models II: One Training Example*  
arXiv:2609.04172, 2026  
https://arxiv.org/abs/2609.04172

这篇论文研究的是另一个问题：**OPD 到底需要多少训练 query？**

它发现：

- 单个 query 也可以持续训练数百步，并恢复 full-data OPD 的大部分收益；
- 单个 query 的 rollout 可以覆盖 full-data OPD state space 的约 **71.5%**；
- 16 个语义多样 query 的 state coverage 可达到约 **98.9%**，性能接近 full-data；
- content-light template 和部分 off-domain query 也可以获得接近真实 query 的效果；
- 因此 OPD 可能是 **data-overfed but algorithm-starved**。

其核心视角是：

$$
\text{Query} \rightarrow \text{Trajectory} \rightarrow \text{State}
$$

一个 query 更像一个 **state generator 的 seed**。真正接受 dense teacher supervision 的，是 rollout 中大量的中间 state：

$$
s_t=(x,y_{<t})
$$

因此：

$$
\boxed{\text{Few Queries} \;\neq\; \text{Few Training States}}
$$

---

# 2. 两篇论文合起来留下的关键缺口

Paper I 主要回答：

> **Which supervision is learnable?**

Paper II 主要回答：

> **Which states are visited?**

但仍然缺少一个问题：

> **Which states are worth visiting?**

这个问题很关键，因为：

$$
\boxed{\text{State Coverage} \neq \text{State Learning Utility}}
$$

覆盖到一个 state，并不意味着这个 state 值得投入 Teacher inference 和 Student update 预算。

例如：

### State A：有意义，但 Student 已经会了

```text
2 + 2 =
```

如果 Student 和 Teacher 都几乎确定答案为 4，那么该 state 很干净，但 learning signal 很小。

### State B：Teacher-Student Gap 很大，但 Student 不一定学得进去

Teacher 与 Student 的分布差异很大，并不代表梯度方向对 Student 有效。Paper I 已经说明，**informative signal 不等于 exploitable signal**。

### State C：覆盖新区域，但与目标能力无关

例如训练 Math OPD 时 rollout 进入长篇闲聊或格式性 meta response。它可能提高 raw state coverage，却不提升目标能力。

### State D：Student 到达了 Teacher 不可靠的区域

在长 trajectory、严重 off-policy prefix 或错误累积后的 state 上，Teacher continuation 未必仍然具有稳定优势。

因此，高质量 OPD 数据不能仅由“query 多样性”或“state 数量”定义。

---

# 3. 核心研究命题：重新定义 OPD Data Quality

传统数据质量通常被理解为：正确、困难、多样、覆盖广。

但在 OPD 中，我们提出：

$$
\boxed{\text{Data Quality} \rightarrow \text{State Learning Utility}}
$$

而且 OPD 数据质量不是 query 的静态属性，而是相对于当前 Student、Teacher、已选数据和目标任务动态定义：

$$
\boxed{
\mathrm{Quality}
\left(
q\mid \pi_S,\pi_T,Q_{\mathrm{selected}},\mathcal{T}
\right)
}
$$

其中：

- $\pi_S$：当前 Student policy；
- $\pi_T$：Teacher policy；
- $Q_{\mathrm{selected}}$：当前已经选择和训练过的 query；
- $\mathcal{T}$：目标任务或能力空间。

因此，同一条 query 在训练早期可能很有价值，在 Student 已经掌握相应 state 后，其价值会显著下降。

---

# 4. Useful State：哪些 state 才值得学习？

我们认为 Useful State 至少应考虑五个维度。

## 4.1 Novelty：是否带来新的 State Region

来自 Paper II 的核心启发。

如果一个 state 与当前已覆盖状态高度重复，则边际价值应降低：

$$
N_t(s)
=
1-
\max_{s'\in S_t}
\mathrm{sim}(h(s),h(s'))
$$

这里 $S_t$ 表示当前训练已经有效覆盖的 state 集合。

---

## 4.2 Information Gain：Teacher 是否真的有新东西可教

Teacher 比 Student 强，并不意味着在每一个局部 state 上都有新知识。

定义局部信息增益 proxy：

$$
I_t(s)
=
D\left(
\pi_T(\cdot|s),
\pi_{S_t}(\cdot|s)
\right)
$$

其中 $D$ 可以使用 JS divergence、top-k log-prob gap 等。

但需要强调：

$$
I_t(s) \text{ 大} \;\not\Rightarrow\; s \text{ 一定 useful}
$$

因为 Paper I 已经说明，较大的 distribution gap 可能并不能被 Student 有效利用。

---

## 4.3 Exploitability：Student 能否利用 Teacher 的信号

这是相对于原 USC proposal 最重要的修改。

我们引入：

$$
E_t(s)=\mathrm{Exploitability}(\pi_{S_t},\pi_T,s)
$$

可以考虑以下 proxy：

- Student/Teacher top-k token overlap ratio；
- overlap token probability mass；
- shared high-probability token alignment；
- 局部 gradient agreement / effective update magnitude；
- entropy gap。

例如一个简单的 overlap 指标：

$$
E_{\mathrm{overlap}}(s)
=
\frac{
|\mathrm{TopK}(\pi_S)\cap\mathrm{TopK}(\pi_T)|
}{k}
$$

更进一步，可以使用 probability-mass-weighted overlap，而不是只统计 token 数量。

核心是区分：

$$
\boxed{\text{Informative Signal} \neq \text{Exploitable Signal}}
$$

---

## 4.4 Reliability：Teacher 在这个 Student-induced State 上是否仍可靠

Student rollout 可能进入 Teacher 不熟悉的 prefix。

尤其在：

- long-horizon reasoning；
- 多轮 Agent trajectory；
- 错误逐步累积；
- tool failure / retry；
- off-policy state；

Teacher 的局部监督质量可能下降。

定义：

$$
L(s)=\mathrm{TeacherReliability}(s)
$$

可采用：

- Teacher entropy；
- 多次 Teacher sampling consistency；
- verifier / execution feedback；
- 多 Teacher agreement；
- 在可验证任务上的 continuation correctness。

---

## 4.5 Relevance：State 是否属于目标能力

如果目标是 Math OPD，那么与数学能力完全无关的 state 不应因为 novelty 高而获得高权重。

定义：

$$
R(s)=\mathrm{Relevance}(s,\mathcal{T})
$$

可以通过：

- target-domain anchor embeddings；
- lightweight classifier；
- task-conditioned representation；
- Teacher relevance judge。

---

# 5. 从“直接相乘”改为“两阶段 Useful State 判定”

原 proposal 简单定义：

$$
U(s)=I(s)\cdot C(s)\cdot R(s)\cdot N(s)
$$

这个形式虽然直观，但问题是：不同 proxy 的尺度不同，而且一个 noisy factor 会把整体 utility 放大或压到接近 0。

因此新版建议使用 **Gate + Utility Score** 两阶段设计。

## Stage A：过滤不可学习或不可靠 state

定义有效 state 集合：

$$
\mathcal{S}_{\mathrm{valid}}
=
\left\{
 s:
R(s)\ge\tau_R,
L(s)\ge\tau_L,
E_t(s)\ge\tau_E
\right\}
$$

即先排除：

- 与目标任务无关的 state；
- Teacher 不可靠的 state；
- Teacher 与 Student thinking pattern 严重不兼容、局部监督难以利用的 state。

## Stage B：在有效 state 中衡量边际学习价值

定义：

$$
U_t(s)
=
I_t(s)
+
\lambda N_t(s)
$$

或者进一步研究可学习的参数化组合：

$$
U_t(s)=f_\phi\left(I_t(s),N_t(s),E_t(s),L(s),R(s)\right)
$$

其中 $f_\phi$ 可以通过小规模真实 OPD gain 数据拟合。

这样论文的重点不再是人为拍一个权重，而是研究：

> **哪些 pre-training signals 最能预测真正的 downstream learning gain？**

---

# 6. Useful State Coverage（USC）

对候选 query pool，使用冻结的初始 Student 做少量 pilot rollout：

$$
q_i
\rightarrow
K\ \text{pilot rollouts}
\rightarrow
S(q_i)
$$

得到所有 candidate states 后，用 Teacher hidden representation 或其他 state encoder 得到：

$$
h(s)
$$

再聚类或进行 density-aware state partition：

$$
\mathcal{C}=\{c_1,c_2,\dots,c_M\}
$$

对每个 state cluster 定义 learning utility：

$$
w_t(c)
=
\mathbb{E}_{s\in c\cap\mathcal{S}_{\mathrm{valid}}}
\left[U_t(s)\right]
$$

最终定义：

$$
\boxed{
\mathrm{USC}_t(Q)
=
\sum_{c\in\mathcal{C}}
 w_t(c)
\left(1-e^{-\beta n_c(Q)}\right)
}
$$

其中 $n_c(Q)$ 表示 query set $Q$ 对 cluster $c$ 的有效访问次数。

这一 saturation term 用于体现 diminishing return：同一 state region 被重复访问很多次，其边际价值应该逐步降低。

---

# 7. Query 的价值：Expected Marginal Useful State Coverage

最终不是直接给 query 打“质量分”，而是计算它相对于当前已选集合还能带来多少新价值：

$$
\Delta \mathrm{USC}_t(q\mid Q)
=
\mathrm{USC}_t(Q\cup\{q\})
-
\mathrm{USC}_t(Q)
$$

进一步考虑 rollout 和 Teacher inference 成本：

$$
\boxed{
q^*
=
\arg\max_q
\frac{
\Delta \mathrm{USC}_t(q\mid Q)
}{
\mathrm{Cost}(q)
}
}
$$

因此一个好的 query 不是“看起来复杂”，也不是“语义上与其他 query 不一样”，而是：

> **能够以较低成本诱导 Student 进入新的、相关的、Teacher 可靠且 Student 能有效吸收监督的 state region。**

---

# 8. 核心算法：Probe → Diagnose → Select → Train

```text
Candidate Query Pool
        │
        ▼
Frozen Student Pilot Rollout
        │
        ▼
Candidate States
        │
        ├── Novelty
        ├── Information Gain
        ├── Exploitability
        ├── Teacher Reliability
        └── Task Relevance
        │
        ▼
Filter Invalid / Unlearnable States
        │
        ▼
Estimate State Learning Utility
        │
        ▼
Marginal Useful State Coverage
        │
        ▼
Select Query Set
        │
        ▼
Full OPD Training
```

这个过程只需要少量 pilot rollouts，不需要预先跑完整 full-data OPD，因此能够解决 Paper II 当前 State Coverage 依赖 full-data reference 的问题。

---

# 9. Dynamic USC：数据质量随 Student 改变

随着 Student 学习，同一个 state 的 utility 会变化。

训练前：

$$
I_0(s) \text{ 很高}
$$

训练一段时间后：

$$
I_t(s) \rightarrow 0
$$

说明这个 state 已经被吸收。

因此：

$$
\boxed{
\mathrm{Quality}_t(q)
\neq
\mathrm{Quality}_{t+k}(q)
}
$$

进一步提出动态过程：

```text
Select
  ↓
Train M steps
  ↓
Re-Probe
  ↓
Update State Utility
  ↓
Select New Queries
  ↓
Continue Training
```

形成：

$$
\boxed{
\text{Explore}
\rightarrow
\text{Learn}
\rightarrow
\text{Re-score}
\rightarrow
\text{Explore}
}
$$

这使 OPD data selection 从一次性静态数据筛选，转化为 **Student-dependent active curriculum**。

---

# 10. 关键实验

## Experiment 1：证明 Raw State Coverage 不等于 Learning Utility

构造或筛选具有相似 raw state coverage、但具有不同局部可学习性的 query groups：

- **High Coverage / Low Information**：覆盖广，但 Student 基本已经会；
- **High Information / Low Exploitability**：Teacher-Student gap 大，但 thinking pattern 不兼容；
- **High Coverage / Low Reliability**：Student 进入 Teacher 不可靠的 state；
- **Useful Coverage**：兼具新颖性、信息增益、可利用性和可靠性。

比较：

$$
\mathrm{corr}(\mathrm{StateCoverage},\Delta \mathrm{Performance})
$$

与：

$$
\mathrm{corr}(\mathrm{USC},\Delta \mathrm{Performance})
$$

核心目标是证明：**State Coverage 是必要但不充分的指标。**

---

## Experiment 2：哪个 State Signal 最能预测真实 Learning Gain？

对大量 query 做少量 pilot rollout，并计算：

- semantic query diversity；
- raw state coverage；
- Teacher-Student divergence；
- top-k overlap / overlap probability mass；
- Teacher entropy / consistency；
- relevance；
- novelty；
- USC。

然后对单 query 或小 query set 进行真实 OPD，测量：

$$
\Delta \mathrm{Performance}(q)
$$

比较每种训练前 proxy 与真实 gain 的相关性和排序质量。

---

## Experiment 3：固定 Query Budget 的数据选择

固定：

$$
|Q|\in\{1,4,16,64\}
$$

比较：

| 方法 | Selection Principle |
|---|---|
| Random | 随机 |
| Difficulty | 难度 |
| Semantic Diversity | Query embedding diversity |
| Raw State Coverage | 最大化 state coverage |
| Gap-only | 最大 Teacher-Student divergence |
| Exploitability-only | 最大 overlap / compatibility |
| USC | 最大 Useful State Coverage |
| Dynamic USC | 周期性重新选择 |
| Full Data | Upper bound |

评价：

- downstream accuracy / pass rate；
- full-data gain recovered；
- rollout tokens；
- Teacher inference tokens；
- wall-clock cost。

---

## Experiment 4：Agent / Long-Horizon 场景

这是非常重要的扩展，因为 Paper I 已经提示 long-horizon OPD 可能存在局部监督退化问题。

在 Agent Tool Use 场景中，将 state 分类为：

- normal planning；
- tool selection；
- tool success；
- tool failure；
- retry；
- recovery；
- replanning；
- malformed observation；
- permission / constraint state。

研究：

> Raw trajectory coverage 是否会因为大量 failure/noisy state 被高估？USC 能否更准确地选择真正提升 Agent 能力的 queries？

---

# 11. Compute-aware Learning Efficiency

“One training example”并不意味着训练成本只有一个样本，因为 OPD 仍需要大量 rollout 和 Teacher forward。

因此定义：

$$
\boxed{
\mathrm{LearningEfficiency}
=
\frac{
\Delta \mathrm{Performance}
}{
\mathrm{StudentRolloutTokens}
+
\alpha\,\mathrm{TeacherTokens}
}
}
$$

最终目标不是单纯减少 query 数，而是：

> **在固定 compute budget 下，把 Student 带到最值得学习的 state。**

---

# 12. 预期贡献

## C1. Conceptual：从 State Coverage 到 State Learning Utility

Paper II 将 OPD 数据单位从 query 下沉到 state。

本文进一步提出：

$$
\boxed{
\text{Query Diversity}
\rightarrow
\text{State Coverage}
\rightarrow
\text{Useful State Coverage}
}
$$

即 state 本身也不是等价的训练单位。

## C2. Mechanistic：连接 Coverage 与 Learnability

将 Paper I 的 **Teacher-Student compatibility / exploitability** 与 Paper II 的 **state coverage** 统一到一个框架中。

核心观点：

$$
\boxed{
\text{Effective Data}
=
\text{Reach Useful States}
+
\text{Learn Effectively at Those States}
}
$$

## C3. Metric：Useful State Coverage

提出一个能够同时考虑：

- novelty；
- information gain；
- exploitability；
- reliability；
- task relevance；

的 state-level 数据价值指标。

## C4. Algorithm：Training-before-Training Query Selection

正式 OPD 前，仅使用冻结 Student 的少量 pilot rollout：

$$
\text{Probe}
\rightarrow
\text{Diagnose}
\rightarrow
\text{Select}
\rightarrow
\text{Train}
$$

无需 full-data OPD reference。

## C5. Dynamic Data Quality

提出 OPD 数据质量是 Student-dependent 的动态属性，并进一步研究 Dynamic USC / active curriculum。

---

# 13. 最核心的 Paper Story

整个 story 可以压缩成三步：

### Paper I

> **到了这个 state，Teacher 的信号 Student 学得进去吗？**

### Paper II

> **这些 query 能把 Student 带到哪些 state？**

### Our Work

> **有限预算下，Student 应该被带到哪些 state？**

因此本文不是简单给 State Coverage 加权，而是把 OPD 数据选择重新定义为：

$$
\boxed{
\text{Data Selection}
=
\text{State Exploration}
+
\text{Local Learnability}
}
$$

一句话 headline：

> **Not all visited states are worth learning from. The best OPD queries are those that induce novel states where the teacher has reliable new information and the student can actually exploit it.**

---

# 14. Candidate Titles

首选：

**Not All States Are Equal: Learning-Utility-Aware Data Selection for On-Policy Distillation**

备选：

- **Beyond State Coverage: Which States Are Worth Learning in On-Policy Distillation?**
- **From State Coverage to State Utility: Rethinking Data Quality for On-Policy Distillation**
- **Where Should the Student Learn? Useful State Coverage for Data-Efficient On-Policy Distillation**

---

# References

1. Yaxuan Li, Yuxin Zuo, Bingxiang He, et al. **Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe.** arXiv:2604.13016, 2026. https://arxiv.org/abs/2604.13016
2. Zixuan Fu, Bingxiang He, Yuxin Zuo, et al. **Rethinking On-Policy Distillation of Large Language Models II: One Training Example.** arXiv:2609.04172, 2026. https://arxiv.org/abs/2609.04172
