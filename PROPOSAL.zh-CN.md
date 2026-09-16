# Not All States Are Equal：从 State Coverage 到 Useful State Coverage

## 0. 研究想法

本文尝试从 **state space** 的视角重新理解 On-Policy Distillation（OPD）中的数据质量。

这个方向主要受到两篇工作的启发：

- **Yaxuan Li et al.**  
  *Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe*  
  arXiv:2604.13016, 2026  
  https://arxiv.org/abs/2604.13016

- **Zixuan Fu et al.**  
  *Rethinking On-Policy Distillation of Large Language Models II: One Training Example*  
  arXiv:2609.04172, 2026  
  https://arxiv.org/abs/2609.04172

两篇工作分别提供了两个重要观察：

1. 有效蒸馏同时依赖 **Teacher 提供新的信息**，以及 Student 对这些监督信号的 **可利用性（exploitability）**。
2. 一个 query 可以通过 Student rollout 诱导出大量不同 states，少量 query 也可能获得很高的 State Coverage。

这让我们自然地得到第三个问题：

> **哪些 Student-visited states 对当前模型真正有学习价值？这种价值能否进一步指导 OPD 数据选择？**

本文的核心工作假设是：

$$ \text{Data Quality} \rightarrow \text{State Learning Utility} $$

一条 query 的价值取决于它让**当前 Student**访问到哪些 states，以及这些 states 是否提供新的、可靠且可利用的学习机会。

这里的 Useful State Coverage（USC）仍然是一个 **working hypothesis**。本文真正希望回答的问题之一，是：**哪些 state-level signals 能够稳定预测实际 downstream learning gain？**

---

# 1. A state-space view of OPD

在 SFT 中，一条训练数据通常在训练开始前已经固定：

```text
prompt → reference response
```

OPD 中，Student 首先从当前 policy rollout：

$$ y \sim \pi_S(\cdot \mid x) $$

Teacher 再在 Student 实际生成的 prefix 上提供 token-level supervision。对第 $t$ 个 token，对应的 state 可以写为：

$$ s_t = (x, y_{1:t-1}) $$

因此，一个 query 更像一个进入 state space 的 seed：

```text
Query
  ↓
Student Rollout
  ↓
Trajectory
  ↓
Visited States
  ↓
Teacher Supervision
```

这意味着同一个 query 可以通过不同 rollout 产生大量训练 states。

*One Training Example* 中，单个 query 可达到约 71.5% 的 full-data State Coverage，16 个语义多样的 query 可达到约 98.9%。这个结果说明，**Student 实际访问到的 state space 是理解 OPD 有效数据规模的重要视角**。

State Coverage 描述了 Student 的探索范围。进一步的问题是：

> **这些被访问到的区域，对当前 Student 有多大的学习价值？**

---

# 2. Research gap：从 Coverage 到 Learning Utility

考虑三个简单的 state。

| State | Student / Teacher 情况 | 直觉上的训练价值 |
| --- | --- | --- |
| `2 + 2 = ?` | Student 对 `4` 已经为 0.99，Teacher 为 0.995 | 低：Student 基本已掌握 |
| 几何证明进入错误分支 | Student 犹豫，Teacher 对正确下一步有明显偏好 | 可能较高：存在新的局部监督 |
| Rollout 已进入乱码或严重 off-task context | Teacher 与 Student gap 可能很大 | 不确定，甚至可能无价值 |

这三个 state 都会贡献 raw State Coverage，但它们对应的 learning opportunity 显然不同。

2604.13016 进一步说明，Teacher–Student gap 的大小只是信息的一部分。有效监督还依赖 Student 是否能够吸收 Teacher 的局部信号。

因此至少需要区分两个维度：

**Is there new information?**

和：

**Can the Student actually use it?**

本文希望把研究问题推进到：

> **“给定有限训练预算，一组 query 能覆盖多少对当前 Student 真正有学习价值的 state regions？”**

这就是本文对 **Useful State Coverage（USC）** 的工作性理解。

---

# 3. 核心命题：Data Quality 是条件性的

传统数据工程通常使用正确性、难度、领域、多样性、去重和覆盖度等属性刻画样本质量。

在 OPD 中，query 的价值还会随着 Student、Teacher、已覆盖区域和目标能力变化。可以写成：

$$ \mathrm{Quality}\!\left(q \mid \pi_S, \pi_T, Q_{\mathrm{selected}}, \mathcal{T}\right) $$

其中：

- $\pi_S$：当前 Student policy；
- $\pi_T$：Teacher policy；
- $Q_{\mathrm{selected}}$：已经选择或训练过的 query；
- $\mathcal{T}$：目标任务 / 能力空间。

一条 query 在训练早期可能持续诱导出 Student 尚未掌握、Teacher 又能有效指导的 states；随着这些区域被吸收，后续 rollout 的边际价值会逐步下降。

因此：

> **Useful State 可以理解为一个 student-dependent learning opportunity。**

---

# 4. What might make a State useful?

当前把以下五类信号视为 **candidate predictors**。

| Signal | 想回答的问题 | 可能的 proxy |
| --- | --- | --- |
| **Novelty** | 这个 region 是否已经被大量访问？ | hidden-state distance、cluster coverage、behavioral embedding |
| **Information** | Teacher 是否有 Student 尚未掌握的信息？ | JS/KL gap、top-k log-prob gap |
| **Exploitability** | Student 能否利用 Teacher supervision？ | high-probability token overlap、local gradient alignment |
| **Reliability** | Teacher 在这个 Student-induced state 上是否仍可信？ | verifier、self-consistency、multi-teacher agreement、execution feedback |
| **Relevance** | 这个 state 是否属于目标能力？ | target-state similarity、domain classifier、task verifier |

为了方便讨论，可以把它抽象写成：

$$ U_t(s) = f\!\left(N_t(s), I_t(s), E_t(s), L_t(s), R_t(s)\right) $$

这里的 $f$ 表示一个待研究的 utility mapping。一个核心问题是：

> **哪些信号真正与 state 的实际 learning gain 相关？它们更适合作为 filter、ranking signal，还是需要联合建模？**

---

# 5. Operationalization：先验证信号，再形成 USC

为了避免过早锁定某个手工公式，本文将几种实现方式作为对照方案。

### A. Gate + ranking

先用 Relevance / Reliability / Exploitability 过滤明显无效 states，再用 Information 和 Novelty 做 ranking。

### B. Learned utility model

用少量真实 OPD gain 数据监督一个函数：

$$ \hat{U}_\phi(s) = f_\phi\!\left(N,I,E,L,R\right) $$

检验它是否能预测 state / cluster 的真实 learning progress。

### C. Direct local learning-progress oracle

对少量 state 或 state cluster 做 micro-update，直接测量：

$$ V(s) = L_{\mathrm{target}}(\theta) - L_{\mathrm{target}}\!\left(\theta - \eta \nabla \ell_{\mathrm{OPD}}(s)\right) $$

这个量计算昂贵，但可以作为 proxy quality 的近似 oracle。

这部分研究的核心是建立：

$$ \text{State Signal} \rightarrow \text{Actual Learning Gain} $$

之间的稳定关系。

---

# 6. Useful State Coverage：一个工作性目标

即使能够估计单个 state 的 utility，数据选择仍需要考虑 coverage，因为重复访问同一 state region 的边际价值会下降。

因此 USC 同时包含两个概念：

- **State Utility**：当前 Student 在某个 state / region 上有多少学习机会；
- **Coverage**：有限 rollout budget 下，覆盖了多少不同的高价值区域。

一个简单的工作性形式是：

$$ \mathrm{USC}_t(Q) = \sum_{c \in \mathcal{C}} w_t(c)\,g\!\left(n_c(Q)\right) $$

其中：

- $c$：state region / cluster；
- $w_t(c)$：该 region 在当前 Student 下的估计学习价值；
- $n_c(Q)$：query 集合 $Q$ 对该 region 的访问次数；
- $g(\cdot)$：具有 diminishing returns 的 coverage function。

例如可以先用：

$$ g(n) = 1 - e^{-\beta n} $$

作为 baseline。

这个式子用于实验性地 operationalize USC。真正需要验证的是：**student-dependent utility 加权后的 state coverage，是否比 raw State Coverage 更能解释和预测 downstream OPD gain。**

---

# 7. Probe before train：训练前如何发现 Useful States？

由于 state 由 query 和当前 Student 联合产生：

$$ s \sim P(s \mid q, \pi_S) $$

query 的真实 OPD 价值需要结合 Student rollout 才能观察。

一个直接方案是先做少量 **pilot rollout**：

```text
Candidate Queries
        ↓
Frozen Student
        ↓
Small Pilot Rollouts
        ↓
Visited States
        ↓
State Probes
        ↓
State Regions / Utility Estimate
        ↓
Query Selection
        ↓
Full OPD
```

这里 probing 阶段保持 Student 参数冻结，只用于观察 query 对当前 Student 所诱导出的 state distribution。

这套方法成立的关键前提是：

> **少量 pilot rollouts 足以预测 query 在完整 OPD 中的真实训练价值，并且 probing 成本明显低于被节省的训练成本。**

这是本文最重要、也最容易被证伪的假设之一。

---

# 8. From query selection to marginal state-space coverage

如果已经选了一组 query $Q$，新 query $q$ 的价值可以通过它带来的 **marginal useful coverage** 衡量：

$$ \Delta \mathrm{USC}_t(q \mid Q) = \mathrm{USC}_t\!\left(Q \cup \{q\}\right) - \mathrm{USC}_t(Q) $$

进一步考虑 rollout 和 Teacher inference 成本，可以使用：

$$ q^* = \arg\max_q \frac{\Delta \mathrm{USC}_t(q \mid Q)}{\mathrm{Cost}(q)} $$

两条 query 在文本语义上可能差异很大，但 Student rollout 后仍可能进入相似 states；表面相似的 query 也可能诱导出完全不同的错误路径和 state regions。

因此，本研究更关注 **model-induced diversity** 以及它带来的新增 learning opportunity。

---

# 9. Dynamic USC：Useful State 会随 Student 改变

如果一个 state 在训练初期具有很大的 Teacher–Student gap，而 Student 在后续训练中已经吸收了该能力，它的 learning utility 会随之下降。

因此 USC 应该是时间相关的：

$$ U_t(s), \quad \mathrm{USC}_t(Q) $$

这自然导向一个动态流程：

```text
probe
  ↓
select
  ↓
train
  ↓
re-probe
  ↓
re-score / re-select
```

也就是：

$$ \text{Explore} \rightarrow \text{Learn} \rightarrow \text{Re-score} \rightarrow \text{Explore} $$

如果这一假设成立，OPD 数据选择将更接近一种 **state-space curriculum learning**：Student 吸收已有高价值区域后，再转向下一批有学习机会的 states。

---

# 10. Core research questions

### RQ1. Raw State Coverage 能解释多少 OPD gain？

在 State Coverage 相近的情况下，Information / Exploitability / Reliability / Relevance 的差异是否仍能显著解释 downstream OPD gain？

### RQ2. 哪些 state-level signals 最能预测真实 learning value？

比较：

- Student loss / difficulty；
- Teacher–Student KL / JS；
- semantic novelty；
- raw State Coverage；
- high-probability token overlap；
- verifier / reliability；
- learned utility model；
- micro-update learning progress。

### RQ3. 少量 pilot rollout 能否预测 query value？

用每条 query 的 2 / 4 / 8 次 pilot rollout 估计 utility，再与真实 one-query OPD gain 做相关性分析。

### RQ4. Useful-State-aware query selection 是否更高效？

在相同 query budget、Student rollout tokens 和 Teacher inference tokens 下，对比 Random、Difficulty、Semantic Diversity、Raw State Coverage、Gap-only 和 USC-aware selection。

### RQ5. Dynamic selection 能带来多少额外收益？

随着 Student 学习后重新 probe / select，评估其减少冗余 states 和提升 learning efficiency 的效果。

---

# 11. Experiment design

## Experiment 1：State Coverage 与 Learning Utility 的差异

构造或筛选三类 query：

- **High-Coverage / Low-Learning**：覆盖广，但 Student 基本已掌握；
- **High-Gap / Low-Exploitability**：Teacher–Student gap 大，但监督难以利用；
- **High-Utility / Diverse**：有新信息、可利用、相关且覆盖不同 regions。

比较不同指标与最终 $\Delta$Performance 的相关性。

重点验证：

> **Raw State Coverage 是否存在稳定的解释缺口？**

## Experiment 2：Pilot rollout 的预测能力

对候选 query：

1. 冻结初始 Student；
2. 每条 query 做 2 / 4 / 8 次 pilot rollout；
3. 计算候选 state signals；
4. 真正进行 one-query OPD；
5. 比较预测分数与真实 $\Delta$Performance。

如果少量 probing 的预测能力较弱，query pre-selection 的实际价值也会相应下降。

## Experiment 3：Query selection under equal budget

Query budget：1 / 4 / 16 / 64。

Baselines：

- Random；
- Difficulty；
- Semantic Diversity；
- Raw State Coverage；
- Gap-only；
- Static USC-aware；
- Dynamic USC-aware；
- Full Data。

Metrics：

- downstream accuracy / reward；
- fraction of full-data gain；
- Student rollout tokens；
- Teacher inference tokens；
- wall-clock time；
- learning efficiency。

可以定义：

$$ \mathrm{LearningEfficiency} = \frac{\Delta \mathrm{Performance}}{\mathrm{StudentTokens} + \alpha\,\mathrm{TeacherTokens}} $$

---

# 12. Agent setting

这个问题在 Agent 场景中可能更重要。

数学 reasoning 的 state 主要是 reasoning prefix；Agent 的 state space 还包括：

```text
plan
→ tool call
→ malformed result
→ wrong interpretation
→ retry
→ permission error
→ replan
→ recovery
```

两个 Agent 数据集即使都有 1,000 个任务，也可能产生完全不同的 state distribution。

一个数据集可能几乎都是：

```text
correct plan → successful tool call → answer
```

另一个数据集则覆盖：

```text
tool failure
partial observation
wrong action
recovery
memory conflict
replanning
```

从 Agent 学习角度看，**Useful Agent State Coverage** 可以进一步刻画任务数据覆盖到多少真正有价值的 execution / recovery states。

---

# 13. What would falsify this idea?

这个方向需要明确可证伪条件。

如果实验发现：

- Raw State Coverage 已经足以稳定预测 OPD gain；
- Exploitability / Reliability / Relevance 等额外信号无法增加解释力；
- 少量 pilot rollout 无法预测完整训练价值；
- probing 成本接近甚至超过节省下来的训练成本；
- USC-aware selection 在相同 compute budget 下不优于简单 baseline；

那么 Useful State Coverage 作为独立研究目标的必要性会显著下降。

反过来，如果 raw coverage 接近的 query 集合表现出明显不同的训练收益，而这种差异可以由 student-dependent learning signals 稳定解释，那么 USC 就具备更强的研究价值。

---

# 14. Expected contribution

如果上述假设成立，本文希望形成以下贡献：

1. **Conceptual**：把 OPD 数据质量从 query-level 静态属性扩展到 student-dependent state learning opportunity；
2. **Empirical**：系统分析 Raw State Coverage 与真实 Learning Utility 的关系；
3. **Measurement**：比较哪些 state-level proxies 最能预测实际 learning gain；
4. **Method**：提出 probe-before-train 的 state-aware query selection；
5. **Dynamic selection**：探索 Student 学习过程中重新估计 state utility 的动态 curriculum；
6. **Generalization**：把 state-space 数据质量视角扩展到 Agent tool-use / recovery 等场景。

一句话概括：

> **OPD 中高价值的 query，往往能够把当前 Student 带到新的、相关的、Teacher 可靠且 Student 能够有效学习的 state regions。**

---

# References

1. Yaxuan Li et al. **Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe.** arXiv:2604.13016, 2026.  
   https://arxiv.org/abs/2604.13016
2. Zixuan Fu et al. **Rethinking On-Policy Distillation of Large Language Models II: One Training Example.** arXiv:2609.04172, 2026.  
   https://arxiv.org/abs/2609.04172
