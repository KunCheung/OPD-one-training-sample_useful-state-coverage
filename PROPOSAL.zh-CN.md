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

1. **Teacher 有信息，不等于 Student 能利用这些信息。** OPD 是否成功与 Teacher–Student compatibility / exploitability 密切相关。
2. **Query 很少，不等于训练 state 很少。** 一个 query 可以通过 Student rollout 诱导出大量不同 states；少量 query 也可以获得很高的 State Coverage。

这让我们自然地得到第三个问题：

> **如果 query count 不是 OPD 数据规模最合适的度量，而 raw State Coverage 又没有区分 state 的学习价值，那么什么才应该被称为“高质量 OPD 数据”？**

本文的核心工作假设是：

$$ \text{Data Quality} \rightarrow \text{State Learning Utility} $$

更具体地说，一条 query 的价值不只是由 query 本身决定，而取决于它让**当前 Student**访问到哪些 states，以及这些 states 是否真的提供新的、可靠且可利用的学习机会。

这仍然是一个 **working hypothesis**，而不是已经定型的 USC 公式。本文真正希望回答的问题之一，就是：**哪些 state-level signals 能够稳定预测实际 downstream learning gain？**

---

# 1. A state-space view of OPD

在 SFT 中，一条训练数据通常在训练开始前就已经固定：

```text
prompt → reference response
```

但 OPD 中，Student 首先从当前 policy rollout：

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

这意味着：

$$ \text{Few Queries} \neq \text{Few Training States} $$

*One Training Example* 中单个 query 可达到约 71.5% 的 full-data State Coverage，16 个语义多样的 query 可达到约 98.9%，进一步说明 **query count 未必是 OPD 有效数据规模最合适的度量**。

但 State Coverage 仍然只回答：

> Student 去过哪些地方？

它没有直接回答：

> 这些地方是否真的值得学习？

---

# 2. Research gap：State Coverage 不等于 Learning Utility

考虑三个简单的 state。

| State | Student / Teacher 情况 | 直觉上的训练价值 |
| --- | --- | --- |
| `2 + 2 = ?` | Student 对 `4` 已经为 0.99，Teacher 为 0.995 | 低：Student 基本已掌握 |
| 几何证明进入错误分支 | Student 犹豫，Teacher 对正确下一步有明显偏好 | 可能较高：存在新的局部监督 |
| Rollout 已进入乱码或严重 off-task context | Teacher 与 Student gap 可能很大 | 不确定，甚至可能无价值 |

这些 state 都可以被 raw State Coverage 统计，但显然不应被同等看待。

与此同时，2604.13016 表明，较大的 Teacher–Student gap 也不一定代表更高学习价值。Teacher 更强、分布差异更大，都不保证 Student 能吸收该监督。

所以至少需要区分：

**Is there new information?**

和：

**Can the Student actually use it?**

因此，我们把研究问题从：

> “覆盖了多少 state？”

推进到：

> **“覆盖了多少对当前 Student 真正有学习价值的 state region？”**

这就是本文对 **Useful State Coverage（USC）** 的工作性理解。

---

# 3. 核心命题：Data Quality 是条件性的

传统数据工程通常把数据质量看成样本本身的静态属性：

- 正确性；
- 难度；
- 领域；
- 多样性；
- 去重；
- 覆盖度。

但在 OPD 中，同一条 query 对不同 Student、不同训练阶段，可能具有完全不同的价值。

因此更合理的表达是：

$$ \mathrm{Quality}\!\left(q \mid \pi_S, \pi_T, Q_{\mathrm{selected}}, \mathcal{T}\right) $$

其中：

- $\pi_S$：当前 Student policy；
- $\pi_T$：Teacher policy；
- $Q_{\mathrm{selected}}$：已经选择或训练过的 query；
- $\mathcal{T}$：目标任务 / 能力空间。

一条 query 在训练早期可能不断诱导出 Student 不会但 Teacher 能教的 states；训练一段时间后，这些区域已经被吸收，再继续采样的价值会下降。

因此：

> **Useful State 不是 state 自身的固定属性，而是一个 student-dependent learning opportunity。**

---

# 4. What might make a State useful?

当前不预设一个最终正确的 Useful State 公式，而是把以下五类信号视为 **candidate predictors**。

| Signal | 想回答的问题 | 可能的 proxy |
| --- | --- | --- |
| **Novelty** | 这个 region 是否已经被大量访问？ | hidden-state distance、cluster coverage、behavioral embedding |
| **Information** | Teacher 是否有 Student 尚未掌握的信息？ | JS/KL gap、top-k log-prob gap |
| **Exploitability** | Student 能否利用 Teacher supervision？ | high-probability token overlap、local gradient alignment |
| **Reliability** | Teacher 在这个 Student-induced state 上是否仍可信？ | verifier、self-consistency、multi-teacher agreement、execution feedback |
| **Relevance** | 这个 state 是否属于目标能力？ | target-state similarity、domain classifier、task verifier |

为了方便讨论，可以把它抽象写成：

$$ U_t(s) = f\!\left(N_t(s), I_t(s), E_t(s), L_t(s), R_t(s)\right) $$

但这里的 $f$ **不是本文预先假设好的固定函数**。

一个关键研究问题正是：

> **哪些信号真正与 state 的实际 learning gain 相关？它们应该作为 filter、ranking signal，还是被联合建模？**

这比直接手工设定一个乘法或加法公式更重要。

---

# 5. Operationalization：先验证信号，再定义 USC

一个容易犯的错误，是过早把 Useful State 定义成：

$$ U(s) = N(s) \times I(s) \times E(s) \times L(s) \times R(s) $$

这种写法直观，但会引入几个问题：

- 不同 proxy 的量纲和校准方式不同；
- proxy 之间可能高度相关；
- 一个 noisy factor 可能让最终分数失真；
- Reliability / Relevance 可能更适合作为 gate，而不是连续权重；
- Novelty 可能奖励罕见但无意义的异常 state。

因此，本文把下面几种方式视为 **待比较的 operationalizations**，而不是最终定义。

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

因此本文的目标不是先宣布一个 USC 公式，而是先建立：

$$ \text{State Signal} \rightarrow \text{Actual Learning Gain} $$

之间是否存在稳定关系。

---

# 6. Useful State Coverage：一个工作性目标

即使能够估计单个 state 的 utility，也不能简单把所有 state utility 相加。

原因是重复访问同一 state region 的边际价值会下降。

因此 Useful State Coverage 应同时包含两个概念：

- **State Utility**：当前 Student 在某个 state / region 上有多少学习机会；
- **Coverage**：有限 rollout budget 下，是否覆盖了不同的高价值区域，而不是反复访问同一区域。

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

需要强调：**这个式子是一个 operational baseline，而不是 USC 的最终理论定义。**

本文真正希望检验的是：用 student-dependent utility 对 state coverage 进行加权，是否比 raw State Coverage 更能解释和预测 downstream OPD gain。

---

# 7. Probe before train：训练前如何发现 Useful States？

定义 Useful State 只是问题的一半。

更实际的问题是：

> **正式训练前，我们如何知道一条 query 会把 Student 带到哪些有价值的 states？**

由于 state 由 query 和当前 Student 联合产生：

$$ s \sim P(s \mid q, \pi_S) $$

仅靠 query 文本、embedding 或静态难度，很难准确判断其 OPD 价值。

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

这里 probing 阶段不更新 Student，只用于观察 query 对当前 Student 所诱导出的 state distribution。

这套方法成立的关键前提是：

> **少量 pilot rollouts 足以预测 query 在完整 OPD 中的真实训练价值，并且 probing 成本明显低于被节省的训练成本。**

这是本文最重要、也最容易被证伪的假设之一。

---

# 8. From query selection to marginal state-space coverage

如果已经选了一组 query $Q$，新 query $q$ 的价值不应看它自己的绝对分数，而应看它带来了多少新的 useful coverage：

$$ \Delta \mathrm{USC}_t(q \mid Q) = \mathrm{USC}_t\!\left(Q \cup \{q\}\right) - \mathrm{USC}_t(Q) $$

如果进一步考虑 rollout 和 Teacher inference 成本，可以使用：

$$ q^* = \arg\max_q \frac{\Delta \mathrm{USC}_t(q \mid Q)}{\mathrm{Cost}(q)} $$

这和传统 semantic-diversity selection 有一个重要区别：

> 两条 query 在文本语义上不同，不代表它们会把 Student 带到不同 states；反过来，两条表面相似的 query，也可能诱导出完全不同的错误路径和 state regions。

因此，本研究真正关心的是 **model-induced diversity**，而不只是 query-space diversity。

---

# 9. Dynamic USC：Useful State 会随 Student 改变

如果一个 state 在训练初期具有很大的 Teacher–Student gap，但 Student 已经在后续训练中吸收了该能力，它的 learning utility 应该下降。

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

如果这一假设成立，OPD 数据选择将更接近一种 **state-space curriculum learning**，而不是训练前一次性构造静态 dataset。

---

# 10. Core research questions

### RQ1. Raw State Coverage 是否足够？

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

### RQ5. Dynamic selection 是否优于一次性静态选择？

随着 Student 学习后重新 probe / select，是否能减少已经被吸收的冗余 states？

---

# 11. Experiment design

## Experiment 1：State Coverage ≠ Learning Utility

构造或筛选三类 query：

- **High-Coverage / Low-Learning**：覆盖广，但 Student 基本已掌握；
- **High-Gap / Low-Exploitability**：Teacher–Student gap 大，但监督难以利用；
- **High-Utility / Diverse**：有新信息、可利用、相关且覆盖不同 regions。

比较不同指标与最终 $\Delta$Performance 的相关性。

目标不是证明 USC 一定成立，而是先验证：

> **Raw State Coverage 是否存在系统性的解释缺口？**

## Experiment 2：Pilot rollout 的预测能力

对候选 query：

1. 冻结初始 Student；
2. 每条 query 做 2 / 4 / 8 次 pilot rollout；
3. 计算候选 state signals；
4. 真正进行 one-query OPD；
5. 比较预测分数与真实 $\Delta$Performance。

如果少量 probing 无法稳定预测真实 gain，那么 query pre-selection 的价值会明显下降。

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

因此，对 Agent 学习来说，Task Coverage 也可能不是最合适的数据质量度量；**Useful Agent State Coverage** 可能更接近真正的 learning opportunity coverage。

---

# 13. What would falsify this idea?

这个方向需要明确可证伪条件。

如果实验发现：

- Raw State Coverage 已经足以稳定预测 OPD gain；
- Exploitability / Reliability / Relevance 等额外信号无法增加解释力；
- 少量 pilot rollout 无法预测完整训练价值；
- probing 成本接近甚至超过节省下来的训练成本；
- USC-aware selection 在相同 compute budget 下不优于简单 baseline；

那么 Useful State Coverage 作为独立研究目标的必要性就会显著下降。

反过来，如果 raw coverage 接近的 query 集合表现出明显不同的训练收益，而这种差异可以由 student-dependent learning signals 稳定解释，那么 USC 才真正具有研究价值。

---

# 14. Expected contribution

如果上述假设成立，本文希望形成以下贡献：

1. **Conceptual**：把 OPD 数据质量从 query-level 静态属性转向 student-dependent state learning opportunity；
2. **Empirical**：系统验证 Raw State Coverage 与真实 Learning Utility 的差异；
3. **Measurement**：比较哪些 state-level proxies 最能预测实际 learning gain；
4. **Method**：提出 probe-before-train 的 state-aware query selection；
5. **Dynamic selection**：探索 Student 学习过程中重新估计 state utility 的动态 curriculum；
6. **Generalization**：把 state-space 数据质量视角扩展到 Agent tool-use / recovery 等场景。

一句话概括：

> **OPD 中最有价值的 query，不一定是文本上最复杂、最困难或最多样的 query，而可能是那些能够把当前 Student 带到新的、相关的、Teacher 可靠且 Student 真正能够学习的 state regions 的 query。**

---

# References

1. Yaxuan Li et al. **Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe.** arXiv:2604.13016, 2026.  
   https://arxiv.org/abs/2604.13016
2. Zixuan Fu et al. **Rethinking On-Policy Distillation of Large Language Models II: One Training Example.** arXiv:2609.04172, 2026.  
   https://arxiv.org/abs/2609.04172
