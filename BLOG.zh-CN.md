# One Query, Many States：从 State Coverage 重新思考 OPD 的数据质量

### 一个关于 Useful State Coverage、数据选择和动态课程学习的工作假设

最近我一直在用 **state space** 的视角想 OPD 的数据问题。

这不是一个已经被证明的理论框架，我也不认为现在就能给出一个严谨的 “Useful State” 定义。它更像是一个我觉得比较有用的 mental model：如果 On-Policy Distillation 的监督发生在 Student 自己访问到的 state 上，那么我们讨论“训练数据好不好”时，也许不应该只盯着输入的 query，而应该看看这些 query 最终把 Student 带到了哪里。

这个想法主要来自两篇最近的工作。

第一篇是 [*Rethinking On-Policy Distillation of Large Language Models II: One Training Example*](https://arxiv.org/abs/2609.04172)。它问了一个很极端的问题：**如果 OPD 只用一个训练 query，会发生什么？**

第二篇是 [*Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe*](https://arxiv.org/abs/2604.13016)。它研究的是另一个问题：**为什么有些 Teacher 能有效教会 Student，而有些更强的 Teacher 却不行？**

我觉得这两篇论文放在一起看，会留下一个很自然的问题：

> 如果 query 数量本身不是关键，而 Teacher-Student gap 也不等于可学习的信号，那么 OPD 中真正有价值的数据到底是什么？

下面是我目前的理解。

---

## A state-space view of OPD

先从最基本的训练过程开始。

在 SFT 里，我们通常把一条数据看成：

```text
prompt → reference response
```

数据在训练开始前就已经存在。我们会讨论 prompt 是否多样、response 是否正确、不同任务有没有覆盖到。

OPD 不一样。Student 先从当前 policy 自己 rollout：

$$ y \sim \pi_S(\cdot \mid x) $$

Teacher 再对 Student 实际产生的 prefix 提供 token-level supervision。对第 $t$ 个 token 来说，Teacher 看到的 state 是：

$$ s_t = (x, y_{1:t-1}) $$

这意味着，真正接受监督的并不是最开始那一条 query，而是 Student rollout 过程中不断产生的 states。

一个 query 更像是一个入口：

```text
Query
  ↓
Student rollout
  ↓
Trajectory
  ↓
State 1, State 2, State 3, ...
  ↓
Teacher supervision
```

这件事看起来只是换了一种描述，但我觉得它改变了我们理解“数据量”的方式。

一条 query 并不等于一个训练状态。一条 query 可以产生很多 rollout；每条 rollout 又包含很多不同 prefix。只要 Student 的采样还有随机性，同一个 query 就可能把模型带到不同的局部状态。

这正是 *One Training Example* 里最反直觉的结果。

论文发现，单个 query 的 OPD 可以持续训练数百步，并恢复 full-data OPD 的大部分收益。更重要的是，作者用 Teacher hidden states 对训练中访问到的 states 做聚类后发现：**一个 query 已经可以覆盖 full-data OPD state space 的约 71.5%；16 个语义多样的 query 可以把 coverage 提高到约 98.9%，同时基本追平 full-data training。**

所以我觉得这篇论文真正重要的结论，不是“以后只需要一条数据”，而是：

> **Query count 可能不是 OPD 数据规模最合适的度量。**

如果一个 query 能不断生成新的 state，那么 query 更像 seed，而不是最终的训练单位。

---

## One training example is not one training state

这个区别也解释了另一个看起来很奇怪的实验。

论文进一步测试了一些 content-light template，甚至 off-domain 的 WildChat query。结果是，某些看起来和数学任务关系不大的 seed，仍然可以接近真实数学 query 的 OPD 效果。

这并不意味着 query 内容完全不重要，也不意味着随便输入一段垃圾文本都能训练数学能力。

我更倾向于这样理解：**query 的文本语义和它最终诱导出的 state distribution，并不是一回事。**

一个很开放的 seed 可能让已经具备数学 reasoning 能力的 Student 自己展开推理。只要 rollout 过程中访问到了大量与目标能力相关的 states，Teacher 就仍然可以在这些 states 上提供有效监督。

因此，数据选择真正要关心的也许不是：

> 这条 query 看起来属于哪个类别？

而是：

> **这条 query 会让当前 Student 去哪里？**

这是我觉得 State Coverage 很有价值的地方。它把“数据覆盖”从 query space 往 model-induced state space 推了一步。

但我觉得还不能停在这里。

---

## Where State Coverage falls short

State Coverage 回答的是：**Student 去过哪些地方？**

它没有直接回答：**这些地方有没有学习价值？**

考虑三个简单的例子。

| State | Student / Teacher 情况 | 直觉上的训练价值 |
| --- | --- | --- |
| `2 + 2 = ?` | Student 对 `4` 已经是 0.99，Teacher 是 0.995 | 很低：Student 基本已经会了 |
| 几何证明走到错误分支 | Student 犹豫，Teacher 对正确下一步有很强偏好 | 较高：这里可能存在新的、可学习的信号 |
| rollout 已经进入乱码或完全跑偏的上下文 | Teacher 和 Student 的分布可能差得很大 | 不确定，甚至可能没有价值 |

这三个 state 都可以被 State Coverage “算进去”，但它们显然不应该被同等对待。

所以我不太愿意把 raw State Coverage 直接等同于数据质量。它更像是一个很重要的第一步：**先知道 Student 到了哪里，再讨论这些 state 值不值得学。**

这里就会自然碰到第二篇论文。

---

## The Student has to be able to use the Teacher signal

[*Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe*](https://arxiv.org/abs/2604.13016) 研究了 OPD 成功和失败的条件。

其中一个我觉得很关键的发现是：**Teacher 更强，并不意味着它一定是更好的 Teacher。**

论文观察到，成功的 OPD 至少依赖两件事情：

1. Teacher 确实提供了 Student 训练过程中没有获得的新能力；
2. Teacher 和 Student 的 thinking pattern 需要有足够的兼容性。

从 token-level 看，成功 OPD 会逐渐在 Student 实际访问到的 state 上，对齐一小组 high-probability tokens；这些共享 token 虽然数量不多，却承载了 97%–99% 的概率质量。

这个结果让我觉得，仅仅使用 Teacher-Student divergence 来衡量 state value 是不够的。

一个很自然的想法是：

$$ D(\pi_T,\pi_S) \text{ 越大} \Rightarrow \text{Student 有越多东西可学} $$

但这个推论并不总成立。

Gap 大可能意味着 Teacher 知道 Student 不知道的东西；也可能意味着两者局部的 reasoning distribution 已经差得太远，Teacher 给出的 dense supervision 对 Student 来说并不好吸收。

换句话说，我觉得至少要区分两件事：

**Is there new information?**

和：

**Can the student actually use it?**

这两件事不是同一个问题。

---

## What I mean by a “Useful State”

如果把前面的观察放在一起，我现在会把 Useful State 理解成一种 **student-dependent learning opportunity**。

它不是 state 自身的静态属性。

同一个 state，对一个弱 Student 可能非常有价值；对已经掌握这个能力的 Student，几乎没有价值。换一个 Teacher，它的价值又可能变化。

我目前会从五个方面去看一个 state：

| Signal | 我想回答的问题 | 可能的 proxy |
| --- | --- | --- |
| **Novelty** | 这个区域是不是已经被反复访问？ | hidden-state distance / cluster coverage |
| **Information** | Teacher 在这里有没有 Student 尚未掌握的东西？ | JS/KL gap、top-k log-prob gap |
| **Exploitability** | Student 能不能利用这个监督？ | high-probability token overlap、local gradient alignment |
| **Reliability** | Teacher 在这个 Student-induced state 上靠谱吗？ | verifier、self-consistency、multi-teacher agreement |
| **Relevance** | 这个 state 和目标能力有关吗？ | target-state similarity、domain classifier、task verifier |

如果一定要写成一个抽象形式，可以记成：

$$ U(s) = f\big(N(s), I(s), E(s), L(s), R(s)\big) $$

但我并不认为现在把这五项乘起来、加起来，就得到了一个正确的 Useful State metric。

相反，我更倾向于把其中一些量当作 **gate**，而不是 score。

例如，一个 state 如果明显 off-task，或者 Teacher 在这里完全不可靠，那么即使 Teacher-Student gap 很大，也没有必要因为这个 gap 给它高分。

通过基本可靠性和相关性过滤之后，再比较 Student 在这里还有多少没学会的东西，以及这个 state 相对于已经访问的区域有多新，可能更合理。

这也是我觉得这个方向真正需要实验而不是继续堆公式的地方。

---

## How would we find Useful States before training?

定义 Useful State 只是问题的一半。

另一半其实更难：**在真正开始大规模 OPD 之前，我们怎么知道一条 query 会不会产生 useful states？**

这是一个有点麻烦的问题，因为 state 本身不是静态数据。它由 query 和当前 Student 一起产生：

$$ s \sim P(s \mid q, \pi_S) $$

只看 query 文本，很难知道 Student 最后会走到哪里。

因此我觉得最直接的方案不是完全静态的数据打分，而是先做一次很小的 **pilot rollout**。

假设我们有一批候选 query。先冻结当前 Student，每条 query 只采样少量 rollout，不做参数更新。然后在这些 rollout 产生的 states 上估计上面的几个信号。

为了把这个过程说清楚，可以定义几个组件：

- **Candidate query**：还没有进入正式训练的数据；
- **Pilot rollout**：冻结 Student 后，为了观察 state distribution 而生成的少量 trajectory；
- **State probe**：在每个 state 上计算 Teacher-Student gap、compatibility、reliability 等信息；
- **State region**：把行为或 representation 相近的 states 聚到一起；
- **Selector**：根据新增的 useful coverage 选择下一批正式训练 query。

整个过程大致是：

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
Useful State Regions
        ↓
Query Selection
        ↓
Full OPD
```

我把它理解成一种 **probe before train**。

这里有一个实际上的 trade-off：pilot rollout 本身也需要 Student sampling 和 Teacher inference。如果 probing 成本接近真正训练，那数据选择就失去意义。所以这个方法成立的前提，是少量 rollout 就足以预测 query 在完整 OPD 中的价值。

这个假设需要被实验验证，而不是默认成立。

---

## Useful State Coverage, not a sum of state scores

即使我们能给每个 state 一个合理的 utility estimate，也不能直接把所有 state utility 相加。

原因很简单：重复访问同一种 state 的边际价值会下降。

如果一个 query 让 Student 一百次进入几乎相同的几何证明 state，那么这不应该被视为一百份独立的高价值数据。

所以 “Coverage” 仍然重要。

一种简单的形式是先把 states 聚成若干 region $c\in\mathcal{C}$，再给每个 region 一个学习价值 $w_c$：

$$ \mathrm{USC}(Q) = \sum_{c\in\mathcal{C}} w_c\,g\!\left(n_c(Q)\right) $$

这里 $n_c(Q)$ 是 query 集合 $Q$ 对 region $c$ 的访问次数，$g$ 是一个具有 diminishing returns 的函数。

它表达的直觉很简单：第一次进入一个新的高价值区域很重要；第十次还有一点价值；第一百次基本是在重复。

因此我现在更愿意把两者区分开：

- **State Utility**：当前 Student 在这个 state 上有多少学习机会；
- **Useful State Coverage**：给定有限 rollout budget，一组 query 覆盖了多少不同的高价值学习区域。

后者才更接近一个可以用来做数据选择的目标。

---

## Data quality becomes conditional

这套视角带来的一个变化是：**“高质量 query”不再是固定标签。**

传统数据筛选经常会给一条样本打静态标签：难度、正确性、领域、质量分数、是否重复。

但 OPD 中，一条 query 的价值更像是条件性的：

$$ \mathrm{Quality}\!\left(q \mid \pi_S, \pi_T, Q_{\mathrm{selected}}, \mathcal{T}\right) $$

同一条 query，在不同训练阶段可能完全不是同一个价值。

训练初期，它可能不断诱导出 Student 不会但 Teacher 能教的 state。训练一段时间后，这些 region 已经被吸收，Teacher-Student gap 降低，再继续采样的收益就会下降。

这也意味着数据选择本身可能需要动态进行：

```text
probe → select → train → re-probe → re-select
```

而不是训练开始前一次性选好一个静态 dataset，然后一路训练到底。

在这个视角下，OPD 的数据选择越来越像 **state-space curriculum learning**：随着 Student 改变，下一阶段应该去探索的区域也跟着改变。

---

## From query selection to state-space exploration

如果已经选了一组 query $Q$，新 query $q$ 的价值也不应该看它自己的绝对分数，而应该看它能带来多少新的 useful coverage：

$$ \Delta \mathrm{USC}(q) = \mathrm{USC}\!\left(Q \cup \{q\}\right) - \mathrm{USC}(Q) $$

这个区别很重要。

两条 query 在语义上可能完全不同，但 Student rollout 后可能走进几乎一样的 reasoning states。反过来，两条表面上很相似的 query，也可能因为 Student 的不同错误路径而产生完全不同的 state coverage。

所以我不太确定 query semantic diversity 最终会不会是 OPD 数据选择里最好的 proxy。

真正要优化的也许是：

> **在有限 sampling 和 Teacher compute 下，怎样让当前 Student 访问尽可能多的高价值、非重复 state regions？**

这已经不太像传统意义上的 dataset curation，更像一个 exploration problem。

---

## Why this may matter even more for agents

数学 reasoning 是一个比较干净的起点，因为 state 主要还是 reasoning prefix。

Agent 场景会复杂得多。

一个任务可能让 Agent 经过：

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

如果我们只按 task 数量看数据覆盖，两个训练集都可能有 1,000 个任务，但它们实际提供的 state distribution 完全不同。

一个数据集可能几乎全部是：

```text
correct plan → successful tool call → answer
```

另一个数据集虽然任务数量更少，却大量覆盖：

```text
tool failure
partial observation
wrong action
recovery
memory conflict
replanning
```

从 Agent 学习的角度，后者可能提供更多有价值的 learning states。

这也是我觉得 Useful State Coverage 不一定只属于数学 OPD 的原因。它更像是在问一个通用的 on-policy learning 问题：**模型到底经历了哪些对当前能力边界有帮助的状态？**

---

## What would convince me this idea is useful?

现在这个方向最大的问题是，它很容易变成一个听起来合理但无法证伪的概念。

所以我觉得最重要的不是先设计一个复杂 USC 公式，而是做几个非常直接的实验。

第一个实验，是验证 **Raw State Coverage 是否真的不足**。找一些 state coverage 相近、但 Teacher-Student gap、compatibility 或 task relevance 明显不同的 query，看看 downstream OPD gain 是否明显不同。

第二个实验，是验证 **pilot rollout 能不能预测真实训练价值**。对每条 query 只做 2、4、8 次 pilot rollout，预测它的 useful coverage；然后真的做 one-query OPD，看预测分数和最终 $\Delta$Performance 的相关性。

第三个实验，才是做数据选择。在相同 query budget、Student rollout tokens 和 Teacher inference tokens 下，对比 Random、Semantic Diversity、Raw State Coverage、Gap-only 和 Useful-State-aware Selection。

如果最后只是发现：

> state coverage 已经足够解释性能，加入其他信号没有稳定收益，

那这个 idea 基本就不成立。

但如果我们发现两组 query 的 raw coverage 很接近，而“Student 是否真的能从这些 state 学到东西”可以显著解释最终差异，那么我觉得 Useful State Coverage 才有存在的必要。

---

## Closing thoughts

我现在比较确定的是一个问题，而不是一个答案。

*One Training Example* 让我重新考虑 OPD 里的“训练数据”到底是什么。Query 显然很重要，但它可能更像是进入 state space 的入口。真正承载 dense supervision 的，是 Student 自己 rollout 时不断访问到的 states。

State Coverage 把这个问题往前推进了一大步：与其问“有多少 query”，不如问“模型到底去了多少地方”。

但如果不同 state 的学习价值差异很大，那么下一步自然是继续问：

> **哪些地方值得去？**

以及更实际的问题：

> **能不能在花掉完整训练预算之前，就大致知道哪些 query 会把 Student 带到这些地方？**

我觉得这两个问题比一个固定的 Useful State 公式本身更重要。

如果这个方向成立，那么 OPD 的数据工程可能会从静态的 query selection，慢慢变成一种动态的 state-space exploration：Student 当前不会什么，就尽量把它带到那里；学会以后，再去找下一片还没吸收的区域。

这也是我目前理解的 **Useful State Coverage**。

---

## References

1. Yaxuan Li et al. **Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe.** arXiv:2604.13016, 2026.  
   https://arxiv.org/abs/2604.13016
2. Zixuan Fu et al. **Rethinking On-Policy Distillation of Large Language Models II: One Training Example.** arXiv:2609.04172, 2026.  
   https://arxiv.org/abs/2609.04172