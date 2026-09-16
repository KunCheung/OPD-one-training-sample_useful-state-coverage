# 一个训练样本真的够吗？从 OPD 的 State Coverage 重新思考“高质量数据”

如果把 17K 条数学训练 Query 缩到 **1 条**，模型还能学到东西吗？

直觉上，答案应该是否定的：数据太少，模型很快就会过拟合。

但 *Rethinking On-Policy Distillation of Large Language Models II: One Training Example* 做了这个极端实验。结果是：**一个 Query 也可以持续做几百步 OPD，并恢复 full-data OPD 的大部分收益。** 更反直觉的是，这一个 Query 的 rollout 覆盖了 full-data OPD 所访问 state space 的约 **71.5%**；当 Query 增加到 16 个且保持语义多样时，State Coverage 提升到 **98.9%**，性能也基本追平 full-data。

这似乎指向一个很诱人的结论：

> **高质量数据，也许根本不需要很多。**

但我觉得这里还少问了一个问题：

> **覆盖了一个 State，就意味着这个 State 值得训练吗？**

这篇文章想讨论的，就是这个问题。

---

## 一个 Query 怎么可能“包含”这么多训练数据？

理解这个结果之前，需要先换一个视角。

在传统 SFT 中，我们通常把一条训练数据理解成：

```text
Question → Answer
```

因此做数据工程时，我们自然会关注：题目够不够多、领域够不够广、难度够不够丰富、答案对不对。

但 OPD 不是这样工作的。

OPD 先让 Student 自己 rollout：

$$
y \sim \pi_S(\cdot|x)
$$

然后 Teacher 不只是给最终答案，而是在 Student 自己生成的每一个 prefix 上提供 token-level supervision。于是每一个 prefix 都是一个 State：

$$
s_t=(x,y_{<t})
$$

所以真正的训练过程更像：

```text
1 Query
   ↓
很多 Student Rollouts
   ↓
很多 Trajectories
   ↓
大量 Intermediate States
   ↓
Teacher 在每个 State 上给监督
```

从这个角度看，Query 更像一个 **State Generator 的 Seed**。

这也解释了为什么：

$$
\boxed{\text{Few Queries} \neq \text{Few Training States}}
$$

One-shot OPD 的关键，不是“一条数据神奇地包含了所有知识”，而是**同一个 Query 可以通过 on-policy rollout 不断诱导出新的训练状态**。

这也是 2609.04172 最重要的启发之一：

> OPD 中，真正值得关注的数据单位，可能不是 Query，而是 Student 实际访问到的 State。

---

## 但 State 多，就一定好吗？

这里马上出现一个问题。

假设我们有下面三个 State。

### State A：很正确，但 Student 已经会了

```text
2 + 2 = ?
```

Student：

$$
P_S(4)=0.99
$$

Teacher：

$$
P_T(4)=0.995
$$

这是一个非常干净、非常正确的 State。

但从训练角度看，它几乎没有新增信息。Student 已经会了。

---

### State B：Student 还不会，而且 Teacher 有明确方向

比如一道几何证明走到中间：

```text
已知 AB = AC，需要证明 ∠B = ∠C，下一步应该……
```

Student：

```text
勾股定理          0.35
等腰三角形性质    0.25
作辅助线          0.20
```

Teacher：

```text
等腰三角形性质    0.80
```

这里 Teacher 明显拥有 Student 尚未掌握的局部知识。

这样的 State 看起来更值得训练。

---

### State C：Teacher 和 Student 差得很大，但 State 本身已经坏了

```text
banana banana @@@ random random ...
```

Teacher 和 Student 的分布差异可能很大。

但这不意味着它值得训练数学能力。

所以，仅仅知道“这个 State 没见过”或者“Teacher 和 Student 差得很大”都不够。

这让我觉得：

$$
\boxed{\text{State Coverage} \neq \text{Learning Utility}}
$$

State Coverage 告诉我们 **模型去了哪里**，但它没有回答：

> **去了那里之后，有没有值得学的东西？**

---

## 第二条线索：Teacher 有东西教，不代表 Student 学得进去

这时另一篇 OPD 工作就变得非常关键。

*Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe*（arXiv:2604.13016）研究的不是“需要多少数据”，而是：

> **为什么有些 OPD 成功，有些 OPD 失败？**

它观察到一个很重要的现象：

**更强的 Teacher，并不一定是更好的 Teacher。**

论文发现，OPD 是否成功至少和两件事情有关：

1. Teacher 是否真的拥有 Student 尚未获得的 **new capabilities**；
2. Teacher 和 Student 的 **thinking pattern 是否足够兼容**。

成功的 OPD 中，Teacher 和 Student 会在 Student 实际访问的 State 上逐渐对齐高概率 token；而失败配置中，即使 Teacher benchmark 更强、全局信号看起来也有信息，局部 token-level supervision 仍可能很难被 Student 利用。

所以：

$$
\boxed{\text{Informative Signal} \neq \text{Exploitable Signal}}
$$

这对 “Useful State” 的定义非常重要。

一开始我们可能会想：

$$
\text{Teacher-Student Gap} \uparrow
\Rightarrow
\text{State Utility} \uparrow
$$

但这并不成立。

一个 State 上 Teacher 和 Student 差得很远，可能说明 Teacher 有新信息；也可能意味着两者局部 policy geometry 根本不兼容，Student 很难沿着这个监督方向有效更新。

于是，两篇论文刚好拼成两块：

```text
Paper I (2604)
到了一个 State 之后，Student 学得进去吗？

Paper II (2609)
一组 Query 会把 Student 带到哪些 State？
```

那么很自然的下一个问题就是：

> **哪些 State 值得我们主动让 Student 去？**

---

## 我更愿意把它叫作 Useful State Coverage

如果 Raw State Coverage 只问：

> “访问过多少 State？”

那么 Useful State Coverage 应该进一步问：

> “访问过多少真正有学习价值的 State？”

我会先用几个直观问题判断一个 State：

```text
这个 State 是新的吗？
        ↓
Teacher 在这里有 Student 不会的东西吗？
        ↓
Student 能利用这个监督吗？
        ↓
Teacher 在这个 State 上仍然可靠吗？
        ↓
这个 State 和目标能力有关吗？
```

对应地，可以概括成：

- **Novelty**：是不是新增 State Region？
- **Information Gain**：Teacher 是否真的提供 Student 尚未掌握的信息？
- **Exploitability**：Student 是否能把这个信号转化成有效更新？
- **Teacher Reliability**：Student-induced prefix 已经很偏时，Teacher 的监督是否仍可信？
- **Task Relevance**：这个 State 是否属于我们希望提升的能力？

这时 Useful State 可以写成一个概念函数：

$$
U(s)=f\big(N(s),I(s),E(s),L(s),R(s)\big)
$$

但我不认为简单把五项直接相乘就是最终答案。

更合理的方式可能是 **先 Gate，再排序**。

例如，先过滤掉：

- 与目标任务无关的 State；
- Teacher 明显不可靠的 State；
- Teacher 和 Student 极度不兼容、局部监督难以利用的 State。

然后只在剩下的 State 中比较：

> **还有多少新信息？又带来了多少新的 State Coverage？**

换句话说，我们真正想最大化的不是 Raw Coverage，而是：

$$
\boxed{\Delta \text{Useful State Coverage}}
$$

---

## 最大的变化：数据质量不再是 Query 的静态属性

传统数据工程很容易把“高质量数据”看成数据本身的属性：

```text
这道题难不难？
是否正确？
属于哪个领域？
是不是和已有数据重复？
```

但在 OPD 中，同一条 Query 对不同 Student 的价值完全可能不同。

训练前，一道几何题可能诱导出大量 Student 不会、但 Teacher 能教的 State。

训练几百步以后，Student 已经掌握这些区域，再继续用同一个 Query 的价值就会下降。

因此我更倾向于这样定义 OPD 的数据质量：

$$
\boxed{
\mathrm{Quality}\left(q\mid\pi_S,\pi_T,Q_{\mathrm{selected}},\mathcal T\right)
}
$$

它取决于：

- 当前 Student；
- 当前 Teacher；
- 已经训练过哪些 Query / States；
- 当前想提升什么能力。

所以：

$$
\boxed{\text{Data Quality} \rightarrow \text{State Learning Utility}}
$$

高质量数据不再是一个固定 Dataset 的标签，而变成一个**随 Student 学习过程动态变化的属性**。

这可能是这两篇 OPD 工作继续往前推，最值得研究的一点。

---

## 但训练前怎么知道一个 Query 值不值钱？

这里有一个现实问题：

> Query 不 rollout，我们根本不知道它会把 Student 带到哪里。

所以只看 Query embedding、领域标签或者难度，很可能还是不够。

一个很自然的做法是：**先进行一次便宜的 Pilot Rollout，再决定要不要正式训练。**

假设有 10,000 条候选 Query。

冻结当前 Student，每个 Query 只采样少量 rollout：

$$
K=2\sim8
$$

得到一批 Candidate States，然后估计：

```text
Candidate Queries
        ↓
Frozen Student
        ↓
Few Pilot Rollouts
        ↓
Candidate States
        ↓
Novel? Informative? Exploitable? Reliable? Relevant?
        ↓
Useful State Estimate
        ↓
Query Selection
        ↓
Full OPD
```

可以把它理解成一种：

> **Training Before Training**

我们不是一开始就消耗大量 Teacher inference 和 rollout budget，而是先问：

> **哪些 Query 最有可能把当前 Student 带到“值得学习”的区域？**

然后再把真正的训练预算集中到这些 Query 上。

---

## Query Selection 可能最终会变成 State-Space Curriculum

还有一个细节很关键：一条 Query 的价值不是独立的。

如果 Q1 已经覆盖了一大片代数 reasoning states，那么另一条几乎产生相同 State 的 Q2，即使单独看很“高质量”，边际价值也会下降。

因此真正要考虑的是：

$$
\mathrm{Value}\left(q\mid Q_{\mathrm{selected}}\right)
$$

选择过程可能更像：

```text
Q1 → 新增代数推理 State → 选
Q2 → 与 Q1 高度重复     → 降权
Q3 → 新增几何证明 State → 选
Q4 → 新增错误恢复 State → 选
```

而且随着 Student 学习，State Utility 还会继续变化：

```text
Select
  ↓
Train
  ↓
Re-Probe
  ↓
Re-Score States
  ↓
Select New Queries
  ↓
Train
```

最终我们得到的可能不再是一个静态 Dataset，而是一种动态的 **State-Space Curriculum**：

$$
\boxed{\text{Explore} \rightarrow \text{Learn} \rightarrow \text{Re-score} \rightarrow \text{Explore}}
$$

---

## 这件事到了 Agent 场景可能更有意思

数学 reasoning 中的 State，大多还是：

```text
公式选择
中间推理
错误假设
自我纠正
结果验证
```

但 Agent 的 State Space 会复杂得多：

```text
正常工具调用
工具调用失败
API timeout
权限不足
返回 malformed
工具选择错误
重新规划
retry
memory conflict
environment change
```

传统 Agent 数据集经常会说：

> “我们有 10,000 个 Task。”

但 10,000 个 Task 如果全部是：

```text
正常调用工具
→ 正常返回
→ 正常回答
```

它们在真正的 Agent State Space 中可能高度重复。

反过来，几十个能稳定诱导出：

```text
failure
→ diagnosis
→ replan
→ retry
→ recover
```

的 Task，可能覆盖更有价值的能力区域。

所以到了 Agent learning，问题也许应该从：

$$
\text{Task Coverage}
$$

进一步变成：

$$
\boxed{\text{Useful Agent State Coverage}}
$$

这也是我觉得这个方向最值得继续往 Agent 场景扩展的原因。

---

## 最后：真正值得问的，也许已经不是“需要多少数据”

*One Training Example* 最抓眼球的结论当然是：

> **一个 Query 也可以做有效 OPD。**

但我觉得它真正重要的意义，并不是告诉我们“以后不需要数据了”。

而是它让“训练数据”这个概念本身开始变得模糊。

在 on-policy learning 中，Query 可能只是入口。

真正决定模型学到什么的，是：

1. Query 把 Student 带到了哪些 State；
2. Teacher 在这些 State 上有没有新的信息；
3. Student 能不能真正吸收这些信息。

所以，下一步的问题也许不再是：

> **我们需要多少高质量数据？**

而是：

> **怎样用最少的探索预算，把模型带到最值得学习的地方？**

如果这个视角成立，那么 OPD 的数据工程最终可能从 **Query Selection**，转变成一种 **State-Space Exploration + Curriculum Design**。

这可能比“一条训练样本就够了”本身，更值得继续研究。

---

## References

1. Yaxuan Li et al. **Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe.** arXiv:2604.13016, 2026.  
   https://arxiv.org/abs/2604.13016

2. Zixuan Fu et al. **Rethinking On-Policy Distillation of Large Language Models II: One Training Example.** arXiv:2609.04172, 2026.  
   https://arxiv.org/abs/2609.04172
