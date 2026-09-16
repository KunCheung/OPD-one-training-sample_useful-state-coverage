# 一个训练样本真的够吗？从 State Coverage 到 Useful State Coverage

如果把 17K 条数学训练 Query 缩到 **1 条**，模型还能学到东西吗？

直觉上，答案应该是否定的：数据太少，模型很快就会过拟合。

但 *Rethinking On-Policy Distillation of Large Language Models II: One Training Example* 做了这个极端实验。结果是：**一个 Query 也可以持续进行数百步 OPD，并恢复 full-data OPD 的大部分收益。** 更反直觉的是，单个 Query 的 rollout 覆盖了 full-data OPD 所访问 State Space 的约 **71.5%**；当 Query 增加到 16 个且保持语义多样时，State Coverage 达到约 **98.9%**，性能也基本追平 full-data。

这似乎指向一个很诱人的结论：

> **高质量数据，也许根本不需要很多。**

但我更关心另一个问题：

> **State 越多，训练效果就一定越好吗？**

如果答案是否定的，那么真正值得研究的就不只是 State Coverage，而是：

> **什么样的 State 才真正有学习价值？以及，我们能不能在正式训练前找到这些 State？**

这篇文章想讨论的，就是这两个问题。

---

## 一个 Query 怎么可能“包含”这么多训练数据？

理解 One-Query OPD 之前，需要先换一个视角。

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

Teacher 也不只是给最终答案，而是在 Student 自己生成的每一个 prefix 上提供 token-level supervision。于是每一个 prefix 都对应一个 State：

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
Teacher 在每个 State 上提供监督
```

从这个角度看，Query 更像一个 **State Generator 的 Seed**。

这也解释了为什么：

$$
\boxed{\text{Few Queries} \neq \text{Few Training States}}
$$

One-Query OPD 的关键，不是“一条数据神奇地包含了所有知识”，而是**同一个 Query 可以通过 on-policy rollout 不断诱导出新的训练状态**。

这也是 2609.04172 最重要的启发之一：

> OPD 中，真正值得关注的数据单位，可能不是 Query，而是 Student 实际访问到的 State。

---

## 但 State 多，就一定好吗？

这里马上出现一个问题。

假设我们看到三个 State。

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

### State B：Student 还不会，而且 Teacher 有明确方向

例如一道几何证明走到中间：

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

### State C：Teacher 和 Student 差得很大，但 State 本身已经偏了

```text
banana banana @@@ random random ...
```

Teacher 和 Student 的分布差异可能很大，但这并不意味着这个 State 对数学能力有训练价值。

所以，仅仅知道“这个 State 没见过”或者“Teacher 和 Student 差得很大”都不够。

这意味着：

$$
\boxed{\text{State Coverage} \neq \text{Learning Utility}}
$$

State Coverage 只能告诉我们 **模型去了哪里**，却没有告诉我们：

> **这些地方里，哪些真的值得花训练预算去学？**

---

## 第二条线索：Teacher 有东西教，不代表 Student 学得进去

这时另一篇 OPD 工作就变得关键。

*Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe*（arXiv:2604.13016）研究的不是“需要多少数据”，而是：

> **为什么有些 OPD 成功，有些 OPD 失败？**

它观察到一个很重要的现象：

> **更强的 Teacher，并不一定是更好的 Teacher。**

论文发现，OPD 是否成功至少和两件事有关：

1. Teacher 是否真的拥有 Student 尚未获得的 **new capabilities**；
2. Teacher 和 Student 的 **thinking pattern 是否足够兼容**。

成功的 OPD 中，Teacher 和 Student 会在 Student 实际访问的 State 上逐渐对齐高概率 token；而在失败配置中，即使 Teacher benchmark 更强、Teacher-Student 差异也很明显，局部 token-level supervision 仍可能很难被 Student 利用。

所以：

$$
\boxed{\text{Informative Signal} \neq \text{Exploitable Signal}}
$$

这对 Useful State 的定义非常重要。

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

## 第一个核心问题：什么样的 State 才算 Useful？

我更愿意先不用复杂公式，而是问五个直观问题：

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

对应地，可以把 Useful State 拆成五个因素：

- **Novelty**：是不是新增 State Region，而不是已有区域的重复访问？
- **Information Gain**：Teacher 是否真的提供 Student 尚未掌握的信息？
- **Exploitability**：Student 是否能把这个监督转化成有效更新？
- **Teacher Reliability**：Student-induced prefix 已经偏离常见分布时，Teacher 的监督是否仍可信？
- **Task Relevance**：这个 State 是否属于我们真正希望提升的能力？

可以概括成：

$$
U(s)=f\big(N(s),I(s),E(s),L(s),R(s)\big)
$$

但我不认为简单把五项直接相乘就是最终答案。

更合理的方式可能是：**先 Gate，再排序。**

例如，先过滤掉：

- 与目标任务明显无关的 State；
- Teacher 明显不可靠的 State；
- Teacher 和 Student 极度不兼容、监督很难利用的 State。

然后只在剩下的 State 中比较：

> **还有多少新信息？又带来了多少新的有效覆盖？**

这里还需要强调一点：Useful State Coverage 仍然是一个 **Coverage** 概念，而不是把所有 State Utility 简单相加。

如果某个高价值 State Region 已经被反复访问 100 次，第 101 次的价值显然不应和第一次一样。

因此可以把它写成：

$$
\mathrm{USC}(Q)
=
\sum_{c\in\mathcal C}
 w_c\,g\big(n_c(Q)\big)
$$

其中：

- $c$ 表示一个 State Region / Cluster；
- $w_c$ 表示该 Region 的学习价值；
- $n_c(Q)$ 表示 Query 集合 $Q$ 对该 Region 的访问次数；
- $g(\cdot)$ 是带饱和效应的 coverage function。

直觉上：

```text
第一次访问高价值 Region    → 很有价值
第二次                       → 还有价值
第 100 次                    → 边际价值很低
```

所以相比 Raw State Coverage：

$$
\text{State Coverage}=\text{Where did the student go?}
$$

Useful State Coverage 更想回答：

$$
\boxed{\text{How much valuable learning space did the student cover?}}
$$

---

## 第二个核心问题：训练前怎么知道哪些 State 是 Useful？

这其实比定义本身更难。

因为 State 不是静态存在的。

它来自 Student 自己的 rollout：

$$
s\sim P(s\mid q,\pi_S)
$$

同一条 Query，对不同 Student，完全可能产生不同的 trajectories 和 States。

所以只看 Query 文本本身，很难判断：

> “这是一条高质量数据。”

这也是 OPD 数据质量和传统 Dataset Quality 最大的区别之一。

我更倾向把它写成：

$$
\boxed{
\mathrm{Quality}
\left(
q\mid\pi_S,\pi_T,Q_{\mathrm{selected}},\mathcal T
\right)
}
$$

也就是说，一条 Query 是否高质量，取决于：

- 当前 Student；
- 当前 Teacher；
- 已经覆盖了哪些 States；
- 当前想提升什么能力。

那么，训练前怎么估计它？

一个很自然的办法是：**先进行一次便宜的 Pilot Rollout，再决定要不要正式训练。**

假设有 10,000 条候选 Query。

先冻结当前 Student，每个 Query 只做少量 rollout：

$$
K=2\sim8
$$

得到一批 Candidate States，再去估计：

```text
Candidate Queries
        ↓
Frozen Student
        ↓
Few Pilot Rollouts
        ↓
Candidate States
        ↓
Novel?
Informative?
Exploitable?
Reliable?
Relevant?
        ↓
Useful State Estimate
        ↓
State Clustering / Coverage
        ↓
Query Selection
        ↓
Full OPD
```

可以把它理解成一种：

> **Training Before Training**

我们不是一开始就消耗大量 Teacher inference 和 rollout budget，而是先侦察一下 State Space：

> **哪些 Query 最有可能把当前 Student 带到真正值得学习的区域？**

---

## 从 Useful State 到 Query Selection

有了 Pilot Rollout 后，真正应该优化的也不是单条 Query 的绝对分数，而是它带来的 **边际 Useful State Coverage**。

假设已经选了一组 Query：

$$
Q_{\mathrm{selected}}
$$

那么新 Query $q$ 的价值应该看：

$$
\Delta \mathrm{USC}(q)
=
\mathrm{USC}
\left(Q_{\mathrm{selected}}\cup\{q\}\right)
-
\mathrm{USC}
\left(Q_{\mathrm{selected}}\right)
$$

这意味着：

```text
Q1 → 新增代数推理 State → 选
Q2 → 与 Q1 高度重复     → 降权
Q3 → 新增几何证明 State → 选
Q4 → 新增错误恢复 State → 选
```

所以 OPD 数据选择的目标会从：

```text
选择“看起来多样”的 Query
```

逐渐变成：

```text
选择能够诱导出最大新增学习价值的 Query
```

这已经更接近 **State-Space Active Learning**，而不是传统的 Query-level Data Selection。

---

## Useful State Coverage 还应该是动态的

还有一个很容易被忽略的问题：

> 今天有价值的 State，训练一段时间以后可能就没那么有价值了。

假设训练前：

$$
D_{JS}(\pi_T,\pi_{S_0})=0.8
$$

Student 明显不会。

训练一段时间后：

$$
D_{JS}(\pi_T,\pi_{S_t})=0.05
$$

说明这个区域已经基本被吸收。

此时继续大量访问同一批 States，边际收益就会下降。

于是数据选择本身也应该形成一个循环：

```text
Pilot
  ↓
Select Queries
  ↓
Train
  ↓
Re-Probe
  ↓
Re-Score States
  ↓
Select New Queries
  ↓
Continue Training
```

即：

$$
\boxed{\text{Explore} \rightarrow \text{Learn} \rightarrow \text{Re-score} \rightarrow \text{Explore}}
$$

最终得到的可能不再是一个静态 Dataset，而是一种：

> **Student-dependent State-Space Curriculum**

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
规划失败
重新规划
memory conflict
环境变化
```

这时传统的 Task Coverage 很可能更容易产生错觉。

比如有 1000 个 Agent Task，但它们全部遵循：

```text
调用工具 → 成功 → 返回结果
```

那么表面上 Task 数量很多，实际访问的 State Region 可能非常窄。

反过来，几十个能够触发：

```text
failure
  ↓
diagnosis
  ↓
replan
  ↓
retry
  ↓
recover
```

的任务，可能提供更丰富、更有价值的训练状态。

所以 Agent 数据工程最终可能同样需要从：

$$
\text{Task Coverage}
$$

走向：

$$
\text{Agent State Coverage}
$$

进一步走向：

$$
\boxed{\text{Useful Agent State Coverage}}
$$

---

## 最后

*One Training Example* 最吸引人的地方，表面上是：

> **一条 Query 也能训练。**

但我觉得它真正重要的地方，是迫使我们重新思考一个更基础的问题：

> **在 on-policy learning 里，到底什么才算“数据”？**

如果 Query 只是 State Generator 的 Seed，那么数据工程真正应该关心的，也许就不再只是：

> “我们有多少高质量 Query？”

而是：

> “这些 Query 会把 Student 带到哪里？”

再结合 2604 关于 Teacher-Student compatibility 的结果，还需要继续问：

> “到了那里以后，Teacher 有没有东西可以教？Student 又能不能真正学进去？”

于是问题最终变成：

$$
\boxed{\text{Data Quality} \rightarrow \text{State Learning Utility}}
$$

而我认为接下来最值得验证的，不只是一个新的 coverage 指标，而是两件事：

> **如何定义 Useful State？**
>
> **如何用尽可能少的探索成本，在正式训练前找到它？**

如果这两个问题能够回答，OPD 的数据选择可能会从 **Query Selection** 进一步变成一种真正的 **State-Space Exploration and Curriculum Design**。

换句话说，下一个问题也许不再是：

> **我们到底需要多少高质量数据？**

而是：

> **怎样用最少的探索预算，把模型带到最值得学习的地方？**

---

## References

1. Yaxuan Li et al. **Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe.** arXiv:2604.13016, 2026.  
   https://arxiv.org/abs/2604.13016
2. Zixuan Fu et al. **Rethinking On-Policy Distillation of Large Language Models II: One Training Example.** arXiv:2609.04172, 2026.  
   https://arxiv.org/abs/2609.04172
