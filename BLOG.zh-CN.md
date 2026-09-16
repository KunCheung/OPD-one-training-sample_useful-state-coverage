# 从 One Training Example 到 Useful State Coverage：重新思考 OPD 的数据质量

最近两篇关于 **On-Policy Distillation（OPD）** 的工作提出了几个非常反直觉的结果。

第一篇 *Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe* 研究的是：

> **为什么有些 Teacher 能把能力蒸馏给 Student，而有些看起来更强的 Teacher 反而不行？**

第二篇 *Rethinking On-Policy Distillation of Large Language Models II: One Training Example* 则把问题推到了另一个极端：

> **OPD 到底需要多少训练数据？如果只有一个训练 Query 会怎么样？**

结果是：**一个 Query 也可以持续训练数百步，并获得 full-data OPD 的大部分收益。**

更有意思的是，这两个问题放在一起，会自然产生第三个问题：

> **如果 Query 数量不是关键，而 State Coverage 也不是全部，那么 OPD 中真正的“高质量数据”到底是什么？**

本文尝试给出一个可能的答案：

$$
\boxed{\text{Data Quality} \rightarrow \text{State Learning Utility}}
$$

也就是说：

> OPD 中，一条训练数据是否“高质量”，不应该只看 Query 本身，而应该看它能否让 Student 进入那些**值得学、Teacher 能教、Student 又学得进去的 State**。

这也是本文想讨论的 **Useful State Coverage**。

---

## 1. 先理解 OPD：真正接受监督的并不是 Query

传统 SFT 的训练过程比较直观：

```text
Question
   ↓
Teacher Answer
   ↓
Student 模仿
```

训练数据基本可以表示成：

$$
(x,y)
$$

因此我们很自然地关注：

- 有多少 Question；
- Question 是否多样；
- Answer 是否正确；
- 不同任务覆盖是否充分。

但 OPD 不一样。

它首先让 **Student 自己 rollout**：

$$
y\sim\pi_S(\cdot|x)
$$

假设 Query 是：

> 一个火车 2 小时行驶 120 km，平均速度是多少？

Student 可能生成：

```text
Average speed = distance / time.
120 / 2 = 60 km/h.
```

在这个生成过程中，每一个 prefix 都对应一个 State：

$$
s_t=(x,y_{<t})
$$

例如：

```text
State 1:
Question

State 2:
Question + "Average"

State 3:
Question + "Average speed"

State 4:
Question + "Average speed = distance"

...
```

Teacher 并不是只在最后告诉 Student：

> 答案是 60 km/h。

而是在这些 Student **真实访问到的 State** 上提供 dense token-level supervision：

$$
\pi_T(\cdot|s_t)
$$

所以一个 Query 实际上对应的可能是：

```text
1 Query
   ↓
64 Rollouts
   ↓
大量 Trajectories
   ↓
数万甚至更多 States
   ↓
大量 Teacher Supervision
```

这也是理解后面所有现象的基础。

---

# 2. Paper I：Teacher 有知识，不代表 Student 学得进去

第一篇论文是：

**Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe**  
Yaxuan Li et al., arXiv:2604.13016, 2026.  
https://arxiv.org/abs/2604.13016

这篇论文关注的核心并不是数据，而是：

> **什么决定一次 OPD 能不能成功？**

它得到一个非常重要的结论：

$$
\boxed{
\text{Teacher is stronger}
\neq
\text{Teacher is better for distillation}
}
$$

也就是说，Teacher benchmark 分数更高，并不能保证 Student 就能从它身上学到更多东西。

作者发现 OPD 成功至少依赖两个条件：

第一，Teacher 必须真的拥有 Student 尚未掌握的 **new capabilities**。

第二，Student 和 Teacher 的 **thinking pattern 需要具有一定兼容性**。

在成功的 OPD 中，两者在 Student 实际访问的 State 上，会逐渐对齐一组 high-probability tokens。这些共享 token 数量并不一定很多，却集中了大部分 probability mass。论文还发现，失败的 OPD 可以通过 off-policy cold start、teacher-aligned prompts 等策略改善。

这带来了一个非常重要的认识：

$$
\boxed{
\text{Informative Signal}
\neq
\text{Exploitable Signal}
}
$$

Teacher 和 Student 分布差异很大，不意味着这个监督一定有价值。

例如：

```text
Student:
我认为下一步可能是 A / B / C

Teacher:
我认为完全应该走 Z
```

两者差异可能非常大。

但如果 Student 当前的 representation、reasoning pattern 和 Teacher 完全不兼容，Teacher 给出的方向可能根本无法形成有效梯度。

所以：

> **Teacher 有东西可教，是一回事；Student 能不能学进去，是另一回事。**

---

# 3. Paper II：一个 Query 为什么也能训练？

第二篇：

**Rethinking On-Policy Distillation of Large Language Models II: One Training Example**  
Zixuan Fu et al., arXiv:2609.04172, 2026.  
https://arxiv.org/abs/2609.04172

作者做了一个极端实验：

原本有：

$$
17K\ Queries
$$

现在只留下：

$$
1\ Query
$$

然后不断用同一个 Query rollout、Teacher supervision、update。

直觉上，我们可能认为模型很快就会过拟合。

但结果却是：

> One-query OPD 可以持续训练几百步，并恢复 full-data OPD 的大部分收益。

作者进一步提出了 **State Coverage** 来解释这一现象。

一个 Query 虽然只有一个输入，但 Student 每次 rollout 不完全一样。

因此：

$$
q
\rightarrow
\tau_1,\tau_2,\tau_3,\dots
$$

每条 trajectory 又产生大量：

$$
s_1,s_2,\dots,s_T
$$

所以：

$$
\boxed{
\text{Few Queries}
\neq
\text{Few Training States}
}
$$

论文发现：

- 单个 Query 可以覆盖 full-data OPD State Space 的约 **71.5%**；
- 16 个语义多样的 Query 可以达到约 **98.9% State Coverage**；
- 16-query 的最终效果基本可以追平 full-data training。

这意味着：

> OPD 的有效数据单位，可能不是 Query，而是 Query 所诱导出的 State。

于是我们可以把 Query 重新理解成：

$$
\boxed{\text{Query}=\text{State Generator 的 Seed}}
$$

---

# 4. `<User><think>` 为什么也能工作？

这篇论文还有一个非常反直觉的结果。

甚至使用类似：

```text
<User><think>
```

这种几乎没有具体任务内容的输入，或者一些 off-domain WildChat Query，数学 OPD 仍然能够获得相当不错的收益。

这并不意味着：

> 随便一条垃圾数据都可以训练数学能力。

更合理的解释是：

Student 本身已经是一个经过大量预训练和 post-training 的模型。

开放式 seed：

```text
<User><think>
```

实际上是在告诉模型：

> 自己开始展开 reasoning。

于是模型可能自主生成：

```text
数学推理
→ 中间验证
→ 错误路线
→ 自我纠正
→ 另一种解法
→ ...
```

换句话说：

$$
\text{Content-light Query}
\neq
\text{Content-light States}
$$

它实际上有一点像：

> **让 Student 自己创造用于训练的 trajectory。**

但这里产生的不是新的用户 Query，而是大量新的 **Student-induced States**。

因此，可以把 OPD 看成一种：

$$
\boxed{
\text{Teacher-Guided Self-Exploration}
}
$$

Student 自己探索 State Space，Teacher 在 Student 实际到达的位置不断提供监督。

---

# 5. 但这里存在一个明显的问题：State 多就一定好吗？

假设我们现在有两个 Query。

Query A rollout 后产生 100 种 State：

```text
简单加法
简单减法
重复解释
模板化回答
闲聊
乱码
无意义重复
...
```

Query B 只产生 30 种 State：

```text
几何条件识别
辅助线选择
三角形相似
错误假设
重新规划
证明验证
...
```

如果我们单纯使用：

$$
State\ Coverage
$$

Query A 可能表现得更好。

但从训练价值看，B 很可能明显更高。

因此：

$$
\boxed{
\text{State Coverage}
\neq
\text{Learning Utility}
}
$$

这就是两篇论文结合之后留下的一个关键问题。

---

# 6. 两篇论文其实分别回答了两个相邻的问题

把两篇工作放到一起看：

### Paper I

关注：

> 已经到达一个 State 后，Teacher 的监督 Student 能不能学进去？

即：

$$
\boxed{
\text{Which supervision is learnable?}
}
$$

### Paper II

关注：

> 一组 Query 能让 Student 到达哪些 State？

即：

$$
\boxed{
\text{Which states are visited?}
}
$$

那么一个非常自然的下一步就是：

$$
\boxed{
\text{Which states are worth visiting?}
}
$$

也就是：

> **有限训练预算下，我们到底应该把 Student 引导到哪些 State？**

这就是 Useful State Coverage 的核心问题。

---

# 7. 什么是 Useful State？

一个 State 是否 Useful，至少不能只看 Teacher-Student Gap。

我们可以从五个维度考虑。

## 7.1 Novelty：是不是新的 State？

如果一个 State 已经被看了几万次，再来一次价值很低。

因此我们关心的是：

$$
N(s|Q_{\text{selected}})
$$

即：

> 当前 State 相对于已经选择的数据，能够提供多少新增覆盖？

## 7.2 Information Gain：Teacher 有没有新东西？

假设：

```text
State:
2 + 2 =
```

Student：

$$
P_S(4)=0.99
$$

Teacher：

$$
P_T(4)=0.995
$$

虽然这是一个非常干净、非常正确的 State：

$$
\pi_S\approx\pi_T
$$

说明 Student 已经基本会了。

继续训练价值非常低。

所以需要考虑：

$$
I(s)
$$

表示：

> Teacher 在这里是否真的拥有 Student 尚未掌握的信息。

---

# 8. Exploitability：有东西教，不代表学得进去

这是 Paper I 对 Useful State 定义非常重要的修正。

一开始我们可能会认为：

$$
TeacherStudentGap\uparrow
\Rightarrow
StateUtility\uparrow
$$

但 2604 的结果说明并不是这样。

如果：

$$
\pi_T
$$

和：

$$
\pi_S
$$

差得太远，甚至 thinking pattern 不兼容，这个 supervision 可能很难被 Student 利用。

因此我们还需要：

$$
E(s)
$$

即：

> **Exploitability：当前 Student 是否能够把 Teacher 的监督转换成有效更新。**

可能的 proxy 包括：

- high-probability token overlap；
- overlap probability mass；
- Teacher/Student entropy relationship；
- gradient alignment；
- local update 后 loss 是否真正下降。

因为：

$$
\boxed{
\text{New Information}
\neq
\text{Learnable Information}
}
$$

---

# 9. Reliability：Teacher 在这个 State 上靠谱吗？

OPD 有一个非常特殊的问题：

State 是 **Student 自己产生的**。

随着 rollout 越来越长，Student 可能进入 Teacher 从未见过甚至非常异常的 prefix。

例如：

```text
正确推理
→ 一个错误假设
→ 第二个错误假设
→ 逻辑开始偏离
→ 奇怪上下文
```

这时候即使 Teacher 给出了 probability distribution，也不一定意味着它的 supervision 仍然可靠。

2604 也观察到了 long-horizon 下 Teacher signal 逐渐弱化的问题，并指出这一现象可能影响长链 reasoning 和 Agent 场景。

所以还需要：

$$
L(s)
$$

表示：

> Teacher 在 Student 当前访问的这个 State 上是否仍然可靠。

---

# 10. Task Relevance：这个 State 是我们想训练的吗？

最后还有一个容易被忽略的问题。

如果我们的目标是 Math reasoning，但 rollout 进入：

```text
Write a romantic poem about the moon.
```

即使：

- Teacher 很确定；
- Teacher 和 Student 差异很大；
- State 很新；

它依然不一定值得用有限训练预算学习。

因此还需要：

$$
R(s)
$$

表示目标任务相关性。

---

# 11. 所以“数据质量”应该是动态的

传统数据工程很容易认为：

> 一道好题永远是一道好题。

例如：

$$
Quality(q)
=
Difficulty + Diversity + Correctness
$$

但对于 OPD，我们认为更合理的是：

$$
\boxed{
Quality
\left(
q\mid
\pi_S,\pi_T,Q_{\mathrm{selected}},\mathcal T
\right)
}
$$

这意味着 Query 的质量取决于：

- 当前 Student；
- 当前 Teacher；
- 已经选过哪些数据；
- 当前目标能力。

一条几何题，在训练早期可能非常有价值。

但经过 500 step 后：

$$
\pi_S\approx\pi_T
$$

那么它就可能不再是高质量数据。

所以：

> **高质量数据不是静态属性，而是 Student-dependent 的动态属性。**

---

# 12. Useful State Coverage

基于上面的讨论，我们可以把 State 的价值概念化为：

$$
U(s)
=
f
\left(
N(s),
I(s),
E(s),
L(s),
R(s)
\right)
$$

其中：

- $N$：Novelty
- $I$：Information Gain
- $E$：Exploitability
- $L$：Teacher Reliability
- $R$：Task Relevance

注意，这里不应该简单地写成：

$$
N\times I\times E\times L\times R
$$

因为这些指标很可能不是同一量纲，而且有一些更适合当 **Gate**。

例如：

```text
Task Relevance 太低
        ↓
直接过滤

Teacher Reliability 太低
        ↓
直接过滤

Exploitability 太低
        ↓
暂时不训练
```

然后只对通过 Gate 的 State 计算：

$$
\text{Information Gain}
+
\text{Marginal Novelty}
$$

因此，一个更合理的设计可能是：

### Stage 1：State Filtering

$$
\mathcal S_{\mathrm{valid}}
=
\{
s:
R(s)\ge\tau_R,
L(s)\ge\tau_L,
E(s)\ge\tau_E
\}
$$

### Stage 2：Utility Ranking

在剩余 State 中：

$$
U(s)=f(I(s),N(s))
$$

再根据 State Utility 反过来评价 Query。

---

# 13. 训练之前怎么知道 Query 有没有价值？

这里还有一个现实问题：

> Query 不 rollout，我们根本不知道它会产生什么 State。

所以完全静态地看 Query 文本来判断，是不够的。

但我们不需要完整训练。

可以先做一次非常便宜的 **Pilot Rollout**。

假设我们有：

$$
10,000
$$

条候选 Query。

冻结初始 Student：

$$
\pi_{S_0}
$$

每个 Query 只 rollout：

$$
K=2\sim8
$$

次。

得到：

$$
q_i
\rightarrow
S_i^{pilot}
$$

然后对这些 State 计算：

```text
Novelty
Information Gain
Exploitability
Teacher Reliability
Task Relevance
```

再估计：

$$
Value(q_i)
$$

整个过程可以表示成：

```text
Candidate Query Pool
        │
        ▼
Frozen Student
        │
        ▼
Few Pilot Rollouts
        │
        ▼
Induced States
        │
        ├── New?
        ├── Informative?
        ├── Learnable?
        ├── Reliable?
        └── Relevant?
        │
        ▼
Useful State Utility
        │
        ▼
Query Selection
        │
        ▼
Full OPD
```

可以把它称作：

> **Training Before Training**

不是直接开始优化 Student，而是先探索：

> 哪些数据真正值得花训练预算？

---

# 14. 数据选择应该最大化“边际 Useful Coverage”

还有一个重要问题。

假设 Query A 和 Query B 都很好。

但它们产生的 State 几乎一模一样。

那么：

$$
Value(A)>0
$$

并不能推出：

$$
Value(B|A)>0
$$

数据价值应该是 **条件性的**：

$$
Value
\left(
q\mid Q_{\mathrm{selected}}
\right)
$$

因此真正优化的是：

$$
\boxed{
\Delta\text{Useful State Coverage}
}
$$

例如：

```text
Q1:
代数 reasoning
↓
选中

Q2:
还是大量代数 reasoning
↓
与 Q1 重复
↓
价值降低

Q3:
几何 + proof verification
↓
大量新 State
↓
选中

Q4:
错误恢复 + self-correction
↓
新的高价值 State Region
↓
选中
```

这已经非常接近一种 **State-Space Active Learning**。

---

# 15. 更进一步：Useful State 还应该动态变化

假设训练刚开始：

$$
I_0(s)=0.8
$$

说明 Student 和 Teacher 差异明显。

训练一段时间以后：

$$
I_t(s)=0.05
$$

说明 Student 已经学会。

此时继续对这个 State 投入 rollout 和 Teacher inference，就比较浪费。

因此数据选择可以变成：

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

即：

$$
\boxed{
Explore
\rightarrow
Learn
\rightarrow
Re\text{-}score
\rightarrow
Explore
}
$$

最终的数据集甚至不再是固定的 Dataset，而更像：

> 随着 Student 能力变化而不断变化的 **curriculum**。

---

# 16. 怎么验证这个想法？

最重要的实验并不是：

> “我们的方法比 Random 高 2 个点。”

而是首先验证一个更基础的问题：

## State Coverage 真的是 Training Utility 吗？

可以人为构造三组 Query。

### Easy-Diverse

产生很多不同 State：

$$
Coverage\uparrow
$$

但 Student 基本已经掌握：

$$
InformationGain\downarrow
$$

### Hard-Unlearnable

Teacher 和 Student 差异很大：

$$
Gap\uparrow
$$

但两者 thinking pattern 不兼容：

$$
Exploitability\downarrow
$$

### Useful-Diverse

同时满足：

$$
Novelty\uparrow
$$

$$
InformationGain\uparrow
$$

$$
Exploitability\uparrow
$$

然后比较下面这些指标，谁最能预测真实 OPD validation gain：

```text
Query Difficulty
Query Semantic Diversity
Raw State Coverage
Teacher-Student Gap
High-probability Token Compatibility
Useful State Score
```

如果最终发现：

$$
corr(USC,\Delta Performance)
>
corr(StateCoverage,\Delta Performance)
$$

那么就说明：

> **State Coverage 是必要视角，但不是最终的数据质量指标。**

---

# 17. 这个问题在 Agent 场景可能更加重要

对于数学 reasoning，State 主要是：

```text
中间推理
公式选择
错误假设
验证
self-correction
```

但 Agent 的 State Space 会复杂得多：

```text
正常工具调用
工具调用失败
权限不足
API timeout
返回结果 malformed
工具选择错误
规划失败
重新规划
信息不足
memory conflict
环境变化
```

传统 Agent 数据集可能强调：

> 我们有 10,000 个 Task。

但真正重要的也许不是：

$$
Task\ Coverage
$$

而是：

$$
\boxed{
Agent\ State\ Coverage
}
$$

甚至进一步是：

$$
\boxed{
Useful\ Agent\ State\ Coverage
}
$$

因为 1000 个 Task 如果全部都是：

```text
正常调用工具
→ 正常返回
→ 正常回答
```

可能远不如几十个能触发：

```text
failure
→ diagnosis
→ replan
→ retry
→ recover
```

的 Task 有价值。

---

# 18. 从两篇论文到一个新的研究问题

把整个逻辑压缩下来：

### Paper I

$$
\boxed{
\text{到了这个 State，Student 学得进去吗？}
}
$$

### Paper II

$$
\boxed{
\text{这些 Query 能带 Student 到哪些 State？}
}
$$

### Next Question

$$
\boxed{
\text{哪些 State 值得我们主动让 Student 去？}
}
$$

所以真正的数据选择目标可能不再是：

$$
\max \text{Query Diversity}
$$

也不只是：

$$
\max \text{State Coverage}
$$

而是：

$$
\boxed{
\max \text{Useful State Coverage}
}
$$

---

# 结语

*One Training Example* 最吸引人的地方可能是：

> 只用一条 Query 也能训练。

但它真正重要的意义不是：

> **以后不需要数据了。**

而是迫使我们重新思考：

> **到底什么才算训练数据？**

在传统 SFT 中，我们以 Query/Response 为单位。

在 OPD 中，一个 Query 更像一个 State Generator。

于是数据工程的问题从：

> “我要准备多少高质量 Query？”

转变成：

> “这些 Query 会把模型带到哪里？”

而结合第一篇关于 Teacher-Student compatibility 的研究之后，还需要再问一步：

> “到了那里以后，Teacher 是否真的有东西可以教？Student 是否真的学得进去？”

因此，更合理的 OPD 数据质量定义可能是：

$$
\boxed{
\text{Data Quality}
\rightarrow
\text{State Learning Utility}
}
$$

未来 OPD 的核心数据工程，也许不再是构建一个越来越大的静态 Dataset，而是：

> **用最少的 Query 和 rollout budget，让 Student 主动探索那些新的、相关的、Teacher 可靠且 Student 可学习的 State。**

如果这个方向成立，那么 OPD 的“数据选择”，本质上将从 **Query Selection** 转变成一种 **State-Space Exploration and Curriculum Design**。

而这可能比“一个训练样本就够了”本身，更值得继续研究。

---

## 参考论文

1. Yaxuan Li et al. **Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe.** arXiv:2604.13016, 2026.  
   https://arxiv.org/abs/2604.13016
2. Zixuan Fu et al. **Rethinking On-Policy Distillation of Large Language Models II: One Training Example.** arXiv:2609.04172, 2026.  
   https://arxiv.org/abs/2609.04172
