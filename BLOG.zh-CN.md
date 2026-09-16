# 一个训练样本真的够吗？我从两篇 OPD 论文里想到的一个问题

最近看了两篇 On-Policy Distillation（OPD）的论文，我一直在想一个问题：我们过去说“高质量数据”的时候，到底是在说什么？

这个问题是被一个很反常识的实验勾起来的。

在 *Rethinking On-Policy Distillation of Large Language Models II: One Training Example* 里，作者把原本上万条数学训练 query 缩到只剩 **1 条**，然后继续做 OPD。按照常见直觉，这应该很快过拟合：模型反复看到同一个问题，还能学到什么？

但结果并不是这样。单个 query 依然能支撑数百步训练，并恢复 full-data OPD 的大部分收益。论文进一步观察到，这一个 query 在 rollout 过程中访问到的 state，竟然覆盖了 full-data OPD state space 的约 **71.5%**；换成 16 个语义多样的 query，coverage 可以到约 **98.9%**，性能也基本追平 full-data。

我第一次看到这个结果时，最直接的反应其实不是“以后只要一条数据了”，而是：**也许我们一直把训练数据的单位看错了。**

## Query 可能只是入口

在 SFT 里，我们很自然地把一条数据理解成 Question → Answer。于是做数据工程时，会关心题目够不够多、领域够不够广、难度够不够丰富、答案是否正确。

但 OPD 的训练过程不太一样。Student 先自己 rollout：

$$
y \sim \pi_S(\cdot|x)
$$

Teacher 再在 Student 实际生成的 prefix 上提供 token-level supervision。也就是说，每一个 prefix 都可以看成一个 state：

$$
s_t=(x,y_{<t})
$$

所以同一个 query，不是只对应一个训练样本。它可能产生几十条 rollout，每条 rollout 又包含很多中间 state。换句话说，query 更像一个起点，真正大量产生训练信号的是后面的 state。

这也是为什么“一条 query”并不等于“只有一个训练状态”。

我觉得 *One Training Example* 真正有意思的地方就在这里。它把注意力从 query 数量移到了 **Student 实际访问了哪些 state**。

不过，顺着这个解释往下想，我又觉得还差一步。

## State 多，就一定好吗？

假设两个 query 最后都产生了很多 state。

第一个 query 反复把模型带到简单算术、模板化解释、重复推理这些区域；第二个 query 虽然覆盖的 state 数量少一些，但里面包含几何证明中的错误路线、重新规划、验证和纠错。

如果只看 raw state coverage，第一个 query 未必差。但直觉上，第二类 state 可能更值得花训练预算。

再举一个更极端的例子。假设当前 state 是：

```text
2 + 2 = ?
```

Student 对 4 的概率已经是 0.99，Teacher 是 0.995。这个 state 很“正确”，也很干净，但 Student 基本已经会了。继续在这里训练，能学到的东西很有限。

反过来，如果 Student 在一道几何证明中明显走错了，而 Teacher 在当前步骤有一个很清晰、Student 尚未掌握的方向，这个 state 的训练价值就可能高很多。

还有第三种情况：Student rollout 已经跑偏到一段非常奇怪的上下文，Teacher 和 Student 的分布差异可能很大，但这个差异未必对目标任务有任何帮助。

所以我现在更愿意把 State Coverage 理解成一个**必要但还不够的统计量**。它告诉我们“模型去了哪里”，却没有告诉我们“这些地方值不值得学”。

这时，另一篇 OPD 工作就变得很关键。

## Teacher 有东西教，不代表 Student 学得进去

*Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe* 研究的是另一个问题：为什么有些 OPD 很成功，有些却不行？

这篇论文里有一个我觉得很重要的观察：**Teacher 更强，并不等于它一定更适合作为 Teacher。**

作者发现，成功的 OPD 不只是要求 Teacher 比 Student “知道得更多”，还要求两者在局部的 reasoning / token distribution 上有一定兼容性。换句话说，Teacher 给出的信号必须是 Student 能够利用的。

这会直接打破一个很自然的想法：

$$
\text{Teacher-Student Gap 大} \Rightarrow \text{训练价值高}
$$

这个推论并不总成立。

Gap 很大，有可能说明 Teacher 确实知道 Student 不会的东西；也可能意味着两者已经处在完全不同的局部 policy 区域，Teacher 的监督虽然“信息很多”，但 Student 根本吸收不了。

所以我觉得两篇论文刚好拼成了两半。

2609 那篇在问：**这些 query 会把 Student 带到哪些 state？**

2604 那篇在问：**到了这些 state 以后，Teacher 的监督 Student 能不能学进去？**

把两者放在一起，一个很自然的问题就是：

> 哪些 state 才真正值得 Student 去访问、去学习？

我暂时把这个问题叫做 **Useful State Coverage**。

## 什么样的 state 才算 useful？

这里我还不想太早给一个“漂亮公式”。因为如果直接写成五个指标相乘，很容易看起来完整，但其实很多东西还没有被验证。

更直观一点，我会先问几个问题。

第一，这个 state 是不是新的？如果 Student 已经在类似区域来回访问了很多次，再多一次意义可能不大。

第二，Teacher 在这里有没有 Student 还不会的东西？如果两者几乎已经一致，就没有多少 learning signal。

第三，即使 Teacher 有新信息，Student 能不能学进去？这正是 2604 那篇提醒我们的地方。Information 不等于 exploitable information。

第四，Teacher 在这个 state 上靠不靠谱？OPD 的 state 是 Student 自己生成的。rollout 越长，Student 越可能进入奇怪、低概率、甚至 off-manifold 的 prefix。Teacher 在这些位置给出的信号不一定和正常分布下同样可靠。

最后，这个 state 和我们想提升的能力有没有关系？如果目标是数学推理，一个非常新、Teacher 也很确定的闲聊 state，价值仍然可能接近零。

所以 Useful State 至少和下面这些因素有关：

- Novelty：是否带来新的 state region；
- Learning potential：Teacher 是否还有东西可教；
- Exploitability：Student 是否能利用这个监督；
- Teacher reliability：Teacher 在这个 prefix 上是否可信；
- Task relevance：是否与目标能力相关。

可以先把它写成一个很宽松的概念：

$$
U(s)=f(N(s), I(s), E(s), L(s), R(s))
$$

但这里真正重要的可能不是这个公式，而是另外一点：**Useful State Coverage 仍然是 coverage。**

如果一个很有价值的 state region 已经被访问了 100 次，第 101 次显然不应该和第一次一样值钱。我们真正关心的是“又覆盖了多少新的、有学习价值的区域”，而不是把所有 state 分数简单相加。

这也是为什么我更喜欢 “Useful State Coverage” 这个说法，而不是 “State Utility Score”。

## 更难的问题其实是：怎么在训练前找到这些 state？

定义只是第一步。真正麻烦的是，state 不是静态存在的。

同一条 query，给不同 Student，或者给同一个 Student 在训练的不同阶段，产生的 rollout 都可能不一样。于是 query 的价值天然是 Student-dependent 的。

一条题今天可能非常有价值，因为它能把 Student 带到大量“不会但能学”的区域；训练几百步以后，这些 state 已经学会了，它的价值自然会下降。

所以我现在越来越不喜欢这种静态表达：

```text
这是一条高质量数据。
```

在 OPD 里，更准确的说法可能是：

$$
\mathrm{Quality}(q\mid \pi_S,\pi_T,Q_{selected},\mathcal T)
$$

也就是说，一条 query 好不好，要看当前 Student、Teacher、已经覆盖过哪些区域，以及我们想提升什么能力。

那训练之前怎么判断？

我觉得一个比较自然的办法是 **Pilot Rollout**。

假设我们手里有一万个候选 query。先不要全部拿去正式训练，而是冻结当前 Student，每条 query 只跑很少几次，例如 2～8 条 rollout。这样成本还可控，但已经能看到每条 query 大概会把 Student 带到什么地方。

然后再在这些 candidate states 上估计几个信号：Student 和 Teacher 差多少、两者高概率 token 是否兼容、Teacher 是否稳定、state 是否和目标任务相关、是否和已经选过的 state 重复。

到这里，数据选择就不再是“从 1 万条 query 里挑看起来最难或最不一样的那些”，而变成：

> 哪些 query 最可能把当前 Student 带到新的、而且真正值得学习的区域？

我很喜欢把这一步叫做 **Training Before Training**。不是因为真的提前训练了一遍，而是正式投入训练预算之前，先侦察一下 state space。

## 从选 Query，变成设计 State-Space Curriculum

再往前一步，query selection 本身也应该是有条件的。

假设 Q1 已经覆盖了一大片代数 reasoning state，那么另一条 query Q2 即使单独看也很“好”，如果它产生的 state 和 Q1 几乎一样，边际价值就不高。

真正想最大化的其实是：加入一条新 query 后，**新增了多少 Useful State Coverage**。

可以写成：

$$
\Delta USC(q)
=
USC(Q_{selected}\cup\{q\})-USC(Q_{selected})
$$

这个想法一旦成立，数据集就很难再是固定的。

Student 学会一批 state 之后，原来 useful 的区域会逐渐饱和；这时应该重新 rollout、重新评估，再去找新的 state。最后形成的更像一个循环：探索一些 state，学一段，再重新看哪里还有东西可学。

这已经不像传统的数据筛选，更像一种 **state-space curriculum**。

## 为什么我觉得 Agent 场景可能更有意思

数学推理里的 state 还相对规整，通常是中间步骤、错误假设、验证、自我纠正这些东西。

Agent 的 state space 会复杂得多。一次任务里，模型可能遇到工具调用失败、权限不足、API timeout、返回格式异常、规划错误、重试、replan、memory conflict，甚至环境状态已经变化。

这时，“我们有多少任务”可能就更加不是一个好的数据质量指标。

一千个任务如果全部都是“调用成功 → 返回正常 → 给出答案”，覆盖的 agent behavior 可能非常窄；反而几十个任务，如果能稳定触发 failure、diagnosis、replan、retry 和 recovery，可能会提供更有价值的训练 state。

所以我觉得 Useful State Coverage 不一定只适用于数学 OPD。它更像是在问一个更一般的问题：**训练一个会行动的模型时，我们究竟希望它经历什么？**

## 最后

*One Training Example* 最吸引人的标题当然是“一条训练样本也够”。但我现在觉得，它更有价值的地方，是逼着我们重新想“数据”到底是什么。

在 on-policy learning 里，query 也许只是入口。真正决定模型学到什么的，是 rollout 把它带到了哪些 state，以及这些 state 上有没有它还不会、Teacher 又确实能教会它的东西。

所以，比“我们还需要多少高质量数据”更让我感兴趣的问题是：

> **怎样用尽可能少的探索预算，把模型带到最值得学习的地方？**

我现在还不知道 Useful State Coverage 最终应该怎样定义，也不确定哪些 proxy 最可靠。Teacher-Student gap、token overlap、verifier、hidden-state clustering，这些都可能只解释其中一部分。

但这恰恰是我觉得值得继续做实验的地方。

---

## References

1. Yaxuan Li et al. **Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe.** arXiv:2604.13016, 2026.  
   https://arxiv.org/abs/2604.13016
2. Zixuan Fu et al. **Rethinking On-Policy Distillation of Large Language Models II: One Training Example.** arXiv:2609.04172, 2026.  
   https://arxiv.org/abs/2609.04172
