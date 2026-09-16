# One Query, Many States: Rethinking Data Quality in On-Policy Distillation

### A working hypothesis about Useful State Coverage, data selection, and dynamic curricula

I have been thinking about OPD data through a **state-space** lens.

This is not a finished theory, and I do not think we have a rigorous definition of a “Useful State” yet. It is better thought of as a working mental model: if On-Policy Distillation supervises the states that the Student actually visits, then maybe data quality should not be judged only by the input query. We should also ask where those queries take the Student.

This view is motivated by two recent papers.

The first is [*Rethinking On-Policy Distillation of Large Language Models II: One Training Example*](https://arxiv.org/abs/2609.04172). It asks an extreme question: **what happens if OPD is trained on only one query?**

The second is [*Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe*](https://arxiv.org/abs/2604.13016). It asks a different question: **why do some Teachers transfer effectively while other, seemingly stronger Teachers do not?**

Taken together, they suggest a question that I find more interesting than either result in isolation:

> If query count is not the right unit of data, and Teacher–Student disagreement is not the same as learnable supervision, what actually makes an OPD example valuable?

Below is my current answer.

---

## A state-space view of OPD

Start with the training process.

In SFT, we usually think of one training example as:

```text
prompt → reference response
```

The data exists before training. We can inspect it directly and ask whether prompts are diverse, responses are correct, domains are covered, and examples are difficult enough.

OPD is different. The Student first rolls out from its current policy:

$$ y \sim \pi_S(\cdot \mid x) $$

The Teacher then provides token-level supervision on the prefixes that the Student actually generates. At token position $t$, the relevant state is:

$$ s_t = (x, y_{1:t-1}) $$

So the objects receiving supervision are not just the original queries. They are the states encountered along Student-generated trajectories.

A query is therefore closer to an entry point:

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

That sounds like a small change in terminology, but I think it changes how we should think about “how much data” OPD is using.

One query is not one training state. One query can produce many rollouts, and every rollout contains many prefixes. As long as the Student samples stochastically, the same query can move the model into different local states.

This is exactly what makes *One Training Example* so surprising.

The paper shows that one-query OPD can train for hundreds of steps and recover a large fraction of the gain from full-data OPD. More importantly, when the authors cluster visited states using Teacher hidden representations, they find that **a single query reaches about 71.5% of the state-space coverage of full-data OPD; 16 semantically diverse queries reach about 98.9%, while matching full-data performance closely.**

To me, the strongest implication is not that “one example is enough.” It is that:

> **Query count may be a poor proxy for the effective amount of OPD training data.**

If one query can keep generating new states, then the query is more like a seed than the final unit of learning.

---

## One training example is not one training state

This distinction also helps explain another unusual result in the paper.

The authors test content-light templates and even some off-domain WildChat prompts. Certain seeds that look only weakly related to mathematics can still produce OPD gains close to those from genuine math queries.

That does **not** mean the query content is irrelevant, and it certainly does not mean arbitrary garbage text is a good math training set.

A more plausible interpretation is that **query semantics and the induced state distribution are different objects**.

A weakly constrained seed can still cause an already capable Student to unfold mathematical reasoning, self-correction, alternative branches, and intermediate verification. If the resulting rollout visits states that matter for the target capability, the Teacher can still provide useful supervision there.

So perhaps the better data-selection question is not:

> What category does this query belong to?

but:

> **Where does this query take the current Student?**

This is why I find State Coverage useful. It moves the idea of “coverage” from query space toward model-induced state space.

But I do not think we can stop there.

---

## Where State Coverage falls short

State Coverage answers: **where did the Student go?**

It does not directly answer: **was there anything worth learning there?**

Consider three states.

| State | Student / Teacher situation | Intuitive training value |
| --- | --- | --- |
| `2 + 2 = ?` | Student already assigns `0.99` to `4`; Teacher assigns `0.995` | Low: the Student essentially knows it |
| A geometry proof has entered a wrong branch | Student is uncertain; Teacher strongly prefers a corrective next step | Potentially high: there is new, actionable supervision |
| A rollout has drifted into malformed or nonsensical context | Teacher and Student distributions may differ dramatically | Unclear, possibly useless |

All three states count toward raw coverage. But they clearly should not count equally toward data quality.

So I would treat State Coverage as an important first step, not as the final objective. It tells us which regions were visited. We still need to ask whether those regions contain useful learning opportunities.

This is where the other OPD paper becomes important.

---

## The Student has to be able to use the Teacher signal

[*Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe*](https://arxiv.org/abs/2604.13016) studies when OPD succeeds and when it fails.

One of its most useful observations is that **a stronger Teacher is not necessarily a better Teacher for a given Student**.

Successful OPD appears to depend on at least two things:

1. the Teacher must actually contain capabilities that the Student has not yet acquired;
2. the Teacher and Student need enough compatibility in their local reasoning / token distributions for the supervision to be usable.

At the token level, successful OPD increasingly aligns a relatively small set of high-probability tokens on Student-visited states. Those shared tokens account for roughly 97–99% of the probability mass.

This makes me skeptical of using Teacher–Student divergence alone as a state-value metric.

A tempting heuristic is:

$$ D(\pi_T,\pi_S) \text{ is large} \Rightarrow \text{the Student has a lot to learn} $$

But that is not always true.

A large gap can mean the Teacher knows something the Student does not. It can also mean the two policies are locally so misaligned that the dense Teacher signal is difficult for the Student to absorb.

So I think we need to separate two questions:

**Is there new information here?**

and

**Can the Student actually use it?**

These are not the same thing.

---

## What I mean by a “Useful State”

Putting the two papers together, I would currently define a Useful State as a **student-dependent learning opportunity**.

The important part is student-dependent.

A state can be valuable for a weak Student and nearly worthless for a Student that has already mastered that region. The value can also change if we swap the Teacher, the target task, or the set of states already covered during training.

I currently think about five signals:

| Signal | Question | Possible proxy |
| --- | --- | --- |
| **Novelty** | Has this region already been visited many times? | hidden-state distance, cluster coverage |
| **Information** | Does the Teacher know something the Student does not? | JS/KL gap, top-k log-prob gap |
| **Exploitability** | Can the Student use this supervision? | high-probability token overlap, local gradient alignment |
| **Reliability** | Is the Teacher trustworthy on this Student-induced state? | verifier, self-consistency, multi-teacher agreement |
| **Relevance** | Does this state matter for the target capability? | target-state similarity, domain classifier, task verifier |

If I had to write an abstract placeholder, it would look like:

$$ U(s) = f\!\left(N(s), I(s), E(s), L(s), R(s)\right) $$

But I do **not** think simply multiplying or summing these five quantities gives us the right metric.

Some of them are probably better treated as gates rather than scores. If a state is clearly off-task, or the Teacher is unreliable there, then a large Teacher–Student gap should not automatically make it valuable.

A more reasonable design might first filter on basic relevance, reliability, and learnability, then rank the remaining states by how much new information and marginal coverage they provide.

This is one of the places where the idea needs experiments more than another layer of notation.

---

## How would we find Useful States before training?

Defining Useful States is only half of the problem.

The harder half is operational: **before spending the full OPD budget, how do we know whether a candidate query will produce useful states?**

This is difficult because the state does not exist independently of the Student. It is induced by both the query and the current policy:

$$ s \sim P(s \mid q, \pi_S) $$

That makes purely static query scoring questionable. Looking only at query embeddings, semantic labels, or difficulty cannot tell us which trajectories the Student will actually enter.

The most direct approach I can think of is a small **pilot rollout**.

Suppose we have a candidate pool. Freeze the current Student, run only a few rollouts for each candidate query, and do not update the model. Then probe the visited states for the signals above.

To make the setup concrete, it helps to separate a few components:

- **Candidate query**: a query not yet selected for full training;
- **Pilot rollout**: a small number of trajectories sampled from a frozen Student;
- **State probe**: measurements such as Teacher–Student gap, compatibility, reliability, and relevance;
- **State region**: a cluster of behaviorally or representationally similar states;
- **Selector**: a policy that chooses queries based on the new useful coverage they are expected to add.

The pipeline would look roughly like this:

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

I think of this as **probe before train**.

There is an obvious catch: pilot rollout is not free. It still consumes Student sampling and Teacher inference. If probing costs almost as much as full training, then the selection mechanism defeats its own purpose.

So one of the key empirical questions is whether a very small pilot budget—say 2, 4, or 8 rollouts—can predict the value of a query under full OPD.

That assumption should be tested, not taken for granted.

---

## Useful State Coverage, not a sum of state scores

Even if we can estimate the utility of individual states, simply summing those scores would still be wrong.

Repeated visits to the same type of state should have diminishing value.

If a query sends the Student into nearly identical geometry-proof states one hundred times, that should not count as one hundred independent high-value examples.

So the “coverage” part still matters.

One simple abstraction is to cluster states into regions $c \in \mathcal{C}$, assign each region a learning value $w_c$, and use a saturating coverage function:

$$ \mathrm{USC}(Q) = \sum_{c \in \mathcal{C}} w_c\,g\!\left(n_c(Q)\right) $$

Here, $n_c(Q)$ is the number of times the selected query set $Q$ visits region $c$, and $g$ is a function with diminishing returns.

The intuition is straightforward: the first visit to a new high-value region matters a lot; the tenth still matters somewhat; the hundredth is mostly repetition.

This distinction is useful:

- **State Utility**: how much learning opportunity exists for the current Student at one state;
- **Useful State Coverage**: under a fixed rollout budget, how much *distinct high-value learning space* a set of queries covers.

The second quantity is closer to what a data-selection objective should optimize.

---

## Data quality becomes conditional

This perspective changes something else: **“high-quality query” stops being a static label.**

Traditional curation often assigns relatively stable attributes to an example: difficulty, correctness, domain, quality score, duplication.

For OPD, query value looks conditional:

$$ \mathrm{Quality}\!\left(q \mid \pi_S, \pi_T, Q_{\mathrm{selected}}, \mathcal{T}\right) $$

The same query can have very different value at different points in training.

Early on, it may repeatedly induce states where the Student is weak and the Teacher can help. Later, those regions may already be absorbed; the Teacher–Student gap shrinks and additional rollouts become redundant.

That suggests data selection should also be dynamic:

```text
probe → select → train → re-probe → re-select
```

rather than choosing one static dataset before training and never revisiting the decision.

Under this view, OPD data selection starts to look like **state-space curriculum learning**: as the Student changes, the next useful region to explore changes as well.

---

## From query selection to state-space exploration

Suppose we have already selected a query set $Q$. The value of a new query $q$ should not be its standalone score. It should be the **marginal useful coverage** it adds:

$$ \Delta \mathrm{USC}(q) = \mathrm{USC}\!\left(Q \cup \{q\}\right) - \mathrm{USC}(Q) $$

This matters because semantic diversity at the query level can be misleading.

Two queries can look very different in text space and still drive the Student into nearly identical reasoning states. Conversely, two queries that look similar may trigger very different error branches, recovery behaviors, or reasoning paths.

So I am not convinced that query semantic diversity is the right objective for OPD data selection.

A more direct objective may be:

> **Given a fixed sampling and Teacher-compute budget, how can we make the current Student visit as many high-value, non-redundant state regions as possible?**

At that point, the problem starts to look less like static dataset curation and more like an exploration problem.

---

## Why this may matter even more for agents

Mathematical reasoning is a relatively clean setting because a state is mostly a reasoning prefix.

Agent systems have a much richer state space.

A single task may go through:

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

If we measure coverage only by task count, two datasets may both contain 1,000 tasks while exposing the Agent to completely different state distributions.

One dataset might consist almost entirely of:

```text
correct plan → successful tool call → final answer
```

Another, even with fewer tasks, might heavily cover:

```text
tool failure
partial observation
wrong action
recovery
memory conflict
replanning
```

From a learning perspective, the second dataset may contain many more valuable states near the Agent’s actual capability boundary.

That is why I do not think Useful State Coverage is necessarily limited to math OPD. It points to a broader on-policy learning question:

> **What kinds of states does the model need to experience in order to improve where it is currently weak?**

---

## What would convince me this idea is useful?

The main risk is that “Useful State Coverage” becomes a concept that sounds reasonable but cannot be falsified.

So I think the right next step is not a more complicated formula. It is a small set of direct experiments.

**First**, test whether raw State Coverage is actually insufficient. Find query sets with similar state coverage but different Teacher–Student gap, compatibility, or task relevance, and see whether downstream OPD gain differs substantially.

**Second**, test whether pilot rollout predicts true training value. For each candidate query, run only 2, 4, or 8 pilot rollouts, estimate useful coverage, then run actual one-query OPD and measure the correlation with downstream performance gain.

**Third**, evaluate selection under a fixed budget. Compare Random, Semantic Diversity, Raw State Coverage, Gap-only, and Useful-State-aware selection while controlling query count, Student rollout tokens, and Teacher inference tokens.

If raw State Coverage already explains the gains and the additional signals do not improve prediction or selection, then this idea probably does not buy us much.

But if two query sets have similar raw coverage and the difference is explained by whether the Student can actually learn from those states, then Useful State Coverage starts to become a meaningful object rather than just a new name.

---

## Closing thoughts

What *One Training Example* changed for me is not the belief that “one query is enough.”

It changed the question of what counts as training data in OPD.

The query clearly matters, but it may function more like an entry point into a state space. The dense supervision is attached to the states the Student actually reaches.

State Coverage pushes the discussion forward: instead of asking only how many queries we have, we ask how much of the state space the Student has visited.

If state value is highly non-uniform, the next question is unavoidable:

> **Which regions are worth visiting?**

And the practical follow-up is:

> **Can we tell, before spending the full training budget, which queries are likely to take the Student there?**

I think those questions are more important than committing too early to any particular Useful State formula.

If the idea holds up, OPD data engineering may gradually shift from static query selection toward dynamic state-space exploration: identify where the current Student can still learn, route it toward those regions, absorb the signal, then move on.

That is what I currently mean by **Useful State Coverage**.

---

## References

1. Yaxuan Li et al. **Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe.** arXiv:2604.13016, 2026.  
   https://arxiv.org/abs/2604.13016
2. Zixuan Fu et al. **Rethinking On-Policy Distillation of Large Language Models II: One Training Example.** arXiv:2609.04172, 2026.  
   https://arxiv.org/abs/2609.04172
