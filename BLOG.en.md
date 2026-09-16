# One Query, Many States: Rethinking Data Quality in On-Policy Distillation

### A working hypothesis about Useful State Coverage, data selection, and dynamic curricula

I have been thinking about OPD data through a **state-space** lens.

The working mental model is simple: if On-Policy Distillation supervises the states that the Student actually visits, then data quality should reflect not only the input query, but also where that query takes the Student.

This view is motivated by two recent papers.

The first is [*Rethinking On-Policy Distillation of Large Language Models II: One Training Example*](https://arxiv.org/abs/2609.04172). It asks an extreme question: **what happens if OPD is trained on only one query?**

The second is [*Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe*](https://arxiv.org/abs/2604.13016). It asks a different question: **why do some Teachers transfer effectively while other, seemingly stronger Teachers do not?**

Taken together, they suggest a broader question:

> If a small number of queries can induce many states, and effective distillation depends on whether the Student can use the Teacher signal, what actually makes an OPD example valuable?

Below is my current answer.

---

## A state-space view of OPD

Start with the training process.

In SFT, we usually think of one training example as:

```text
prompt → reference response
```

The data exists before training. We can inspect it directly and ask whether prompts are diverse, responses are correct, domains are covered, and examples are difficult enough.

OPD is more dynamic. The Student first rolls out from its current policy:

$$ y \sim \pi_S(\cdot \mid x) $$

The Teacher then provides token-level supervision on the prefixes that the Student actually generates. At token position $t$, the relevant state is:

$$ s_t = (x, y_{1:t-1}) $$

A query is therefore closer to an entry point into a state space:

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

That small shift in perspective changes how we should think about the amount of training data in OPD.

A single query can produce many rollouts, and every rollout contains many prefixes. With stochastic sampling, the same query can move the Student into different local states.

This is what makes *One Training Example* so surprising.

The paper shows that one-query OPD can train for hundreds of steps and recover a large fraction of the gain from full-data OPD. More importantly, when the authors cluster visited states using Teacher hidden representations, they find that **a single query reaches about 71.5% of the state-space coverage of full-data OPD; 16 semantically diverse queries reach about 98.9%, while matching full-data performance closely.**

To me, the strongest implication is:

> **State-space coverage may be a better proxy for effective OPD data volume than query count alone.**

The query acts more like a seed; the continuing learning opportunities come from the states the Student visits afterward.

---

## One training example is not one training state

This is one contrast worth keeping: **one training example is not one training state**.

The authors also test content-light templates and even some off-domain WildChat prompts. Certain seeds that look only weakly related to mathematics can still produce OPD gains close to those from genuine math queries.

A useful interpretation is that **query semantics and the induced state distribution live at different levels**.

A weakly constrained seed can cause an already capable Student to unfold mathematical reasoning, self-correction, alternative branches, and intermediate verification. If the resulting rollout visits states that matter for the target capability, the Teacher can provide useful supervision there.

So a more informative data-selection question is:

> **Where does this query take the current Student?**

That is why I find State Coverage useful. It moves the idea of coverage from query space toward model-induced state space.

The next step is to ask how much learning value those visited states actually carry.

---

## Where State Coverage falls short

State Coverage answers: **where did the Student go?**

Data quality also depends on: **how much was there to learn in those regions?**

Consider three states.

| State | Student / Teacher situation | Intuitive training value |
| --- | --- | --- |
| `2 + 2 = ?` | Student already assigns `0.99` to `4`; Teacher assigns `0.995` | Low: the Student essentially knows it |
| A geometry proof has entered a wrong branch | Student is uncertain; Teacher strongly prefers a corrective next step | Potentially high: there is new, actionable supervision |
| A rollout has drifted into malformed or nonsensical context | Teacher and Student distributions may differ dramatically | Unclear, possibly useless |

All three states count toward raw coverage, but they represent very different learning opportunities.

I therefore think of State Coverage as the first layer: it tells us which regions were visited. The second layer asks how valuable those regions are for the current Student.

This is where the other OPD paper becomes important.

---

## The Student has to be able to use the Teacher signal

[*Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe*](https://arxiv.org/abs/2604.13016) studies when OPD succeeds and when it fails.

One of its most useful observations is that **a stronger Teacher is not necessarily a better Teacher for a given Student**.

I would keep this negative formulation because it corrects a very natural intuition.

Successful OPD appears to depend on at least two things:

1. the Teacher must actually contain capabilities that the Student has not yet acquired;
2. the Teacher and Student need enough compatibility in their local reasoning / token distributions for the supervision to be usable.

At the token level, successful OPD increasingly aligns a relatively small set of high-probability tokens on Student-visited states. Those shared tokens account for roughly 97–99% of the probability mass.

This means Teacher–Student divergence captures only one part of the learning opportunity.

A large gap can mean the Teacher knows something the Student does not. It can also mean the two policies are locally so misaligned that the dense Teacher signal is difficult for the Student to absorb.

So I think we need to separate two questions:

**Is there new information here?**

and

**Can the Student actually use it?**

Together, those two dimensions help characterize whether a state is worth training on.

---

## What I mean by a “Useful State”

Putting the two papers together, I would currently define a Useful State as a **student-dependent learning opportunity**.

Its value depends on the current Student, the Teacher, the target task, and which regions have already been covered.

A state can be valuable for a weak Student and much less useful for a Student that has already mastered that region. The value can also change if we swap the Teacher or change the target capability.

I currently think about five signals:

| Signal | Question | Possible proxy |
| --- | --- | --- |
| **Novelty** | Has this region already been visited many times? | hidden-state distance, cluster coverage |
| **Information** | Does the Teacher know something the Student has not learned yet? | JS/KL gap, top-k log-prob gap |
| **Exploitability** | Can the Student use this supervision? | high-probability token overlap, local gradient alignment |
| **Reliability** | Is the Teacher trustworthy on this Student-induced state? | verifier, self-consistency, multi-teacher agreement |
| **Relevance** | Does this state matter for the target capability? | target-state similarity, domain classifier, task verifier |

As an abstract placeholder:

$$ U(s) = f\!\left(N(s), I(s), E(s), L(s), R(s)\right) $$

The important research question is what $f$ should actually look like. Some signals may work better as gates, others as ranking features, and some may only become useful when modeled jointly.

For example, a clearly off-task state or a state where the Teacher is unreliable is unlikely to be valuable even with a large Teacher–Student gap. After basic relevance and reliability filtering, it may make sense to rank states by how much new information and marginal coverage they provide.

This is exactly the kind of question that needs experiments.

---

## How would we find Useful States before training?

Defining Useful States is only half of the problem.

The operational question is: **before spending the full OPD budget, how do we know what kinds of states a candidate query will induce?**

A state is generated jointly by the query and the current Student:

$$ s \sim P(s \mid q, \pi_S) $$

That suggests a practical estimation strategy: run a small **pilot rollout** first.

Suppose we have a candidate pool. Freeze the current Student, run only a few rollouts for each candidate query, and keep the model parameters fixed. Then probe the visited states for the signals above.

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

There is a real trade-off here: pilot rollout still consumes Student sampling and Teacher inference. The approach becomes useful when a small pilot budget—say 2, 4, or 8 rollouts—predicts full-training value well enough to save more compute than it costs.

That assumption should be tested directly.

---

## Useful State Coverage

Even with a reasonable estimate of individual-state utility, data selection still needs a coverage term because repeated visits to the same region have diminishing value.

If a query sends the Student into nearly identical geometry-proof states one hundred times, the hundredth visit should contribute far less than the first.

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

This perspective makes **“high-quality query” a conditional concept**.

Traditional curation often assigns relatively stable attributes to an example: difficulty, correctness, domain, quality score, duplication.

For OPD, query value looks more like:

$$ \mathrm{Quality}\!\left(q \mid \pi_S, \pi_T, Q_{\mathrm{selected}}, \mathcal{T}\right) $$

The same query can have very different value at different points in training.

Early on, it may repeatedly induce states where the Student is weak and the Teacher can help. Later, those regions may already be absorbed; the Teacher–Student gap shrinks and additional rollouts become redundant.

That suggests a dynamic selection loop:

```text
probe → select → train → re-probe → re-select
```

Under this view, OPD data selection starts to look like **state-space curriculum learning**: as the Student changes, the next useful region to explore changes as well.

---

## From query selection to state-space exploration

Suppose we have already selected a query set $Q$. The value of a new query $q$ can be expressed through the **marginal useful coverage** it adds:

$$ \Delta \mathrm{USC}(q) = \mathrm{USC}\!\left(Q \cup \{q\}\right) - \mathrm{USC}(Q) $$

This matters because semantic diversity at the query level and diversity in Student-induced states can diverge.

Two queries can look very different in text space and still drive the Student into nearly identical reasoning states. Conversely, two queries that look similar may trigger very different error branches, recovery behaviors, or reasoning paths.

A more direct objective may therefore be:

> **Given a fixed sampling and Teacher-compute budget, how can we make the current Student visit as many high-value, non-redundant state regions as possible?**

At that point, OPD data selection starts to look like an exploration problem.

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

This makes Useful State Coverage a broader on-policy learning question:

> **What kinds of states does the model need to experience in order to improve where it is currently weak?**

---

## What would convince me this idea is useful?

The main risk is that “Useful State Coverage” becomes a concept that sounds reasonable but is hard to falsify.

So the next step should be a small set of direct experiments.

**First**, measure how much raw State Coverage already explains. Find query sets with similar state coverage but different Teacher–Student gap, compatibility, or task relevance, and compare downstream OPD gain.

**Second**, test whether pilot rollout predicts true training value. For each candidate query, run only 2, 4, or 8 pilot rollouts, estimate useful coverage, then run actual one-query OPD and measure the correlation with downstream performance gain.

**Third**, evaluate selection under a fixed budget. Compare Random, Semantic Diversity, Raw State Coverage, Gap-only, and Useful-State-aware selection while controlling query count, Student rollout tokens, and Teacher inference tokens.

If raw State Coverage already explains the gains and the additional signals do not improve prediction or selection, then the incremental value of USC is limited.

If two query sets have similar raw coverage and the difference is explained by whether the Student can actually learn from those states, then Useful State Coverage becomes a meaningful object to study.

---

## Closing thoughts

*One Training Example* changed how I think about what counts as training data in OPD.

The query matters, but it may function more like an entry point into a state space. The dense supervision is attached to the states the Student actually reaches.

State Coverage pushes the discussion forward: besides asking how many queries we have, we can ask how much of the state space the Student has visited.

If state value is highly non-uniform, the next question is:

> **Which regions are worth visiting?**

And the practical follow-up is:

> **Can we tell, before spending the full training budget, which queries are likely to take the Student there?**

Those questions seem more important to me than committing too early to a particular Useful State formula.

If the idea holds up, OPD data engineering may gradually shift toward dynamic state-space exploration: identify where the current Student still has clear learning opportunities, route more budget toward those regions, absorb the signal, then move on.

That is what I currently mean by **Useful State Coverage**.

---

## References

1. Yaxuan Li et al. **Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe.** arXiv:2604.13016, 2026.  
   https://arxiv.org/abs/2604.13016
2. Zixuan Fu et al. **Rethinking On-Policy Distillation of Large Language Models II: One Training Example.** arXiv:2609.04172, 2026.  
   https://arxiv.org/abs/2609.04172
