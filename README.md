# OPD One-Training-Sample → Useful State Coverage

This repository explores a working hypothesis for **On-Policy Distillation (OPD)**:

> **OPD data quality may be better understood in the state space induced by the current Student, rather than only in the query space.**

The idea is motivated by two recent works:

- **Yaxuan Li et al.**  
  *Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe*  
  arXiv:2604.13016, 2026  
  https://arxiv.org/abs/2604.13016

- **Zixuan Fu et al.**  
  *Rethinking On-Policy Distillation of Large Language Models II: One Training Example*  
  arXiv:2609.04172, 2026  
  https://arxiv.org/abs/2609.04172

The first suggests that **Teacher information is not automatically exploitable by the Student**. The second shows that **few queries can still induce a surprisingly broad set of Student-visited states**.

Taken together, they motivate a question:

> **If query count is not the right unit of data, and raw State Coverage does not tell us whether a state is actually learnable, what should high-quality OPD data mean?**

## Working view

```text
Query
  ↓
Student-induced States
  ↓
State Coverage
  ↓
State Learning Utility
  ↓
Useful State Coverage
  ↓
Dynamic State-Space Data Selection
```

A **Useful State** is not treated here as a fixed property of the state itself. It is a **student-dependent learning opportunity**: its value may depend on the current Student, the Teacher, previously visited states, and the target capability.

We currently consider several candidate signals:

- **Novelty** — is this state region already heavily covered?
- **Information** — does the Teacher contain something the Student has not yet learned here?
- **Exploitability** — can the Student actually use the Teacher signal?
- **Reliability** — is the Teacher still trustworthy on this Student-induced state?
- **Relevance** — does this state matter for the target capability?

These are **candidate signals, not a finalized USC formula**. A central research question is which of them actually predict downstream learning gain.

## Probe before train

Because states are induced by Student rollouts, query value cannot be judged reliably from query text alone. A practical direction is:

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

The key empirical assumption is that **a small number of pilot rollouts can predict the training value of a query well enough to justify their cost**.

## Repository contents

- [中文研究 Proposal](./PROPOSAL.zh-CN.md)
- [中文技术 Blog：One Query, Many States](./BLOG.zh-CN.md)
- [English Blog: One Query, Many States](./BLOG.en.md)

## Main research questions

1. Is raw State Coverage sufficient to explain OPD gain?
2. What makes a Student-visited state useful for learning?
3. Which pre-training signals best predict actual state-level or query-level learning gain?
4. Can a few pilot rollouts estimate query value before full OPD training?
5. Can Useful-State-aware selection outperform random, semantic-diversity, raw-coverage, and gap-based selection under the same compute budget?
6. Should data selection be dynamic as the Student absorbs previously useful states?

## Status

Research idea / early-stage proposal. The current goal is to turn the state-space view into **testable hypotheses**, rather than assume a fixed USC metric in advance.
