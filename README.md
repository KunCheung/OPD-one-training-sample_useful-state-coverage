# OPD One-Training-Sample → Useful State Coverage

This repository explores a working hypothesis for **On-Policy Distillation (OPD)**:

> **OPD data quality may be better understood through the state space induced by the current Student.**

The idea is motivated by two recent works:

- **Yaxuan Li et al.**  
  *Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe*  
  arXiv:2604.13016, 2026  
  https://arxiv.org/abs/2604.13016

- **Zixuan Fu et al.**  
  *Rethinking On-Policy Distillation of Large Language Models II: One Training Example*  
  arXiv:2609.04172, 2026  
  https://arxiv.org/abs/2609.04172

Together they suggest two important observations:

- effective distillation depends on both **new Teacher information** and the Student's ability to exploit that signal;
- a small number of queries can still induce a surprisingly broad set of Student-visited states.

These observations motivate the central question of this project:

> **What makes a Student-visited state valuable for learning, and can that value guide OPD data selection?**

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

A **Useful State** is treated as a **student-dependent learning opportunity**. Its value may depend on the current Student, the Teacher, previously visited states, and the target capability.

We currently study several candidate signals:

- **Novelty** — is this state region already heavily covered?
- **Information** — does the Teacher contain something the Student has not yet learned here?
- **Exploitability** — can the Student actually use the Teacher signal?
- **Reliability** — is the Teacher trustworthy on this Student-induced state?
- **Relevance** — does this state matter for the target capability?

These signals are hypotheses to test. A central goal is to identify which of them actually predict downstream learning gain and how they should be combined.

## Probe before train

Because states emerge from Student rollouts, a practical way to estimate query value is to probe the induced state distribution first:

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

The key empirical assumption is that **a small number of pilot rollouts can predict query value well enough to justify their cost**.

## Repository contents

- [中文研究 Proposal](./PROPOSAL.zh-CN.md)
- [中文技术 Blog：One Query, Many States](./BLOG.zh-CN.md)
- [English Blog: One Query, Many States](./BLOG.en.md)

## Main research questions

1. How well does raw State Coverage explain OPD gain?
2. What makes a Student-visited state useful for learning?
3. Which state-level signals best predict actual learning gain?
4. Can a few pilot rollouts estimate query value before full OPD training?
5. Can Useful-State-aware selection improve over random, semantic-diversity, raw-coverage, and gap-based selection under the same compute budget?
6. How should data selection adapt as the Student absorbs previously useful states?

## Status

Research idea / early-stage proposal. The current goal is to turn the state-space view into **testable hypotheses and measurable learning signals**.
