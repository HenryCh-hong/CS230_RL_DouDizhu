# CS230 hand-belief study in DouDizhu

Honglong Chen's Stanford CS230 final project. It asks whether a DouZero agent plays better when its input
explicitly represents beliefs about both hidden hands, rather than about the next player's hand only. The
design is in `DouZero/CS230_DESIGN.md`; how training works, step by step, is in `docs/TRAINING_WALKTHROUGH.md`.

## Layout

- `DouZero/`: the training code, kwai/DouZero vendored and modified (see below).
- `proposal/`: the project proposal (LaTeX).
- `docs/`: the training walkthrough, the implementation map and the project reviews.
- `notes/`: reading notes on the related papers. `links/`, `papers.csv` and `download_all.py` fetch the
  papers; the PDFs themselves are not in the repository.
- `tools/`: evaluation helpers (`dup_eval.py`).
- `cloud/`: scripts for the Stage 4 predictor run on AWS, prepared but not yet run
  (`docs/STAGE4_AWS_PROPOSAL.md`).

## DouZero (third-party code)

`DouZero/` starts from [kwai/DouZero](https://github.com/kwai/DouZero) at commit `718a5c9`, released under
the Apache License 2.0 (`DouZero/LICENSE`). The first commit of this repository imports it unmodified, and
every change made for this project is in the later commits, so `git diff <import commit> -- DouZero` shows
exactly what was changed. The released DouZero checkpoints (`DouZero/baselines/`) are not included.
DouZero is the work of its authors; please cite it:

> Zha, Daochen et al. "DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning." ICML (2021).

```bibtex
@InProceedings{pmlr-v139-zha21a,
  title     = {DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning},
  author    = {Zha, Daochen and Xie, Jingru and Ma, Wenye and Zhang, Sheng and Lian, Xiangru and Hu, Xia and Liu, Ji},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  pages     = {12333--12344},
  year      = {2021},
  editor    = {Meila, Marina and Zhang, Tong},
  volume    = {139},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {http://proceedings.mlr.press/v139/zha21a.html}
}
```

# Paper pack

This pack is curated for the CS230 final-project direction: reinforcement learning, deep RL, self-play, multi-agent imperfect-information card games, and DouDizhu.

## Start here

Do **not** try to read everything at once. Use this order:

### Tier A — Must read first (directly shapes the proposal)
1. **DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning** (ICML 2021)
2. **DouZero+: Improving DouDizhu AI by Opponent Modeling and Coach-guided Learning** (2022)
3. **Combinatorial Q-Learning for Dou Di Zhu** (AIIDE 2020 / arXiv 2019)
4. **DeltaDou: Expert-level Doudizhu AI through Self-play** (IJCAI 2019)

Goal: understand the evolution of DouDizhu methods, what each paper changes, and what limitations remain.

### Tier B — Foundations for imperfect-information self-play
5. **Deep Reinforcement Learning from Self-Play in Imperfect-Information Games (NFSP)** (2016)
6. **Deep Counterfactual Regret Minimization (Deep CFR)** (ICML 2019)

Goal: understand why imperfect-information games are different and how self-play / game-theoretic approaches are evaluated.

### Tier C — Useful extensions / project inspiration
7. **RLCard: A Platform for Reinforcement Learning in Card Games** (IJCAI 2020)
8. **Suphx: Mastering Mahjong with Deep Reinforcement Learning** (2020)
9. **ReBeL: Combining Deep Reinforcement Learning and Search for Imperfect-Information Games** (NeurIPS 2020)

Goal: environment design, related card-game techniques, and more advanced future-work ideas.

## How to download all PDFs

From this folder, run:

```bash
python3 download_all.py
```

The script downloads the papers from their public official/open-access URLs into a `papers/` folder.

## How we will read every paper

For each paper, answer:

1. What exact problem is the paper solving?
2. Why is previous work insufficient?
3. What is the baseline?
4. What component did the authors change or introduce?
5. What objective / loss is optimized?
6. How is training data generated (self-play, fixed opponents, etc.)?
7. How is the policy evaluated?
8. What limitations and future-work ideas remain?
9. Which idea could become **our** CS230 modification or ablation?

Use `PAPER_NOTES_TEMPLATE.md` for each paper.
