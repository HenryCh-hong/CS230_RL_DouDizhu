# Notes index: CS230 RL / Dou Dizhu paper pack

Start with **`00_SYNTHESIS_AND_RECOMMENDATION.md`**. It contains the research map, comparison tables, the evaluation audit, five ranked project directions, and the final recommendation (section 20).

| File | Paper | Calibrated verdict |
|---|---|---|
| `01_DouZero.md` | DouZero (ICML 2021) | READ DEEPLY |
| `02_DouZeroPlus.md` | DouZero+ (2022) | READ DEEPLY |
| `08_Suphx.md` | Suphx: Mahjong (2020) | READ SELECTIVELY |
| `07_RLCard.md` | RLCard (IJCAI 2020) | READ SELECTIVELY |
| `04_DeltaDou.md` | DeltaDou (IJCAI 2019) | READ SELECTIVELY |
| `05_NFSP.md` | NFSP (2016) | READ SELECTIVELY |
| `03_Combinational_Q_Learning.md` | Combinational Q-Learning / CQN (2019/2020) | READ SELECTIVELY (skim) |
| `09_ReBeL.md` | ReBeL (NeurIPS 2020) | REFERENCE ONLY |
| `06_Deep_CFR.md` | Deep CFR (ICML 2019) | REFERENCE ONLY |

Each per-paper file follows the same structure:

- §0: TL;DR.
- §1–§19: the standard sections. These run from Overview, RL Formulation, Architecture, Training, Math, Loss, Self-Play and Imperfect Information, through Evaluation, Experiments, Compute, Limitations and Future Work, to Connections, Project Ideas, Implementation, Staff Questions, Concepts and Scorecard.
- An appendix of "Synthesis Facts".

Conventions:

- **[Paper]** marks what the authors state.
- **[Interpretation]** marks analysis and inference.
- "The paper does not specify this clearly." marks gaps in the paper.
- *[From public code repo, not the paper]* marks code facts.
- "p. N" means the PDF page number.

## Suggested reading plan

1. **Days 1–3:** read `01_DouZero.md` alongside the DouZero paper (§3–5, App. C). Learn the NOW concepts in its §18 first: MDP/POMDP, return, Q-function, Monte Carlo vs TD, ε-greedy.
2. **Day 4:** read `02_DouZeroPlus.md` and the paper; focus on its §9 critique of the learning curves.
3. **Day 5:** read `08_Suphx.md` §3.3 (oracle guiding) and §13.
4. **Day 6:** skim `04_DeltaDou.md`, `07_RLCard.md` and `03_Combinational_Q_Learning.md`, mainly their §9 (evaluation) and §14 (connections).
5. **Week 2:** read `05_NFSP.md` (why naive self-play fails in imperfect-information games). Skim `06_Deep_CFR.md` and `09_ReBeL.md` for concepts only.
6. **Then:** re-read `00_SYNTHESIS_AND_RECOMMENDATION.md` §20.6–20.8 and draft the proposal.

## Source files

- PDFs: `../papers/`. They were downloaded with `curl`, because `download_all.py` failed on this Mac's Python SSL certificates.
- Extracted text: `../papers_text/`.
  - `*.paged.txt` has `[PDF page N]` markers.
  - `*.layout.txt` preserves table layout.

## How these notes were checked

- Every per-paper file was written from the full paper text.
- Each file was then Each file was then checked against the paper, including the figures for any numbers taken from plots..
- The most consequential claims were re-checked by hand.
- See `00_SYNTHESIS_AND_RECOMMENDATION.md` §20.10 for the corrections that were made and the caveats that remain.
