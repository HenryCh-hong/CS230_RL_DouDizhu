# 20. Cross-Paper Synthesis and Project Recommendation

> Scope: the nine papers in this pack. Facts are cited as (paper, section/page). Anything marked **[Interpretation]** is my judgment, not a paper claim. Figure-derived numbers are approximate readings, labelled "≈".
> Per-paper deep dives (sections 1–19 for each paper) are in `01_…` to `09_…` in this folder.

---

## 20.0 Read this first: the five things that matter most

1. **DouZero is the baseline, and its code is the infrastructure.** Deep Monte Carlo (DMC) beat DQN, A3C, CQN, DeltaDou and a supervised human-data agent in Dou Dizhu. It did this without search and without human heuristics, on one 4-GPU server (DouZero §5, Table 1). The public repo ships training (`train.py`, with `--objective adp|wp`) and evaluation (`generate_eval_data.py`, `evaluate.py`). It also ships random and RLCard-rule agents, and downloadable pretrained SL / DouZero-ADP / DouZero-WP baselines. *[From public code repo, not the paper.]*
2. **The weakest part of the whole Dou Dizhu literature is statistical evaluation.** No Dou Dizhu paper in this set reports multiple training seeds or confidence intervals on its head-to-head results (see the evaluation audit in §20.3-D). DeltaDou's "human expert level" claim rests on 100 deals. DouZero's SARSA and actor-critic comparison was a single run. DouZero+'s improvements are single-run learning curves with gaps of ≈0.05–0.2 ADP. Rigorous evaluation alone would set this project apart.
3. **The most promising open thread is hidden information that is available only during training.** In a simulator you *know* every player's cards while training, even though the agent must play without them. The papers use this in three different ways:
   - DouZero+ uses opponents' hands as supervised labels for a hand predictor.
   - DouZero+'s "maphack" pre-experiment feeds the true hand in as a cheating upper bound.
   - Suphx (Mahjong) gradually drops the hidden features out of an "oracle" agent's input during training.

   Nobody has compared these head-to-head in Dou Dizhu. The one hint is striking: the true next-player hand helps Peasants by ≈0.6 ADP, while DouZero+'s learned prediction of it helps by ≈0.1 ADP (DouZero+ Figs. 5–6, at different training budgets).
4. **Game-theoretic methods (NFSP, Deep CFR, ReBeL) are background, not baselines.** Their guarantees are for two-player zero-sum games. Dou Dizhu is a three-player, 2-vs-1 team game with a ~10⁴-sized combinatorial action space. DouZero says CFR-style methods are "often not sound in this complex three-player setting" and that "DouDizhu has not seen an effective CFR-like solution" (DouZero §1, §6). Read them for concepts (information sets, exploitability, why averaging policies stabilizes self-play), not to implement.
5. **My recommendation:** a controlled study of *how a Deep Monte Carlo agent should use training-time-only hidden information*. It compares an oracle ceiling, DouZero+-style prediction-as-input, a shared-trunk auxiliary loss, and Suphx-style oracle guiding, all at matched compute with multi-seed duplicate-deal evaluation. Details in §20.7.

---

## 20.1 Chronological research map

```
                    GAME-THEORETIC LINE (2-player zero-sum poker)            DOU DIZHU LINE                               TRANSFER (Mahjong)
2007  CFR (outside pack) ── tabular; needs hand-made abstraction
          │
2016  NFSP ── first end-to-end deep RL for approx. Nash via
          │   fictitious self-play (Q-net best response + avg-policy net)
          │   limitation: slow, 2p0s theory only, small action sets
          │
2019  Deep CFR ── CFR with neural nets, no abstraction;          DeltaDou ── AlphaZero-style MCTS (FPMCTS) +
          │       beats NFSP in FHP/HULH                                 Bayesian hand inference + kicker net
          │       limitation: tree traversal, 2p0s only                  + heuristic bootstrap; ~human-expert on 100 deals
          │                                                              limitation: 2 months on 68 CPUs; heuristics
          │                                                       CQN ── decompose hand (dancing links) → Q-learn over
          │                                                              decompositions; order-invariant pooling
          │                                                              limitation: heuristic decomposition, slow,
          │                                                              DQN-style bootstrapping; 20 days on 1 GPU
          │                                                       RLCard ── toolkit + env API; Dou Dizhu action
          │                                                              abstraction (>3×10⁴ → 309); DQN/NFSP
          │                                                              "plateau" on Dou Dizhu vs random
          ▼                                                                │
2020  ReBeL ── RL + search over public belief states;                     │                                  Suphx ── SL from human logs → RL;
              superhuman HUNL; limitation: input grows with              │                                  global reward prediction; ORACLE
              #infostates per public state; 2p0s theory only              │                                  GUIDING; run-time adaptation
                                                                          ▼                                  limitation: 44 GPUs/agent; hand-built features
2021                                                              DouZero ── Deep Monte Carlo + 4×15 action
                                                                         encoding as Q-net INPUT + parallel actors;
                                                                         no search, no human knowledge; #1 Botzone
                                                                         limitation: no explicit beliefs, bidding is SL,
                                                                         no search, single-run ablations
                                                                          ▼
2022                                                              DouZero+ ── + separate hand-PREDICTION model (next
                                                                         player's hand → extra input) + COACH net that
                                                                         filters unbalanced deals; top-5 Botzone
                                                                         limitation: small single-run gains, early-training
                                                                         curves only, many details unspecified
                                                                          ▼
OPEN  ── how best to use training-time-only information? (oracle guiding never tried in DMC/Dou Dizhu in this set)
      ── statistically sound evaluation (seeds, CIs, cross-play, unfamiliar partners)
      ── variance from deal luck in training targets (coach filters deals; Suphx suggests difficulty-aware rewards)
      ── MC vs TD: when does bootstrapping help? (DouZero §5.5: single run, "more studies are needed")
      ── search at test time on top of DMC (DouZero & DouZero+ future work); bidding inside RL (DouZero future work)
      ── game-theoretic solution concepts for 2-vs-1 team games (mostly theoretical)
```

### Evolution chains (what each step kept, changed, and left open)

**Chain A: Dou Dizhu**
- **Plain DQN / A3C** fail: under 20% WP against rule-based agents after 20 days, as reported in CQN and cited by DouZero §1. RLCard's DQN and NFSP plateau against random agents (RLCard Fig. 2).
  - *Limitation:* the combinatorial, state-dependent action space (DouZero App. A.2 counts 27,472 action types). DQN overestimates over huge max-operators; classifier-style policy heads can't generalize across actions (DouZero §3.2–3.3).
- **CQN** keeps Q-learning but *changes the action space*. A heuristic dancing-links search decomposes the hand into card groups; the network Q-scores decompositions, then moves within them.
  - *New limitation:* the decomposition is heuristic and slow. DouZero reports CQN "can not even beat simple heuristic rules after twenty days of training". In DouZero's Table 1, CQN ranks 7th of 8 and narrowly loses to the RLCard rule agent (WP 0.478).
- **DeltaDou** *changes the paradigm to search*: AlphaZero-style MCTS over sampled hidden hands (Bayesian inference) plus policy-value nets, with a heuristic kicker network and a heuristic bootstrap.
  - Result: strongest program of its time. 59% WP vs XDou on a fixed 100-deal set; 48.5% WP vs a human expert team on 100 deals.
  - Its own ablation shows hand inference matters: the agent with inference beats the agent without it at WP 54.5%, ADP +0.145 (DeltaDou §5.2).
  - *New limitation:* 2 months on 68 CPUs, plus reliance on human heuristics.
- **DouZero** *drops search, heuristics and explicit inference*. It keeps per-position Q-networks but replaces bootstrapped Q-learning with Monte-Carlo regression. It *puts the action into the network input* (4×15 card matrix) instead of having one output per action, and parallelizes with 45 actors.
  - Result: beats DeltaDou in 3 (WP) / 10 (ADP) days and is #1 of 344 bots on Botzone.
  - *New limitations:* hidden information is handled only implicitly (history LSTM, plus the union of the other two players' remaining cards). Bidding is a separately trained supervised network. There is no search, and the ablations are single runs.
- **DouZero+** *keeps DouZero intact* and adds two modules:
  - a separate prediction model that predicts the next player's hand and feeds the prediction into the decision network;
  - a coach network that predicts the Landlord's win probability from the dealt hands and skips unbalanced deals during training.

  Result: small gains, mostly for Peasants, in single-run early-training curves; top-5 on Botzone.
  - *New limitations:* effects are small and statistically unvalidated. Only the next player's hand is predicted, and prediction quality is never measured. The coach changes the training distribution without studying the effect on unbalanced deals.

**Chain B: game-theoretic self-play (poker)**
- **NFSP** fixes plain self-play's failure to converge in imperfect-information games. It plays a mixture of a best response (DQN) and a supervised *average* policy.
  - *Limitation:* slow; guarantees only via fictitious play in 2-player zero-sum (and potential) games; no proof with neural networks.
- **Deep CFR** brings regret minimization to neural networks. It beats NFSP: 37 vs 47 mbb/g exploitability in FHP, and +43 ± 2 mbb/g head-to-head in HULH.
  - *Limitation:* game-tree traversal and 2-player zero-sum assumptions.
- **ReBeL** adds search over public belief states. It is provably sound in 2-player zero-sum and superhuman in HUNL.
  - *Limitation (authors):* input grows linearly with the number of infostates per public state; theory is limited to 2-player zero-sum.
  - *For Dou Dizhu [Interpretation]:* none of these three gives guarantees for a 3-player, 2-vs-1 game. DouZero states CFR-style methods are not sound there.

**Chain C: transfer from Mahjong**
- **Suphx** shows three Dou-Dizhu-relevant ideas in another hidden-information, multi-player, shedding/drawing game:
  - *oracle guiding:* train with opponents' hidden tiles visible, then drop them out;
  - *global reward prediction:* turning a multi-round final score into per-round rewards;
  - *run-time policy adaptation.*

  Its future work explicitly suggests (a) training an oracle and a normal agent together with distillation, (b) an "oracle critic", and (c) reward signals that account for how lucky the initial hand was (Suphx §6).
  - *What the Dou Dizhu papers did with these ideas:* DouZero+'s "maphack" uses the oracle idea only as a one-off upper-bound experiment. Its coach network is related in spirit to (c), but filters deals instead of adjusting rewards. **Oracle guiding itself has not been tried in Dou Dizhu in this set.**

---

## 20.2 Comparison table

The table is split into two halves so it stays readable. Rows are in chronological order.

### Part 1: what the method is

| Paper | Year | Environment | Main algorithm | State representation | Action representation | Self-play? | Opponent modeling? |
|---|---|---|---|---|---|---|---|
| NFSP | 2016 | Leduc, Limit Texas Hold'em (2-player) | Fictitious self-play with a DQN best-response net + supervised average-policy net; two memories (circular RL buffer, reservoir SL buffer); anticipatory η = 0.1 | Vector encoding of cards + betting history | Tiny discrete set (fold/call/raise); one output per action | Yes, core | No (implicit via history) |
| Deep CFR | 2019 | Flop Hold'em, Limit Hold'em (2-player) | External-sampling MCCFR; per-player advantage net retrained from scratch each iteration on reservoir memory with linear iteration weighting; final average-strategy net | Infoset: card embeddings (rank/suit/card) + bet features; 98,948-parameter net | Small discrete set; one output per action | Yes (alternating traverser) | No |
| DeltaDou | 2019 | Dou Dizhu (card play; bidding not learned) | FPMCTS (AlphaZero-like) + Bayesian inference of hidden hands + policy-value nets; heuristic bootstrap (200k games) | Condensed 15×7 matrix | Rank+category from the network; kicker from a separate kicker network | Yes (episodes of 8,000 games) | Yes, implicitly: Bayesian posterior over hands using other players' policy nets |
| CQN | 2019/20 | Dou Dizhu (bidding omitted) | Two-stage Q-learning: heuristic hand decomposition (dancing links) + Q over decompositions/moves; double Q-learning + replay | 180-d (own hand + uniform "remaining-card" distributions for the two others) + 512-d last-round groups | Card-group encodings, order-invariant max-pooling | Yes (3 agents, async) | No (only counting of remaining cards) |
| RLCard | 2019/20 | 7 card games incl. Dou Dizhu | Toolkit; baselines DQN, NFSP, CFR (DeepCFR in later versions *[repo]*) | Dou Dizhu: 6 card planes (exact shape not in paper) | Integer index; Dou Dizhu abstracted from >3×10⁴ to 309 actions (rule-based kickers) | NFSP yes; DQN trained vs random | No |
| Suphx | 2020 | 4-player Riichi Mahjong (Tenhou) | Supervised learning on human logs → policy-gradient RL (importance sampling + entropy reg.) + global reward predictor (GRU) + oracle guiding + pMCPA | 34×838 / 34×958 feature planes incl. hand-built look-ahead features | Separate decision models (discard: 34-way; Riichi/Chow/Pong/Kong: binary) | Yes | No explicit model; *oracle* sees hidden tiles in training |
| ReBeL | 2020 | HUNL poker, Liar's Dice, Turn Endgame Hold'em (2-player) | Self-play RL + CFR-D/AVG subgame search at public belief states; value net (Huber) + policy net (MSE) | Public belief state: distribution over both players' private states (HUNL value-net input 2,660-d) | Small discrete (bet abstraction) | Yes | Beliefs are explicit (common-knowledge Bayes), not learned opponent models |
| DouZero | 2021 | Dou Dizhu (card play; SL bidding net for one eval) | Deep Monte Carlo: Q(s,a) regressed to MC return, ε-greedy, 45 actors + 1 learner, 3 per-position nets | Card matrices (hand, union of others' cards, recent/played moves) + one-hots (cards left, bombs) + LSTM over last 15 moves (5×162) | **Action is an input**: 4×15 card matrix (→54-d) concatenated with state; Q scores each legal action | Yes (latest policies only) | No |
| DouZero+ | 2022 | Dou Dizhu | DouZero + separate hand-prediction model (concatenated to state) + coach net filtering deals | DouZero features + predicted next-player hand | Same as DouZero | Yes | **Yes**: supervised prediction of the next player's hand (per card kind, cross-entropy) |

### Part 2: how it is trained, evaluated and what to take from it

| Paper | Main loss | Main contribution | Evaluation method | Compute | Main limitation | Best idea I could borrow |
|---|---|---|---|---|---|---|
| NFSP | DQN TD-MSE (target net) + cross-entropy for average policy | First end-to-end deep RL approach to approximate Nash from self-play; exploitability shown to fall in Leduc | Exact exploitability (Leduc); vs ACPC-2014 bots in LHE (−52.1 ± 8.5, −17.4 ± 9.0, −13.6 ± 9.2 mbb/h) | Not reported | 2p0s theory; no proof with function approximation; small action sets | Play against an **average of past policies**, not just the latest one; reservoir sampling |
| Deep CFR | Iteration-weighted MSE on sampled advantages; MSE for strategy net | First non-tabular CFR to work in large games without abstraction | Exact exploitability (FHP: 37 vs NFSP 47 mbb/g); HULH head-to-head ±2 mbb/g; ablations | Not reported in PMLR version | 2p0s only; tree traversal; needs scalable sampling and variance reduction (authors) | Ablations "retrain-from-scratch vs fine-tune" and "reservoir vs window" are model experiments for a clean study |
| DeltaDou | AlphaZero-style: value MSE + policy cross-entropy to MCTS visits (+ regularization) | First Dou Dizhu program reported at roughly human-expert level | Duplicate deals; fixed 100-deal set vs XDou/RHCP; 100 deals vs human team | 68 CPUs, 2 months (640k self-play games) | Expensive; heuristics; tiny eval samples | **Explicit hand inference helps** (54.5% WP vs no-inference); duplicate-deal evaluation |
| CQN | Double-DQN TD-MSE at both stages | Q-learning over a decomposed combinatorial action space | vs RHCP/random (50 episodes per epoch; 100 episodes × 10 eval seeds); 20 humans, 717 games | 32-core CPU + 1×1080Ti, 20 days | Heuristic, slow decomposition; bootstrapped Q unstable; weak in DouZero's tournament | Order-invariant pooling over card groups; a cautionary baseline |
| RLCard | Baselines' own losses | Standard environments + API + baselines for card-game RL | Learning curves vs random agents (3 runs each); small tournaments | 24-core CPU server; Dou Dizhu run 87,271 s → 7,276 s with 16 processes | Weak evaluation opponents (random); naive baselines don't learn Dou Dizhu well | Clean env API; seeding; the honest finding that DQN/NFSP struggle here |
| Suphx | Policy gradient with importance sampling + entropy reg.; GRP regression | Superhuman-rated Mahjong AI (stable rank 8.74 dan on Tenhou) | Offline: 1M games vs 3 fixed agents with **bootstrap resampling (800K × 1000)**; online: 5,760 Tenhou games | 44 GPUs × 2 days per agent; 1.5M training games | Heavy compute; hand-built features; pMCPA not deployed | **Oracle guiding** (perfect-feature dropout γₜ: 1 → 0); bootstrap CIs over games |
| ReBeL | Huber (value), MSE (policy) | RL + search with convergence guarantees in 2p0s; superhuman HUNL | Exact exploitability; HUNL vs bots and one human (7,500 hands, ±) | HUNL data generation on 90 DGX-1 (8×V100 each) | 2p0s only; belief input scales with infostates | Explicit **belief features** as network input; search as a future add-on |
| DouZero | MSE(Q(s,a), MC return); no bootstrapping, no target net | Simple MC + action-as-input + parallel actors beats all prior Dou Dizhu AIs | Duplicate decks; 10k decks per pair (WP, ADP); Elo on 100k decks × 5 resamples; per-role tables; Botzone | 48 cores + 4×1080Ti; 30 days (beats SL in ~1–2 days, DeltaDou in 3–10) | Implicit beliefs only; no search; SL bidding; single-run ablations; no CIs | **The whole system** as your baseline; the WP-vs-ADP objective switch |
| DouZero+ | DMC MSE (decision) + cross-entropy (prediction); coach loss unspecified | Opponent modeling + deal-filtering curriculum on DouZero | ADP vs released DouZero, duplicate, 10k episodes every 30 min; single runs; Botzone | RTX 2080Ti + Xeon server; duration not stated | Small, unvalidated gains; next player only; many unspecified details | Treat training-time hidden cards as **labels**; deal difficulty matters |

---

## 20.3 Cross-cutting themes

### A. Four ways to handle a huge combinatorial action space

| Strategy | Papers | Pros | Cons |
|---|---|---|---|
| Hand-abstract the action space (kickers chosen by rules or a separate network) | RLCard (→309), DeltaDou (kicker net) | Small output layer; any RL algorithm fits | DouZero argues kickers "can not be easily abstracted": a bad kicker can break a chain and lose the game |
| Hierarchical decomposition | CQN | Exploits card-group structure | Heuristic, slow, still bootstrapped Q |
| **Action as network input, score each legal action** | DouZero, DouZero+ (also DouZero's SL baseline) | Generalizes across similar actions ("3KKK → 3JJJ"); only legal actions are scored | One forward pass per legal action (batched); needs a value-based or MC learner |
| Naturally small action space | NFSP, Deep CFR, ReBeL (poker), Suphx (per-decision models) | Simple | Not available in Dou Dizhu |

**Takeaway [Interpretation]:** if you change the learner (e.g., TD vs MC), keep DouZero's action-as-input design. It is the component every later Dou Dizhu result depends on.

### B. Hidden information: a spectrum from implicit to explicit

| Approach | Where | What is actually used |
|---|---|---|
| Implicit, from history only | DouZero (LSTM over 15 moves), NFSP, Deep CFR | Only observable information; any "belief" lives inside hidden activations |
| Counting / constraint features | CQN (uniform distribution over remaining cards), DouZero (union of the other two hands, which is deducible), RLCard | Exact public deductions, no learning |
| Learned prediction of hidden cards | DouZero+ (next player's hand, per card kind) | True hands from the simulator as *training labels* |
| Bayesian posterior + sampling inside search | DeltaDou | Other players' policy networks give likelihoods; hands are sampled for MCTS |
| Privileged ("oracle") inputs during training | Suphx (dropped out gradually); DouZero+ "maphack" (test-time cheat, upper bound only); CQN's *A3C baseline* critic sees all cards | True hidden state as *inputs* during training |
| Explicit public belief states + equilibrium search | ReBeL | Exact Bayesian beliefs under known policies (2-player) |

**Gap [Interpretation]:** the ladder of uses for training-time hidden information has never been compared under one protocol in Dou Dizhu. The rungs are: labels (DouZero+), inputs then dropout (Suphx), inputs as a ceiling (maphack), and an auxiliary loss on a shared trunk (no paper here). This is the basis of my top recommendation.

### C. Self-play, non-stationarity and who you train against

- **Latest-policy self-play** (all players use the current networks): DouZero, DouZero+, CQN, and Suphx (opponents are clones of the current policy). None of the Dou Dizhu papers uses an opponent pool or past checkpoints.
- **Averaging over history**: NFSP explicitly plays against an average policy. Its Leduc experiments show DQN self-play, even with its behaviour averaged, fails to approach a Nash equilibrium (NFSP §4.2).
- **Equilibrium computation**: Deep CFR and ReBeL. Principled, but only for 2-player zero-sum games.
- **[Interpretation]** Dou Dizhu's Peasants are *teammates* trained together. Latest-policy self-play can produce private conventions that fail with unfamiliar partners. No paper in the set measures this ("ad-hoc teamwork"); the only cooperation evidence is a single case study (DouZero Fig. 10).

### D. Evaluation audit (the section to cite in your proposal)

| Paper | Opponents | Format | Sample size | Per-role results? | Training seeds | Uncertainty reported? |
|---|---|---|---|---|---|---|
| DeltaDou | XDou, RHCP, policy-only, human team | Duplicate deals (seats swapped) | **Fixed set of 100 deals** for learning curves; 100 deals vs humans | Yes | 1 | No (a "5 games out of 100" rule of thumb) |
| CQN | RHCP, random; A3C/DQN Landlord-only comparison; 20 humans | Agent vs fixed bots | 50 episodes per epoch; 100 episodes × 10 eval seeds; 717 human games | Yes | 1 (not stated) | Std bars on the human figure only |
| RLCard | Random agents; DQN vs NFSP | Learning curves | Each experiment run 3 times | Partly | 3 | Curves |
| NFSP | Exact best response (Leduc); ACPC bots (LHE) | Exploitability; head-to-head | 25,000 hands per evaluation vs SmooCT | n/a | Not reported | Standard error < 10 mbb/h |
| Deep CFR | Exact best response (FHP); head-to-head (HULH) | Exploitability; matches | Not stated | n/a | Not stated | ± 2 mbb/g; std 2.25 mbb/g in one ablation |
| Suphx | 3 fixed SL-weak agents (offline); Tenhou humans | Bootstrap resampling | 1M offline games; 5,760 online games | n/a | 1 per variant | **IQR over 1,000 bootstrap samples** |
| ReBeL | Exact exploitability; bots; one human | Exploitability; matches | 7,500 hands vs human | n/a | Not stated | ± (caption says SD, appendix says SE) |
| DouZero | 7 baselines incl. DeltaDou, CQN, SL, rule-based | **Duplicate decks** | 10k decks per pair; Elo on 100k decks × 5 resamples; bidding 100k decks × 6 seatings | Yes (App. D.1) | **1 per variant** (SARSA/Actor-Critic "single run") | Elo mean ± std over deck resamples only |
| DouZero+ | Released DouZero (ADP) | Duplicate decks | 10k episodes per checkpoint | Yes | **1** | **None** |

**What this means for this project [Interpretation]:**
1. **Duplicate deals with seat swapping** (DeltaDou, DouZero, DouZero+, Botzone) are the right *format*. Keep them.
2. **Deal-level noise is under control** with ~10k duplicate deals. **Training-run noise is not controlled anywhere**: one training run per method is the norm. With effects of ≈0.1 ADP (DouZero+), you cannot tell a method from a lucky seed. Use ≥3 seeds per variant.
3. **Report uncertainty** the way Suphx and the poker papers do: bootstrap over deals within a run, and spread across runs.
4. **Report per-role results** (Landlord vs Peasants). Several effects in this literature live almost entirely on one side (DouZero+ Peasants; DeltaDou vs humans).
5. **Use fixed reference opponents** (the released DouZero-ADP/WP checkpoints, the RLCard rule agent, random) *and* a cross-play matrix among your own variants. Head-to-head against a single opponent can be non-transitive.
6. **Match compute.** Compare at equal environment frames *and* equal wall-clock. DouZero shows both views (Fig. 6); DouZero+ shows only "training steps".

### E. Compute reality check

| Paper | Reported compute | Quarter-feasible for you? [Interpretation] |
|---|---|---|
| DouZero | 4×1080Ti + 48 cores, 30 days (beats SL in ~1–2 days, DeltaDou in 3–10 days) | Full: no. **Budgeted runs (hours–days on 1 GPU) + released checkpoints as reference: yes** |
| DouZero+ | 2080Ti server; duration unstated; curves to 6×10⁵ "training steps" | Yes, at small budgets (its comparisons are early-training curves) |
| DeltaDou | 68 CPUs × 2 months | No (search at both train and test time) |
| CQN | 1 GPU, 20 days | No, and not worth it given its results |
| RLCard | 24-core CPU | Yes (but use DMC, not the original DQN baseline) |
| Suphx | 44 GPUs × 2 days per agent | No; borrow ideas only |
| NFSP / Deep CFR | Not reported; small games are cheap | Leduc-scale yes; not Dou Dizhu |
| ReBeL | 90 DGX-1 machines for HUNL | No; borrow ideas only |

---

## 20.5 Remaining open problems (after all nine papers)

1. **Training-time-only information.** Which use of the simulator's hidden state works best in DMC: labels, auxiliary losses, oracle inputs with dropout, or distillation? How much of the oracle ceiling can each close? *(DouZero+, Suphx; not compared anywhere.)*
2. **Statistical validity.** No Dou Dizhu paper reports multi-seed results or CIs for head-to-head comparisons. *(All.)*
3. **Variance from deal luck in *training*.** DouZero+ filters deals; Suphx suggests difficulty-aware rewards. Nobody has tried using a deal-value estimate as a *control variate* (a variance-reducing baseline) on DMC's target. *(DouZero+, Suphx §6.)*
4. **Robustness to unfamiliar partners and opponents.** All agents are trained by latest-policy self-play and evaluated against fixed bots. *(NFSP's averaging idea; DouZero's cooperation claims.)*
5. **MC vs TD.** DouZero's only test of this is a single run ("more studies are needed to understand when TD learning will help", §5.5).
6. **Search on top of a learned DMC policy.** DouZero and DouZero+ future work; DeltaDou and ReBeL show it can help.
7. **Bidding inside the RL loop.** DouZero future work; not done by DouZero+.
8. **Solution concepts for 2-vs-1 team games.** Mostly theoretical; out of scope for CS230.

---

## 20.6 The five most promising CS230 directions (ranked strongest to weakest)

### Direction 1: How should a Deep Monte Carlo agent use hidden information that exists only during training? ★ recommended

- **Research question:** What happens if a DouZero agent gets access to opponents' true hands *during training only*? Compare four ways of using them: as labels for a separate predictor, as labels for an auxiliary loss on a shared trunk, and as oracle inputs that are gradually dropped out. Which closes the largest fraction of the gap to a test-time oracle?
- **Baseline:** DouZero (DMC), trained with the released code at your compute budget; plus the released DouZero-ADP/WP checkpoints as fixed reference opponents.
- **Modification (variants):**
  - **V0** DouZero as-is.
  - **V1 Oracle ceiling (not a legal agent):** add the two other players' true hands (2×54-d) to the input at train *and* test time. This measures how much the hidden information is worth (cf. DouZero+ "maphack").
  - **V2 Prediction-as-input (DouZero+-style):** a separate predictor trained with cross-entropy on true hands; its per-card-kind count distributions are concatenated to the decision network's input. Extend it to *both* opponents, not just the next player.
  - **V3 Auxiliary loss (new for Dou Dizhu):** no new inputs. Add a hand-prediction head on the decision network's shared trunk; loss = DMC MSE + λ·CE. At test time the head is unused; the question is whether it shapes better representations.
  - **V4 Oracle guiding (Suphx §3.3, new for DMC/Dou Dizhu):** V1's network with oracle features multiplied by a Bernoulli mask whose keep-probability γₜ decays 1 → 0 over the first part of the budget; then continue as a normal agent (oracle inputs = 0) with learning rate ÷10, as Suphx does.
- **Expected contribution:**
  - The first controlled comparison of these mechanisms in Dou Dizhu.
  - A "fraction of the oracle gap closed" metric.
  - Evidence on whether belief quality (prediction accuracy and calibration) predicts playing strength.
- **Experiment design:**
  - Same budget (frames and wall-clock) for every variant; ≥3 seeds for V0, V2, V3, V4; 1–2 seeds for V1 (it is only a ceiling).
  - ADP objective, matching DouZero+, so results are comparable.
  - Cheaper fallback if compute is tight: train only the two Peasant networks against a *frozen* released DouZero Landlord. This is where DouZero+ found the effects, and it removes most non-stationarity. It requires a modest change to the actor loop.
- **Evaluation protocol:**
  - A held-out set of ≥10,000 deals generated once with a fixed seed (`generate_eval_data.py`).
  - Duplicate format (each deal played with seats swapped), against: (a) the released DouZero-ADP, (b) the released DouZero-WP, (c) the RLCard rule agent (sanity check), and (d) a round-robin cross-play among V0–V4.
  - Report WP and ADP **separately for Landlord and Peasants**.
  - Uncertainty: bootstrap 95% CIs over deals within each run, and mean ± spread across seeds.
  - Learning curves vs frames *and* vs wall-clock.
  - Diagnostics: hand-prediction accuracy, NLL and calibration (V2, V3); a linear probe on V0's and V4's hidden features to measure how much belief they encode implicitly; and gap closed = (Vᵢ − V0)/(V1 − V0).
- **Minimum viable version:** V0 and V1 at a fixed budget plus the evaluation harness with CIs. Measuring the size of the oracle gap at your budget is already a result. Then add V2 (or V3).
- **Stretch goal:** V4 oracle guiding. Also: predicting both opponents vs only the next; per-hand-size analysis (does belief help most in the endgame?); a quick PIMC-style lookahead using the predictor (Direction 5).
- **Biggest risk:** effects may be small (DouZero+: ≈0.1 ADP), so 3 seeds at small budgets may not resolve them.
  - *Mitigations:* the Peasant-only variant; ≥10k duplicate deals; pre-registering the comparison; reporting null results honestly. A well-measured null result with a clear oracle-gap measurement is still a solid CS230 report.
  - *Subtle risk for V4 [Interpretation]:* while γₜ > 0, actors play with partial oracle information. The Monte-Carlo returns then reflect continuation by stronger, partly cheating play, a bias that disappears only as γₜ → 0. Suphx needed extra tricks after removing the oracle features; expect to need them too.
- **Compute:** Medium. Five variants × ~3 seeds × a fixed budget (e.g., 12–48 h each on 1 GPU + CPU actors) [estimate]. Measure your frames/sec in week 1 and size the budget from that.
- **Supported by:** DouZero (baseline, evaluation protocol, implicit beliefs); DouZero+ (prediction model, maphack ceiling, "Peasants benefit most"); Suphx (oracle guiding and its future-work variants); DeltaDou (its inference ablation, 54.5% WP, shows explicit beliefs help); CQN (its A3C baseline's privileged critic shows the idea is old but untested here); ReBeL (beliefs as network input).

### Direction 2: Luck-adjusted Monte Carlo targets (a deal-value control variate)

- **Research question:** What happens to DMC's learning speed if we subtract a learned estimate of "how good was this deal?" from every Monte-Carlo return, instead of filtering unbalanced deals out (DouZero+'s coach)?
- **Baseline:** DouZero; DouZero + coach-style filtering (re-implemented; DouZero+ leaves β's schedule and accept rule unspecified, so you document your choice).
- **Modification:** Train a deal-value network b(D) on the full initial deal D (all hands + the Landlord's 3 extra cards) to predict the final return. It can be trained offline on games simulated with the released DouZero, then frozen. The DMC regression target for each position becomes G − b(D), with the sign appropriate to that role.
- **Why this is principled [Interpretation; a short derivation you should write out]:** The deal is fixed before any action. Given the agent's information Iₜ, its action aₜ is chosen from Iₜ alone, so E[b(D) | Iₜ, aₜ] = E[b(D) | Iₜ]. Subtracting b(D) therefore shifts all of an information set's Q-values by the same constant: the greedy action is unchanged while target variance can drop substantially. This is the same logic as a baseline in policy gradients.
- **Expected contribution:**
  - A variance-reduction alternative to deal filtering that never discards data or changes the training distribution.
  - A measurement of whether coach filtering hurts on extreme deals.
- **Experiment design:** plain DMC vs filtering vs control variate (vs both), same budget, ≥3 seeds.
- **Evaluation:**
  - Same duplicate protocol as Direction 1.
  - Results *stratified by deal-difficulty bins* (bin the evaluation deals by b(D)).
  - Measured variance of the training targets.
  - Learning curves.
- **Minimum viable version:** offline-trained frozen b(D) and a single ADP objective.
- **Stretch goal:** train b(D) concurrently; extend it to b(D, t) using public information only (a proper state-dependent baseline); try the WP objective.
- **Biggest risk:** if b(D) is inaccurate the variance reduction is small. Also, the ADP scale (bombs double scores) needs care.
- **Compute:** Low–Medium.
- **Supported by:** DouZero+ (coach; "the luck factor plays an important role"); Suphx §6 (difficulty-aware rewards using perfect information); DouZero §3 (MC's high variance is its known weakness); DouZero/DeltaDou duplicate evaluation (the same idea used at *evaluation* time).

### Direction 3: Partner and opponent diversity, and robustness in cross-play

- **Research question:** What happens to a self-play-trained Peasant when its *teammate* is a policy it never trained with? Does training against a pool of past checkpoints (NFSP-inspired) make Peasants more robust?
- **Baseline:** DouZero latest-policy self-play.
- **Modification:** with probability p, the other seats are filled from a pool of past checkpoints (uniform or recency-weighted); optionally, also partners from other sources (SL, RLCard rule agent).
- **Expected contribution:** the first measurement of ad-hoc teamwork in Dou Dizhu, and whether population training helps it.
- **Experiment design:** p ∈ {0, 0.3, 0.6}, ≥3 seeds.
- **Evaluation:**
  - Mixed-team cross-play matrix, with Peasant pairs = (your agent, X) for X ∈ {itself, DouZero-ADP, DouZero-WP, SL, RLCard rule}, against fixed Landlords.
  - Coordination gap = performance with self-partner − performance with foreign partner.
- **Minimum viable version:** *evaluation only*: measure the coordination gap of the released DouZero agents using mixed teams. This needs no training at all.
- **Stretch goal:** the checkpoint pool.
- **Biggest risk:** the actor-loop changes needed for a pool; effects could be small.
- **Compute:** Low (MVP) to Medium.
- **Supported by:** NFSP (average-policy self-play), DouZero (cooperation case study only), DouZero+ (cooperation as the motivation for Peasant gains), Suphx (clone self-play), RLCard (multiple baseline agents).

### Direction 4: Where on the Monte-Carlo ↔ TD spectrum should Dou Dizhu learning sit?

- **Research question:** What happens if DouZero's MC target is replaced by n-step or λ-return targets (n ∈ {1 (SARSA), 3, 8, ∞ (MC)})? Does the answer depend on game phase or role?
- **Baseline:** DouZero DMC.
- **Modification:** n-step targets that bootstrap from the *same player's* next decision point (careful: other players act in between), with a periodically updated target network.
- **Expected contribution:** a multi-seed, controlled version of DouZero's single-run SARSA comparison (§5.5), on a question the authors explicitly left open.
- **Experiment design:** same budget, ≥3 seeds.
- **Evaluation:** the duplicate protocol from Direction 1, plus TD-error and target-variance statistics.
- **Minimum viable version:** MC vs 1-step SARSA vs one intermediate n.
- **Stretch goal:** λ-returns; an adaptive n.
- **Biggest risk:** getting bootstrapping right in multi-agent turn order; "MC wins again" is a likely (still publishable for CS230) outcome.
- **Compute:** Medium.
- **Supported by:** DouZero (§3.3, §5.5), CQN and RLCard (bootstrapped methods struggling), NFSP (DQN instability in self-play).

### Direction 5: Belief-sampled lookahead at test time on top of DMC

- **Research question:** What happens if, at test time, a trained DouZero agent samples K plausible hidden-hand assignments from a hand predictor and evaluates its top candidate actions by short rollouts with DouZero policies?
- **Baseline:** greedy DouZero.
- **Modification:** perfect-information Monte Carlo (PIMC: sample hidden hands, evaluate each, average) restricted to the endgame (e.g., when ≤ N cards remain), using the Direction 1 predictor for sampling.
- **Expected contribution:** a measure of how much cheap search adds to a strong model-free agent, and of how belief quality changes that.
- **Evaluation:** duplicate head-to-head vs greedy DouZero, as a function of K and per-move time budget.
- **Minimum viable version:** uniform sampling over consistent hands, endgame only.
- **Stretch goal:** learned-belief sampling; search in both roles.
- **Biggest risk [Interpretation]:** known PIMC pathologies ("strategy fusion": each sampled world assumes perfect information); engineering of game-state cloning; inference cost.
- **Compute:** Low for training, high for evaluation.
- **Supported by:** DeltaDou (search + inference; XDou uses a PIMC endgame solver), ReBeL (search + beliefs), DouZero and DouZero+ future work.

**Why this ranking [Interpretation]:**
- Direction 1 has the best mix of deep-learning content (architectures, auxiliary losses, representation probes), a clean ceiling/floor framing, multiple paper threads converging on it, and a natural milestone.
- Direction 2 is the most *novel* and cheapest, but its signal could be small, and the deep-learning content is thinner.
- Direction 3's evaluation-only MVP is almost free, but the training part needs more engineering.
- Direction 4 has the highest *learning* value for RL fundamentals but lower novelty.
- Direction 5 is the riskiest engineering.

---

## 20.7 Final recommendation: Direction 1

**Title suggestion:** *"Seeing the Cards While Learning: Privileged Hidden Information for Deep Monte-Carlo Agents in Dou Dizhu."*

**Why it has the best balance:**

| Criterion | Why Direction 1 scores well |
|---|---|
| **Learning value** | You will implement and understand DMC (the MC-regression view of Q-learning), auxiliary losses, supervised belief prediction, feature dropout schedules and representation probing. These are the core concepts of both RL and CS230's deep learning material. |
| **Technical depth** | Four mechanisms with different inductive biases (input features vs gradient signal vs curriculum), plus a real subtlety (the oracle-guiding bias while γₜ > 0) that you can analyze. |
| **Novelty** | Oracle guiding has not been applied to DMC or Dou Dizhu in this literature. DouZero+ tested only prediction-as-input, for the next player, in single runs. A matched-compute, multi-seed comparison with an oracle-gap metric and belief diagnostics goes beyond reproduction. |
| **Feasibility** | Every variant is a change to the input features, the loss, or a mask inside an existing, working codebase. The environment, legal-move generation, actors and evaluation scripts are reused. True hands are available in the simulator state. |
| **Evaluation quality** | A built-in ceiling (V1) and floor (V0) make effect sizes interpretable. The duplicate-deal protocol, released reference checkpoints, per-role reporting, seeds and bootstrap CIs address the main weakness of every Dou Dizhu paper in the pack. |
| **CS230 suitability** | It is a deep-learning question (what representations and training signals help a network act under uncertainty?) inside an RL system, with clear figures: learning curves, gap-closed bars, calibration plots, probe accuracy. |
| **Quarter-long scope** | Clear MVP by the milestone (V0 + V1 + harness), a mid-quarter core (V2/V3), and a stretch (V4) that can be dropped without sinking the project. |

### Suggested timeline (10-week quarter; adjust to your actual deadlines)

| Week | Goal | Deliverable |
|---|---|---|
| 1 | Install DouZero; download the pretrained baselines; run `evaluate.py` to reproduce a few Table 1 entries (e.g., DouZero-ADP vs RLCard, vs random) | Sanity numbers within the noise of DouZero's Table 1; frames/sec on your hardware |
| 2 | Build the evaluation harness: fixed 10k-deal set, duplicate play, per-role WP/ADP, bootstrap CIs, cross-play script | `eval/` module + one CI table |
| 3 | Train V0 at your chosen budget (×2 seeds); write the proposal around the evaluation audit (§20.3-D) | V0 curves; budget decision |
| 4–5 | V1 oracle ceiling (oracle features in the observation encoder + model input) | **Milestone:** V0 vs V1 gap with CIs, per role |
| 6–7 | V2 (prediction-as-input, both opponents) and/or V3 (auxiliary head); prediction-quality metrics | Gap-closed numbers for V2/V3 |
| 8 | V4 oracle guiding (γₜ schedule; post-oracle fine-tuning at lr/10) | V4 curves |
| 9 | Extra seeds; cross-play matrix; linear probes; per-hand-size analysis | Final tables and figures |
| 10 | Write-up: method, evaluation protocol, results, honest limitations | Report + poster |

### Where to make the changes in the DouZero code *[From public code repo, not the paper; file/function names checked against `kwai/DouZero` main branch on 2026-10-06]*
- **Observation features:** `douzero/env/env.py`. `Env.reset()` deals the hands into `card_play_data` (so the true hands are available), and `get_obs()` / `_get_obs_landlord()` (and the peasant equivalents) build each position's features with `_cards2array()`. Add the oracle features and per-opponent hand labels here.
- **Networks:** `douzero/dmc/models.py`, classes `LandlordLstmModel`, `FarmerLstmModel`, and the `Model` wrapper. You'll need a wider input for V1/V2/V4 and an extra head for V3. ε-greedy exploration lives in the models' `forward()` (`flags.exp_epsilon`).
- **Learner and loss:** `douzero/dmc/dmc.py`, functions `compute_loss()`, `learn()`, `train()`.
- **Actors, buffers, optimizer:** `douzero/dmc/utils.py`, functions `act()`, `create_buffers()`, `get_batch()`, `create_optimizers()` (RMSprop). Flags are in `douzero/dmc/arguments.py`.
- **Evaluation:** `generate_eval_data.py`, `evaluate.py`, and `douzero/evaluation/` (`deep_agent.py`, `random_agent.py`, `rlcard_agent.py`, `simulation.py`).

### Statistical sizing (rules of thumb) [Interpretation]
- **Win rate.** For a win rate near 0.5 from n independent games, the standard error is ≤ 0.5/√n. With n = 10,000, the 95% CI half-width is ≈ ±0.01. Duplicate (paired) evaluation is usually tighter.
- **ADP.** Its per-game spread depends on bombs, so **measure the standard deviation of the per-deal paired difference in a pilot run**. For example, if it is ≈2, detecting a 0.1-ADP difference at 95% confidence needs roughly (1.96 × 2 / 0.1)² ≈ 1,500 deals.
- **Training seeds dominate.** Report spread across seeds, not just deal-level CIs. With 3 seeds, show all three, not only the mean.

---

## 20.8 Questions to bring to CS230 office hours (cross-paper)

1. "My baseline is DouZero's released code, and my contribution is a controlled comparison of four ways to use training-time-only hidden information (oracle ceiling, prediction-as-input, auxiliary loss, oracle guiding). Is a well-measured comparison with a possible null result enough novelty for the final project, or do you expect a new method to *win*?"
2. "Each DouZero run needs hours to days per seed on one GPU. Would you rather see 4 variants × 3 seeds at a small budget, or 2 variants × 5 seeds at a larger one? Can the course's cloud credits support ~15–20 GPU-days in total?"
3. "For evaluation, I plan 10k duplicate deals per comparison against the released DouZero checkpoints, per-role results, bootstrap CIs within runs, and spread across seeds. Is that sufficient statistical rigor for CS230, and should I pre-register the primary metric (ADP vs WP)?"
4. "Is it acceptable to include an oracle agent that cheats at test time purely as an upper bound, as long as it is clearly labelled and never compared as a competitor?"
5. "If full self-play is too expensive, is training only the two Peasant networks against a frozen released Landlord a legitimate simplification, given that it changes the problem from 3-player self-play to a 2-agent cooperative task against a fixed opponent?"

---

## 20.9 Calibrated reading verdicts (these override the per-file scorecards)

Each per-paper file looks at its paper alone, and most of their scorecards rate the paper "READ DEEPLY". Viewed across the whole pack and this project, I would prioritize as follows:

| Paper | Verdict | What to read |
|---|---|---|
| DouZero | **READ DEEPLY** | All of §3–§5, App. C (features), D.2 (WP vs ADP) |
| DouZero+ | **READ DEEPLY** (it's short) | §IV (both methods), §V + Figs. 5–8. Read the curves critically (see `02_DouZeroPlus.md` §9) |
| Suphx | **READ SELECTIVELY** | §3.3 oracle guiding, §3.2 global reward prediction, §4 offline evaluation, §6 future work |
| RLCard | **READ SELECTIVELY** | Table 1 (game sizes), Dou Dizhu encoding/abstraction, evaluation; mostly as API docs |
| DeltaDou | **READ SELECTIVELY** | §4 inference, §5 experiments (note the 100-deal samples), conclusion |
| NFSP | **READ SELECTIVELY** | §2–§3 (fictitious play, the two networks and memories), §4.2 (why DQN self-play fails) |
| CQN | **READ SELECTIVELY** (skim) | §1, §4.2 idea, §5 results; mostly as a cautionary baseline |
| ReBeL | **REFERENCE ONLY** | Intro, the PBS idea, conclusion (limitations) |
| Deep CFR | **REFERENCE ONLY** | §2 background on CFR/exploitability; Fig. 4 ablations as a model of clean experiments |

---

## 20.10 Notes on sources and checks

- Each paper was read in full from its PDF text (and figure images where numbers mattered) and analyzed in its own file. Every file was then checked against the source, and I spot-checked the most consequential claims myself.
- **Corrections made during verification** (so you know what to be skeptical of if you see these claims elsewhere):
  - DouZero+ never states a combined "decision + α·prediction" loss or shared layers; its prediction and decision models are separate.
  - DouZero+'s figure values were re-read at high resolution.
  - DouZero never claims to be "superhuman".
  - The ReBeL vs Dong Kim result is +165 ± 69 mbb/g in ReBeL's favour.
  - CQN's human-study role win rates were inverted in a draft: agent-as-Landlord wins ≈60%, agent-as-Peasants ≈30%.
  - DeltaDou's human table reports *Landlord* win rates per seating (see `04_DeltaDou.md`).
- Remaining caveats:
  - Several hyperparameters in DouZero+ are genuinely unspecified by the paper.
  - Any statement about the current DouZero/RLCard code is labelled *[From public code repo, not the paper]* and should be verified when you clone them.
  - Effort and compute estimates for this project are my estimates, not paper facts.
