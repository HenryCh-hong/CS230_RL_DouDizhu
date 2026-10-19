# Deep Review: Joint Hidden-Hand Modeling for DouZero (CS230, solo)

Prepared 2026-10-17.

**Sources:**
- the papers in `papers/` and `papers_text/`;
- the kwai/DouZero code at commit `718a5c9`, in `DouZero/`;
- the DouZero+ authors' code at commit `64a4dc2`, in `reference_code/Doudizhu_plus/`;
- measurements run on this Mac.

**Evidence tags:** **[code]** means I read it in the code. **[run]** means I measured it by running code. **[paper]** means a paper states it. **[unverified]** means I could not check it.

---

## 0. Executive summary

**Verdict: keep the direction. There is no fatal flaw.** Four findings, though, change how the proposal has to be written and how the experiment has to be designed.

1. **Card conservation makes the second hidden hand redundant information.** DouZero's observation already contains the *union* of the two hidden hands (`other_hand_cards`) and both hand sizes. Once one hidden hand is known, the other is fixed: `H_B = U − H_A`, and per rank `P(c_B = k) = P(c_A = u − k)`.
   - So two-hand prediction adds no new information to the Q-network's input and no new supervision signal.
   - Any gain M2 shows over M1 has to come from **representation**: making something explicit that the network would otherwise have to derive.
   - Reframe the hypothesis this way and add the control **M2c**: the next-player predictor plus the analytically derived complement, at zero extra parameters.
   - [run: the union identity held in 18,631 of 18,631 sampled states]
2. **DouZero+'s "next player" depends on the seat.** In the authors' code the Landlord predicts `landlord_down` (an opponent), `landlord_up` predicts the Landlord (an opponent), and `landlord_down` predicts `landlord_up`, its **teammate** [code]. So M1 already models the teammate for one Peasant and the Landlord for the other. M2 adds something different at each seat, which means results must be reported per seat.
3. **The baseline effect you are extending is unproven.**
   - DouZero+'s reported gains come from single-run learning curves and are about 0.1 ADP [paper].
   - An independent group (IJCAI-24, which includes a DouZero author) retrained DouZero+ with the released code at smaller compute. Their DouZero+ came out **0.226 ± 0.031 ADP below** the released DouZero [paper].
   - AlphaDou (2024) says the DouZero+ variants "do not provide quantifiable improvements over the original DouZero" [paper].
   - So "M1 > M0" is itself an open question you must measure. That makes your project more valuable, as long as you measure it rigorously.
4. **Practical findings:**
   - The official pretrained-checkpoint link is dead (Google Drive returns 404).
   - I found a mirror and verified it (Appendix B). Its WP files are byte-identical across two independent mirrors, and its ADP files reproduce DouZero's Table 1 [run]. **Stage 0 is done** (table below).
   - The DouZero+ repo is **not runnable as-is**: missing modules, an undefined flag, CUDA-only device strings, a `torch==1.6` pin, and hard-coded `/root/...` paths [code]. Port its predictor into current kwai/DouZero instead.

**Stage 0 results** [run]. 10,000 fixed deals (seed 2026), each played in both seatings, duplicate format. CIs are 95% bootstrap intervals over deals.

| A vs B | WP of A | ADP of A | DouZero paper, Table 1 | A as Landlord (WP / ADP) | A as Peasants (WP / ADP) | per-deal paired SD (ADP) |
|---|---|---|---|---|---|---|
| DouZero-ADP vs RLCard rule agent | 0.855 [0.851, 0.860] | **2.340** [2.302, 2.378] | ADP 2.288 | 0.842 / 2.258 | 0.869 / 2.421 | 1.97 |
| DouZero-WP vs RLCard rule agent | **0.885** [0.880, 0.889] | 2.147 [2.115, 2.181] | WP 0.889 | 0.874 / 2.080 | 0.895 / 2.215 | 1.74 |
| DouZero-ADP vs SL | 0.614 [0.609, 0.620] | 0.778 [0.746, 0.809] | ADP 0.700 | 0.523 / 0.349 | 0.706 / 1.207 | 1.61 |
| DouZero-ADP vs DouZero-WP | 0.454 [0.450, 0.459] | 0.131 [0.101, 0.162] | (not reported) | 0.376 / −0.283 | 0.532 / 0.546 | **1.59** |

How to read this table:
- The WP checkpoint matches Table 1 almost exactly.
- The ADP checkpoint lands 0.05–0.08 ADP above the paper's numbers. That is plausibly due to a different deal sample and/or a later checkpoint than the snapshot the paper used.
- Each objective wins its own metric (ADP agent vs WP agent: WP < 0.5 but ADP > 0), exactly as DouZero §5 claims. Good sanity check.
- **The last column is the noise level you will face.** Between two strong agents, the per-deal paired SD is about 1.6 ADP.

**Novelty.** Your narrow claim works as a *description* of what you add to DouZero+. It is not safe as a *"first"* claim.
- Several 2024–2026 papers report learned prediction of hidden opponent information in DouZero-based agents. I could not verify their prediction targets because the papers are paywalled.
- By conservation, DouZero+ already models the second hand implicitly.
- Claim a **controlled, conservation-aware study** instead (§2.4).

**Compute.** One actor process runs at about 2,560 frames/s on this M5 [run], and a learner update of 3,200 samples takes 23–39 ms [run]. The recommended warm-start protocol needs tens of AWS CPU instance-hours, not weeks.

**Design recommendation (details in §3, §7 and §8):**
1. Train the predictors **offline** first, on states from self-play of the frozen DouZero-ADP. This is cheap and gives clean belief-quality results.
2. Run the main RL comparison as **warm-start fine-tuning from DouZero-ADP**: zero-initialize the new input columns so every arm starts from the same policy; continue training M0 as the control ("M0c"); add an **oracle positive control**; use 3 seeds.
3. Run a from-scratch replication only as a secondary experiment, if your AWS credits allow it.

---

## Part 1: CS230 proposal review

| # | Category | Score | Why |
|---|---|---|---|
| 1 | Problem clearly defined | **7** | The game, roles and question are clear. But "explicitly modeling both hidden players" is never distinguished from what DouZero already observes (the union of both hidden hands). "Improves decision making" is only operationalized later, in the Evaluation section. |
| 2 | Input → model → output | **7** | `Q(o_t, a_t)` is stated cleanly. Missing: the predictor's input and output (per-rank count distributions, 15×5 per hidden player), how predictions enter the Q-network (concatenated probabilities), and that predictions are stop-gradient inputs. |
| 3 | Substantive DL training | **8** | Three Q-networks and three multi-head predictors, trained with self-play DMC plus supervised cross-entropy. That is clearly enough deep learning for CS230. |
| 4 | Contribution non-trivial | **5** (7 after fixes) | As written, it is "one extra output block on DouZero+". Under conservation that block is informationally redundant (Part 4). It becomes a solid contribution once framed as a representation question, with the M2c control, belief-quality analysis and multi-seed evaluation. |
| 5 | More than reproduction | **6** | Yes in principle (M2, M2c). Re-implementing and validating M1 is real work and a real result, since DouZero+'s gain is contested. Say so explicitly. |
| 6 | Novelty phrasing conservative enough | **6** | The draft doesn't overclaim. But the related-work list leaves out the closest work (Full DouZero+ TG'24, PerfectDou NeurIPS'22, OADMCDou IJCAI'24, AlphaDou'24) while including distant work (Deep CFR, ReBeL, NFSP). |
| 7 | Feasible for one student | **7** | Feasible with the warm-start design and the scope caps in Part 12. Four or five from-scratch variants × 3 seeds at DouZero+'s regime (gains appeared only after very long training) would be borderline. |
| 8 | Baselines meaningful | **7** | M0 and M1 are the right baselines. Missing: the conservation control (M2c), an oracle ceiling, and fixed external references (released DouZero-ADP/WP, SL). |
| 9 | Evaluation scientifically defensible | **6** | Duplicate deals, ADP as primary metric, per-role reporting and the fixed-partner test are all right. Missing: number of deals, number of seeds, CI method, matched frames vs wall-clock, a checkpoint-selection rule, and a pre-registered primary comparison. |
| 10 | Dataset / self-play description | **7** | Good, and it correctly says labels are training-only. Add: eval deal-set size and seed, the frames budget, where labels come from (`InfoSet.all_handcards`), and how you prevent leakage. |
| 11 | Overpromising | **7** | Mostly calibrated ("if compute permits"). But "success means consistent positive ADP improvement" sets up a binary outcome. State what you will conclude from a null result, which is the likely outcome given conservation. |
| 12 | Transformer: helps or distracts | **6** | Fine as one sentence. It becomes a distraction if it takes more than about a week. GtDou (2026) already pairs a Transformer with hand guessing, so there is little novelty left there. |
| 13 | Technical accuracy | **5** | Five problems: (i) the proposal implies two-hand prediction adds information, but the union feature makes that false. (ii) `G_t` is undefined; it is the terminal ADP reward ±2^{#bombs}, signed by role, undiscounted. (iii) `CE(Ĥ₁,H₁)` hides that it is 15 masked categorical cross-entropies. (iv) "Reuse pretrained baselines" assumes the official link works; it's dead (a mirror exists). (v) M1's target is seat-dependent and the proposal doesn't say so. |
| 14 | Equations appropriate | **6** | Right objectives, but under-specified: no stop-gradient, no legal-count mask, no hand representation, no complement. |
| 15 | Important things missing | **5** | Conservation; seat semantics; compute budget; seeds; statistics; leakage tests; matched budgets; belief shift against foreign opponents. Also, the three face-up landlord cards are public information that DouZero's features omit. |

**Overall: about 6.3/10 as written, about 8.5/10 after the must-fix list.**

### A. Must fix before proposal submission (small, targeted; LaTeX in Appendix A)
1. **Conservation and reframing** (2–3 sentences). Say that the union of the hidden hands is observed, so the second hand is determined by the first. State the hypothesis as: *does an explicit representation of both hidden hands help the DMC learner?*
2. **Define M1 precisely.** It predicts the *next* player's hand, so the target depends on the seat (Landlord → `landlord_down`; `landlord_up` → Landlord; `landlord_down` → teammate). Say it is run *without* the coach network.
3. **Add M2c** (M1 plus analytic complement) and, if space allows, an oracle ceiling.
4. **Make the equations precise**: masked per-rank cross-entropy, stop-gradient into the Q-network, and the definition of `G_t`.
5. **Put numbers on the evaluation**:
   - ≥10,000 duplicate deals with a fixed seed;
   - ADP as the primary metric, with M2 vs M1 head-to-head pre-registered as the primary comparison;
   - paired bootstrap CIs;
   - 3 seeds per model, or say exactly what you'll do if compute runs short;
   - a matched frame budget, evaluating the final checkpoint;
   - DouZero-ADP, DouZero-WP and SL as fixed references.
6. **Related work**: cite Full DouZero+, PerfectDou, OADMCDou and AlphaDou. Compress NFSP, Deep CFR and ReBeL into one clause. Phrase novelty as a study (§2.4).
7. **One compute line**: AWS CPU instances; warm-start from the released DouZero-ADP via the verified mirror; from-scratch runs if feasible.

### B. Nice to improve
- One sentence on leakage prevention: a re-deal invariance test (§5.3).
- Name the prediction metrics: masked cross-entropy vs a card-counting (hypergeometric) baseline, and calibration.
- Mention the test-time belief ablation, i.e., whether the Q-network actually uses the beliefs (§9).
- Acknowledge the partner-swap confounds (§8).
- Shrink the Transformer to one "stretch" line.
- Note that the DouZero+ evidence is contested. This motivates a rigorous re-test.

### C. Already strong
- A fixed, narrow scope that reuses DouZero's infrastructure.
- A baseline-first plan: verify before you modify.
- Mirrored deals with role swap; ADP primary and WP secondary; per-role results.
- You noticed that no published number exists for M1 and that you must establish it yourself.
- You plan to relate prediction error to playing strength.
- The fixed-landlord / fixed-teammate test is a good idea (with the caveats in §8).

---

## Part 2: Verify the research idea

### 2.1 DouZero, reconstructed

| Component | What DouZero does | Where |
|---|---|---|
| **Observation** | **Landlord** `x` is 319-d: own hand, union of the other two hands, last move, cards played by `landlord_up` and by `landlord_down` (54-d each), cards left for each peasant (17-way one-hot each), and bombs played so far (15-way one-hot). **Peasant** `x` is 430-d: own hand, union, cards played by the Landlord and by the teammate, last move, the Landlord's last move, the teammate's last move (54-d each), Landlord cards left (20-way), teammate cards left (17-way), and bombs (15-way). **History** `z`: the last 15 moves, each a 54-d card vector, grouped 3 per row into a (5, 162) array. **Card encoding**: a 4×13 count "thermometer" plus 2 joker bits, 54-d in total. | `env.py:230` `_cards2array`, `:252`, `:268`, `:290`, `:374`, `:472` [code]. Shapes confirmed [run]. |
| Omitted | The 3 face-up landlord cards are public but are **not** a feature. This matters for predicting the Landlord's hand. | `env.py:290-570` [code] |
| **Action** | The move's 54-d card vector, appended to `x` (373-d for the Landlord, 484-d for Peasants). | `env.py:304-306` [code] |
| Legal moves | `MovesGener` generates candidates, filtered by the last move and move type. | `game.py:177`, `move_generator.py`, `move_detector.py`, `move_selector.py` [code] |
| **Q-network** | LSTM(162 → 128); its last step is concatenated with `x` and the action, then 5 × Linear(512)+ReLU, then Linear(1). 1,457,665 params (Landlord) / 1,514,497 (Peasant). | `models.py:11`, `:46` [code, run] |
| Role-specific nets | Three independent networks (`landlord`, `landlord_up`, `landlord_down`). No weight sharing. | `models.py:87-121` [code] |
| Action selection | All legal actions are scored in one batch. ε-greedy with ε = 0.01 during training; greedy argmax at evaluation. | `models.py:38-43`, `deep_agent.py:27`, `arguments.py` [code] |
| Self-play | All three seats are played by the current actor copies. Deals are uniformly random. There is no bidding: the Landlord seat is fixed and gets the 3 extra cards. | `utils.py:110`, `env.py:50` [code] |
| **DMC target** | Every-visit Monte Carlo with γ = 1. Every decision a seat makes in an episode gets that seat's final reward. With the ADP objective, the Landlord gets ±2^{bombs} and each Peasant gets the negation. | `env.py:97`, `utils.py:145-156` [code] |
| **Actor / learner** | `num_actors` processes per device, each with its own environment. Per-seat shared-memory buffers (50 buffers × T = 100 steps) with free/full queues. Four learner threads per seat. A batch is B = 32 chunks = 3,200 samples. | `utils.py:78`, `:110`; `dmc.py:61-190` [code] |
| Buffer contents | `obs_x_no_action` (T, 319/430) int8; `obs_action` (T, 54) int8; `obs_z` (T, 5, 162) int8; `target`, `done`, `episode_return`. | `utils.py:91-98` [code] |
| **Updates** | MSE between `Q(x, a_taken)` and `G`, averaged over 3,200 samples. RMSprop (lr 1e-4, α 0.99, ε 1e-5, momentum 0), grad-norm clip 40. **Weights are copied into the shared-memory actor models after every update.** No target network, no bootstrapping. | `dmc.py:19-59`, `utils.py:62` [code] |
| Evaluation | `evaluate.py` runs one seating at a time and prints aggregate WP and ADP. It outputs no per-deal results. | `simulation.py:45` [code] |

### 2.2 DouZero+, reconstructed from the authors' code

| Question | Answer | Evidence |
|---|---|---|
| What the predictor sees | The same public features as DouZero, minus the action: `x_no_action` (319/430) plus `z` (5×162), fed through its **own** LSTM. | `oppo_modeling/dmc/models.py`, `utils.py act()` [code] |
| Whose hand it predicts | "The player behind current agent", i.e., the next to act. Play order is L → D → U (`game.py:154-167`), so: **Landlord → `landlord_down` (opponent); `landlord_up` → Landlord (opponent); `landlord_down` → `landlord_up` (teammate).** | `oppo_modeling/env/env.py` (per-seat `down_label`) [code]; paper §IV-A |
| Target representation | That player's **current** remaining hand (initial hand minus cards played), as 15 per-rank counts: 13 ranks with 0–4 copies each, and 2 jokers with 0–1 each. This gives 15 categorical labels. | `env.py get_label` (196-207) [code] |
| "Legal label" mask | For each rank, the allowed counts are 0..u(r), where u(r) is the number of that rank in the *union* of the other two hands (public). Illegal logits are set to −1e7. | `env.py legal_predict` (182-193); `models.py` [code] |
| Architecture | LSTM(162 → 128) ⊕ `x` → 4 × Linear(512)+ReLU → 13 rank heads (512 → 256 → 5) and 2 joker heads (512 → 5). **2,895,947 params (Landlord) / 2,952,779 (Peasant), about 2× the Q-network.** | `models.py` [code, run] |
| Loss | `nn.CrossEntropyLoss` over B·T·15 classifications, averaged. | `oppo_modeling/dmc/dmc.py predict_loss` [code] |
| How predictions reach the Q-net | The softmax probabilities (15 × 5 = 75) are concatenated as `[LSTM(z), x(with action), p]`, so dense1 takes 373+128+75 (Landlord) or 484+128+75 (Peasant). The actor runs the predictor once per decision and broadcasts `p` across all legal actions. | `models.py`, `utils.py act()` [code] |
| Separate from the decision model? | **Yes.** Separate modules and weights (`Pre_model` vs `Model`), no shared trunk. | [code] |
| How they are trained together | In the same learner step: the predictor runs on the batch to get the CE loss; then **`prob.detach()`** goes into the Q-net to get the MSE loss; `loss = MSE + CE`. One RMSprop optimizer with **two parameter groups**, separate grad clipping, and both models synced to the actors. Because of the detach, CE trains only the predictor and MSE trains only the Q-net. Predictions are **recomputed at learn time** with the current predictor rather than stored. | `oppo_modeling/dmc/dmc.py learn()`, `utils.py create_optimizers` [code] |
| Paper vs code | The paper mentions neither the stop-gradient nor the shared optimizer. It says "five shared layers" and "each head ... a fully connected layer", but the code has 4 shared layers and 2-layer rank heads. I can't confirm that the released code is exactly what produced the paper's figures. | paper §IV-A vs `models.py` |
| Coach network | Per-seat card embedding (31 × 256), mean-pooled, then MLPs, outputting P(Landlord wins \| three initial hands). Trained with BCE (binary cross-entropy) on self-play outcomes using Adam, concurrently with RL. A deal is **rejected unless β ≤ p ≤ 1 − β**; β is 0 at first, then steps 0.1 → 0.3 between learner steps 20k and 100k. It affects only the training-deal distribution. | `coach_netwrok/dmc/models.py:112-157`, `dmc/utils.py:168-174`, `dmc.py:316-340` [code] |
| What stays as in DouZero | The environment, features (apart from the +75 dims), action encoding, DMC target, ε, actor–learner layout, and hyperparameters ("kept the same as the default ones"). | paper §V; [code] |
| How DouZero+ evaluates | ADP against the released DouZero-ADP (and SL), on 10,000 **fresh, unseeded** deals per checkpoint, with the Landlord and Peasant seatings as separate runs. One training run, no CIs. | `oppo_modeling/test/ADP_test.py`; paper Figs. 5–7 [code, paper] |
| Can you run it? | **No, not as released.** `oppo_modeling/dmc` lacks `file_writer.py`, `arguments.py` and `__init__.py`; `flags.oppo_interval` is undefined; `import traceback` is missing; device strings are CUDA-only; paths are hard-coded to `/root/doudizhu/...`; and it pins `torch==1.6`. | [code] |

**A note on a false alarm.** An earlier check flagged DouZero+'s test-time use of `infoset.other_hand_cards` as leakage. **It isn't.** It is the union of the two hidden hands, which anyone can deduce (54 cards − own hand − all played cards). The real leakage hazard is that every agent receives an `InfoSet` that also contains `all_handcards`, the true hands (`game.py:331`) [code]. See §3 #13.

### 2.3 Exactly where your model differs

- **DouZero+ (M1):** `o_t → p_φ(H_next | o_t)` (15×5) → `Q_θ(o_t, a, sg[p])`.
- **Yours (M2):** `o_t → (p_φ(H_next | o_t), p_φ(H_other | o_t))` → `Q_θ(o_t, a, sg[p_next], sg[p_other])`.

The changes are exactly these:
1. One more 75-d block in the Q-network input. That adds 38,400 weights to dense1, +2.6% on the Q-net.
2. Each predictor head outputs 2 × 5 logits instead of 5.
3. One extra CE term, on the labels of the *other* hidden player.
4. What the added block means depends on the seat: the Landlord gains explicit beliefs about `landlord_up`; `landlord_up` gains explicit **teammate** beliefs; `landlord_down` gains explicit **Landlord** beliefs.

These stay the same: predictor inputs, the legal mask (the same union applies to both hidden players), the training loop, the DMC target and evaluation. Part 4 adds the caveat: at the optimum, the added block is a fixed function (a "reflection") of the M1 block and inputs the Q-net already has.

### 2.4 Is the narrow novelty claim defensible?

The nearest work, and how much I could verify:

| Work | Framework | How it uses hidden hands | Relation to yours |
|---|---|---|---|
| DeltaDou (IJCAI'19) | MCTS + policy/value nets | Bayesian inference over **both** opponents' hands, used to sample for search | Models both hands, but via inference inside search, not learned prediction in DMC |
| DouZero (ICML'21) | DMC | Union of the hidden hands as an input (implicit) | Your M0 |
| DouZero+ (CoG'22) | DMC | Learned **next-player** prediction fed to Q | Your M1 |
| Full DouZero+ (IEEE ToG 2024, DOI 10.1109/TG.2023.3299612) | DMC + bidding | Opponent modeling; whether it goes beyond the next player is **[unverified]** (paywalled) | **Check** |
| PerfectDou (NeurIPS'22) | PPO, perfect-information critic | True hands only in the critic; no prediction head [paper] | Different mechanism |
| Luo & Tan (Applied Soft Computing 2024) | DMC + "opponent model" | Predicted hidden information as an input; targets **[unverified]** | Possibly close |
| OADMCDou (IJCAI'24) | DMC + oracle guiding | True hands as inputs, with a λ dropout decayed from 1 to 0 [paper] | Different mechanism. Already shows the oracle-guiding idea in DMC. Retrained DouZero+ came out below released DouZero |
| AlphaDou (arXiv'24) | DMC-like + bidding | None; says DouZero+ gives "no quantifiable improvements" [paper] | Context |
| CAPRE_DMC (Applied Soft Computing 2025) | DouZero-based + peasant credit assignment | LSTM + attention "prediction of hidden opponent information" (abstract); targets **[unverified]** | Possibly close |
| GtDou (conference, 2026) | Transformer + "opponent's hand card guessing model" | **[unverified]**, no full text found | Possibly close to both your main idea and the Transformer idea |

**Verdict.** "I extend DouZero+'s learned next-player hidden-hand prediction within the model-free DouZero/DMC framework to model both hidden players" holds as a description of your delta. It does **not** hold as a priority claim, for two reasons: (i) three recent papers report learned hidden-information prediction in DouZero-based agents, and I could not check what they predict; (ii) by conservation, DouZero+ already models the second hand implicitly. Before the final report, use Stanford library access to read Full DouZero+, Luo & Tan 2024, CAPRE_DMC 2025 and GtDou 2026.

**Wording to use instead:**

> "Prior DouZero-line work feeds learned predictions of the *next* player's hand into the Q-network (DouZero+), feeds true hands during training (OADMCDou), or uses perfect information in a critic (PerfectDou). We do not claim a new mechanism. We contribute a controlled, conservation-aware comparison of next-player vs. both-player hand modeling in DMC, under matched compute and with multi-seed duplicate-deal evaluation, including a control that derives the second hand analytically."

---

## Part 3: Minimum viable model

**Model set:**
- **M0**: DouZero.
- **M1**: next-player predictor (DouZero+ without the coach).
- **M2**: your model, both hidden players with independent heads.
- **M2c**: the M1 predictor plus the analytic complement. Same 150-d Q input as M2, zero extra predictor parameters, consistent by construction.
- **O**: oracle, not a legal agent. True counts of both hidden hands are fed in the same 150 slots, at train *and* test time. It gives the ceiling and checks whether the protocol is sensitive at all.

| # | Design choice | Recommendation | Status |
|---|---|---|---|
| 1 | Predictor input | Exactly DouZero+'s: `x_no_action` (319/430) + `z` (5×162) through its own LSTM. Don't change inputs between arms. | **REQUIRED** |
| | | Add the remaining face-up landlord cards (public), but to *every* arm including M0, never to just one | OPTIONAL |
| 2 | Outputs | For each hidden player j ∈ {next, other} and each of the 15 ranks: 5 logits for counts 0–4, masked to ≤ u(r). M1: K = 1. M2: K = 2, with each head outputting 5K logits. | **REQUIRED** |
| 3 | Hand representation | Per-rank count categorical, as in DouZero+. Not 54-d card bits: suits are irrelevant, and DouZero's encoding is count-based anyway. | **REQUIRED** |
| 4 | Heads / dims | 15 heads: 13 rank MLPs (512 → 256 → 5K) and 2 joker linears (512 → 5K). The Q-net gets 75K extra input dims, so dense1 takes 501+75K (Landlord) or 612+75K (Peasant). | **REQUIRED** |
| 5 | Labels | Each hidden player's **current** hand at decision time, before acting, read from `infoset.all_handcards[target]` as a (K, 15) int8 tensor. Computed **only** in the training actor. | **REQUIRED** |
| 6 | Loss | Masked CE **averaged** over the K·15 heads, so the loss scale matches between M1 and M2. Weight 1, as in DouZero+. | **REQUIRED** |
| | | A consistency penalty for M2 | NOT initially (M2c makes it unnecessary) |
| 7 | Decision-network input | `concat[LSTM(z), x ⊕ a, sg(p_next), sg(p_other)]` | **REQUIRED** |
| 8 | What to feed | Probabilities | **REQUIRED** |
| | | Expected counts (15 per player). These make conservation linear: E_B = u − E_A. | OPTIONAL ablation |
| | | Logits (masked at −1e7, which would wreck the Q-net); sampled hands (noise; PIMC territory) | NOT worth doing |
| 9 | Q-gradients into the predictor | **No.** Stop-gradient, as in DouZero+'s code. This keeps beliefs measurable and keeps M1 faithful to DouZero+. | **REQUIRED** |
| | | End-to-end training (no detach) | NOT initially |
| 10 | Role-specific predictors | Yes, three (one per seat), like the Q-nets. | **REQUIRED** |
| 11 | Shared trunk for the two hands | Yes. Only the output width changes, adding about 17k parameters (not 1.7M). | **REQUIRED** |
| 12 | Player identity and semantics | Fixed slot order per seat: slot 1 = next player (DouZero+'s target), so M1 is nested inside M2; slot 2 = the other player. Seat-specific networks make the teammate/opponent role implicit. Document the mapping table. | **REQUIRED** |
| | | Explicit role embeddings | NOT (networks are already per seat) |
| 13 | Leakage prevention | Labels exist only in the training actor. The evaluation agent builds features from public fields only. Add a re-deal invariance unit test (§5.3), and assert that the evaluation path never reads `all_handcards`. | **REQUIRED** |

---

## Part 4: The card-conservation issue

**Setup.** Let `s` be the acting seat, `A` = next(s) and `B` = the other hidden player. At time t these are public: own hand, every played card, and both hand sizes `n_A`, `n_B`. Let `U_t` = deck − own hand − played cards, and let `u_t(r)` be the count of rank r in `U_t`. Then:

1. `H_A ⊎ H_B = U_t`, with `|H_A| = n_A` and `|H_B| = n_B`. This holds exactly.
2. DouZero's `x` already contains `U_t` (`other_hand_cards`) and `n_A`, `n_B` [code; checked in 18,631 of 18,631 states, run].

**Consequences.**
- (a) **Labels:** `c_B(r) = u(r) − c_A(r)`, deterministically.
- (b) **Beliefs:** `P(c_B(r)=k | o) = P(c_A(r) = u(r) − k | o)` for every rank and every k. Expected counts satisfy `E[c_B(r)] = u(r) − E[c_A(r)]`.
- (c) So A's per-rank marginals fully determine B's. M1's 75-d belief vector together with `x` already contains everything M2's 150-d vector does.
- (d) The supervision is identical too. B's labels are a fixed function of A's labels and `x`, so `CE_B(reflect(p_A)) = CE_A` exactly.
- (e) **What both M1 and M2 ignore**: the hand-size coupling `Σ_r c_A(r) = n_A` and all cross-rank correlations. Real "joint" modeling would live there.

**Questions, answered:**
- **Is predicting both hands independently redundant or inconsistent?** Both. It is redundant at the optimum. In practice it is inconsistent: two separately trained heads won't satisfy `p_B = reflect(p_A)` or `Σ E_A = n_A`. Measure the size of those violations (Part 9).
- **Can the other hand be inferred from one?** Yes, exactly. For each rank, the reflection is a fixed permutation of the 5 probabilities, determined by u(r).
- **Does that make the extension trivial?** It is trivial **in information content** but not **in learning**. M1's Q-net has to learn the complement on its own, from sparse Monte Carlo returns. This is the same implicit-to-explicit argument DouZero+ makes for the first hand: *"the information about the hand cards of the next player is included implicitly while the idea of opponent modeling is essentially making such information explicit"* (DouZero+ §V-B). Whether explicitness helps a second time is a falsifiable empirical question, but your prior should be a small effect.
- **Is there still a meaningful distinction?** Yes, three:
  1. **Representation.** M2 hands the Q-net explicit beliefs about the player M1 leaves implicit, and that player differs by seat (§2.3). This yields a seat-level prediction: *if representation matters, the gap should be largest for `landlord_down`*, whose M1 target is its teammate rather than the Landlord it must beat. That is an interpretation; pre-register it as exploratory.
  2. **Training signal on the trunk.** M2's extra head adds a second, redundant gradient. That is equivalent to re-weighting, which averaging the loss controls for.
  3. **Consistency.** M2c is consistent by construction; M2 is not.
- **Simplest way to enforce or exploit conservation.** Use the **reflection** (M2c): `p_B(r,k) = p_A(r, u(r)−k) · 1[k ≤ u(r)]`. It is one gather with an index tensor and has zero parameters. Later, if wanted: project expected counts so `Σ_r E_A = n_A` with `0 ≤ E_A ≤ u`, via iterative scaling. A soft penalty is not needed.
- **MVP or later ablation?** **Put M2c in the MVP.** It costs about 20 lines and is the control that makes any M2 result interpretable:
  - M2 ≈ M2c > M1: explicit representation helps.
  - M2 > M2c ≈ M1: the effect comes from the extra training signal or the extra estimator. Unexpected; investigate.
  - All three ≈: redundancy. This is the expected result, explained by conservation, and it is a clean, reportable finding.
  - Modeling the hand-size and cross-rank coupling is a stretch goal.

**A 30-minute diagnostic that predicts much of the answer before any RL.** Fit a linear probe from M1's Q-net first-layer activations to B's true counts. If B decodes as well as A, M1 already represents the complement internally.

---

## Part 5: Implementation on the DouZero codebase

### 5.1 Where everything lives (kwai/DouZero @ `718a5c9`)

| Concern | Location |
|---|---|
| Environment wrapper | `douzero/env/env.py`: `Env` (`:21`), `reset` (`:50`), `step` (`:75`), `_get_reward` (`:97`), `DummyAgent` (`:161`) |
| Game engine and turn order | `douzero/env/game.py`: `GameEnv` (`:18`), `card_play_init` (`:56`), `step` (`:103`), `get_acting_player_position` (`:154`, order L → D → U), `get_infoset` (`:292`; `other_hand_cards` at `:316-321`, **`all_handcards` (true hands!) at `:331-333`**), `InfoSet` (`:337`) |
| Observation features | `env.py`: `get_obs` (`:188`), `_get_obs_landlord` (`:290`), `_get_obs_landlord_up` (`:374`), `_get_obs_landlord_down` (`:472`), helpers `_cards2array` (`:230`), `_action_seq_list2array` (`:252`), `_process_action_seq` (`:268`), `_get_one_hot_bomb` (`:281`) |
| Legal-action generation | `game.py:177` `get_legal_card_play_actions`; `move_generator.py` (`MovesGener`); `move_detector.py`; `move_selector.py` |
| Networks, LSTM, action encoding | `douzero/dmc/models.py`: `LandlordLstmModel` (`:11`), `FarmerLstmModel` (`:46`), `model_dict`, `Model` wrapper (`:87`). LSTM at `:14`. The action is the raw 54-d vector (no learned encoder). |
| Role-specific models | `Model.models[position]` (`models.py:92-98`) |
| Actor processes | `douzero/dmc/utils.py:110` `act()`, spawned at `dmc.py:145-152` |
| ε-greedy | `models.py:38-43` (`flags.exp_epsilon`, default 0.01 in `arguments.py`) |
| Trajectory storage | `utils.py:124-177` (per-seat Python lists, flushed into buffers in chunks of T) |
| Monte Carlo return | `utils.py:145-156` |
| Shared buffers | `utils.py:78` `create_buffers`; `utils.py:42` `get_batch` |
| Learner update | `dmc.py:23` `learn()`, `:19` `compute_loss`, `:154` `batch_and_learn` |
| Optimizer | `utils.py:62` `create_optimizers` (RMSprop per seat) |
| Checkpointing | `dmc.py:191` `checkpoint()` (`model.tar` plus per-seat `*_weights_<frames>.ckpt`) |
| Model synchronization | `dmc.py:57-58` (after every update, into shared-memory actor models) |
| Evaluation | `evaluate.py`, `generate_eval_data.py`, `douzero/evaluation/simulation.py` (`:45` `evaluate`), `deep_agent.py` (`:22`), `rlcard_agent.py`, `random_agent.py` |
| Pretrained agents | `baselines/{sl, douzero_ADP, douzero_WP}/` (installed from the verified mirror; Appendix B) |

### 5.2 Change map: current code → DouZero+-style M1 → two-hand M2

Implement M0, M1, M2 and M2c in **one** codebase, behind a single flag `--hand_model {none, next, both, both_cons, oracle}`, so the arms differ only by that flag. Port DouZero+'s predictor rather than running its repo.

**C1. `env.py`: new `_get_hand_legal(infoset)` and `_get_hand_labels(infoset, K)`.**
- *Current:* nothing.
- *Desired:* `hand_legal` (15, 5) bool, where [r, k] = k ≤ u(r), built from `infoset.other_hand_cards` (public). `hand_label` (K, 15) int8 per-rank counts of `all_handcards[next(s)]` (and of `[other(s)]` when K = 2), using DouZero's rank order (3..A, 2 → 0..12, jokers 20 → 13 and 30 → 14).
- *Scope:* `hand_legal` is used in training and inference. `hand_label` is **training only**.
- *Bugs to expect:* wrong next-seat mapping (order is L → D → U); labels taken *after* the action; the joker index; the rank `2` (value 17) mapping to column 12.
- *Test first:* the label/mask consistency tests in §5.3.

**C2. `env.py`: `get_obs` (`:188`) and `Env.reset`/`step` (`:50`, `:75`).**
- *Current:* returns only public features.
- *Desired:* add `hand_legal` always. Add `hand_label` only when `Env(..., with_labels=True)`, which only the training actor uses; the evaluation path never does.
- *Scope:* both.
- *Bugs to expect:* the evaluation agent accidentally calling the labelled path.
- *Test first:* the leakage test.

**C3. `dmc/env_utils.py`: `_format_observation`.**
- *Desired:* pass `hand_legal` and `hand_label` through as tensors in `env_output`.
- *Scope:* training.

**C4. `utils.py`: `create_buffers` (`:78`).**
- *Desired:* add buffer specs `hand_legal (T, 15, 5) int8` and `hand_label (T, K, 15) int8`.
- *Scope:* training.
- *Bugs to expect:* dtype mismatches.
- *Test first:* buffer shape test.

**C5. `utils.py`: `act()` (`:110`).**
- *Current:* Q over `(z_batch, x_batch)`.
- *Desired:*
  - Under `no_grad`, run the predictor on `z` (1, 5, 162), `x_no_action` (1, 319|430) and `legal` (1, 15, 5), giving logits (1, K, 15, 5) and softmax `p`.
  - For M2c, apply the reflection to get (1, 2, 15, 5). For the oracle, use one-hot true counts.
  - Flatten to (1, 75K) and expand to (n_legal, 75K), then feed the Q-net.
  - Append `hand_legal` and `hand_label` to the per-seat lists, flush them, and **slice them by T exactly like the other lists**.
- *Scope:* training (actor inference).
- *Bugs to expect:* **misaligned lists**, which corrupt labels silently; `x_no_action` stays int8 unless you call `.float()`; per-actor thread oversubscription (set `torch.set_num_threads(1)`).
- *Test first:* the buffer-alignment test.

**C6. `models.py`: new `HandPredictor(seat, K)`, Q-nets taking `pred_dim`, `Model.forward(position, z, x, pred=None, ...)` (`:100`).**
- *Desired:*
  - `HandPredictor`: LSTM(162, 128) → 4 × Linear(512) → 15 heads with 5K outputs; masked logits (−1e7); returns `(logits, probs)` of shape (B, K, 15, 5).
  - Q-net dense1 takes 501|612 + 75K inputs.
  - With `--init_from`, load DouZero-ADP weights and **zero-initialize the new dense1 columns**.
- *Scope:* both.
- *Bugs to expect:* applying the mask per player wrongly (both players share the union mask); feeding masked logits instead of probabilities.
- *Test first:* shape tests; probabilities sum to 1; zero mass on illegal counts; the function-preserving init test.

**C7. `dmc.py`: `learn()` (`:23`).**
- *Desired:*
  - `logits = φ(z, x_no_action, legal)` on (3200, …).
  - `L_pred` = CE(logits.view(−1, 5), labels.view(−1)).
  - `p = softmax(logits).detach()`, reflected for M2c.
  - `Q = θ(z, x ⊕ a, p)`; `L_DMC` = MSE.
  - Step both models, then copy θ and φ to the actors (as at `:57-58`).
- *Scope:* training.
- *Bugs to expect:* a missing `.detach()`, which silently changes M1's semantics; long vs int8 labels; an illegal label under the mask, which makes the loss about 1e7 and explodes.
- *Test first:* the detach test; assert labels are legal.

**C8. `dmc.py`: `train()` (`:61`) and `checkpoint()` (`:191`).**
- *Desired:* create, share and evaluate the predictor models for actors and learner; include φ (and its optimizer state) in `model.tar` and the per-seat checkpoints; log CE and accuracy per seat; load the predictor in the `load_model` and `--init_from` paths.
- *Scope:* training.
- *Bugs to expect:* actors not receiving predictor updates.
- *Test first:* the checkpoint round-trip test.

**C9. `utils.py`: `create_optimizers` (`:62`).**
- *Desired:* an RMSprop optimizer for φ, with the same hyperparameters as DouZero+.
- *Warm-start only:* LR warm-up, because a fresh RMSprop starts with ~0 second-moment estimates and takes roughly 10× steps at first.

**C10. `arguments.py`.**
- *Desired:* add `--hand_model`, `--pred_weight 1.0`, `--seed`, `--init_from`, `--pred_init_from` and `--lr_warmup`.
- *Note:* **DouZero has no seed control.** Seed numpy and torch per actor (`seed*1000 + actor_id`) and for the learner's initialization.

**C11. `evaluation/deep_agent.py`: `DeepAgent` (`:22`, `act` at `:27`).**
- *Desired:* load φ and θ; build public features plus `hand_legal`; run the predictor to get probabilities; feed them to Q.
- *Scope:* inference.
- *Bugs to expect:* calling the labelled `get_obs`.
- *Test first:* the re-deal invariance test.

**C12. `evaluation/simulation.py` (`:21`, `:45`) and `generate_eval_data.py`.**
- *Desired:* return **per-deal** outcomes (deal id, winner, bombs) so you can pair deals and bootstrap; take a fixed `--seed`; keep supporting mixed teams. `tools/dup_eval.py` is a working template.
- *Scope:* inference.
- *Bugs to expect:* the evaluation deal file changing between runs.

**C13. New `analysis/` scripts.** The bootstrap and table builder; offline prediction-quality evaluation (log predictor probabilities and true labels on evaluation games *after* each game; never fed back); the belief-ablation runner (§9).

### 5.3 Tests to write before writing features (pytest; all cheap)
1. **Label consistency.** Over random games: `label[next] + label[other] == counts(other_hand_cards)`; `sum(label[j]) == num_cards_left[target_j]`; the next-seat mapping matches the table in §2.2.
2. **Mask legality.** Labels are always legal under the mask, and `mask[r, k] == (k ≤ u(r))`.
3. **Reflection.** `reflect(p_A)` sums to 1; `E_A + E_B == u`; `CE_B(reflect(p_A), label_B) == CE_A` to 1e-6.
4. **Shapes.** Actor and learner forward passes for K ∈ {0, 1, 2}.
5. **No leakage.** For a fixed infoset, re-deal the two hidden hands *consistently* (same union, same sizes, same history and played cards). The evaluation agent's predictor output and chosen action must be identical. Also grep-assert that the evaluation path never reads `all_handcards`.
6. **Function-preserving warm start.** With zero-initialized columns, `Q_M1(x, p) == Q_DouZero(x)` for random `p`.
7. **Buffer alignment.** Run the actor for a few seeded episodes and check that the label stored at buffer position t equals the label recomputed for the observation stored at t.
8. **Detach.** With CE weight 0, the predictor's parameters don't change after a learner step. With the Q-loss only, the predictor gets no gradient.

---

## Part 6: Training pipeline (end to end)

1. **Random deal.** `Env.reset` shuffles 54 cards: the Landlord gets 20 (including the 3 face-up cards), each Peasant 17 (`env.py:50-73`).
2. **The actor receives an observation.** The acting seat `s` gets an `InfoSet` (`game.py:292`). It contains perfect information, but the feature code reads only public fields. Build `z` (5, 162), `x_no_action` (319|430) and `hand_legal` (15, 5). In training only, also build `hand_label` (K, 15) from `all_handcards`.
3. **Predict hidden hands.** φ_s runs with no gradient and returns `p` (1, 75K). For M2c, apply the reflection; for the oracle, use one-hot truth.
4. **Score legal actions.** θ_s scores `[LSTM(z_batch), x_batch, p broadcast]` with shape (n_legal, 501|612 + 75K), giving (n_legal, 1).
5. **ε-greedy.** With probability 0.01 pick a uniformly random legal action; otherwise take the argmax.
6. **Continue the episode.** Append (`x_no_action`, `z`, action, `hand_legal`, `hand_label`) to seat s's lists, then `env.step(a)`. Repeat until someone empties their hand.
7. **Terminal reward.** With the ADP objective, r = ±2^{bombs} from the Landlord's perspective (`env.py:97`); Peasants get −r.
8. **Monte Carlo target.** Every step seat p took in the episode gets G = r_p (`utils.py:145-156`). No discount, no bootstrap.
9. **Prediction labels.** These were already stored per step at observation time. Never compute them after the action.
10. **Buffers.** Once a seat has more than T = 100 steps, a T-chunk is copied into a free shared buffer and its index goes on the full queue (`utils.py:158-177`).
11. **Learner updates.** Four threads per seat each pull B = 32 chunks (3,200 samples), then run two updates:
    - **A. DMC decision learning (θ_s):** `p = softmax(φ_s(...)).detach()`, recomputed with the **current** φ; `L_DMC = mean (Q_θs(z, x ⊕ a, p) − G)²`; clip at 40; RMSprop step.
    - **B. Supervised hidden-hand prediction (φ_s):** `L_pred` = masked CE averaged over K·15 heads, on the same batch; clip; RMSprop step.
12. **Actor synchronization.** Copy θ_s and φ_s into the shared-memory actor models after each update. Checkpoint every 30 minutes, including φ.

**Which loss trains which parameters:** `L_DMC` trains θ_s only, because the detach blocks it from φ. `L_pred` trains φ_s only. Nothing is shared across seats.

**About the DouZero+ paper:** it says only that the prediction models are trained "concurrently" with cross-entropy, and **does not say whether gradients flow between the two models**. The released code isolates them with `.detach()` and sums both losses into one RMSprop with two parameter groups. That is equivalent to two separate optimizers here, because RMSprop's state is per-parameter.

**Recommendation:**
- Use detach.
- Use two optimizers (cleaner logs) or DouZero+'s summed loss; they are equivalent.
- Recompute predictions at learn time.
- Store predictor *inputs* and labels, not predictions.
- Initialize φ from offline-pretrained weights (Stage 2), so the Q-net never sees garbage beliefs.

**A subtlety to write down in the report:** beliefs are *policy-conditional*. The history → hand mapping depends on how the other players behave, so a predictor trained in self-play will be miscalibrated against unfamiliar opponents (§8).

---

## Part 7: Staged plan

**Compute numbers I measured:**
- One actor process runs at about 2,560 frames/s on one M5 thread.
- A learner update of 3,200 samples takes 39 ms on CPU (4 threads) and 23 ms on MPS.
- A 10k-deal duplicate evaluation costs about 8 core-minutes.

Cloud figures below are estimates: plan on roughly half the M5's per-core speed on x86 cloud vCPUs, and **measure in Stage 1**. **On the laptop, always check load first** (`top -l 1 -n 0`, `memory_pressure`, `pmset -g therm`) and use at most 4 processes under `nice`.

| Stage | Exact success criterion | Compute | Outputs | Sanity checks | Common failure modes | Stop debugging when… |
|---|---|---|---|---|---|---|
| **0. Pretrained eval** (**done**, [run]) | Reproduce Table 1 to within about 0.02 WP / 0.1 ADP. **Met**: WP 0.885 vs 0.889; ADP 2.340 vs 2.288. | 8 core-minutes per 10k-deal pair | Table above; per-deal `.npz`; `tools/dup_eval.py` | ADP(A as Landlord) and ADP(A as Peasants) both plausible; ADP > 0 and WP < 0.5 for ADP vs WP (each objective wins its own metric) | Dead checkpoint link (solved); unpaired deals; multiprocessing counters under `spawn` (DouZero+'s test code) | If a reference doesn't reproduce within 1 day, use a self-trained M0 as the reference instead |
| **1. Pipeline check (M0)** | (a) Local smoke run: 2 actors on CPU, about 2–5M frames, loss falling, checkpoints load into `DeepAgent`, and WP vs random rising across checkpoints. (b) AWS: M0 from scratch for about 1e8 frames, recording WP/ADP vs RLCard every checkpoint (this curve calibrates every later budget). (c) Warm-start sanity: DouZero-ADP continued for 1e7 frames keeps ADP vs DouZero-ADP within ±0.05 (no collapse from re-initializing the optimizer). | (a) ~15 min local, capped. (b) ~2–4 h on a 16-vCPU instance. (c) ~15 min. | Frames/s per actor; learning curve; budget decision | Episode returns shrink in magnitude as play improves; per-seat frame counts are balanced | CUDA assert (use `--actor_device_cpu --training_device cpu`); thread oversubscription; RMSprop re-init shock (use LR warm-up); no seeding | Cap Stage 1 at about 4 days. If from-scratch is slow, go warm-start-only. |
| **2. M1 port** | All §5.3 tests pass. Offline: an M1 predictor trained on frozen DouZero-ADP self-play states (~20k games) beats the **hypergeometric card-counting baseline** CE on held-out games, for every seat. Online: a 5e7-frame warm-start M1 pilot is stable. | Data generation ~10 core-min; offline training minutes on CPU/MPS | CE/MAE/ECE per seat and phase; pilot curve | Equal-length per-seat lists; labels legal; CE drops quickly in the first epoch | Misaligned labels; mask bug (loss ≈ 1e7); CE stuck at the prior | If CE ≈ baseline after 2 days, assume a bug (alignment test). If truly stuck, report that beliefs ≈ counting (a finding) |
| **3. M2, M2c, oracle** | K = 2 tests pass; reflection identity holds to 1e-6; offline M2 `CE_next` ≈ M1 `CE_next`; conservation violations reported. **Oracle pilot shows a clear ADP gain over M0c within the pilot budget.** That is the positive control. | ≈ Stage 2 | Offline table M1/M2/M2c; oracle pilot curve | E_A + E_B = u for M2c | Slot-order bug; M2 head mixing up players | **If the oracle shows no gain**, the protocol is insensitive: raise the budget, or switch to Peasant-only or from-scratch training *before* running the main grid |
| **4. Controlled comparison** | {M0c, M1, M2, M2c} × 3 seeds at a fixed budget F′ (set from the pilots, e.g. 2–3e8 frames), plus O × 1–2 seeds. Evaluate per Part 8; report the primary comparison whatever its sign. | About 14 runs × 4–6 h ≈ 60–80 instance-hours (estimate). Optional from-scratch tier: {M0, M1, M2} × 2 seeds × 5e8–1e9 frames ≈ 60–100 h. | Tables 1–3 and Figures 1–3 (§8) | Seeds of the same arm agree within their CIs; M0c ≈ DouZero-ADP | One seed diverging; checkpoint mix-ups | No extra seeds or budget-hunting after seeing the results (pre-registration) |
| **5. Transformer (optional)** | Only if Stage 4 is done ≥ 2 weeks before the deadline. One parameter-matched variant, 2 seeds, same budget, on M0 or as the predictor's encoder (§11). | ~10–30 instance-hours | One extra row plus attention plots | Parameter counts within ±15% | LR mismatch; padding vs "pass" tokens confused | Drop it after one week regardless of outcome |

---

## Part 8: Evaluation design

**Protocol:**
- **Held-out deals.** Generate 10,000 deals once with a fixed seed never used in training, save them, version them, and never regenerate them. DouZero+'s test script regenerates unseeded deals every round; don't copy that.
- **Duplicate pairing.** Play every deal twice with roles swapped: A as Landlord vs B as Peasants, then B as Landlord vs A as Peasants. Per-deal scores: `ADP_d = (L₁ − L₂)/2` and `WP_d = (1[A wins game 1] + 1[A wins game 2])/2`. Evaluation is greedy and deterministic, so within a run the only noise is the deal sample.
- **Fixed references:** **DouZero-ADP** (primary), DouZero-WP and SL. RLCard and random are sanity checks only; never use them as evidence.
- **Head-to-head:** M2 vs M1 directly is the **pre-registered primary comparison**. The full cross-play matrix is secondary.
- **Per role:** report overall, A-as-Landlord and A-as-Peasants. For the Peasant analyses, also report per seat (`landlord_up` vs `landlord_down`).
- **Matched compute:** compare at equal environment frames (primary), and report wall-clock too, since the predictor costs actor time.
- **Checkpoint rule:** evaluate the final checkpoint at the fixed budget. Draw learning curves on a separate 2k-deal *validation* set. Never pick the best checkpoint using the test deals.

**The fixed-landlord / fixed-teammate experiment.** For each Peasant seat q ∈ {`landlord_up`, `landlord_down`}: the Landlord is DouZero-ADP (fixed), the other Peasant seat is a fixed teammate T, and seat q is filled by M1 or by M2. Use the same deals and compare the per-deal paired team points. Use several teammates: T ∈ {DouZero-ADP's Peasant, M0c, M1's own partner, M2's own partner}.

*Does it isolate the tested Peasant's contribution?* It isolates the policy **given that teammate and that Landlord**, and removes deal luck and opponent differences. Six confounds remain:
1. **Co-adaptation.** Each model learned with its own partner. With a foreign T, both are off-distribution, possibly unequally, so you partly measure *compatibility* rather than skill. Report the gap between self-partner and foreign-partner results.
2. **Belief shift.** Predictors were trained on self-play behavior. Against DouZero-ADP's Landlord and T they are miscalibrated, which may hurt M2 more because it relies more on beliefs. **Measure the predictor's CE in each pairing**: labels can be computed *after* each game, offline.
3. **Seat asymmetry.** M1's target differs by seat (U predicts the Landlord, D predicts the teammate). Analyze each seat separately.
4. **Fixed-Landlord specificity.** One non-adapting Landlord may not generalize, so add DouZero-WP as a second Landlord.
5. **What "teammate modeling" means here.** M2 differs from M1 by explicit beliefs about whichever player M1 left implicit. Only in seat U is that added block exactly the teammate.
6. **Multiple comparisons** across teammates × seats. Treat these as secondary and report all of them.

**Final-report tables and figures:**

*Table 1 (main).* Mean over 3 seeds, 95% CI from a hierarchical bootstrap (resample seeds, then deals), 10,000 duplicate deals vs DouZero-ADP:

| Model | Extra Q input | ADP | WP | ADP as Landlord | ADP as Peasants | Δ ADP vs M1 (head-to-head) | CE next / other (vs card counting) | Frames | Actor fps |
|---|---|---|---|---|---|---|---|---|---|
| M0c | 0 | | | | | | — | | |
| M1 | 75 | | | | | 0 | | | |
| M2 | 150 | | | | | | | | |
| M2c | 150 (derived) | | | | | | (= M1 by construction) | | |
| *O (oracle, cheats)* | *150* | | | | | | *0 (truth)* | | |

- *Table 2*: cross-play matrix (row's ADP vs column).
- *Table 3*: fixed-partner results, per seat × teammate.
- *Figure 1*: ADP vs DouZero-ADP against frames, mean and seed range.
- *Figure 2*: belief quality (CE gain over card counting) by game phase and seat, plus a calibration plot.
- *Figure 3*: the belief-ablation dose-response (§9).

**Numbers to use:**
- **Deals.** Use 10,000 per final comparison (20,000 games). The measured per-deal paired SD between two strong agents is about **1.6 ADP** [run], so SE ≈ 0.016 and the 95% CI is about ±0.031. Detecting Δ = 0.05 with 80% power needs about (2.8 × 1.6 / 0.05)² ≈ **8,000 deals**; Δ = 0.03 needs about 22,000. Policies that share an initialization (M1 vs M2) produce more correlated outcomes, so their SD should be smaller; measure it in the pilot.
- **Seeds.** At least 3 per arm for M0c, M1, M2 and M2c; 1–2 for the oracle. Once the deal-level CI is below the seed SD, extra seeds help more than extra deals.
- **What to report.** Mean ± std across seeds, plus every individual seed's value (with 3 seeds, show all three).
- **CIs.** Within a run, a percentile bootstrap over deals (2,000–10,000 resamples). Across runs, a hierarchical bootstrap. With 3 seeds that interval will be wide; say so.
- **Significance and effect size.** Pre-register one primary test (M2 vs M1 ADP). With 3 vs 3 seeds, an exact permutation test can't go below p = 0.05 (one-sided), so **lead with estimation**: ΔADP with its CI, the standardized paired effect (Δ / SD_deal), and the **fraction of the oracle gap closed**, (M_i − M0c) / (O − M0c). Apply Holm correction to secondary tests, or label them exploratory.

---

## Part 9: Prediction diagnostics

| Metric | Definition | Useful? |
|---|---|---|
| Masked NLL (cross-entropy), nats per head | −log p(true count) | **Yes, primary.** It's a proper scoring rule and equals the training loss |
| **Information gain over card counting** | CE(hypergeometric) − CE(model) | **Yes, the key number.** How much the predictor extracts from *play* beyond public counting |
| Expected-count MAE | Σ_r \|E[c(r)] − c(r)\|, in cards | Yes; interpretable |
| Per-rank argmax accuracy | | Coarse. Dominated by trivial ranks with u(r) ≤ 1 |
| Exact-hand accuracy | Every rank correct | Endgame only (≤ 5 cards) |
| Calibration (ECE, reliability diagram) | Over (rank, count) probabilities | **Yes.** Beliefs are network inputs, so miscalibration matters |
| Conservation violation (M2 only) | Σ_r \|E_A + E_B − u\| and TV(p_B, reflect(p_A)) | **Yes**, for M2 |
| Hand-size violation | Σ_r E_A(r) − n_A | Yes; cheap; applies to every predictor |
| Decision-relevant events | AUROC for: holds a bomb, holds the rocket, can beat the last move | Yes, as a small set |
| Slices | seat × game phase (buckets of cards left) × opponent (self-play vs DouZero-ADP) | Yes |

**The card-counting baseline.** Under a uniformly random split of `U` into hands of sizes `n_A` and `n_B`, the count of rank r in A is hypergeometric: `P(c_A(r)=k) = C(u_r,k)·C(N−u_r, n_A−k) / C(N, n_A)`, where `N = n_A + n_B`. It is exact, parameter-free, and the floor any learned predictor must clear.

**Testing whether better beliefs cause better play** (cheapest and strongest first):
1. **Test-time belief ablation (interventional; no training needed).** Feed the trained Q-net (a) the card-counting prior, (b) the mixture (1−α)·p + α·prior for α ∈ {0, 0.25, 0.5, 0.75, 1}, or (c) beliefs shuffled across states. If ADP falls monotonically as α grows, the Q-net *uses* the beliefs, and within that model better beliefs cause better decisions. A flat curve means the Q-net ignores them (failure mode 3).
2. **Oracle gap.** How much of (O − M0c) each arm closes.
3. **Probes.** Linear probes from M0's and M1's Q-net activations to the true counts, to see what is already encoded implicitly.
4. **Correlation across checkpoints and seeds.** ADP vs CE, with training frames partialled out. This is weak evidence, because both improve with training. Use it only as a supplement.

---

## Part 10: Ablation plan, ranked by value/cost

| Ablation | Value | Cost | Verdict |
|---|---|---|---|
| **M2c: conservation-constrained** (analytic complement) | **Very high**: decides how to interpret everything | **Very low** (~20 lines; same run cost as M1) | Do |
| **M1: next-player only** | Essential baseline | Medium | Do |
| **M2: both hidden players** (independent heads) | Essential: it is your method | Medium | Do |
| **M0: no explicit prediction** | Essential | Medium | Do (as M0c in warm-start) |
| **True-hand oracle** | High: ceiling plus sensitivity check | Low–medium (1–2 seeds, no predictor) | Do |
| Opponent-only / teammate-only (Peasants) | Medium. Under conservation this is mostly a relabeling, and M1 already covers one of each across U and D | Medium | Optional, as a "seat-target swap" |
| Independent two-hand prediction | This *is* M2 | — | — |
| Transformer history encoder | Low for your question | Medium–high | Stretch |
| Different optimizer | Low: doesn't touch the hypothesis | Medium | Skip |
| End-to-end gradients (no detach) | Low–medium | Low | Skip initially |
| Coach network | Off-topic | Medium | Skip, and say "without coach" |

**The four experiments that make the strongest solo report:**
1. The main comparison of M0c, M1, M2 and M2c: 3 seeds, warm-start, duplicate evaluation.
2. The oracle ceiling, reported as the fraction of the oracle gap each arm closes.
3. An offline belief-quality study (information gain over card counting, calibration, violations, by phase and seat) plus the test-time belief ablation.
4. Fixed-partner evaluation, per seat.

---

## Part 11: Optional modernization (Transformer)

Is it technically sensible? Yes, but your history is short (15 moves), and most belief-relevant information already sits in the count features. Expect parity.

- **What to encode.** One token per move for the last 15 moves, which matches DouZero. Optionally, the *full* history (≲ 100 moves) with padding.
- **Tokens.** Each token is the 54-d card vector, the mover's seat (one-hot of L/U/D, or relative to you), and an "is pass" flag, projected linearly to d. DouZero encodes both passes and padding as all-zeros. **Separate them**: use a padding mask for padding, and real "pass" tokens for passes.
- **Positional encoding.** A learned embedding of recency (0 = most recent) plus the seat embedding. DouZero's 3-moves-per-row grouping aligns seats implicitly; per-move tokens need the seat made explicit.
- **Attention mask.** A key-padding mask for the early game. No causal mask is needed, since the encoder only sees the past.
- **Pooling.** A [CLS] token or the most recent token, projected to 128-d so the Q-head stays unchanged.
- **Parameter matching.** The LSTM(162 → 128) has **149,504** parameters. A 2-layer encoder with d = 96, 4 heads and FF = 192 has about 150k, plus ~19k for the input/output projections and positions, which is within 15%.
- **Fair comparison.** Change only the encoder. Keep frames, seeds (≥ 2), learning rate and evaluation identical, and report parameters and fps. Run it on M0 (cleanest) *or* as the predictor's encoder. Don't stack it with the M2 change.
- **Compute.** DouZero re-encodes the identical history once per legal action (`z_batch`). Encode once and broadcast, in *both* arms. A 15-token encoder costs about the same as the LSTM.
- **Likely to help?** Probably not for the Q-net. **The more interesting placement is the predictor**: who passed on what, across the whole game, is exactly what hand inference needs. It is also measurable cheaply offline by CE and tied to your hypothesis. If you do one Transformer experiment, do this one.
- **If it fails**, the result is still informative: the 15-move LSTM isn't the bottleneck, or beliefs are limited by information rather than by the encoder (the CE gap to card counting tells you which).
- **Context:** GtDou (2026) already pairs a Transformer with hand guessing; cite it.

---

## Part 12: Compute and scope

| Risk | Tasks |
|---|---|
| **LOW** | Stage-0 evaluation and harness (done); unit tests; offline dataset and predictor training; belief ablations; statistics; plots |
| **MEDIUM** | Wiring the predictor into the multiprocess actor/learner and shared buffers; AWS setup and cost control; warm-start runs; 3 seeds; partner evaluations |
| **HIGH** | From-scratch training into DouZero+'s regime (very long runs × variants × seeds); coach network; Transformer; bidding; debugging multiprocessing deadlocks on macOS |

**Necessary to answer the research question:** M0c, M1, M2 and M2c at a matched budget; duplicate evaluation with CIs; prediction metrics; ≥ 2–3 seeds.

**Cut if time runs out:** Transformer, coach, from-scratch replication, seat-target swap, then partner-swap breadth. Keep the oracle: it's cheap and makes everything interpretable.

| Level | Contents |
|---|---|
| **Minimum viable** | Warm-start M0c, M1, M2 and M2c (2 seeds each); 10k-deal duplicate evaluation vs DouZero-ADP plus M2 vs M1 head-to-head, with CIs, per role; offline CE vs card counting per seat; a conservation-based explanation of whatever you find |
| **Strong target** | MVP + 3 seeds + oracle and gap closed + belief-ablation dose-response + per-seat analysis + fixed-partner evaluation + probes |
| **Stretch** | From-scratch replication (≥ 5e8 frames); Transformer predictor encoder; joint predictor with the hand-size constraint; seat-target swap |

**AWS.** DouZero's networks are small, so CPU instances suffice: 16-vCPU compute-optimized instances, with roughly 12 actors, a CPU learner, and `OMP_NUM_THREADS=1` per actor. Set up a **budget alarm first** and prefer spot instances. My rough estimate: at a typical rate of about $0.6–0.8/h on demand, Stage 4 costs tens of dollars; check current prices.

---

## Part 13: Failure modes

| Failure | How you'd detect it | Diagnostic experiment | Scientific conclusion |
|---|---|---|---|
| M2's beliefs are more accurate, but ADP doesn't move | CE / violations better; ADP CIs overlap | Belief ablation; M2c vs M2; probe M1 for B | The Q-net already derives B or doesn't need it. Redundancy confirmed, which is expected under conservation |
| Predictor collapses | CE ≈ hypergeometric and flat; probabilities ≈ prior | Alignment and mask tests; overfit one batch; check the LR | Most likely a bug. If not, play history carries little information beyond counting at this budget |
| Q-net ignores predictions | Flat belief-ablation curve; near-zero first-layer weights on belief columns | Prior/shuffle swap; oracle control | If the oracle gains but M1/M2 don't, the beliefs aren't informative enough. If the oracle is flat too, the protocol is insensitive |
| Unstable training | Loss spikes; Q divergence; one seed diverging | Return scale (2^bombs); grad norms; repeat the seed; try `logadp` | Report it; compare under a stable objective; never tune per arm |
| Overfits to self-play partners | Strong in self-play, weak vs references | Cross-play matrix; belief CE under foreign opponents | Beliefs and policies are population-specific; report the generalization gap |
| Teammate modeling helps only with the self-play teammate | Fixed-partner gap larger for M2 | Several teammates T | M2 learned partner-specific conventions, not general teammate modeling |
| Gain comes from capacity | M2 > M1 | M2c (same input width); a "prior-as-input" 75-d control | If the prior-input control also helps, it isn't the beliefs. The Q-net is only +2.6% larger, so this is unlikely |
| Second prediction adds nothing (conservation) | M2c ≈ M2 ≈ M1 | — | The expected result. Report it as a clean negative result with a mechanism |
| M1 ≤ M0 (DouZero+ doesn't replicate) | ADP CI ≤ 0 | Budget sensitivity; oracle gap | Consistent with IJCAI-24 and AlphaDou; a replication finding in its own right |
| Leakage | Evaluation ADP suspiciously near the oracle; re-deal test fails | Unit test; code audit | Fix and re-run; never report leaked numbers |
| Seat or label misalignment | CE stuck; label sums wrong | Tests 1, 2 and 7 | Bug |
| Seed variance swamps the effect | Seed SD > Δ | Power analysis | Inconclusive; report CIs and the power analysis |
| Transformer hurts | Lower ADP or CE | Parameter/LR check; 2 seeds | The LSTM is sufficient; report it |

A negative result is fine for CS230 **if** M2c, the oracle and the belief ablation are in place. Together they turn "it didn't work" into "here is why, mechanistically."

---

## Part 14: Timeline

*Corrected 2026-10-17 from the CS230 project page.* The proposal was due **Tue Oct 13**, the milestone is due **Tue Nov 10** (you must also meet your TA before it), the final report is due **Fri Dec 4 with no late days**, and the poster session is **Wed Dec 9**. An earlier version of this table assumed Nov 13 and Dec 11, so everything is now one week tighter: the main training grid must be running by about Nov 10.

| Week | Focus | Deliverable |
|---|---|---|
| **W1** (Oct 19–25) | Proposal (if not yet submitted, check the late policy on Ed now). Add seeding and the duplicate-eval harness. Set up AWS (budget alarm first). Stage 1a (local, capped) and 1b (1e8 frames on AWS); warm-start sanity check. | Proposal; frames/s on AWS; M0 learning curve |
| **W2** (Oct 26–Nov 1) | §5.3 tests, then labels, mask and predictor behind `--hand_model`. Generate offline data from DouZero-ADP self-play; train M1, M2 and M2c predictors offline; build the hypergeometric baseline. | **First real result**: belief-quality table per seat and phase |
| **W3** (Nov 2–9) | Wire the predictor into actor and learner; function-preserving warm start; pilots of M1 and the oracle (1 seed, ~5e7 frames); fix F′. Meet your TA. Write the milestone. | Sensitivity check (oracle gains); **milestone due Tue Nov 10** |
| **W4** (Nov 10–15) | Launch the main grid: M0c, M1, M2, M2c × 3 seeds, plus the oracle. | All runs started |
| **W5** (Nov 16–22) | Evaluate all final checkpoints (references plus head-to-head); bootstrap; belief ablations; fixed-partner evaluations. | Tables 1–3 |
| **W6** (Nov 23–29, Thanksgiving) | Buffer week; probes; figures. **Drop the Transformer** unless everything above is finished by about Nov 20. | Figures 1–3 |
| **W7** (Nov 30–Dec 4) | Write the final report and reproducibility package (configs, seeds, deal file, scripts). | **Final report due Fri Dec 4 (no late days)** |
| **Dec 5–9** | Poster. | **Poster session Wed Dec 9** |

---

## Part 15: Final verdict

1. **Is the current proposal acceptable for CS230?** Yes, after the must-fix edits. As written it is borderline: a sharp TA will catch the conservation issue and the missing statistics.
2. **Is the main hypothesis technically meaningful?** Yes, **as a representation hypothesis**: "does explicitly representing both hidden hands help the DMC learner?" As an information hypothesis ("more information helps") it is vacuous, because the second hand adds no information.
3. **Can one student implement it on the DouZero codebase?** Yes. Expect roughly 400–600 changed lines across about 8 files plus tests. Compute is affordable with warm-start on AWS CPU instances.
4. **Top 5 risks:**
   1. A null result driven by redundancy (likely). Mitigation: the framing, M2c, and the belief analysis.
   2. M1 doesn't beat M0 (the literature contests it). Mitigation: measure it and report it.
   3. Seed variance swamping small effects. Mitigation: paired evaluation, 3 seeds, the oracle sensitivity check.
   4. Silent pipeline bugs: label alignment, seat mapping, leakage. Mitigation: write the tests first.
   5. Time overruns from from-scratch training and multiprocess debugging. Mitigation: warm-start, AWS, stop rules.
5. **The 5 things to do first:**
   1. Apply Appendix A and submit.
   2. Keep the Stage-0 harness.
   3. Set up AWS with a budget alarm and run Stage 1.
   4. Write the label/mask/reflection/leakage tests, then port DouZero+'s predictor behind a flag.
   5. Generate offline DouZero-ADP self-play data and train the M1, M2 and M2c predictors offline.
6. **What to absolutely NOT spend time on:** running DouZero+'s repo as-is; the coach network; bidding; GPU tuning; optimizer ablations; the Transformer before the main result; a 1e10-frame from-scratch reproduction; random or RLCard agents as evidence; adding seeds after looking at the results.
7. **What the minimum viable final result looks like:** Table 1 with M0c, M1, M2 and M2c (≥ 2–3 seeds) under warm-start at a fixed budget, evaluated on 10k duplicate deals vs DouZero-ADP plus the M2 vs M1 head-to-head, with CIs per role; a belief-quality table (information gain over card counting) per seat; and a clear conservation-based explanation of whatever you find.
8. **What would make this an excellent CS230 project:** everything in the MVP, plus:
   - the oracle ceiling with the fraction of the gap closed;
   - the belief-ablation dose-response showing whether beliefs are actually used;
   - a per-seat test of the pre-registered `landlord_down` prediction;
   - multi-partner fixed-partner evaluation;
   - probes revealing what M0 and M1 encode implicitly;
   - an honest treatment of the contested DouZero+ effect;
   - clean, tested, reproducible code.

---

## Appendix A: Targeted LaTeX changes to the proposal

To stay within the 2-page limit, cut the NFSP / Deep CFR / RLCard / Suphx / ReBeL clause in Learning Method down to one short clause to make room.

**Problem Description**: after "My proposed model instead predicts both hidden hands." add:

```latex
DouZero+ predicts the hand of the \emph{next} player to act, so its target depends on the seat:
the Landlord and the up-Peasant model an opponent, while the down-Peasant models its teammate.
Moreover, DouZero's observation already contains the union of the two hidden hands and their
sizes, so the hidden state is only the \emph{partition} of a known multiset: once one hidden hand
is known, the other is determined (card conservation). Predicting both hands therefore adds no new
information; I test whether representing both hidden hands \emph{explicitly} helps the Deep Monte
Carlo learner, using a control that derives the second hand analytically from the first.
```

**Learning Method**: replace the two equations with:

```latex
\[
\mathcal{L}_{\mathrm{DMC}}(\theta_s)=\mathbb{E}\big[(Q_{\theta_s}(o_t,a_t,\mathrm{sg}[\hat b_t])-G_t)^2\big],
\qquad G_t=\pm 2^{\#\mathrm{bombs}},
\]
\[
\mathcal{L}_{\mathrm{pred}}(\phi_s)=-\frac{1}{15K}\sum_{j=1}^{K}\sum_{r=1}^{15}
\log p_{\phi_s}\!\big(c_j(r)=c^{*}_j(r)\mid o_t\big),
\]
```

and follow them with:

```latex
where $G_t$ is the undiscounted terminal ADP reward from seat $s$'s perspective, $p_{\phi_s}$ is a
5-way softmax over the number of rank-$r$ cards held by hidden player $j$, masked to counts at most
$u_t(r)$ (the unseen rank-$r$ cards), $\hat b_t$ concatenates the $K$ predicted distributions, and
$\mathrm{sg}$ is a stop-gradient (as in the released DouZero+ code). $K=1$ gives DouZero+ (M1,
without the coach network) and $K=2$ my model (M2). The control M2c uses the M1 predictor and
derives the second hand by conservation, $p(c_2(r)=k)=p(c_1(r)=u_t(r)-k)$, giving the Q-network the
same input as M2 with no extra parameters. An oracle that sees the true hands bounds achievable gains.
```

**Evaluation**: replace the mirrored-deal sentences with:

```latex
I evaluate on a fixed set of 10{,}000 held-out deals, each played twice with roles swapped, against
the released DouZero-ADP, DouZero-WP and SL agents and head-to-head (M2 vs.\ M1 is the
pre-registered primary comparison). I report ADP (primary) and WP per role, with paired bootstrap
95\% CIs over deals and mean $\pm$ std over three training seeds at a matched number of environment
frames. To keep compute feasible, all models start from the released DouZero-ADP weights (new input
columns zero-initialized, so every arm starts from the same policy) and are fine-tuned by self-play.
I also measure prediction quality (cross-entropy relative to a card-counting baseline, calibration)
and test whether the Q-network uses the beliefs by replacing them at test time.
```

**Related work** (one sentence):

```latex
Closest to this work are DouZero+ and its journal version~\cite{zhao2024fulldouzeroplus},
PerfectDou~\cite{yang2022perfectdou}, which uses perfect information in a PPO critic, and oracle
guiding for DMC~\cite{luo2024oadmcdou}; an independent re-training at smaller compute placed DouZero+
below the released DouZero~\cite{luo2024oadmcdou}, and AlphaDou~\cite{lei2024alphadou} reports no
quantifiable gain, which motivates a controlled multi-seed re-test rather than a new architecture.
```

**BibTeX** (verify page and issue details via the library before the final report):

```bibtex
@article{zhao2024fulldouzeroplus,
  title   = {Full {DouZero+}: Improving {DouDizhu} {AI} by Opponent Modeling, Coach-Guided Training and Bidding Learning},
  author  = {Zhao, Youpeng and Zhao, Jian and Hu, Xunhan and Zhou, Wengang and Li, Houqiang},
  journal = {IEEE Transactions on Games},
  volume  = {16},
  pages   = {518--529},
  year    = {2024},
  doi     = {10.1109/TG.2023.3299612}
}
@inproceedings{yang2022perfectdou,
  title     = {{PerfectDou}: Dominating {DouDizhu} with Perfect Information Distillation},
  author    = {Yang, Guan and Liu, Minghuan and Hong, Weijun and Zhang, Weinan and Fang, Fei and Zeng, Guangjun and Lin, Yue},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022}
}
@inproceedings{luo2024oadmcdou,
  title     = {Enhanced {DouDiZhu} Card Game Strategy Using Oracle Guiding and Adaptive Deep {Monte Carlo} Method},
  author    = {Luo, Qian and Tan, Tien Ping and Zha, Daochen and Zhang, Tianqiao},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI-24)},
  year      = {2024}
}
@article{lei2024alphadou,
  title   = {{AlphaDou}: High-Performance End-to-End {Doudizhu} {AI} Integrating Bidding},
  author  = {Lei, Chang and Lei, Huan},
  journal = {arXiv preprint arXiv:2407.10279},
  year    = {2024}
}
```

---

## Appendix B: Stage-0 artifacts and how to reproduce them

- **Checkpoints** are in `DouZero/baselines/{douzero_ADP,douzero_WP,sl}/`. They come from the Hugging Face mirror `palemoky/douzero-baselines`; an untouched copy is in `reference_code/hf_douzero_baselines/`.
  - Provenance: the WP files' git blob hashes (`75456a66d5`, `ae3e490357`, `8dd68da61a`) equal those in the independent GitHub fork `tianqiraf/DouZero_For_HappyDouDiZhu`.
  - All nine files load with `torch.load(..., weights_only=True)` and match DouZero's architecture exactly.
  - The official Google Drive folder returned HTTP 404 on 2026-10-17.
- **Environment:** a `.venv/` at the project root (Python 3.14.0, torch 2.14.1, GitPython, rlcard, gdown). Caches are kept inside the project.
- **Evaluation script:** `tools/dup_eval.py`. It does duplicate deals, records per-deal outcomes, computes bootstrap CIs, and writes a `.npz` to the current directory. Check load first and cap the workers:
  ```bash
  top -l 1 -n 0 | head -4; memory_pressure | tail -1
  nice -n 19 timeout 3600 .venv/bin/python tools/dup_eval.py DouZero DouZero/baselines/douzero_ADP rlcard 10000 2026 4
  ```
  The table at the top used 8 workers, about 1 minute per pair. That overheated the laptop; use ≤ 4 locally, or run it on AWS.
- **Measured throughput:** 2,559 frames/s per actor process (one M5 thread, untrained weights); learner 39 ms (CPU) / 23 ms (MPS) per 3,200-sample update.

## Appendix C: Sources

- DouZero: Zha et al., ICML 2021, `papers/01_DouZero_ICML2021.pdf`; code: https://github.com/kwai/DouZero
- DouZero+: Zhao et al., CoG 2022, `papers/02_DouZeroPlus_2022.pdf`; code: https://github.com/submit-paper/Doudizhu_plus
- Full DouZero+: IEEE ToG, DOI 10.1109/TG.2023.3299612 (paywalled, not read): https://ieeexplore.ieee.org/document/10197166/
- OADMCDou: Luo, Tan, Zha, Zhang, IJCAI-24, `papers/10_OracleGuiding_AdaptiveDMC_IJCAI2024.pdf`; https://www.ijcai.org/proceedings/2024/0660.pdf
- AlphaDou: Lei & Lei, arXiv:2407.10279, https://arxiv.org/abs/2407.10279
- PerfectDou: Yang et al., NeurIPS 2022, https://arxiv.org/abs/2203.16406; code (eval-only): https://github.com/Netease-Games-AI-Lab-Guangzhou/PerfectDou
- Luo & Tan, Applied Soft Computing 2024, DOI 10.1016/j.asoc.2024.111545 (paywalled): https://www.sciencedirect.com/science/article/abs/pii/S1568494624003193
- CAPRE_DMC, Applied Soft Computing 2025 (paywalled): https://www.sciencedirect.com/science/article/abs/pii/S1568494625011366
- GtDou, Kong, 2026 conference paper (full text not found; ResearchGate listing)
- DouRN: arXiv:2403.14102; Douzero_Resnet: https://github.com/Vincentzyx/Douzero_Resnet; Big 2 self-play: arXiv:2605.28863 (none of these predict hidden hands, per an earlier literature check)
- Checkpoint mirrors: https://huggingface.co/palemoky/douzero-baselines and https://github.com/tianqiraf/DouZero_For_HappyDouDiZhu
