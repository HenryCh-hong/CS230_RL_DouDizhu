# CS230 hand-belief study: design notes

Local branch `cs230-hand-belief`, based on kwai/DouZero `718a5c9`. Nothing is pushed.

## Question

Does explicitly representing the recoverable second hidden-hand belief help a Deep Monte Carlo agent learn better decisions than a next-player-only belief representation?

From the acting seat, the two hidden hands satisfy **H_next + H_other = U** for every card kind. Here U is the public union of unseen cards, and both hand sizes are public. A belief about the next player's hand therefore determines the belief about the other hidden hand.

## Arms

All arms share one code path and one Q-network. They differ only in what fills the belief input.

| Arm | `--hand_model` (planned) | Slot 1 (75) | Slot 2 (75) | Predictor |
|---|---|---|---|---|
| M0c | `none` | 0 | 0 | none (continued DouZero) |
| M1 | `next` | p_next | 0 | frozen next-player predictor (DouZero+-style) |
| M2c | `derived_both` | p_next | reflect(p_next, U) | **the same frozen checkpoint as M1** |
| M2 | `learned_both` | p_next (two-hand model) | p_other (two-hand model) | separately pretrained two-hand predictor, frozen |
| Oracle | `oracle` | one-hot true next | one-hot true other | none (positive control only) |

**Comparisons** (decided 2026-10-17):

| Comparison | Role | Reading |
|---|---|---|
| **M2c vs M1** | Primary, clean | Same frozen predictor and same information. The only difference is whether the second belief is represented |
| **M2 vs M2c** | Secondary | A learned two-belief system against a conservation-completed one. It is not an isolated test of slot 2: M2's slot 1 comes from its own predictor, and that predictor has 1,729,099 more parameters (the second set of 15 heads) |

- **Per-head quality.** M2's prediction quality is tracked separately as CE_next and CE_other, so any difference in slot-1 quality is visible.
- **No M2\* variant.** M2\* would share slot 1 with M1. It may become an optional later ablation, and only if the results require it.

## Belief input: 150-d, identical for every arm

- **Layout.** b = [slot 1 | slot 2]. Each slot holds 15 card kinds × 5 count classes, flattened kind-major (index 5·r + k), with kinds in `hand_belief.KINDS` order.
- **Q-network input.** `dense1` takes [LSTM 128 | x 373/484 | b 150], i.e. 651 inputs for the landlord and 762 for the peasants. The belief is appended after x, following the DouZero+ convention.
- **No gradient into predictors.** Beliefs enter as detached probabilities, so the RL loss never updates a predictor.
- **Warm start from DouZero-ADP.** All existing weights are copied, and the 150 new `dense1` columns start at exactly zero.
  - At step 0 the belief cannot change Q at all; a test checks this bit for bit.
  - Every arm matches DouZero-ADP's Q-values within 1e-6; the measured difference on the Mac is exactly 0.
- **Unused slots stay exactly zero.** A zero input gives its column zero gradient. M0c's two slots and M1's second slot therefore never change: M0c is DouZero training continued, and M1 has the 75-d DouZero+ input.

## Predictors are frozen in the main experiment

- **Frozen by default.** Predictors are pretrained offline on DouZero-ADP self-play and then frozen. The setting is controlled by `--freeze_predictor` (planned; default on).
- **One shared checkpoint for M1 and M2c.** The two arms load exactly the same next-player checkpoint. They differ only in whether slot 2 holds reflect(p_next, U).
- **M2.** Its two-hand predictor is pretrained on the same data and frozen.
- **M0c and Oracle.** Neither uses a predictor.
- **Deviation from the proposal.** `proposal/main.tex` (line 46) says predictors are "pretrained … then fine-tuned jointly".
  - Under joint fine-tuning, each arm keeps training its predictor on its own self-play data, so M1's and M2c's predictors would drift apart. A difference between the arms could then come from the predictors rather than from the second slot.
  - A single shared frozen checkpoint removes this confound. That makes the comparison cleaner, and M2c vs M1 is the primary comparison.
  - Joint fine-tuning remains an optional ablation (`--freeze_predictor` off). The main experiment does not need it.
- **The submitted proposal stays unchanged.** The frozen-predictor design and the refined M2c control are implementation-stage methodological improvements. Each deviation and its rationale is recorded here for the milestone report.

## Model layer (stage 2)

**`douzero/dmc/belief.py`** holds the torch counterparts of `hand_belief.py`, cross-checked by T12:
- `union_counts_from_obs` and `legal_count_mask` (k ≤ u);
- `mask_logits` (sets impossible counts to −inf) and `masked_softmax`;
- `reflect_belief`, `one_hot_counts`, and `flatten_belief` (kind-major, index 5·kind + count);
- `hand_cross_entropy`, which returns one value per predicted hand, shape (K,): CE_next and CE_other.

`build_belief(hand_model, x_no_action, probs, labels)` is the single place where the arms differ.
- It returns the (B, 150) belief.
- It detaches the probabilities, so a Q-loss can never reach a predictor.
- It raises an error if true hands are passed to any arm except the oracle.
- `PREDICTOR_TARGETS` gives the predictor size K for each `--hand_model` (0 means no predictor).

**`douzero/dmc/models.py`:**
- **Q-networks.** `LandlordLstmModel` and `FarmerLstmModel` gain `belief_dim=0` and a `belief=` argument to `forward`, appended after x as DouZero+ does. `belief_dim=0` is the original network, so the DouZero training code, `DeepAgent` and the released checkpoints are untouched. `forward` raises an error if a belief is missing, unexpected, or still carries gradients. A wiring mistake therefore cannot silently turn an arm into DouZero, or let a Q-loss reach a predictor.
- **`HandPredictor(position, num_targets)`** returns `logits` and `probs` of shape (B, K, 15, 5). Logits of impossible counts are −inf, and the probabilities there are exactly 0.
- **`make_arm(position, hand_model, predictor=None)`** returns the arm's Q-network (identical for all arms) and its predictor. M1, M2c and M2 must be given their predictor explicitly, never a silently created untrained one. It checks that the predictor's seat and K match the arm.
- **`warm_start_from_douzero(position, state_dict)`** copies every weight and adds 150 zero columns to `dense1`.
- **`save_hand_predictor` / `load_hand_predictor(path, freeze=True)`.** Loading freezes by default: eval mode with `requires_grad_(False)`.

**Parameter counts:**

| Network | Landlord | Peasant |
|---|---|---|
| DouZero Q-network | 1,457,665 | 1,514,497 |
| Belief Q-network (+150 × 512) | 1,534,465 | 1,591,297 |
| HandPredictor, K = 1 (identical to DouZero+) | 2,895,947 | 2,952,779 |
| HandPredictor, K = 2 | 4,625,046 | 4,681,878 |

**Warm start.** On this Mac, the measured max |Q_new − Q_DouZero-ADP| is exactly 0 for all three seats and for zero, random, large and constant beliefs. The test tolerance is 1e-6, for portability across BLAS backends.

**Fidelity to DouZero+** (`oppo_modeling/dmc/models.py`). The architecture is the same: LSTM 162→128, 4 × 512 ReLU layers, 13 rank heads of the form 512→256→5, 2 joker heads of the form 512→5, and heads in kinds order (DouZero+ `pre3..pre13, pre1, pre2, presmall, prebig`). Deliberate differences:
1. Impossible counts get logits of −inf rather than "minus 1e7". Exact zeros then hold by construction, matching `hand_belief.masked_softmax`.
2. The predictor computes its own mask from `x_no_action[54:108]` instead of taking a `legal` input. A caller therefore cannot pass in a wrong or leaky mask.
3. The head ReLUs are not in-place. This has no functional effect.
4. Predictors are trained offline and frozen, not trained jointly with the Q-loss (see above).

## Conventions (`douzero/env/hand_belief.py`)

| Item | Convention |
|---|---|
| Card kinds | `KINDS` = 3…14, 17 ('2'), 20, 30 (the jokers). Counts run 0–4, or 0–1 for jokers |
| Hidden players | `hidden_players(position)` returns (next, other). "Next" is the seat that acts next: L→D, D→U, U→L. This is the DouZero+ target, and for `landlord_down` it is the teammate |
| Labels | `hand_labels(infoset)` returns true counts with shape (2, 15), rows ordered (next, other) |
| Union | `union_counts(infoset)`, or `union_counts_from_obs(x_no_action)`, which decodes `x[54:108]` |
| Hand sizes | `hidden_sizes_from_obs(position, x_no_action)` |
| Legal-count mask | `legal_count_mask(u)`: count k is allowed iff k ≤ u_r. `masked_softmax` gives exactly zero mass outside the mask |
| Reflection (M2c) | `reflect_belief(p_next, u)`: p_other(r, k) = p_next(r, u_r − k) for k ≤ u_r, otherwise 0 |
| Baseline | `hypergeometric_belief(u, n)` is the card-counting reference used in prediction analysis |
| Metric | `belief_cross_entropy` is the mean over kinds of −log p(true count), in nats |

## Leakage policy

- **`InfoSet.all_handcards` holds the true hands**, and it is present in every agent's infoset. Only `hand_belief.hand_labels` may read it.
- **`hand_labels` callers.** It may be called only by offline predictor-data generation and by the Oracle arm. It is never called on a non-oracle decision path.
- **Order of `InfoSet.other_hand_cards` leaks.** It is the union, but it is concatenated in seat order, so its order reveals the split. Use only order-invariant functions of it, such as card counts.
- **Enforced by `tests/test_leakage.py`:**
  - T6a swaps cards between the hidden hands while keeping sizes and union fixed. The observation, the DouZero-ADP Q-values and the chosen action must not change. Positive controls show that the swap exposes a feature that reads either leak channel. Each new belief agent will be added to T6a.
  - T6b is a static allow-list of the files that mention `all_handcards` or `hand_labels`. Adding a file to it is a reviewed decision. The only addition so far is `trace_decision.py`, a debug tool that displays the labels and runs the oracle arm.
  - In stage 2, T6a was extended to the M0c/M1/M2c/M2 belief and Q path: after a swap, beliefs and Q-values must be bit-identical. The Oracle's belief must change.

## Tests and compute

- **Running the tests.** From `DouZero/`, run `../.venv/bin/python -m pytest tests -p no:cacheprovider -q -rs`. This is single-process and takes a few seconds. pytest is installed only in the project `.venv`.
- **Checkpoints needed.** Six tests need `baselines/douzero_ADP/`: the ADP swap test, the weight-copy check, the three warm-start checks and the Q-network checkpoint round trip. Without it they are skipped, so confirm the summary reports 0 skipped. `baselines/` is git-ignored.
- **Tested versions.** Python 3.14.0, torch 2.14.1, numpy 2.5.4, pytest 9.1.1. `requirements.txt` only says `torch>=1.6.0`, so pin these on the remote machine.
- **Local vs remote.** This Mac runs correctness tests only. Dataset generation, RL training, pilots and large evaluations run remotely, after approval.

## Making training visible

Every stage should make the data flow, tensor shapes, losses and training behavior visible enough to follow.
- **`docs/TRAINING_WALKTHROUGH.md`** (project root, outside this repo) walks through the pipeline in 22 steps. Each step names its file, function, tensor shapes, information class and gradient flow. It also has the arm comparison, the gradient-flow diagrams, and worked examples of the CE and Q-losses.
- **`trace_decision.py`** prints one decision for every arm: observation shapes, predictor outputs against card counting, belief layouts, Q over the legal actions, and the epsilon-greedy choice. It is single-process and takes about a second.
- **Planned.**
  - Stage 3: `trace_episode.py`, which writes one full game to `debug/episode_trace.txt`.
  - Training runs: predictor and RL metrics logged at sensible intervals, as CSV/JSON with plots under `artifacts/plots/` and a `summary.md` per run under `artifacts/runs/<run_name>/`. These will come from remote runs, never from the laptop.

## Open items before pilots

These come from the stage-2 review and must be settled in the stages named.
- **Seed variance (evaluation design).** Paired, duplicate deals remove card luck, not training-seed variance. Plan at least 3 seeds per arm and analyze paired per-seed differences. Fix the minimum detectable effect and the deal count in advance.
- **Predictor drift (Stage 3/4 metrics).** Predictors are trained on DouZero-ADP self-play, and P(hand | public history) depends on the policies that produced that history, so beliefs can drift as RL changes the policy. M1 and M2c share the predictor, which keeps the primary comparison matched. Still, log predictor CE on current-policy self-play at every checkpoint.
- **M2 capacity.** Report the 1,729,099 extra predictor parameters alongside M2 vs M2c.
- **Strict loading in evaluation agents.** DouZero's `DeepAgent` silently keeps only matching checkpoint keys. The belief-aware agent and the `OracleAgent` must load strictly.
- **Seeded evaluation deals.** DouZero's `generate_eval_data.py` is unseeded; `tools/dup_eval.py` seeds its deals.
- **ADP units.** Evaluation scores the landlord side at ±2·2^bombs, twice the training target. State the units with every reported number.

## Status

1. **Done (stage 1).** `hand_belief.py` pure functions and tests T1–T6b. No change to the Q-network, actor, learner, buffers or training loop.
2. **Done (stage 2).** The model layer: `HandPredictor`, the common 150-d belief interface, all five arms, the function-preserving warm start, stop-gradient checks, checkpoint round trips, and tests T8–T12 with the T6a extension. The trace tool and the walkthrough were added too. The actor, learner, buffers and training loop are still unchanged.
3. **Next (awaiting approval).** Actor/learner wiring (T7), the `--hand_model` and `--freeze_predictor` flags, and `trace_episode.py`.
