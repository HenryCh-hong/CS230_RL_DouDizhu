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

## Belief input: 150-d, identical for every arm

- **Layout.** b = [slot 1 | slot 2]. Each slot holds 15 card kinds × 5 count classes, flattened kind-major (index 5·r + k), with kinds in `hand_belief.KINDS` order.
- **Q-network input.** `dense1` takes [LSTM 128 | x 373/484 | b 150], i.e. 651 inputs for the landlord and 762 for the peasants. The belief is appended after x, following the DouZero+ convention.
- **No gradient into predictors.** Beliefs enter as detached probabilities, so the RL loss never updates a predictor.
- **Warm start from DouZero-ADP.** All existing weights are copied. The 150 new `dense1` columns start at exactly zero, so at step 0 every arm computes DouZero-ADP's Q-values exactly; a test checks this as an exact Q match.
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
  - T6b is a static allow-list of the files that mention `all_handcards` or `hand_labels`. Adding a file to it is a reviewed decision.

## Tests and compute

- **Running the tests.** From `DouZero/`, run `../.venv/bin/python -m pytest tests -p no:cacheprovider -q`. This is single-process and takes a few seconds. The agent test needs `baselines/douzero_ADP/`; without it, that test is skipped. pytest is installed only in the project `.venv`.
- **Local vs remote.** This Mac runs correctness tests only. Dataset generation, RL training, pilots and large evaluations run remotely, after approval.

## Status

1. **Done (stage 1).** `hand_belief.py` pure functions and tests T1–T6b. No change to the Q-network, actor, learner, buffers or training loop.
2. **Next (awaiting approval).** Predictors and the belief-input Q-network (T8–T10), then actor/learner wiring (T7).
