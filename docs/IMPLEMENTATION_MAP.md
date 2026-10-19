# Implementation Map: Hidden-Hand Beliefs in DouZero

Verified on 2026-10-17 against `DouZero/` (kwai/DouZero `main` @ `718a5c9`, unmodified: `git status` shows only the untracked `baselines/` checkpoints). Line numbers refer to that commit.

**Evidence:**
- I read the code directly.
- An empirical probe ran 300 seeded random-play games (18,631 decision states), decoding the network's actual observation vectors.
- Each of the 28 file:line claims was checked a second time, independently: 26 confirmed, 2 corrected (folded in below).

## 1. Code path (current code)

| # | Component | File : lines | Role | Key shapes | Change? |
|---|---|---|---|---|---|
| 1 | Env wrapper | `douzero/env/env.py` `Env` 21; `reset` 50–73; `step` 75–95; `_get_reward` 97–119 | Deals with an **unseeded** `np.random.shuffle` (60). Landlord gets `_deck[:20]` (bottom cards `_deck[17:20]`), up `[20:37]`, down `[37:54]`. `card_play_data` is a local, not stored. Terminal ADP reward is ±2^bombs from the landlord's perspective | obs dict | **Yes**: training-only label/mask attachment (flag); seeding |
| 2 | GameEnv / InfoSet | `douzero/env/game.py` `GameEnv` 18; `card_play_init` 56–65; `step` 103–135; turn order 154–168; `get_infoset` 292–335; `InfoSet` 337–372 | Rules. Turn order is L→D→U→L. The infoset holds the union `other_hand_cards` (316–321), sizes `num_cards_left_dict` (312–314), and **true hands `all_handcards` (330–333)**. A deepcopy goes to every agent (335) | — | **No** |
| 3 | Observation | `env.py` `get_obs` 188–219; `_get_obs_landlord` 290–372; `_get_obs_landlord_up` 374–470; `_get_obs_landlord_down` 472–574; encoders `_cards2array` 230–250, `_action_seq_list2array` 252–266, `_process_action_seq` 268–279, `_get_one_hot_array` 221–228, `_get_one_hot_bomb` 281–288 | Features from the acting player's own information. Never reads `all_handcards`. The obs dict also has a `position` key | `x_no_action`: 319 (L) / 430 (P) int8. `x_batch`: (n_legal, 373/484) float32. `z`: (5, 162) int8. `z_batch`: (n_legal, 5, 162) float32 | **No** (stays as is) |
| 4 | Legal actions | `game.py` `get_legal_card_play_actions` 177–263; `move_generator.py`, `move_detector.py`, `move_selector.py` | Candidate moves, including bombs and pass | list of card lists | **No** |
| 5 | Actor | `douzero/dmc/utils.py` `act` 110–185; `douzero/dmc/env_utils.py` `Environment` (auto-reset when done) | Self-play rollout process | Appends `x_no_action` and `z` **before** acting (136–137) | **Yes**: predictor inference → belief (1, 150) broadcast to (n_legal, 150); store labels and mask |
| 6 | Model inference | `douzero/dmc/models.py` `Model.forward` 100; `LandlordLstmModel` 11–44; `FarmerLstmModel` 46–79 | Q(o, a) for every legal action | LSTM(162→128) (14/49); last step (24/59); `cat[lstm_out, x]` (25/60); dense1 Linear(501 / 612 → 512); dense2–5 512; dense6 → 1. Params 1,457,665 (L) / 1,514,497 (P) | **Yes**: belief appended after `x`, so dense1 takes 651 / 762; new `HandPredictor` |
| 7 | ε-greedy | `models.py` 38–43 / 73–78 | Random legal action with probability `exp_epsilon` = 0.01 | — | No |
| 8 | Trajectory storage | `utils.py` 124–130 (per-seat lists), 136–143 | Per-seat Python lists | — | **Yes**: append `hand_label` and `hand_legal` at the same point as the observation |
| 9 | Monte Carlo return | `utils.py` 145–156 | Undiscounted; peasants get −return | — | No |
| 10 | Shared buffers | `utils.py` `create_buffers` 78–108; flush 158–177; `get_batch` 42–60 | Shared-memory transfer: 50 buffers × T = 100 per seat | `done` (T,), `episode_return` (T,), `target` (T,), `obs_x_no_action` (T, 319/430) int8, `obs_action` (T, 54) int8, `obs_z` (T, 5, 162) int8; batch is (T, B = 32, …) | **Yes**: add `hand_label` (T, 2, 15) int8 and `hand_legal` (T, 15, 5) int8 |
| 11 | Learner | `douzero/dmc/dmc.py` `learn` 23–59; `batch_and_learn` 154–169 | MSE on T·B = 3,200 samples. `obs_x = cat(x_no_action, action)` (37); clip 40 (54) | `obs_x` (3200, 373/484); `obs_z` (3200, 5, 162) | **Yes**: build belief (3200, 150) per `--hand_model` |
| 12 | Optimizer | `utils.py` `create_optimizers` 62–76 | One RMSprop per seat (lr 1e-4, α 0.99, eps 1e-5, momentum 0) | — | Only if the predictor is trainable during RL |
| 13 | Actor sync | `dmc.py` 57–58 (after **every** update, under a per-seat lock); actor models in shared memory (92), eval mode (93) | Learner → actors | — | Only if the predictor is trainable during RL |
| 14 | Checkpointing | `dmc.py` `checkpoint` 191–209; every 30 min (221–223); resume with `--load_model` (130–142) | `model.tar` plus per-seat `<pos>_weights_<frames>.ckpt` | — | **Yes**: save the predictor and the `hand_model` config; warm-start loader |
| 15 | Training entry | `train.py`; `douzero/dmc/arguments.py`; CUDA assert at `dmc.py` 68–70 (CPU needs `--actor_device_cpu --training_device cpu`); `file_writer.py` tolerates non-git directories (40) | Flags | — | **Yes**: new flags |
| 16 | Evaluation | `evaluate.py`; `douzero/evaluation/simulation.py` `evaluate` 45, `mp_simulate` 21 (aggregates only); `deep_agent.py` `DeepAgent` 22–43 (single legal action → returned without the network, 28–29); `generate_eval_data.py` (unseeded); `tools/dup_eval.py` (mine: per-deal duplicate results plus bootstrap) | Plays deals | — | **Yes**: hand-model-aware `DeepAgent`, a separate `OracleAgent`, per-deal and head-to-head outputs |

## 2. Phase-1 assumptions

| # | Assumption | Verdict | Evidence |
|---|---|---|---|
| 1 | `other_hand_cards` = union of the two hidden remaining hands | ✅ | `game.py` 316–321 (concatenation). Probe: 0 / 18,631 mismatches, also when decoded from `x_no_action[54:108]` |
| 2 | Each hidden player's remaining count is observable | ✅ | `num_cards_left_dict` (`game.py` 312–314). Observation one-hots: Landlord `x[270:287]` (up, 17), `x[287:304]` (down, 17); Peasants `x[378:398]` (Landlord, 20), `x[398:415]` (teammate, 17). Probe: 0 decode errors |
| 3 | H_next + H_other = U for every card kind, throughout play | ✅ | Probe: 0 / 18,631 |
| 4 | Turn order L → D → U → L | ✅ | `game.py` 154–168; probe true in all 300 games |
| 5 | DouZero+ targets: L→D, D→U (teammate), U→L | ✅ | `reference_code/Doudizhu_plus/oppo_modeling/env/env.py` 251 / 460 / 337 |
| 6 | True hands are stored in `InfoSet.all_handcards` | ✅ | `game.py` 330–333. **Every agent receives this field** in its deepcopied InfoSet (335) |
| 7 | Labels are OK for training; the evaluation path must not read them | ✅ (currently) | Nothing in the repo reads `all_handcards`. Evaluation agents read only `legal_actions`, `last_move`, `last_two_moves`, `last_pid`, the own hand, and whatever `get_obs` reads. Hazard: the field is *present* in eval infosets |
| 8 | Encoding | ✅ | `Card2Column` (`env.py` 6–7): 3..14 → 0..11, 17 ('2') → 12. 54-d = 4×13 thermometer flattened column-major + jokers (20 → index 52, 30 → 53). Proposed 15-kind order `[3,4,…,13,14,17,20,30]` (index 13 = small joker, 14 = big joker) matches DouZero+ `get_label`. Counts 0–4; jokers 0–1. Maximum observed: 4 |
| 9 | Q-network architecture | ✅ | Row 6 above |
| 10 | Actor/learner sync | ✅ | `dmc.py` 57–58 copy after every learner step; actors read shared weights without locks (benign races, as in DouZero) |

## 3. The 150-wide matched belief input is sound

- Layout: `dense1` input = `[lstm 128 | x 373/484 | b 150]`, so 651 / 762. This is DouZero+'s convention (append the belief after `x`).
- With the new columns zero-initialized, any slot that is always zero (both slots for M0c, slot 2 for M1) gets **exactly zero gradient**: ∂L/∂W = δ·0. RMSprop updates are therefore 0, and those weights stay exactly zero.
  - Consequence: every arm has identical architecture and code.
  - M0c is numerically "DouZero continued", and M1 is functionally the 75-wide DouZero+ input.
- M2c's derived slot uses u(r), which is decodable from `x[54:108]`. Its information is identical to M1's.
- Oracle: one-hot true counts in the same 150 slots.

## 4. Implementation notes (from verification)

- No seeds anywhere: Env deck shuffle (`env.py` 60), `generate_eval_data.py`, torch initialization. This must be added.
- `env_utils._format_observation` moves only `x_batch` and `z_batch` to the device. `legal_actions` stays a Python list, and `x_no_action` and `z` stay CPU tensors. The new `hand_label` and `hand_legal` should follow `x_no_action` (CPU int8).
- The DouZero+ repo is reference only: missing modules, CUDA-only, hard-coded `/root/...` paths, `torch==1.6` pin.
- Not yet verified on this machine: the DouZero **training loop** (`train.py` → `dmc.train`) under Python 3.14 / torch 2.14 / CPU flags, including checkpoint save and resume under torch's `weights_only=True` default. This is the first smoke-test item.
