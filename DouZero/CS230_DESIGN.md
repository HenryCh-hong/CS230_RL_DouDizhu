# CS230 hand-belief study: design notes

`DouZero/` is part of the project repository (one local git repository at the project root; nothing is pushed). Its first commit imports kwai/DouZero `718a5c9` unmodified (Apache-2.0), so `git diff <import commit> -- DouZero` shows every change made for this study. `run_dir.git_info` records the git state of `DouZero/` only.

## Question

Does explicitly representing the recoverable second hidden-hand belief help a Deep Monte Carlo agent learn better decisions than a next-player-only belief representation?

From the acting seat, the two hidden hands satisfy **H_next + H_other = U** for every card kind. Here U is the public union of unseen cards, and both hand sizes are public. A belief about the next player's hand therefore determines the belief about the other hidden hand.

**Terms used in this note.**
- **DMC (Deep Monte Carlo).** DouZero's training method. A Q-network predicts the final game return of each legal move, trained on the actual return of finished games (Monte Carlo targets), not on bootstrapped estimates.
- **Q-value.** The predicted return of playing one move in one state.
- **ADP and WP.** DouZero's two reward objectives: Average Difference in Points and Winning Percentage. DouZero-ADP is the released agent trained on ADP.
- **x and z.** `x` is the non-history observation (`x_no_action`: the hand, unseen-card information and similar features) plus the 54-bit candidate move, so 373 values for the landlord and 484 for the peasants. `z` is the history of earlier moves, which the LSTM reads.
- **Slot.** One 75-value half of the 150-d belief (15 card kinds × 5 count classes).
- **CE (cross-entropy).** Mean of −log p(true count) over the card kinds, in nats. Lower is better.
- **RMSprop.** The adaptive gradient optimizer DouZero uses.
- **Detached.** A tensor cut from the computation graph, so no gradient flows back through it.
- **Clipping.** Rescaling the gradient when its norm exceeds 40, so one large step cannot dominate.
- **reflect(p_next, U).** Defined under Conventions (M2c).

## Arms

All arms share one code path and one Q-network. They differ only in what fills the belief input.

| Arm | `--hand_model` | Slot 1 (75) | Slot 2 (75) | Predictor |
|---|---|---|---|---|
| M0c | `none` | 0 | 0 | none (continued DouZero) |
| M1 | `next` | p_next | 0 | frozen next-player predictor (DouZero+-style) |
| M2c | `derived_both` | p_next | reflect(p_next, U) | **the same frozen checkpoint as M1** |
| M2 | `learned_both` | p_next (two-hand model) | p_other (two-hand model) | separately pretrained two-hand predictor, frozen |
| Oracle | `oracle` | one-hot true next | one-hot true other | none (positive control only) |

**Comparisons** (decided 2026-10-17):

| Comparison | Role | Reading |
|---|---|---|
| **M2c vs M1** | Primary, clean | Same frozen predictor and same information. The inputs differ only in slot 2. Outcome differences can still come from seed and asynchronous training noise, so they need paired, multi-seed evaluation (see Open items before pilots) |
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

- **Frozen by default.** Predictors are pretrained offline on DouZero-ADP self-play and then frozen. `--freeze_predictor` is implemented and defaults to on. `--no-freeze_predictor` makes `check_flags` raise `NotImplementedError`, so joint fine-tuning cannot start by accident.
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
- **Q-networks.** `LandlordLstmModel` and `FarmerLstmModel` gain `belief_dim=0` and a `belief=` argument to `forward`, appended after x as DouZero+ does. `belief_dim=0` is the original network, and released DouZero checkpoints still load into it. The DouZero training loop and `DeepAgent` were changed in stage 3 (see Fixes to DouZero's training loop and Strict reloading). `forward` raises an error if a belief is missing, unexpected, or still carries gradients. A wiring mistake therefore cannot silently turn an arm into DouZero, or let a Q-loss reach a predictor.
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

## Actor, buffer and learner wiring (stage 3)

**Flags** (`douzero/dmc/arguments.py`):

| Flag | Meaning |
|---|---|
| `--hand_model` | `baseline` (default: the original DouZero network, without a belief input), `none` (M0c), `next` (M1), `derived_both` (M2c), `learned_both` (M2) or `oracle` |
| `--predictor_dir` | The `<seat>.ckpt` frozen predictors. Required by M1, M2c and M2, and rejected for the other arms |
| `--load_model` | Resumes `<savedir>/<xpid>/checkpoints/model.tar`. Mutually exclusive with `--warm_start_dir`. The resume rules are below |
| `--allow_untrained_predictors` | Smoke runs only. Without it, `train()` refuses predictor files that are not marked `trained`. The evaluation side refuses them the same way (`load_arm_checkpoint`, `BeliefDeepAgent`) unless `allow_untrained=True`; the evaluation CLI has no such flag yet, so it is a Python argument only. A run started with it is warned in `summary.md` |
| `--warm_start_dir` | The DouZero checkpoints to start from. Mutually exclusive with `--load_model` |
| `--freeze_predictor` | Default on. `--no-freeze_predictor` raises `NotImplementedError` |
| `--seed` | Drawn at random when omitted (`run_dir.fresh_seed`) and always recorded. A resume without `--seed` inherits the saved seed; a resume with a different one is refused |
| `--allow_dirty` | Required for a dirty git tree or an unreadable git state: without it, `run_dir.check_git` raises. With it, the run records `diff_sha256` (tracked diff) and `code_sha256` (tracked diff plus the sha256 of each untracked file) in `config.json` and `attempts.jsonl`, and `summary.md` warns |
| `--savedir` | Defaults to `artifacts/runs` |

**Stored, not recomputed.** The actor stores, for every decision, the exact 150-d belief it fed to the Q-network, in a new shared buffer `belief` of shape (T, 150), float32. The learner uses the stored belief.
- **Correctness.** The stored belief is the actor's own vector, copied row by row, and T14 checks it for every arm and seat. The learner then recomputes Q with its current weights, while actors play with the weights of their last sync. So for the same row, the learner's Q can differ from the actor's. This is off-policy staleness, which asynchronous DMC tolerates. Each target is still the Monte Carlo return of the actor's game.
- **Cost.** 600 bytes per decision, and no predictor compute in the learner. Recomputing would cost 3,200 predictor passes per batch and could differ between CPU and GPU.
- **Reproducibility.** The buffer is a complete record of the inputs the Q-network saw. It does not store Q-values, which the learner recomputes (see Correctness).
- **Fine-tuning later.** `x_no_action` and `z` are stored anyway, so a recompute variant remains possible.

**Actor** (`utils.act`):
- `actor_belief(hand_model, predictor, position, x_no_action, z, labels)` builds the belief from the observation tensors only. It refuses a predictor that is not frozen or belongs to another seat.
- For the oracle arm only, `act()` first calls `oracle.oracle_labels(env.env.infoset, position, x_no_action)`. That module is the single place where true hands are handed to an arm, and it checks that the infoset matches this exact decision.
- `seed_actor` seeds actor i's numpy and torch streams from `utils.actor_seeds(seed, device, i, attempt)`. Both seeds come from `np.random.SeedSequence([seed, device_index, i, attempt])`, where `device_index` is 0 for the CPU and k+1 for GPU k. `attempt` counts the earlier starts of the run (from `attempts.jsonl`), so a resumed attempt gets fresh streams under the same seed. The old rule `--seed + 1 + i` is gone: it made seed s, actor i+1 equal seed s+1, actor i. The numpy seed has 32 bits, so two combinations collide with probability about 2^-32 per pair, which is negligible here.
- `--seed` fixes initialization and each actor's random stream, not the whole run. Actors and learners are asynchronous, and exploration draws from the same numpy stream, so the deal sequences of two arms diverge after each actor's first game. Training seeds are therefore not deal-paired; pairing happens only in evaluation, on shared deal files.
- `predictors` and `max_episodes` are new optional parameters. `max_episodes` is for tests and tools only.

**Learner** (`dmc.learn`):
- `learner_inputs` checks every batch shape and that beliefs are finite, then flattens row t × B + b.
- The loss is unchanged: mean (Q − G)². Monte Carlo targets, RMSprop settings, clipping at 40 and the per-seat learners are all DouZero's.
- New per-update stats: `q_mean`, `q_std`, `grad_norm` (before clipping) and `clipped`. `train()` counts updates per seat.
- No CE is computed and no predictor is in any optimizer.

**Fixes to DouZero's training loop:**
1. **Actors start from the learner's weights.** `initial_models` copies the learner's weights (warm start, resume or random) into every actor before the actors start. DouZero copied them only on resume. A run started from `--warm_start_dir` (a feature of this project) would otherwise have had random actor networks until the first learner update.
2. **Clean, monitored shutdown.** Learner threads are daemons that a `None` sentinel wakes, so they stop cleanly. Actors are daemon processes, terminated at the end. A watchdog fails the run if any worker dies. The final checkpoint and `summary.md` are written on completion, interruption, SIGTERM/SIGHUP or failure (see the next subsection). DouZero never stopped its actors, so a finite run hung on exit.
3. **Little environment in metadata.** `FileWriter` no longer dumps `os.environ` into `meta.json`; it records only `CUDA_VISIBLE_DEVICES` and `OMP_NUM_THREADS`, plus the `SLURM_*` variables when `SLURM_JOB_ID` is set (not used on AWS). It also no longer crashes on a detached HEAD.

**Predictor provenance.** `save_hand_predictor(..., provenance)` marks predictor files as trained or not; untrained is the default.

**Shutdown and signals** (`dmc.train`, `install_stop_handlers`, `wake_learners`, `stop_actors`). SIGTERM and SIGHUP raise `Terminated` in the main thread (status `terminated (SIGTERM)`); Ctrl-C raises `KeyboardInterrupt` (status `interrupted`). A signal the process was started with ignored stays ignored, so `nohup` keeps SIGHUP ignored and the run survives a closed terminal. Every way of ending (completed, interrupted, terminated or failed) runs the same `finally` block, and each step is guarded so that later steps still run:
1. **Stop the learners.** `wake_learners` puts one `None` sentinel (a stop marker that `get_batch` turns into `StopLearning`) per learner thread on each full queue, from daemon helper threads. The helpers matter because an actor killed while holding a queue's cross-process write lock (for example by the out-of-memory killer) would block a plain put for ever; this hang was observed in an integration run and reproduced. The wait is at most `STOP_TIMEOUT_S` = 60 s. Learners still running are named in the status.
2. **Final checkpoint.** Its locks are taken with a 60 s timeout, so a `TimeoutError` is recorded instead of a hang.
3. **Stop the actors.** Terminate, join for 10 s, then kill.

Then the signal handlers are restored, a final metrics row is logged if needed, `summary.md` is written and the FileWriter is closed. `train()` returns the status. `train.py` exits 0 only for `completed`; otherwise it exits 1, and a failure re-raises.

**Watchdog** (`check_workers`). Every 5 s tick, the run fails if a learner thread raised, an actor process died, or a learner thread stopped early. It runs once more after the loop, so a failure in the last tick (after another learner reached `--total_frames`) still fails the run.

**Run directory** (`<savedir>/<xpid>/`, via `run_dir.RunDir`):

| File | Contents |
|---|---|
| `config.json` | Written once, at the first start, with an exclusive create (`open(..., 'x')`). Flags, arm, seed, warm-start and predictor files with sha256, the git record of the first start (commit, branch, dirty state, untracked files, `diff_sha256`, `code_sha256`), versions, machine. Never environment variables |
| `attempts.jsonl` | One line per start, including each resume: attempt index, seed, git record, `resumed_from_frames` (None for the first start) and that attempt's full flags. `flags.attempt` is the number of earlier lines |
| `checkpoints/model.tar` | The latest checkpoint, written to a temporary file and renamed, from a lock-protected snapshot: weights, optimizer state, counters and flags. Includes the run metadata under `cs230`, in plain types only, so `weights_only=True` loads it |
| `checkpoints/weights_<frames>.tar` | Written at every save: all seats' weights, frames and run metadata. `models.load_arm_checkpoint` loads it |
| `checkpoints/<seat>_weights_<frames>.ckpt` | Only for `--hand_model baseline`: plain per-seat DouZero state dicts, for DouZero's `evaluate.py` |
| `metrics.csv` | The authoritative training record: a fixed schema, one row every 5 s plus a final row. A resume truncates it to the checkpoint's frames |
| `summary.md` | Status, arm, seed, attempts, checkpoints used, frames, updates, final stats and warnings (listed below). Rewritten at the end of every start |
| `logs.csv`, `fields.csv`, `meta.json`, `out.log` | DouZero's FileWriter files, kept for compatibility. `logs.csv` is appended across attempts and never truncated, so use `metrics.csv` |
| `traces/` | Created, reserved for traces |

**Start-up order** (`dmc.train`). The CUDA availability check is the first statement of `train()`. Every check of flags, folder, checkpoint, seed, predictors and git state runs before anything is written. The order is: `check_flags` → `run_dir.check_run_path` (refuses an unusable folder; creates nothing) → read the checkpoint if `--load_model` → seed (a resume inherits the saved seed when `--seed` is omitted) → `load_predictors` → `run_metadata` → `run_dir.check_git` → `check_resume` (resume only) → GPU count → `initial_models` → `frozen_copies` → `create_buffers` → **only then** the `RunDir` creation, `config.json` (new run) or `read_config` (resume), `flags.attempt`, `truncate_metrics` (resume), `record_attempt`, `FileWriter` → `install_stop_handlers` → actors and learner threads. Failures after the `RunDir` creation (queue and process start-up, for example OS limits on open files) can still stop a run and leave its folder. Two integration tests check that a refused start leaves no run folder: a missing predictor directory, and a warm start whose seat files do not fit. A start that fails after the folder exists but before its first save leaves a folder with no checkpoint. Reusing its `--xpid` is refused with a message saying that it cannot be resumed.

**Run folder rules.** An existing run folder is never reused unless the run is resumed with `--load_model`. `check_run_path` refuses a folder that holds a checkpoint, and one that holds files but no checkpoint.

**Resume rules** (`dmc.check_resume`). `--load_model` resumes `checkpoints/model.tar` of the run folder. `check_resume` refuses to resume when:
- `hand_model` or `belief_dim` differs from the saved run;
- the predictor files differ. They are compared by sha256, so a copied run still resumes;
- any flag in `RESUME_FIXED_FLAGS` differs: `objective`, `unroll_length`, `batch_size`, `learning_rate`, `alpha`, `momentum`, `epsilon`, `max_grad_norm`, `exp_epsilon` and `seed`.

Flags such as `--total_frames` may change, so a finished run can be extended. `training_device`, `num_actors`, `num_threads` and `disable_checkpoint` are not in `RESUME_FIXED_FLAGS`; each attempt's flags are in `attempts.jsonl`. The resume restores the optimizer state, the counters (frames, position_frames, updates, stats, elapsed_s) and the weights together. It truncates `metrics.csv` to the checkpoint's frames, because later rows describe lost progress. Its actors get fresh random streams (`flags.attempt`). The git gate runs again for every start. A resume on changed code is accepted, but `summary.md` warns and `attempts.jsonl` records each attempt's git state. Whether to refuse it is an open decision (see Open items before pilots).

**Checkpoint consistency.** `train()`'s `checkpoint()` passes a `progress` callback to `dmc.save_checkpoint`, which reads the frames, position_frames, updates and stats under the same locks as the weights. The checkpoint therefore never mixes an update's weights with another update's counters. In every `model.tar`, frames == unroll_length × batch_size × (sum of updates) == sum(position_frames). The integration tests check this invariant.

**Summary warnings.** `summary.md` lists these warnings: the run did not complete; the git state of an attempt could not be read; an attempt trained from uncommitted changes (its code sha256 is listed); the attempts ran different code (a resume on changed code is accepted but flagged); the predictors are untrained (a smoke run, not an experiment). The warnings cover every attempt in `attempts.jsonl`, not only the latest.

**Strict reloading.** `models.load_arm_checkpoint` reads the arm from the metadata, loads every network strictly and verifies the predictor files' sha256, seat and K. It refuses untrained predictors unless `allow_untrained=True`. The evaluation agents use it:
- `evaluation/belief_agent.py` `BeliefDeepAgent` plays M0c, M1, M2c and M2;
- `evaluation/oracle_agent.py` `OracleAgent` plays only the oracle.

Each refuses the other kind of checkpoint. Neither is wired into `evaluate.py` yet; that is Stage 4/5 (`../tools/dup_eval.py`, at the project root). That script already plays paired deals from a seeded `make_deals`, but it does not load belief arms.

DouZero's `evaluation/deep_agent.py` `DeepAgent` now loads strictly. It refuses a run checkpoint (`model.tar` or `weights_<frames>.tar`, which carry `model_state_dict` or `cs230` keys) and any state dict that is missing a weight, so a checkpoint cannot play with random weights. `simulation.evaluate` raises `RuntimeError` when a worker exits non-zero. It used to hang waiting for a result that never came.

**Known limitations of the Stage-3 wiring** (recorded, not fixed in Stage 3):
- `logs.csv` (DouZero's FileWriter) is appended across attempts and never truncated, so it can hold rows of lost progress and repeat frame numbers. Use `metrics.csv`.
- A resume on different code is accepted, flagged in `summary.md` and recorded per attempt. Whether to refuse it is still open.
- `RESUME_FIXED_FLAGS` does not include `training_device`, `num_actors`, `num_threads` or `disable_checkpoint`. `--disable_checkpoint` on a resume truncates `metrics.csv` and then saves nothing.
- A signal during start-up, before `install_stop_handlers`, ends the run without a summary (a window of seconds). A signal in the few bytecodes between a lock acquire and its release registration could leak a lock; the final checkpoint then times out after 60 s instead of hanging.
- No stall watchdog: a learner that is alive but stalled is not detected during training. It is only bounded at shutdown. A learner stuck while holding its seat lock also blocks the next periodic checkpoint, and with it the main loop and its watchdog, until the run is stopped.
- After a stop timeout (60 s), a learner that is still running could change the weights and counters after the final snapshot, and log to the closed `FileWriter`. The status records the stuck threads, so such a run is never reported as clean.
- The last steps of the shutdown (the final metrics row, `summary.md`, closing the `FileWriter`) are not guarded one by one, and a torn last line of `attempts.jsonl` (a kill in the middle of writing it) makes the summary fail.
- Actors are orphaned if the parent itself is SIGKILLed. Daemon processes are stopped only on a normal exit.
- Checkpoints are renamed atomically but not fsynced, so a power loss can lose the latest save.
- The running mean of episode returns (last 100) is not checkpointed, so it restarts after a resume.
- Predictor and warm-start files are hashed in a separate read from the one that loads them. Do not change them while a run starts.
- A predictor's `trained` flag is self-attested provenance, not a verification.
- Evaluation: `simulation.evaluate` can still hang on a stuck (not dead) worker, and it divides by zero on an empty deal file. `generate_eval_data.py` is unseeded and records no hash (open item).
- `trace_decision.py` passes the true hands from `hand_labels` straight into `belief.build_belief` for the oracle. Training and evaluation go through `oracle.oracle_labels` instead. It is a teaching tool only.

**Teaching tools:**
- `trace_episode.py` prints one full game of one arm and the rows the actor stores. T20 checks that those rows equal what `act()` stores.
- `trace_learner_update.py` prints one real `learn()` update: batch shapes, Q and loss before and after, gradient norm, the change of each parameter group, predictor change = 0, and the actor sync.
- `douzero/dmc/inprocess.py` runs the real `act()` in one process for tests and tools.

**Finding: a fresh optimizer takes large first steps.** The DouZero-ADP release has no optimizer state, so every arm starts with an empty RMSprop. Its running average of squared gradients starts at 0, so the first update moves each weight by about lr/√(1 − α) = 10 × lr. `trace_learner_update.py` shows Q's mean moving 0.87 → 1.01 in one step. The effect fades over about 100–300 updates. It hits all arms equally, but it means M0c is "DouZero-ADP weights with a fresh optimizer", and the warm-started policy may dip at first. Decide before the pilots whether to warm up the learning rate (or the optimizer state) for all arms.

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
- **`hand_labels` callers.** Only `oracle.py` (which the Oracle arm reaches through `oracle_labels`) and the debug tool `trace_decision.py` call it. The file allow-list in `tests/test_leakage.py` (`MENTIONS_HAND_LABELS`) is `hand_belief.py`, `oracle.py` and `trace_decision.py`. Offline predictor-data generation (Stage 4) will need a reviewed addition. In training and evaluation code, the Oracle arm gets true hands only through `oracle_labels`. The callers of `oracle_labels` are allow-listed separately (`MENTIONS_ORACLE_LABELS`): `oracle.py`, the actor's Oracle branch in `utils.act`, `evaluation/oracle_agent.py`, and the debug tool `trace_episode.py`. A runtime test makes `oracle_labels` raise during the non-oracle arms' games, and they still complete. `actor_belief` itself takes tensors, never an infoset.
- **Order of `InfoSet.other_hand_cards` leaks.** It is the union, but it is concatenated in seat order, so its order reveals the split. Use only order-invariant functions of it, such as card counts.
- **Enforced by `tests/test_leakage.py`:**
  - T6a swaps cards between the hidden hands while keeping sizes and union fixed. The observation, the DouZero-ADP Q-values and the chosen action must not change. Positive controls show that the swap exposes a feature that reads either leak channel.
  - T6b is a static allow-list of the files that mention `all_handcards`, `hand_labels` or `oracle_labels`. Adding a file to any of them is a reviewed decision. The additions so far are `trace_decision.py` (to the `hand_labels` list: a debug tool that displays the labels and runs the oracle arm) and `trace_episode.py` (to the `oracle_labels` list).
  - In stage 2, T6a was extended to the M0c/M1/M2c/M2 belief and Q path: after a swap, beliefs and Q-values must be bit-identical. The Oracle's belief must change.
  - Belief agents and the oracle agent are covered by T21 (`test_evaluation_agents_are_blind_to_the_swap_except_the_oracle`, `tests/test_leakage.py`). Add any new agent there.
  - In stage 3, T21 checks the wired code. The real `act()`, run with every infoset swapped, must store exactly the same games for the four public arms and different games for the Oracle. `BeliefDeepAgent` must be blind to swaps, and `OracleAgent` must not be.

## Tests and compute

- **Running the tests.** From `DouZero/`, run `../.venv/bin/python -m pytest tests -p no:cacheprovider -q -rs`. The unit suite has 188 tests. It runs in one process and takes about 17 s on an idle Mac. pytest is installed only in the project `.venv`.
- **Integration tests (opt-in).** `tests/test_integration.py` holds 8 end-to-end tests of `train()`. They start real processes, with one actor and two learner threads per seat, and take about 75 s in all:
  - a tiny run completes, resumes and refuses a changed flag;
  - SIGTERM, SIGHUP and SIGINT each still save a final checkpoint and summary, and leave no actor behind;
  - a killed actor fails the run loudly and still saves;
  - a refused start (a bad predictor directory, or a warm start that does not fit) leaves no run folder;
  - evaluation fails loudly when a worker dies.

  They are collected by the default run above but skipped unless `CS230_INTEGRATION=1` is set. A default run therefore shows 188 passed and 8 skipped (`-rs` prints the reasons). To run them:

  ```
  CS230_INTEGRATION=1 ../.venv/bin/python -m pytest tests/test_integration.py -p no:cacheprovider -q
  ```
- **macOS pipe gotcha.** A fresh pipe holds only 512 bytes on this Mac. Never start `train.py` with `stderr=PIPE` unless something reads the pipe while it runs: the run blocks on its next log line.
- **Checkpoints needed.** Seven tests need `baselines/douzero_ADP/`: the ADP swap test, the weight-copy check, the three warm-start checks, the Q-network checkpoint round trip and the actor warm-start check. Without it they are skipped. The only skips allowed in a default run are the 8 integration tests; any other skip means a missing checkpoint. The trace-tool tests fall back to random weights. `baselines/`, `artifacts/` and `debug/` are git-ignored.
- **Smoke run (stage 3).** The command used:

  ```
  OMP_NUM_THREADS=1 nice -n 19 gtimeout --kill-after=30 300 ../.venv/bin/python train.py \
    --hand_model derived_both --predictor_dir artifacts/predictors/untrained_k1_seed0 --allow_untrained_predictors \
    --warm_start_dir baselines/douzero_ADP --actor_device_cpu --training_device cpu --num_actors 1 --num_threads 1 \
    --num_buffers 6 --unroll_length 16 --batch_size 2 --total_frames 300 --save_interval 1 --seed 0 \
    --xpid smoke_m2c_seed0 --allow_dirty
  ```
- **Tested versions.** Python 3.14.0, torch 2.14.1, numpy 2.5.4, pytest 9.1.1. `requirements.txt` lists `torch>=1.6.0`, `GitPython`, `gitdb2` and `rlcard` without versions (GitPython is used by `FileWriter`), so pin these on the remote machine.
- **Local vs remote.** This Mac runs correctness tests only. Dataset generation, RL training, pilots and large evaluations run remotely, after approval.

## Making training visible

Every stage should make the data flow, tensor shapes, losses and training behavior visible enough to follow.
- **`docs/TRAINING_WALKTHROUGH.md`** (project root) walks through the pipeline in 22 steps. Each step names its file, function, tensor shapes, information class and gradient flow. It also has the arm comparison, the gradient-flow diagrams, and worked examples of the CE and Q-losses.
- **`trace_decision.py`** prints one decision for every arm: observation shapes, predictor outputs against card counting, belief layouts, Q over the legal actions, and the epsilon-greedy choice. It is single-process and takes about a second.
- **`trace_episode.py`** (stage 3) prints one full game of one arm and writes it to `debug/episode_trace.txt`. **`trace_learner_update.py`** (stage 3) prints one real learner update.
- **Run directories** (stage 3) log `metrics.csv` every 5 s, one line per start in `attempts.jsonl`, and `summary.md` at the end of every start.
- **Planned.** Predictor metrics in Stage 4, and plots under `artifacts/plots/` built from the raw `metrics.csv` of remote runs, never from the laptop.

## Open items before pilots

These come from the stage-2 and stage-3 reviews and must be settled in the stages named.
- **Seed variance (evaluation design).** Paired, duplicate deals remove card luck, not training-seed variance. Plan at least 3 seeds per arm and analyze paired per-seed differences, where a pair is seed s of M2c and seed s of M1. Such a pair shares only the initialization and each actor's first game (see the wiring section), so it does not control the rest of the training noise. Three seeds leave 2 degrees of freedom for seed-to-seed variance, so a paired test over 3 seeds has little power. Use a bootstrap or t-interval over seeds and report the seed-level spread. Fix the minimum detectable effect (MDE) and the deal count in advance; if the MDE needs more than 3 seeds, call 3 seeds a pilot. `--seed` fixes initialization and each actor's stream of deals and exploration, but asynchronous actors and learners still make trajectories differ, and training seeds are not deal-paired (see the wiring section).
- **Refusing a resume on changed code (open decision).** A resume on different code is accepted, flagged in `summary.md`, and recorded per attempt in `attempts.jsonl`. Decide whether it should be refused outright, or allowed only with an explicit flag.
- **Predictor drift (Stage 4).** Predictors are trained on DouZero-ADP self-play, and P(hand | public history) depends on the policies that produced that history, so beliefs can drift as RL changes the policy. M1 and M2c share the predictor, which keeps the primary comparison matched. Still, measure predictor CE on the current policy's games at every checkpoint. Do it **offline** from saved trajectories in a separate allow-listed script, because labels must never enter the training processes.
- **Separate predictor-training deals from evaluation deals (Stage 4).** Record the predictor's training-deal seeds or deal-list hash in its provenance, and have evaluation assert the two sets are disjoint.
- **Fresh-optimizer transient (before pilots).** See the stage-3 finding above: decide on a learning-rate or optimizer-state warm-up applied equally to all arms.
- **M2 capacity.** Report the 1,729,099 extra predictor parameters alongside M2 vs M2c.
- **Clipping as a possible confound.** Each update logs `clipped`. Report the fraction of clipped updates per arm, because clipping on the global gradient norm couples the belief columns' gradients to the rest of the network.
- **Matched frames.** Compare arms at equal `total_frames`, and report fps and wall time as a cost, not as a control.
- **Evaluation wiring (evaluation stage).**
  - Route `BeliefDeepAgent` and `OracleAgent` into `evaluate.py` / `../tools/dup_eval.py` by checkpoint metadata.
  - Seed and hash the evaluation deals: `generate_eval_data.py` is unseeded and records no hash, while `../tools/dup_eval.py` seeds its deals (`make_deals`) but records no hash either. Every evaluation output should record the seed and the sha256 of its deal list.
- **Remote (AWS) checks before the first run.**
  - A short CUDA smoke run of the GPU actor path.
  - `ulimit -n` against the number of shared tensors.
  - A disk budget for checkpoints. Each save writes one `weights_<frames>.tar` with all three seats (about 19 MB for the belief arms: 4,717,059 float32 parameters) and overwrites `model.tar`, which also holds RMSprop state (about twice that). Baseline runs also write three per-seat `.ckpt` files per save. Budget = (run length / `--save_interval`) × these sizes.
- **ADP units.** Evaluation scores the landlord side at ±2·2^bombs, twice the training target. State the units with every reported number.

## Status

1. **Done (stage 1).** `hand_belief.py` pure functions and tests T1–T6b. No change to the Q-network, actor, learner, buffers or training loop.
2. **Done (stage 2).** The model layer: `HandPredictor`, the common 150-d belief interface, all five arms, the function-preserving warm start, stop-gradient checks, checkpoint round trips, and tests T8–T12 with the T6a extension. The trace tool and the walkthrough were added too. The actor, learner, buffers and training loop are still unchanged.
3. **Done (stage 3), committed locally; awaiting review before stage 4.** It covers the actor, buffer, learner, checkpoint and reload wiring for all arms; the run directory, with start-up validation, resume rules, `attempts.jsonl` and checkpoint consistency; clean shutdown on SIGTERM, SIGHUP and SIGINT; the watchdog; strict loading and fail-loud evaluation (`DeepAgent`, `simulation.evaluate`); the evaluation agents; `trace_episode.py` and `trace_learner_update.py`; tests T7 and T13–T21; 188 unit tests; and 8 opt-in integration tests, which run tiny `train()` runs end to end with the current code (completion, resume, three stop signals, a killed actor). The older smoke artifact `artifacts/runs/smoke_m2c_seed0` (1 actor, 320 frames) predates the review fixes: it has no `attempts.jsonl` and no `weights_<frames>.tar`.
4. **Next (after stage 3 is approved).** Stage 4: offline predictor dataset and training, which needs remote compute. No real RL training or large evaluation has been run.
