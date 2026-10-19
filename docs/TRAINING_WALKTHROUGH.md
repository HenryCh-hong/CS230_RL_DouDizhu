# Training walkthrough: from one deal to one gradient step

This document follows the pipeline that trains DouZero and our hand-belief arms. It starts with one deal, goes through one decision, and ends with one learner update, a checkpoint and an evaluation. It is written for someone still learning reinforcement learning (RL). Every step names the real file, function and tensor shapes in `DouZero/` of the project repository, as of Stage 3: the actor, buffers, learner, run folder, resume and shutdown are wired and tested, and the opt-in integration tests run a tiny `train()` end to end.

**How to read it.** All code paths are relative to `DouZero/` unless they start with `docs/` or `tools/`. Line numbers are approximate guides; the function names are exact. Each step carries tags:

| Tag | Meaning |
|---|---|
| **[DouZero]** | Original kwai/DouZero code, unchanged |
| **[Stage 3]** | Our addition, wired into the real actor/learner and tested |
| **[planned]** | Not built yet (Stage 4: offline predictor training; later: evaluation runs) |
| PUBLIC | The acting player is allowed to know this |
| LABEL | True hidden hands. Used only to train a predictor, never as an input |
| ORACLE-ONLY | True hidden hands used as an *input*. Only the Oracle positive control may do this |

Names such as T14 are test IDs. Each is listed in the docstring of the `tests/test_*.py` file that holds it (for example, T14 is in `tests/test_pipeline.py`).

Three teaching tools print the pipeline on real data. Each runs from `DouZero/` on one CPU thread in seconds:

```
../.venv/bin/python trace_decision.py --seed 7 --num_random_moves 8       # one decision, all five arms
../.venv/bin/python trace_episode.py --hand_model derived_both --seed 0    # one full game, one arm
../.venv/bin/python trace_learner_update.py --hand_model derived_both      # one learner update
```

Example outputs are in [the trace tools](#the-trace-tools) at the end.

---

## The big picture

DouZero learns with **Deep Monte Carlo (DMC)**. A neural network Q(o, a) learns to predict, for the observation o and a candidate action a, the final score the acting player will get.

- **What it trains on.** The training target is simply the score that was actually obtained when the game ended. That is the "Monte Carlo" part: there is no bootstrapping from the network's own estimates.
- **How it plays.** It picks the action with the highest predicted score.

Training runs as **self-play** with an **actor/learner** split:

```
 actor processes (5 by default)                      learner threads (4 per seat)
 +----------------------------------+                +-------------------------------+
 | play full games with copies of   |  shared-memory | sample batches, compute the   |
 | the current networks (all three  |  buffers       | loss, take gradient steps on  |
 | seats), record each decision,    | -------------> | the three seat networks       |
 | its belief, and its final score  |  (per seat)    |                               |
 +----------------------------------+                +-------------------------------+
            ^                                                       |
            +------------- new weights after every update ----------+
```

> **Self-play:** all three seats are played by the current networks, so the agent learns from games against copies of itself. No human data is needed.
>
> **Actor / learner:** actors only *play*. They run the networks in inference mode, with no gradients, to produce data. Learners only *learn*: they take gradient steps on that data. The two run at the same time in separate processes and threads, and talk through shared buffers.
>
> **Role-specific networks:** there is one Q-network per seat (`landlord`, `landlord_up`, `landlord_down`), because the roles play very differently. Our hand predictors are role-specific too.

**Where the run starts.** `train()` (`douzero/dmc/dmc.py`) validates everything before it creates anything, then starts the workers. The order matters: a refused start must leave no run folder.
1. **Check the devices and flags.** If a GPU is used, `train()` asserts that CUDA is available and that there are enough GPUs. `check_flags` then rejects contradictions: `--predictor_dir` is required by M1, M2c and M2 and refused for every other arm; both `--load_model` and `--warm_start_dir` are given; `--num_buffers` is below `--batch_size`; or joint predictor fine-tuning is asked for (not implemented).
2. **Check the run folder.** `run_dir.check_run_path` refuses a folder that cannot be used (a new run into a folder that already holds a run, or a resume without `config.json`). It creates nothing.
3. **Read the checkpoint** if `--load_model` is given (`read_checkpoint`).
4. **Seed.** `train()` seeds numpy and torch with `--seed`. A resume inherits the saved seed when `--seed` is omitted. A new run without `--seed` draws one with `run_dir.fresh_seed` and records it.
5. **Load the predictors.** `models.load_predictors` loads the arm's frozen predictors strictly; untrained files are refused unless `--allow_untrained_predictors` is given.
6. **Record the metadata and git state.** `run_metadata` builds what every checkpoint carries (arm, predictor files, warm-start files, git state). `run_dir.check_git` refuses a dirty or unreadable git tree unless `--allow_dirty` is given.
7. **Check the resume** (resume only). `check_resume` refuses a different experiment (see **Resume** below).
8. **Build the models.** `initial_models` builds the learner from the checkpoint being resumed, else from `--warm_start_dir`, else from random initialization. Every actor copy then gets exactly the learner's weights. `frozen_copies` places the predictors for each actor device, and `create_buffers` allocates the shared buffers. DouZero used to start actors from random weights; this was a Stage-3 fix.
9. **Only now, create the run record.** `run_dir.RunDir` creates `checkpoints/` and `traces/`. A new run writes `config.json` (once, with `'x'` mode, so it never overwrites another run's record). A resume reads it and truncates `metrics.csv` to the checkpoint's frames. `attempts.jsonl` gets one line for this start, and `FileWriter` opens its log files.
10. **Install the stop handlers and start the workers.** `install_stop_handlers` maps SIGTERM and SIGHUP to a clean stop (see Shutdown in step 21). Then the actor processes and the learner threads start, and the watchdog loop runs every 5 s (a check that fails the run if a worker has died; see step 21). SIGTERM and SIGHUP are the signals a system sends to ask a program to stop; SIGHUP also arrives when a terminal closes.

> **Resume.** `--load_model` resumes `<savedir>/<xpid>/checkpoints/model.tar`. It must not be combined with `--warm_start_dir`. `check_resume` refuses the run if the arm (`hand_model`, `belief_dim`) or the predictor files (compared by sha256, so a copied run still resumes) differ, or if any flag in `RESUME_FIXED_FLAGS` differs: objective, unroll_length, batch_size, learning_rate, alpha, momentum, epsilon, max_grad_norm, exp_epsilon and seed. Flags such as `--total_frames` may change. A resume restores the weights, the optimizer states and the counters (frames, position_frames, updates, stats, elapsed_s) together. It starts the actors with a new attempt number, so they get fresh random streams. A run folder with no `model.tar` (a start that failed before its first save) cannot be resumed, and the start says so.
>
> **macOS pipe gotcha.** A fresh pipe holds only 512 bytes on this Mac. Never start `train.py` with `stderr=PIPE` unless something reads the pipe while it runs: the run blocks on its next log line.

---

## Part A: one decision

### 1. One game begins [DouZero + Stage 3 seeding]

- **Code:** `douzero/dmc/utils.py` `act()` is the actor's loop. It owns an `Environment` (`douzero/dmc/env_utils.py`), which wraps `Env` (`douzero/env/env.py`). `env.initial()` starts the first game, and `Environment.step` automatically resets when a game ends.
- **Seeding:** `utils.seed_actor` seeds the actor's numpy and torch streams (deals and exploration) from `utils.actor_seeds(seed, device, i, attempt)`. That function uses numpy's `SeedSequence` on four numbers: the run seed, the device (CPU, or GPU k), the actor index i and the attempt. The attempt is the number of earlier starts of the run, so a resumed run's actors get new streams. A simpler rule, `--seed + 1 + i`, would have made seed s, actor i+1 equal seed s+1, actor i, so we do not use it. The seed fixes each actor's stream and the initialization, not the whole run. Actors and learners are asynchronous, and exploration draws from the same numpy stream as the deals, so two arms' deal sequences drift apart after each actor's first game. Training seeds are therefore not deal-paired; pairing happens only in evaluation, with shared deal files. `tests/test_pipeline.py` checks that nearby seeds, devices and actors never collide.
- **Two parameters for tests and tools:**
  - `predictors` is a dict mapping each seat to its frozen predictor (or None);
  - `max_episodes` stops after N games (tests and tools only; training runs forever).
- **Tensors:** none yet.
- **Gradients:** none. Actors never compute gradients.

### 2. Cards are dealt [DouZero]

- **Code:** `Env.reset()` (`env.py` 50–73) shuffles the 54-card deck with numpy. `GameEnv.card_play_init` (`douzero/env/game.py` 56–65) stores the hands.
- **The deal:**
  - The landlord gets 20 cards: 17 plus the 3 bottom cards (`three_landlord_cards`, which everyone sees).
  - `landlord_up` and `landlord_down` get 17 each.
  - There is no bidding in DouZero: the seat called "landlord" is the landlord.
- **Turn order:** landlord → `landlord_down` → `landlord_up` → landlord … (`game.py` 154–168). `landlord_down` plays right after the landlord, and `landlord_up` right before.
- **Information:** your own hand is private to you. The bottom cards are PUBLIC under the rules, although DouZero's observation does not encode them separately: for a peasant they are simply part of the unseen union until the landlord plays them. The other two hands are hidden. The belief inputs also do not encode who holds the bottom cards (the landlord), so they are a subset of the public information, not the full public posterior.
- **Gradients:** none.

### 3. The acting player receives an observation [DouZero]

- **Code:** `GameEnv.get_infoset()` (`game.py` 292–335) builds an `InfoSet` for the acting player and deep-copies it.
- **What is in it:**
  - PUBLIC: own hand, every card played so far and by whom, the sequence of moves, the number of cards each player holds (`num_cards_left_dict`), and `other_hand_cards`, the *union* of the two hidden hands. The union is public because 54 cards − own hand − played cards = the cards the other two hold together.
  - **Two leakage hazards** (both handled; see [leakage](#leakage-rules)):
    - The engine (`game.py`) fills `InfoSet.all_handcards` with everyone's true hands. Only `hand_belief.hand_labels` may read it, and only `oracle.oracle_labels` may hand it to an arm.
    - `other_hand_cards` is the two hidden hands *joined in seat order*, so its order reveals the split. Only order-free counts of it may be used.
- **Gradients:** none.

### 4. Observation tensors are constructed [DouZero]

- **Code:**
  - `env.py` `get_obs` (188–219) calls `_get_obs_landlord` (290–372), `_get_obs_landlord_up` (374–470) or `_get_obs_landlord_down` (472–574).
  - `douzero/dmc/env_utils.py` `_format_observation` turns the result into torch tensors.
- **Tensors (all PUBLIC):**

| Name | Shape | dtype | Meaning |
|---|---|---|---|
| `x_no_action` | (319,) landlord, (430,) peasant | int8 | Everything about the state except the candidate action |
| `z` | (5, 162) | int8 | The last 15 moves, 3 moves of 54 bits per row: the input to the history LSTM |
| `x_batch` | (n_legal, 373 / 484) | float32 | One row per legal action: `[x_no_action \| action (54)]` |
| `z_batch` | (n_legal, 5, 162) | float32 | `z` repeated once per legal action |
| `legal_actions` | list of n_legal moves | Python lists | The moves you may play |

**The 54-bit card encoding** (`_cards2array`, env.py 230–250). Each of the 13 ranks (3 … A, 2) gets 4 bits that count copies like a thermometer: two 7s → `[1,1,0,0]`. Two more bits mark the small and the big joker.

**What is inside `x_no_action`:**

| Offsets | Landlord (319) | Offsets | Peasant (430) |
|---|---|---|---|
| 0:54 | own hand | 0:54 | own hand |
| 54:108 | union of the two hidden hands | 54:108 | union of the two hidden hands |
| 108:162 | the move to beat | 108:162 | cards the landlord played |
| 162:216 | cards `landlord_up` played | 162:216 | cards the teammate played |
| 216:270 | cards `landlord_down` played | 216:270 | the move to beat |
| 270:287 | `landlord_up` cards left (one-hot 1–17) | 270:324 | the landlord's last move |
| 287:304 | `landlord_down` cards left (one-hot 1–17) | 324:378 | the teammate's last move |
| 304:319 | bombs so far (one-hot 0–14) | 378:398 | landlord cards left (one-hot 1–20) |
| | | 398:415 | teammate cards left (one-hot 1–17) |
| | | 415:430 | bombs so far (one-hot 0–14) |

`x_batch` and `z_batch` go to the actor's device. `x_no_action` and `z` stay on the CPU, because those two are what gets stored (step 10).

- **Gradients:** none.

### 5. The hidden-hand predictor runs [Stage 3]

- **Code:**
  - `douzero/dmc/models.py` `HandPredictor(position, num_targets)` is a copy of DouZero+'s predictor.
  - The actor calls it through `utils.actor_belief`, from the decision loop of `act()`.
- **Loading:** `train()` loads the frozen predictors once per run with `models.load_predictors`, which checks:
  - each seat's file exists;
  - it was saved for that seat and the arm's K;
  - its sha256 matches when given;
  - it is marked as trained.

  `frozen_copies` then places one copy per actor device in shared memory, and every actor process receives it.
- **Input (PUBLIC):**
  - `z` (1, 5, 162) and `x_no_action` (1, 319 / 430), both float32. B = 1 because an actor decides once at a time.
  - No other input is possible: the predictor's own `forward` takes only these two tensors.
- **Inside:**
  1. The LSTM (162 → 128) reads `z`, and its last state goes into `cat[history 128, x_no_action]`, which is (B, 447) or (B, 558).
  2. Four 512-unit layers with ReLU.
  3. Each predicted hand has 15 heads, one per card kind. The 13 ranks use `Linear(512, 256)`, ReLU, `Linear(256, 5)`; each joker uses `Linear(512, 5)`. Every head outputs 5 scores, for "holds 0, 1, 2, 3 or 4 copies".
  4. **Legal-count mask.** The unseen count u of each kind is decoded from `x_no_action[54:108]` (`belief.union_counts_from_obs`). Counts above u are impossible, so `belief.legal_count_mask` and `mask_logits` set their logits to −inf.
  5. Softmax over the 5 counts.
- **Output:**
  - `logits` and `probs` of shape (1, K, 15, 5).
  - K = 1 predicts the next player's hand (M1, M2c). K = 2 predicts (next, other) (M2).
  - `probs` are exactly 0 on impossible counts, and each kind's 5 probabilities sum to 1.
- **Size:**

| Seat | K = 1 | K = 2 |
|---|---|---|
| landlord | 2,895,947 | 4,625,046 |
| peasant | 2,952,779 | 4,681,878 |

  The K = 1 counts are identical to DouZero+'s reference model.
- **Gradients:** none. `actor_belief` refuses a predictor that is in training mode, has any `requires_grad`, or belongs to another seat. It runs the predictor under `torch.no_grad()`.

> **Frozen model:** a network whose weights no longer change. Freezing the predictor means M1 and M2c see *exactly* the same first slot (the next-hand belief) for the whole experiment. Their inputs differ only in the second slot. But that slot is also learned: M2c's second-slot weights get gradient, M1's stay at exactly zero, and global gradient clipping (shrinking all gradients together when their combined norm is above a limit; see step 19) couples all parameters. Runs are also asynchronous, so outcomes vary with the seed. Outcome differences therefore need paired, multi-seed evaluation before they can be credited to the second slot.

### 6. The belief tensor is built [Stage 3]

- **Code:** `utils.actor_belief` calls `douzero/dmc/belief.py` `build_belief(hand_model, x_no_action, probs, labels)`. That function is the only place where the arms differ.
- **The Oracle's input.** For the Oracle arm only, `act()` first calls `douzero/dmc/oracle.py` `oracle_labels(env.env.infoset, position, x_no_action)`. It checks that the infoset belongs to this seat and this exact state, then returns the true counts (1, 2, 15), which are ORACLE-ONLY.
- **Output:** `belief` (1, 150) = `[next-player slot (75) | other-player slot (75)]`.
  - Each slot is a (15, 5) belief flattened kind-major: index = 5 × kind + count.
  - Kinds run in the order 3, 4, …, K, A, 2, small joker, big joker.

| Arm | `--hand_model` | Slot 1 (next player) | Slot 2 (other hidden player) | Information |
|---|---|---|---|---|
| DouZero | `baseline` (default) | no belief input at all | | PUBLIC |
| M0c | `none` | zeros | zeros | PUBLIC |
| M1 | `next` | p_next (from the shared next-hand predictor) | zeros | PUBLIC |
| M2c | `derived_both` | p_next (the same predictor) | reflect(p_next, u) | PUBLIC |
| M2 | `learned_both` | p_next (two-hand predictor) | p_other (two-hand predictor) | PUBLIC |
| Oracle | `oracle` | one-hot true next counts | one-hot true other counts | ORACLE-ONLY |

- **Reflection (M2c).** Card conservation says next + other = u for every kind. So if the next player holds k of the u unseen 7s, the other player holds u − k: p_other(k) = p_next(u − k). This adds no new information, only a different *representation* of the same belief.
- **Guard rails.** `build_belief` raises an error if a non-Oracle arm is given true hands, if the Oracle is not given them, or if the predictor output has the wrong K. A runtime test makes `oracle_labels` raise during the non-oracle arms' games, and those games still complete.
- **Gradients:** `probs` are **detached** inside `build_belief`.

> **Stop-gradient (`.detach()`):** the value flows forward into the Q-network, but the backward pass stops there. However large the Q-loss, it cannot change the predictor. Tests T9 and T15 check this.

The actor stores `belief[0]` (150,) in `belief_buf[position]` (step 10). It then expands the belief to (n_legal, 150), so that every legal action is scored with the same belief.

### 7. Legal actions are enumerated [DouZero]

- **Code:** `GameEnv.get_legal_card_play_actions()` (`game.py` 177–263), built on `move_generator.py`, `move_detector.py` and `move_selector.py`.
- **What it does:** it lists every move the rules allow, given your hand and the move to beat. "Pass" is included unless you must lead. There can be one legal action or several hundred.
- **Tensors:** `legal_actions` (n_legal moves). Each move also appears as the last 54 columns of its `x_batch` row.
- **Gradients:** none.

> **Action as input:** DouZero does not have one output per possible move; there are far too many. The move is part of the *input* (`x_batch`), and the network outputs one number per (state, move) pair. That is why `x_batch` has one row per legal action.

### 8. Q-values are computed for every legal action [DouZero network, Stage 3 belief input]

- **Code:** `act()` calls `model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags, belief=...)` through the `Model` wrapper (`models.py`, class `Model`). That reaches `LandlordLstmModel` / `FarmerLstmModel` `.forward`, built with `belief_dim=150` for every belief arm (M0c, M1, M2c, M2, Oracle); `belief_dim=0` is the DouZero baseline.
- **Forward pass:**
  1. The LSTM reads `z_batch` (n, 5, 162) and outputs (n, 128).
  2. Concatenate `[LSTM 128 | x_batch 373/484 | belief 150]` = (n, 651) for the landlord or (n, 762) for peasants.
  3. Five layers of 512 units with ReLU, then one output: `Q` (n, 1).
- **Meaning:** Q is the predicted final score for *this player* if they play that action and then keep playing as the policy does.
- **Size:** 1,534,465 parameters for the landlord and 1,591,297 for peasants. That is the original 1,457,665 / 1,514,497 plus 150 × 512 new weights.
- **Warm start:** `models.warm_start_from_douzero` copies every DouZero-ADP weight and sets the 150 new input columns to exactly 0.
  - At step 0 the belief cannot change Q at all, and T10 checks this bit for bit.
  - Q also matches DouZero-ADP within the test tolerance of 1e-6 (T10 asserts this).
  - Arms can only diverge once training moves those new weights.
- **Guard:** `forward` raises an error if the belief is missing, is passed to a network built without one, or still carries gradients.
- **Gradients:** none in the actor.

### 9. Epsilon-greedy chooses an action [DouZero]

- **Code:** the end of `LandlordLstmModel.forward` / `FarmerLstmModel.forward` (`models.py`), when `return_value=False`.
- **The rule:**
  - With probability `--exp_epsilon` (default 0.01), play a uniformly random legal action (`torch.randint`).
  - Otherwise play `argmax Q`.
- **In the actor:** `act()` maps the returned index to `obs['legal_actions'][index]`.
- **Gradients:** none.

> **Epsilon-greedy:** a small amount of random exploration, so that training data also contains moves the network currently thinks are worse. Evaluation agents play pure argmax with no exploration, and skip the network when there is only one legal action.

---

## Part B: one episode

### 10. The decision is stored [DouZero + Stage 3 belief]

- **Code:** `act()` appends to per-seat Python lists:
  - before acting: `env_output['obs_x_no_action']` (int8) and `env_output['obs_z']` (int8);
  - for a belief arm: the belief, (150,) float32;
  - after choosing: `_cards2tensor(action)` (54,) int8, the encoding of the chosen move.
- **Not stored:** the other legal actions. The learner only needs the move that was actually played, and it rebuilds that move's input row as `[x_no_action | action]`.
- **Information:** PUBLIC for every arm except the Oracle, whose stored belief holds the true hands (ORACLE-ONLY). Each run trains one arm, so an Oracle belief can never reach another arm's buffers.

**Decision: store the belief, don't recompute it.** For the frozen-predictor experiment, the actor stores the exact 150 numbers it fed to the Q-network:

| Concern | A: store the belief (chosen) | B: store inputs, recompute in the learner |
|---|---|---|
| Correctness | The learner trains on exactly what the actor used, by construction | Correct only if the recomputation matches; it could differ across CPU and GPU, or if the wrong predictor were loaded |
| Memory | 600 bytes per decision: about 3 MB per seat at the default 50 × 100 buffers | None extra |
| Compute | None in the learner | 3,200 extra predictor forward passes per batch; the K=2 predictor has 4.6M parameters |
| Reproducibility | The buffer is a complete record of what the Q-network saw | Depends on loading the same predictor again |
| Future fine-tuning | Stored beliefs would go stale. But `x_no_action` and `z` are stored anyway, so B can be added later | Natural fit |

Test T14 shows that the learner's Q inputs (z, `[x_no_action | action]`, belief) equal the actor's inputs for the chosen action exactly, row by row, for every arm and seat.

### 11. The episode continues [DouZero]

`env.step(action)` passes through `Environment.step` → `Env.step` → `GameEnv.step`. The engine plays the move and moves to the next seat, and the next player gets its own observation (back to step 3). One actor process plays all three seats, each with its own seat network.

### 12. The game ends [DouZero]

A game ends as soon as any player has no cards left (`GameEnv.game_done`). The side of that player wins: either the landlord alone, or the two peasants together.

### 13. The terminal reward is assigned [DouZero]

- **Code:** `Env._get_reward` (`env.py` 97–119).
- **ADP objective** (what we train): the landlord gets +2^bombs on a win and −2^bombs on a loss, where `bombs` counts the bombs and rockets played during the game. The WP objective uses ±1.
- **Peasants:** the value is from the landlord's point of view. Both peasants get its negative.
- **Every other step has reward 0.** Only the end of the game says how well you played.

### 14. Monte Carlo targets are created [DouZero]

- **Code:** `act()`, in the `if env_output['done']:` block at the end of each game.
- **Rule:** every decision a seat made in this game gets the same target G, the final return from that seat's point of view: `episode_return` for the landlord and `-episode_return` for the peasants.
- **Buffers filled:** `target_buf` (G for each decision), plus `done_buf` (True on the seat's last decision of the game) and `episode_return_buf` (G there, 0 elsewhere), which are used only for logging.

> **Monte Carlo return:** the target is the *actual* outcome of the game, with no discounting (γ = 1). There is no "bootstrapping" from the network's own guess of the next state's value, which is what TD methods such as Q-learning do. A good move followed by a lost game still gets a bad target. The network sorts out credit statistically over millions of games.

### 15. Samples enter the shared buffers [DouZero + Stage 3 belief]

- **Code:** `utils.create_buffers` and the flush loop at the end of `act()`, which runs after each game and writes a full slot whenever a seat has more than T decisions waiting (so the newest decisions wait for the next game to complete a slot).
- **Layout:** per seat there are `--num_buffers` (50) slots. Each slot holds T = `--unroll_length` (100) consecutive decisions of that seat:

| Buffer | Shape per slot | dtype |
|---|---|---|
| `done` | (100,) | bool |
| `episode_return` | (100,) | float32 |
| `target` | (100,) | float32 |
| `obs_x_no_action` | (100, 319 / 430) | int8 |
| `obs_action` | (100, 54) | int8 |
| `obs_z` | (100, 5, 162) | int8 |
| `belief` (belief arms) | (100, 150) | float32 |

- **Protocol:** an actor takes a free slot index from `free_queue[seat]`, writes 100 decisions (all seven fields, row by row) and puts the index into `full_queue[seat]`. It then drops those 100 rows from every per-seat list together. Before each flush it checks that the belief list and the observation list have the same length.

> **Shared buffers:** tensors placed in shared memory (`share_memory_()`), so that actor processes and learner threads read and write the same memory without copying. The queues only pass slot numbers, which tells each side which slots are free and which are full.

**Alignment test (T7).** A silent one-row shift between observations, actions, beliefs and targets would invalidate everything, so T7 runs the real `act()` for six seeded games per arm, in one process. It then checks every flushed row of every seat:
- the stored observation and history equal `get_obs` of that decision;
- the stored action is the move the Q-network chose;
- the own hand shrinks by exactly the stored action between a seat's consecutive decisions;
- each game's first row has a full hand;
- the belief equals the actor's input and is recomputed exactly from the stored observation;
- the Oracle's slots equal the true hands;
- the target equals the engine's real outcome with the right sign, and `done` marks the seat's last decision.

Its positive controls show that shifting the beliefs, actions or targets by one row, or dropping the first row, is caught.

---

## Part C: one learner update

### 16. The learner samples a batch [DouZero + Stage 3 checks]

- **Code:**
  - `train()` starts `--num_threads` (4) `batch_and_learn` threads per seat.
  - `utils.get_batch` takes B = `--batch_size` (32) full slots from `full_queue` (it blocks until they arrive) and stacks them into tensors of shape (100, 32, …). It then returns the slots to `free_queue`. A `None` in the queue is the shutdown signal and raises `StopLearning`.
  - `dmc.learn()` calls `learner_inputs(batch, device)`. It checks every shape and that the beliefs are finite, then flattens row-major (row = t × B + b):
    - `obs_x` = cat(`obs_x_no_action`, `obs_action`), giving (3200, 373 / 484) float;
    - `obs_z` (3200, 5, 162);
    - `belief` (3200, 150);
    - `target` (3200,).
- **Information:** PUBLIC inputs, the stored beliefs and the realized returns.

### 17. Predictor loss [planned: offline pretraining, Stage 4]

- **When:** in the main experiment, predictors are trained **once, offline, before RL**, then frozen. During RL there is no predictor loss, and the learner never computes CE.
- **Data:** decisions from DouZero-ADP self-play. The labels are true counts from `hand_belief.hand_labels` (LABEL), shape (B, K, 15): rows (next) or (next, other).
- **Loss:** `belief.hand_cross_entropy(logits, labels)` returns one cross-entropy per predicted hand, shape (K,). For M2, the entries are CE_next and CE_other. The training loss is their sum.
- **Provenance:** the saved file will carry `trained=True` and a description of the training data, and `train()` refuses predictor files without it.
- **Contrast with DouZero+:** DouZero+ trains its predictor *jointly* inside `learn()`, adding `pre_loss` to the Q-loss. We keep that only as an optional later ablation; see `CS230_DESIGN.md` for why.

### 18. Q-loss [DouZero + Stage 3 stats]

- **Code:** `dmc.compute_loss`: `((Q.squeeze(-1) - target) ** 2).mean()`, the mean squared error over 3,200 samples. `learn()` raises a RuntimeError before any weight changes if the loss is not finite.
- **Q:** comes from `model(obs_z, obs_x, return_value=True, belief=belief)`.
- **Logged with every update:** `loss`, `q_mean`, `q_std` and `mean_episode_return` (the mean over the last 100 batches that held a finished game).
- **Gradients:** into the seat's Q-network only.

### 19. Gradients are applied [DouZero + Stage 3 stats]

- **Code:** `learn()` runs `optimizer.zero_grad()`, `loss.backward()`, `clip_grad_norm_(model.parameters(), --max_grad_norm 40)`, then `optimizer.step()`. If the gradient norm is not finite, it raises before the step. The run does not skip the update: the learner thread fails, the watchdog ends the run with status `failed`, and `train.py` exits 1.
- **Logged:** the total gradient norm before clipping (`grad_norm`), and whether it was clipped (`clipped`).
- **Optimizer:** one RMSprop per seat (`utils.create_optimizers`; RMSprop scales each step by a running average of squared gradients): lr 1e-4, alpha 0.99, eps 1e-5, momentum 0. Predictors are in no optimizer.
- **Unused slots stay at zero.** A belief slot that is always 0 (both slots of M0c, slot 2 of M1) contributes exactly 0 gradient to its weights, so a warm-started arm keeps those weights at exactly 0.
  - T17 runs a real `learn()` for M0c and finds that its update equals a belief-free DouZero network's update on the same batch.
  - M1's Q-network behaves exactly like one that has only the 75 next-hand inputs, the DouZero+ input design.
- **Watch out: a fresh optimizer takes big first steps.** The DouZero-ADP release has no optimizer state, so every arm starts with an empty RMSprop. Its running average of squared gradients starts at 0, so the first step moves each weight by about lr/√(1 − alpha) = 10 × lr, whatever the gradient size. The effect fades over roughly 100–300 updates.
  - `trace_learner_update.py` shows it: one update moves Q's mean from 0.87 to 1.01.
  - It hits all arms equally. But M0c is precisely "DouZero-ADP weights with a fresh optimizer", not literally continued DouZero training.

### 20. Updated parameters are synchronized to actors [DouZero]

- **Code:** at the end of `learn()`, for every actor device: `actor_model.get_model(position).load_state_dict(model.state_dict())`. This runs after **every** update, so each actor receives the new weights right away.
- **What actors play with.** An actor plays with the weights of the last sync. Between two syncs it uses the same weights, and a few decisions may use slightly older weights than the learner has. The copy is not locked against actor forward passes, so one decision can even overlap a copy. This race is inherited from DouZero.
- **Off-policy staleness.** A stored decision was chosen by the actor's weights at that time. A row can then wait in `full_queue` while the learner makes other updates, and the learner trains it with its current weights. This is off-policy staleness: the training data comes from a slightly older policy than the one being trained. It is the normal cost of asynchronous actor-learner training, and DMC tolerates it. The target is still the Monte Carlo return of the game the actor actually played.
- **Frozen predictors:** they never change, so they are never synced. Actors receive them once, at start-up.

### 21. Checkpoints are saved [Stage 3]

- **Code:** the `checkpoint()` closure inside `train()` calls `dmc.save_checkpoint` every `--save_interval` (30) minutes and once more at the end. It takes no frame count: the counters are read under the same locks as the weights.
- **Consistent snapshot:** it holds the three seat locks and the stats lock while copying the weights, optimizer states and counters, so no update is half-applied. A checkpoint therefore satisfies frames = unroll_length × batch_size × (sum of updates) = sum of position_frames. The integration tests check this. Each save writes `model.tar` (to a temporary file, then renamed into place, so a crash cannot corrupt the previous one) and `weights_<frames>.tar` beside it.
- **Git provenance.** `run_dir.git_info` records the state of the code folder `DouZero/` only (notes and docs elsewhere in the repository do not count): the commit, the branch (None when detached), whether the code is dirty, every untracked non-ignored file with its sha256, `diff_sha256` (the tracked diff) and `code_sha256` (the tracked diff plus the untracked hashes). `code_sha256` therefore pins the exact code a run used, even from a dirty tree. `run_dir.check_git` refuses a dirty tree, and an unreadable git state, unless `--allow_dirty` is given. The git record is stored in `config.json` and in every line of `attempts.jsonl`.
- **The run folder `artifacts/runs/<xpid>/`:**

| File | Contents |
|---|---|
| `config.json` | Written once, at the first start: all flags; the arm; the seed; the warm-start and predictor files with sha256, K, frozen=True and provenance; the git record (see above); library versions; the machine. **No environment variables** |
| `attempts.jsonl` | One line per start, including each resume: the start time, the attempt number, the seed, the git record, `resumed_from_frames` (None on the first start) and that attempt's full flags. Each attempt's flags can differ (for example `--total_frames`) |
| `checkpoints/model.tar` | The resumable checkpoint: all seats' networks and optimizer states, the counters (frames, position_frames, updates, stats, elapsed_s), the flags, and the run metadata under `cs230`. Plain types only, so `torch.load(..., weights_only=True)` reads it |
| `checkpoints/weights_<frames>.tar` | Written at every save: all seats' weights, frames and run metadata. `models.load_arm_checkpoint` and `BeliefDeepAgent` read these. Not resumable |
| `checkpoints/<seat>_weights_<frames>.ckpt` | Only for `--hand_model baseline` runs: one plain state dict per seat, for DouZero's `evaluate.py`. Belief arms have none |
| `metrics.csv` | **The authoritative training record.** One row every 5 s plus a final row: wall time, frames, fps, and per seat the update count, loss, Q mean, Q std, grad norm, clipped and mean episode return. The columns are fixed by the first row. On resume, rows after the checkpoint's frames are dropped, because that progress was lost |
| `summary.md` | Written at the end of every run, however it ends: status, arm, seed, number of attempts, starting checkpoint, predictor, frames, updates, wall-clock time, final stats, checkpoints, warnings, machine and any failure traceback |
| `logs.csv`, `fields.csv`, `meta.json`, `out.log` | DouZero's own `FileWriter` output, kept for compatibility. **Use `metrics.csv`, not `logs.csv`** (see Known limitations). `meta.json` holds the FileWriter metadata: all flags, the run id, dates, and a git block (commit, branch, is_dirty, path) that is less complete than `config.json`'s git record. Of the environment it keeps only `CUDA_VISIBLE_DEVICES` and `OMP_NUM_THREADS` (plus `SLURM_*` when `SLURM_JOB_ID` is set) |
| `traces/` | Created empty; reserved for trace output |

- **Summary warnings.** `summary.md` warns when the run did not complete; when the git state of any attempt could not be read; when any attempt trained from uncommitted changes (it lists their code sha256s); when the attempts ran different code (a resume on changed code is accepted but flagged; see `attempts.jsonl`); and when the predictors are untrained (the beliefs are then noise: a smoke run, not an experiment).

- **Reloading fails loudly.** `models.load_arm_checkpoint` reads the arm from the metadata and loads every network strictly. It raises if:
  - the file is missing or has no metadata;
  - a weight is missing or has the wrong shape;
  - a predictor file is missing, changed since training (sha256), or has the wrong seat or K.
  - `--load_model` also refuses to resume a run as a different arm. T18 and T19 test all of this.
- **Shutdown.** However a run ends, `train()`'s `finally` block runs the same steps. Each step is guarded, so a failure in one does not skip the others:
  1. **Stop the learners.** `wake_learners` puts one `None` sentinel per learner thread on each full queue. A sentinel is a stop message: a learner waiting for a batch receives it and exits. The puts run in daemon helper threads (background threads that the program does not wait for at exit), because an actor killed while holding a queue's write lock (by the out-of-memory killer, which ends processes when memory runs out, say) would otherwise block a put forever. The wait is at most `STOP_TIMEOUT_S` (60 s). Learners still running are named in the status.
  2. **Save the final checkpoint.** Each of its four locks (the three seats and the stats lock) waits at most 60 s, so a stuck learner that holds a lock is recorded as an error (a `TimeoutError`), not a hang.
  3. **Stop the actors.** All actors are terminated; each is then joined for at most 10 s and killed if it is still alive.
  4. **Finish up.** The signal handlers are restored, a final `metrics.csv` row is written if needed, `summary.md` is written and the `FileWriter` is closed.

  How the run ended is its **status**: `completed`; `interrupted` (Ctrl-C, SIGINT); `terminated (SIGTERM)` or `terminated (SIGHUP)`; or `failed: ...` (an exception, which is also re-raised). SIGTERM and SIGHUP (a timeout, a preemption, a closed terminal) are mapped to a `Terminated` exception in the main thread, so they take the same clean path. A signal the run was started with ignored stays ignored: `nohup` ignores SIGHUP, and that must keep working. `train.py` exits 0 only for `completed`, and 1 otherwise. A signal during shutdown is handled per step.

- **Watchdog.** Every 5 s `check_workers` fails the run loudly if a learner thread raised, an actor process died, or a learner thread stopped before training ended. It runs once more after the loop, so a failure in the last tick, after another learner reached `--total_frames`, still fails the run. DouZero never stopped its actors, so a finished run used to hang on exit; this is the stop-and-watch design that replaced that.

---

## Part D: evaluation

### 22. Evaluation runs [DouZero + Stage 3 agents; runs planned]

- **Code:**
  - `evaluate.py` calls `douzero/evaluation/simulation.py` `evaluate()`, which plays a fixed list of deals (`eval_data.pkl`, made by `generate_eval_data.py`). Each seat is played by a `DeepAgent` (`douzero/evaluation/deep_agent.py`), which plays greedily. `DeepAgent` loads strictly: it refuses a run checkpoint (`model.tar` or `weights_<frames>.tar`, recognised by its `model_state_dict` or `cs230` keys) and refuses any state dict that is missing a weight or has the wrong shape. Without this, a missing weight used to load as a random network.
  - `simulation.evaluate` fails loudly: it raises `RuntimeError` if any worker exits non-zero. Before, a dead worker's result never arrived, so evaluation hung. It still cannot tell a stuck worker from a slow one (see Known limitations).
  - It reports **WP**, the share of deals won by each side, and **ADP**, the average points per deal. Evaluation scores a game as ±2 × 2^bombs for the landlord side, because the landlord plays for a double stake against two peasants who each win or lose 2^bombs. Evaluation ADP is therefore on twice the scale of the training target in step 13. `tools/dup_eval.py` uses the same units.
  - `tools/dup_eval.py` (project root) runs **paired, duplicate** evaluation and stores per-deal outcomes, which allows bootstrap confidence intervals.
- **Stage 3 agents:**
  - `douzero/evaluation/belief_agent.py` `BeliefDeepAgent(position, checkpoint_path, predictor_dir=None, allow_untrained=False)` plays the four public arms (M0c, M1, M2c and M2). It reads the arm from the checkpoint's metadata, never from the caller.
  - `douzero/evaluation/oracle_agent.py` `OracleAgent` plays only the Oracle.
  - Both load strictly. Each refuses the other kind of checkpoint, and `BeliefDeepAgent` refuses untrained predictors unless `allow_untrained=True`. Evaluation therefore refuses untrained predictors by default, as training does.
  - Blindness to the hidden-hand swap is tested in two places. `test_evaluation_agents_are_blind_to_the_swap_except_the_oracle` checks it at the agent level, for M2c through `BeliefDeepAgent` and for the Oracle through `OracleAgent`. The four public arms are checked through `actor_belief` and the Q-network, and every public arm is checked through the real `act()` swap test. Extending the agent-level test to all four public arms is not done.
  - **[planned]** wiring them into `evaluate.py`/`dup_eval.py`. `evaluate.py` cannot run a belief arm yet, and the planned Stage 4/5 work (offline predictor training, then evaluation runs) will add it.

> **Paired (duplicate) evaluation:** the same deal is played twice with the two agents' seats swapped. Card luck then hits both agents equally and mostly cancels in the difference. This lets far fewer deals detect a real difference between M2c and M1. It does *not* remove training-seed luck, which needs several seeds per arm.

---

## One concrete transition from actor to learner

This is a real sample from `trace_episode.py --hand_model derived_both --seed 0`: M2c, the DouZero-ADP warm start, and an untrained K=1 predictor.

**1. The decision (landlord, t = 0, the opening lead).**

| | |
|---|---|
| Own hand (20 cards) | 3 3 5 7 8 9 T T J J J Q K K K A A A 2 B |
| Unseen union, decoded from `x_no_action[54:108]` (34 cards) | 3344445556666777888999TTJQQQKA222R |
| Public sizes of the two hidden hands | next = `landlord_down` 17, other = `landlord_up` 17 |
| Legal actions | 147 |
| Chosen move | play 5 (greedy) |

**2. The belief built by `actor_belief`.**
- **Slot sums:** 15.0 / 15.0, one distribution per card kind in each slot.
- **Expected sizes:** E|next| = 17.1 and E|other| = 16.9, against the true public 17 / 17. The predictor is untrained, so it is near-uniform.
- **Example, kind "3" (u = 2):** p_next = [0.321, 0.329, 0.349], and M2c's reflection gives p_other = [0.349, 0.329, 0.321].

**3. What the buffer stores for this row.**

| Key | Shape | dtype |
|---|---|---|
| `obs_x_no_action` | (319,) | int8 |
| `obs_action` | (54,) | int8 |
| `obs_z` | (5, 162) | int8 (no earlier moves yet) |
| `belief` | (150,) | float32 |
| `target`, `done`, `episode_return` | scalars | float32, bool, float32 |

**4. The game ends.** The landlord wins with no bombs, so G = +2^0 = **+1** for the landlord and −1 for both peasants. All 6 landlord decisions get target +1, `done` is True on the last one, and `episode_return` is +1 there and 0 elsewhere.

**5. What the learner sees.** `learner_inputs` turns the row into `obs_z` (1, 5, 162), `obs_x` = `[x_no_action | action]` (1, 373) and `belief` (1, 150), with target (1,).

**6. The learner's Q-loss for this row.**
- **The learner recomputes Q with its current weights.** The actor computed Q = **+0.1773** for this move, with the weights it held when it chose the move. The inputs are identical. If no update came between the two, the learner gets the same value. This trace has no learner-side Q to show: `trace_episode.py` runs no learner, and `trace_learner_update.py` prints only batch statistics. In training, a row can wait in `full_queue` across several updates, so its learner Q can differ from the stored actor Q (off-policy staleness, step 20). The stored target G is the only label; the loss uses the learner's Q.
- **Loss (using Q = +0.1773, the case with no intervening update):** (Q − G)² = (0.1773 − 1)² = **0.677**.
- **Gradient with respect to Q:** 2(Q − G) = −1.65. In a real batch it is divided by 3,200, and gradient descent pushes this Q *up*, toward +1.

---

## The five arms side by side

| Arm | Belief input `[slot 1 \| slot 2]` | Learned during RL | Derived | Frozen | Information available | Question it answers |
|---|---|---|---|---|---|---|
| **M0c** | `[0 \| 0]` | Q-network | nothing | nothing (no predictor) | public observation, as in DouZero | How much does *more DouZero training alone* give, with DouZero-ADP's weights and a fresh optimizer? This is the "extra training" control |
| **M1** | `[p_next \| 0]` | Q-network | nothing | next-hand predictor (pretrained offline) | public observation + a learned estimate of the next hand | Does a DouZero+-style next-player belief help? |
| **M2c** | `[p_next \| reflect(p_next)]` | Q-network | slot 2, by card conservation | **the same** predictor as M1 | exactly M1's information, represented for both hands | **Primary:** does explicitly representing the second hidden hand help, with the same information and the same predictor? (M2c vs M1) |
| **M2** | `[p_next_M2 \| p_other_M2]` | Q-network | nothing | its own two-hand predictor (pretrained offline) | public observation + learned estimates of both hands | **Secondary:** learned two-belief system vs conservation-completed two-belief system. CE_next and CE_other are tracked separately |
| **Oracle** | `[true next \| true other]` | Q-network | nothing | nothing | **the true hidden hands** (cheating) | Positive control: how much could perfect hand knowledge help at all? If the Oracle barely beats M0c, belief inputs cannot be expected to help |

---

## Gradient flow

```
MAIN EXPERIMENT, RL PHASE (wired in Stage 3)

  public observation: x_no_action (1, 319/430), z (1, 5, 162)       [actor]
        |
        v
  HandPredictor              <- frozen: eval(), requires_grad=False, run under no_grad
        |
        v
  probs (1, K, 15, 5)
        |
     .detach()               <- stop-gradient inside build_belief
        |
        v
  belief (1, 150) --- stored in the buffer ---> belief (3200, 150)  [learner]
                                                   |
  x (3200, 373/484), z (3200, 5, 162) -------------+--> Q-network --> Q (3200, 1) --> Q-loss = mean (Q - G)^2
                                                            ^                              |
                                                            +-------- gradients -----------+
                                                      (only Q-network parameters change)


OFFLINE PREDICTOR PRETRAINING (Stage 4, before RL)

  public observation --> HandPredictor --> masked logits (B, K, 15, 5) --+
                              ^                                          |
                              |      true hidden-hand counts (B, K, 15) -+  [LABEL: training only]
                              |                                          v
                              +------------ gradients ------- CE = hand_cross_entropy(logits, labels)
                                    (only predictor parameters change)
```

- **The predictor's CE loss updates the predictor**, offline, before RL.
- **The Q-loss updates the Q-network.**
- **The Q-loss never updates the predictor.** Several things prevent it:
  - the belief is detached;
  - the stored belief has no gradient history;
  - the predictor is frozen, and in no optimizer;
  - the Q-network refuses any belief that carries gradients.

  T15 checks that after a real `learn()` every predictor parameter is unchanged and has no gradient. `trace_learner_update.py` prints "predictor 0".
- **Predictors are frozen for the whole main RL experiment** after offline pretraining.

---

## The two losses with real numbers

### Predictor cross-entropy (CE)

Suppose the next player truly holds **2** sevens, and the predictor's head for the sevens outputs

```
count:        0     1     2     3     4
p_next(7):  0.05  0.10  0.70  0.10  0.05
```

The cross-entropy for this kind is −ln p(true count) = −ln 0.70 = **0.357** nats.

- **A confident wrong guess costs much more.** With p(2) = 0.10 the term is −ln 0.10 = 2.303.
- **A perfect guess costs nothing.** With p(2) = 1 the term is 0.
- **Averaging.** The per-decision loss is the mean of these terms over the 15 kinds, and the batch loss averages over decisions. That is what `hand_cross_entropy` returns, once per predicted hand.

**The mask matters.** Suppose only u = 3 sevens are unseen; the other one is in your hand or already played. Then "holds 4" is impossible: the mask sets that logit to −inf, and the softmax renormalizes over counts 0–3:

```
p_next(7) = [0.053, 0.105, 0.737, 0.105]      CE = -ln 0.737 = 0.305
```

**The card-counting baseline.** Without learning anything, you can still count cards. Suppose 24 cards are unseen, the next player holds 11 of them, and 3 of the unseen cards are sevens. If the next hand were a random 11 of those 24 cards, the number of sevens in it would follow the hypergeometric distribution:

```
P(count = k) = C(3, k) C(21, 11-k) / C(24, 11)  ->  [0.141, 0.424, 0.353, 0.082]
```

So card counting puts p(2) = 3 × 293,930 / 2,496,144 = 0.353, and its CE term is −ln 0.353 = **1.041**. Our predictor's 0.305 would beat card counting by 0.74 nats on this kind. Card counting ignores the public play history, so a useful predictor must at least beat it. Beating it shows the predictor uses that history, but not yet that its beliefs help play. `hand_belief.hypergeometric_belief` computes this baseline.

**Reflection (M2c), same example.** If p_next(7) = [0.053, 0.105, 0.737, 0.105] with u = 3, the other hidden player holds 3 − k sevens whenever the next player holds k:

```
p_other(7) = [p_next(3), p_next(2), p_next(1), p_next(0)] = [0.105, 0.737, 0.105, 0.053]
```

### DMC Q-loss

The landlord chose an action, and the network's Q for it was **0.8**. The game went on, one bomb was played, and the landlord won: the ADP return is G = +2^1 = **2**. That decision's squared error is

```
(Q - G)^2 = (0.8 - 2)^2 = (-1.2)^2 = 1.44
```

- **Direction of the update.** The gradient of the squared error with respect to Q is 2(Q − G) = −2.4. The real loss averages over 3,200 samples, so this sample's gradient is −2.4 / 3,200. Gradient descent therefore pushes this Q *up*, toward 2.
- **Every landlord decision in that game gets the same target G = 2.** That is the Monte Carlo return.
- **Both peasants get G = −2** for the same game. A peasant decision with Q = −0.5 has error (−0.5 − (−2))² = 2.25, and its Q is pushed *down*, toward −2.

[The concrete transition](#one-concrete-transition-from-actor-to-learner) above does the same calculation on a real stored row.

---

## Key tensors

| Tensor | Shape | dtype | Made by | Information | Gradient |
|---|---|---|---|---|---|
| `x_no_action` | (319,) landlord / (430,) peasant per decision | int8 (float32 when fed to a model) | `env.get_obs` | PUBLIC | — |
| `z` | (5, 162) per decision | int8 / float32 | `env.get_obs` | PUBLIC | — |
| `x_batch`, `z_batch` | (n_legal, 373 / 484), (n_legal, 5, 162) | float32 | `get_obs` | PUBLIC | — |
| union u | (15,) | int64 | `belief.union_counts_from_obs(x_no_action)` | PUBLIC (decoded) | — |
| legal mask | (15, 5) | bool | `belief.legal_count_mask(u)` | PUBLIC | — |
| predictor `logits`, `probs` | (1, K, 15, 5) | float32 (−inf / 0 on impossible counts) | `HandPredictor.forward` | estimate | none during RL |
| Oracle labels | (1, 2, 15) | int64 counts 0–4 | `oracle.oracle_labels` | ORACLE-ONLY | — |
| `belief` | (1, 150) in the actor, (150,) stored | float32 | `utils.actor_belief` → `belief.build_belief` | per arm | none (detached) |
| buffer slot | done, return, target (T,); obs (T, 319/430); action (T, 54); z (T, 5, 162); belief (T, 150) | bool, float32, int8, float32 | `act()` | as above | — |
| learner batch | (T, B, …) → flattened to T × B = 3,200 rows | — | `get_batch` → `learner_inputs` | as above | — |
| dense1 input | (n, 651 / 762) | float32 | inside the Q-network's `forward` | — | — |
| `Q` | (n, 1) | float32 | Q-network `forward` | — | yes, into the Q-network (learner) |
| CE | (K,) | float32 | `belief.hand_cross_entropy` | — | yes, into the predictor (offline only) |

K is 1 for M1/M2c and 2 for M2. n is the number of scored (decision, action) rows: the legal actions in the actor, or 3,200 in a learner batch.

---

## Leakage rules

1. **True hands are read in one place only.** Only `hand_belief.hand_labels` may read `InfoSet.all_handcards`. Test T6b keeps an allow-list for each name:
   - `all_handcards` may appear only in `game.py` and `hand_belief.py`;
   - `hand_labels` only in `hand_belief.py`, `douzero/dmc/oracle.py`, and the debug tool `trace_decision.py`;
   - `oracle_labels` only in `oracle.py`, `utils.py` (the actor's Oracle branch), `evaluation/oracle_agent.py`, and the debug tool `trace_episode.py`;
   - the engine's internal state (`info_sets`, `_env`) only inside `game.py` and `env.py`.
2. **Only order-free counts of the union.** `other_hand_cards` may be used only as counts, never in its seat-ordered form. The predictor gets the union exclusively by decoding `x_no_action[54:108]`.
3. **Only the Oracle arm receives true hands as input.** `build_belief` refuses labels for every other arm. `actor_belief` takes only tensors, never an infoset.
4. **Swap tests (T6a, T21).** Move cards between the two hidden hands while keeping sizes and union fixed. A real player cannot tell the two states apart, and neither may we:
   - the observation, the DouZero-ADP decision, and the M0c/M1/M2c/M2 beliefs and Q-values must be bit-for-bit identical, through `actor_belief` and the Q-network for all four arms, and through `BeliefDeepAgent` for M2c (the agent-level check covers that arm only);
   - running the real `act()` with *every* infoset swapped must store exactly the same games for those arms.

   The Oracle's belief must change, and does.

---

## How we will know it is working

- **Logged now, in every run's `metrics.csv`** (not `logs.csv`): per seat, the update count, loss, Q mean, Q std, grad norm, clipped and mean episode return; plus frames, fps and wall time. `summary.md` writes up the run.
- **Are the tests passing?** The unit suite (188 tests at Stage 3, one process): `../.venv/bin/python -m pytest tests -p no:cacheprovider -q`. The integration file is collected too, but its 8 tests are skipped by default, so the output reads `188 passed, 8 skipped`. The opt-in end-to-end suite runs real `train()` processes, about 75 s: `CS230_INTEGRATION=1 ../.venv/bin/python -m pytest tests/test_integration.py -p no:cacheprovider -q`. Its 8 tests check that:
  - a tiny run completes, writes its checkpoints and summary, resumes, and refuses a changed flag (`--learning_rate`) before recording the attempt;
  - SIGTERM, SIGHUP and SIGINT each still save a final checkpoint and summary, and leave no actor process behind;
  - a killed actor fails the run loudly (exit code 1, `RuntimeError` in the summary) and still saves;
  - a refused start (a missing predictor directory, or a warm start that does not fit) leaves no run directory;
  - evaluation raises when a worker dies, instead of hanging.

  Every checkpoint these tests read must satisfy the frame-count invariant from step 21.
- **Unit tests behind the shutdown and watchdog claims (steps 19 and 21).** `tests/test_checkpoints.py` checks that the watchdog reports dead actors and failed learners (`test_the_watchdog_reports_dead_actors_and_failed_learners`), that waking the learners cannot hang on a queue lock a dead actor held (`test_waking_the_learners_cannot_hang_on_a_queue_lock_a_dead_actor_held`), that the final checkpoint gives up on a lock a stuck learner holds (`test_the_final_checkpoint_gives_up_on_a_lock_a_stuck_learner_holds`), that an ignored signal stays ignored (`test_stop_handlers_leave_an_ignored_signal_ignored`), and that git provenance pins untracked code and fails closed (`test_git_provenance_pins_untracked_code_and_fails_closed`). `tests/test_pipeline.py` checks that actor seeds never collide (`test_actor_seeds_never_collide_across_nearby_runs_devices_actors_or_attempts`).
- **Is the Q-network learning?**
  - The loss trends down; it is noisy because targets are ±2^bombs.
  - Q mean and Q std stay in a sensible range.
  - The grad norm is mostly under 40.
  - Above all, validation **ADP and WP from paired evaluation**, across several seeds per arm (planned).
- **Is the belief input being used?**
  - The norm of `dense1`'s 150 belief columns starts at exactly 0. No logged metric records it yet. The `trace_learner_update.py` output below shows the first update: each belief slot's columns move by an L2 norm of about 0.11, which is also their norm after that update, since they started at zero.
  - `trace_decision.py`'s line "identical for all five arms: True" will turn False once the networks have trained.
  - An evaluation-time ablation will replace the belief with zeros or with another decision's belief and measure the ADP drop (planned).
- **Is the predictor learning? (Stage 4, planned)**
  - Validation CE falls **below the hypergeometric card-counting CE**.
  - Also track: the mean absolute error of expected counts, per-seat CE, CE_next and CE_other for M2, and M2's conservation violation |E_next + E_other − u|.
  - Track drift as well: CE on the *current* RL policy's games, computed offline from saved trajectories.
- **Plots** of these metrics will come from the remote runs (`artifacts/plots/`, built from the raw `metrics.csv`), never from the laptop.

---

## Known limitations (Stage 3)

Not fixed in Stage 3. Each is either small or an open decision for the user.

- **`logs.csv` is not the training record.** DouZero's `FileWriter` appends to it across attempts and never truncates it, so it can hold lost progress and repeated frame numbers. Use `metrics.csv`.
- **A resume on changed code is accepted**, flagged in `summary.md` and recorded per attempt. Whether to refuse it is an open decision.
- **Flags not in `RESUME_FIXED_FLAGS`** (`training_device`, `num_actors`, `num_threads`, `disable_checkpoint`) may change on resume; each attempt's flags are in `attempts.jsonl`. `--disable_checkpoint` on a resume still truncates `metrics.csv`, then saves nothing.
- **Not checkpointed:** the running mean of the last 100 episode returns restarts after a resume.
- **Durability:** checkpoints are renamed atomically but not fsynced, so a power loss can lose the latest save.
- **No stall watchdog.** A learner that is alive but stuck is not detected during training; it is only bounded at shutdown. If it is stuck while holding its seat lock, the next periodic checkpoint, and so the main loop and its watchdog, wait for it too.
- **After a stop timeout.** A learner still running after the 60 s stop timeout could change the weights after the final snapshot; the status names the stuck threads, so the run is not reported as clean.
- **Orphaned actors.** If the parent process is SIGKILLed, its actors are orphaned (the daemon flag only helps on a normal exit).
- **Start-up window.** A signal before the stop handlers are installed ends the run without a summary. A signal in the few bytecodes between a lock acquire and its release can leak a lock; the final checkpoint then times out after 60 s instead of hanging.
- **Predictor files are read twice.** They are hashed in one read and loaded in another, so do not change them while a run starts.
- **`trained` is self-attested.** A predictor's `trained` flag records provenance; nothing verifies it.
- **Evaluation hang and division.** `simulation.evaluate` can still hang on a stuck (not dead) worker, and divides by zero on an empty deal file. `generate_eval_data.py` is unseeded and records no hash (open item: seed the evaluation deals).
- **Training seeds are not deal-paired** (see step 1). Pairing is evaluation-only.
- **Teaching tool shortcut.** `trace_decision.py` gives the Oracle belief the true hands directly, not through `oracle.oracle_labels`. Training and evaluation go through `oracle_labels`.

## What to understand before real training

1. **Actors don't need gradients.** An actor only *chooses moves*. It runs the networks forward under `torch.no_grad()`, which is faster and uses less memory, and nothing it does changes a weight.
2. **The learner does need them.** Learning means changing weights to reduce a loss, and the gradient says which way each weight should move. Only the learner computes them, on a batch of 3,200 stored decisions.
3. **Every decision in a game gets the final Monte Carlo return.** DouDizhu only scores at the end, so DMC uses the final score G as the target for every decision a seat made in that game. Good moves in lost games get a bad target too; averaged over millions of games, moves that tend to lead to wins get higher Q.
4. **M1 and M2c can share one predictor.** M2c's second slot is computed from M1's first slot by card conservation (other = unseen − next). With the same frozen predictor, the two arms see the same information. Their inputs differ only in whether the second hand is written out explicitly, which is the representation question we are testing. The trained networks can still differ (M2c's second-slot weights learn, and the shared trunk and global gradient clipping couple them), so training differences are measured by paired, multi-seed evaluation, not read off the input.
5. **The predictor is frozen** so that it cannot drift differently in different arms. If it kept learning inside each run, M1's and M2c's predictors would diverge, and a difference between the arms might come from the predictors rather than from the second slot.
6. **The Q-network can still learn to use a frozen belief.** The belief is just more input numbers. The Q-network's own weights on those 150 inputs (dense1's belief columns, starting at 0) are trained by the Q-loss like every other weight. Freezing fixes what the belief *says*, not how the Q-network *uses* it.
7. **What changes during RL:** the three seat Q-networks (LSTM, all six dense layers, including the belief columns) and their RMSprop states. Changes are copied to the actors after every update.
8. **What does not change:**
   - the predictors and the game rules;
   - the always-zero belief columns of M0c, and of M1's second slot;
   - the stored data, which is consumed, not edited.
9. **What a buffer does.** It is a shelf of slots in shared memory. An actor fills a slot with 100 consecutive decisions of one seat and hands over its number. A learner takes 32 full slots, trains on them and hands the numbers back. The queues pass slot numbers, so no data is copied.
10. **What model synchronization means.** After each update, the learner copies its new weights into the actors' copies of the networks. An actor plays with the weights of its last sync, so its data comes from a policy that is one or a few updates old. That is off-policy staleness (step 20): normal for asynchronous training, and the reason the data is never assumed to be exactly from the current network.

---

## The trace tools

### `trace_decision.py`: one decision, all five arms

It plays a seeded deal with random legal moves, then prints one decision for every arm:
- the seat, hands and legal actions;
- the observation shapes;
- the true hands, for comparison;
- predictor outputs next to the card-counting baseline;
- each arm's belief layout;
- Q over the legal actions, the top 5, and the epsilon-greedy choice.

```
$ ../.venv/bin/python trace_decision.py --seed 7 --num_random_moves 8   (excerpt)
ACTING PLAYER
  seat           landlord_up (peasant). Acts next: landlord. Other hidden seat: landlord_down (teammate)
  own hand       10 cards  33445JAAA2
  hidden union   23 cards = landlord 15 + landlord_down 8 (sizes public, split hidden)
  legal actions  17
PREDICTORS (untrained, random weights; input z (1, 5, 162) + x_no_action (1, 430))
    kind u  hand   true  card counting                M1 / M2c                            M2
    Q    4  next      2  [0.01 0.09 0.33 0.41 0.15]   [0.21 0.20 0.21 0.19 0.19]          [0.20 0.20 0.20 0.19 0.20]
            other     2  [0.15 0.41 0.33 0.09 0.01]   reflect [0.19 0.19 0.21 0.20 0.21]  [0.20 0.21 0.20 0.19 0.20]
Q-NETWORK (one network for every arm; DouZero-ADP warm start)
  input   LSTM(z_batch (17, 5, 162)) -> 128 | x_batch (17, 484) | belief (17, 150)  ->  dense1 input 762
  output  Q (17, 1): min -2.424  max -0.889  mean -1.393
  identical for all five arms: True (the new belief weights are still zero)
    1. Q = -0.889  play 44
Epsilon-greedy (epsilon 0.01, the rule in models.py forward): draw 0.873 >= 0.01 -> greedy (highest Q): play 44
```

### `trace_episode.py`: one full game, one arm

The game is played the way actor 0 of a `--seed` run plays it. Test T20 checks that the rows it shows equal what the real `act()` stores. The output is also written to `debug/episode_trace.txt`.

```
$ ../.venv/bin/python trace_episode.py --hand_model derived_both --seed 0   (excerpt)
ONE GAME, ARM M2c (--hand_model derived_both), seed 0
Q-network: DouZero-ADP warm start, belief weights zero. Predictors: UNTRAINED, so beliefs are noise. Exploration epsilon 0.01.

DECISIONS
    t  seat           own  hidden  legal  E|next|  E|other|  Q(chosen)  move                choice
    0  landlord        20   17/17    147     17.1      16.9     +0.177  play 5              greedy
    1  landlord_down   17   17/19      9     18.1      17.9     -0.286  play 9              greedy
    2  landlord_up     17   19/16      6     17.7      17.3     +0.457  play A              greedy
   ...
   12  landlord        16   15/15     92     15.0      15.0     +0.572  play 789TJQ         greedy
   15  landlord        10   15/15     27     15.0      15.0     +0.988  play 33JJKKKAAA     greedy

GAME OVER
  winner: landlord  |  bombs and rockets played: 0  |  ADP return for the landlord = +2^0 = +1
  training return G per seat: landlord +1, landlord_up -1, landlord_down -1
  decisions per seat: landlord 6, landlord_up 5, landlord_down 5

WHAT THE BUFFER STORES (3 of the 16 rows; the learner never sees the other legal moves)
  landlord, first row (t=0): target +1, done False, episode_return +0
    obs_x_no_action (319,) int8: own hand 335789TTJJJQKKKAAA2B;  obs_action (54,) int8: play 5
    obs_z (5, 162) int8: 0 non-pass moves among the last 15;  belief (150,) float32: slot sums 15.0 / 15.0
  landlord, last row (t=15): target +1, done True, episode_return +1
    obs_x_no_action (319,) int8: own hand 33JJKKKAAA;  obs_action (54,) int8: play 33JJKKKAAA
```

### `trace_learner_update.py`: one learner update

The real `act()` plays 4 games in this process, the real `get_batch()` forms a batch, and the real `learn()` makes one update.

```
$ ../.venv/bin/python trace_learner_update.py --hand_model derived_both --seat landlord --seed 0
BATCH (the real get_batch() after 4 games of the real act(): T=8 steps x B=2 buffers = 16 decisions)
  obs_x_no_action (8, 2, 319) int8 + obs_action (8, 2, 54) int8  ->  obs_x (16, 373) float32 = [x_no_action | action]
  obs_z           (8, 2, 5, 162) int8  ->  (16, 5, 162) float32
  belief          (8, 2, 150) float32  ->  (16, 150), exactly as the actor computed it
  target G        (8, 2) float32  ->  (16,): {'+1': 16}
BEFORE THE UPDATE
  Q mean +0.8657  std 0.4931  |  loss = mean (Q - G)^2 = 0.2460
THE UPDATE (dmc.learn: loss.backward(), clip_grad_norm_ to 40.0, one RMSprop step, sync to actors)
  gradient norm 22.1313 (not clipped)
AFTER THE UPDATE (same batch)
  Q mean +1.0129  std 0.3821  |  loss = 0.1370 (-0.1090)
PARAMETER CHANGE (L2 norm of new minus old weights)
  Q-network, all parameters             0.412664
    LSTM over z                          0.179635
    dense1, its 501 original inputs      0.248009
    dense1, belief slot 1 (75 inputs)    0.113605
    dense1, belief slot 2 (75 inputs)    0.113554
    dense2 .. dense6                     0.225197
  predictor                              0 (frozen: no gradient, no optimizer)
  actor copy equals the learner         True (learn() syncs after every update)
```

**How to read it.**
- **The loss fell on this batch, and the predictor did not move.** One update lowered the loss on this batch from 0.246 to 0.137. The predictor moved by exactly 0.
- **The belief columns moved off zero.** The Q-network started to use the belief.
- **M0c shows the opposite.** With `--hand_model none`, the same trace shows both belief slots moving by exactly 0.000000, while every other layer moves by the same amount as here.
- **The first step was big.** Q's mean jumped from 0.87 to 1.01 in a single step. That is the fresh-RMSprop effect described in step 19.
