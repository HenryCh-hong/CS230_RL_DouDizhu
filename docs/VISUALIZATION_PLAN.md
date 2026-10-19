# Visualization plan

The project needs results, and it also needs a way to **see** what happens inside a game and during
training. This document describes the visualization tools: what they show, what data they need,
which parts of that data are public and which are hidden, and how to use them. The current tools
are a first, small prototype. They are deliberately simple and correct rather than fancy.

## 1. Architecture: record, then render

```
one seeded game   ── douzero/viz/record.py ──▶  JSON record  ── douzero/viz/render.py ──▶  one HTML page
(or one decision)    plays it exactly as an       (versioned       pure function: the same      (self-contained,
                     actor would                   schema)          record, the same page)       works offline)
```

- **Recording** (`douzero/viz/record.py`) plays one game the way actor 0 of a CPU-actor training run
  (`--actor_device_cpu`) plays it at its first start (GPU actors derive their seeds from the device
  index and play other games). It uses the same seeding (`utils.seed_actor`), the same belief function (`utils.actor_belief`) and
  the same epsilon-greedy forward pass. A test checks that the recorded game is move for move the
  game `trace_episode.py` prints, and `trace_episode.py` is itself checked against `act()`. So the
  replay shows the real pipeline, not an imitation of it.
- **Rendering** (`douzero/viz/render.py`) turns a record into HTML. It does no game logic and no
  model calls, so a page can be regenerated from its JSON at any time. A later dashboard can read the
  same JSON.
- **Two command-line tools** sit next to the existing `trace_*.py` tools in `DouZero/`:
  - `render_episode_replay.py` records and renders one game;
  - `render_state_comparison.py` rebuilds one decision of a recorded game and compares the five arms on it.

  They are not in `debug/`, because `DouZero/debug/` is git-ignored output. Their default output
  goes to `DouZero/debug/replays/`.
- **Isolation.** Training and evaluation never import `douzero/viz` (a test checks this). The only
  path from the viewer to the true hands is `oracle.oracle_labels`, the reviewed access point,
  which also checks that the labels belong to the state shown. `douzero/viz/record.py` is on the
  leakage test's allow-list as a reviewed reader.

## 2. Why self-contained HTML

| Option | Verdict |
|---|---|
| **One HTML file** (chosen) | It can show the three seats in space, with tables, colours and hover details. It runs offline: no server, no install, no network, no CDN. It steps through decisions with ← → and still works without JavaScript (every step is then shown one after another). It is deterministic, so tests can check it. |
| Markdown + PNG | It cannot step through 30–60 decisions, and a three-seat layout in Markdown is awkward. Plots belong in PNG; game states do not. |
| Local web app or dashboard | Too heavy for now (a server, dependencies, state). The JSON records are designed so that one can be built on top later. |

Text traces (`trace_decision.py`, `trace_episode.py`) remain the quickest view in a terminal.

## 3. What is recorded, and who may know it

Every field has one of four information classes. NORMAL mode shows only the first three.

| Class | Fields (per decision) | Shown in |
|---|---|---|
| **PUBLIC**: every player sees it | cards left per seat (so the hidden hand sizes), cards each seat has played, the move to beat and who made it, bombs so far, the bottom cards the landlord must still hold, the move history | both modes |
| **ACTOR-PRIVATE**: only the acting player knows it | the actor's own hand, and the unseen-card counts derived from it (54 cards minus its hand minus everything played; another player, not knowing this hand, counts differently) | both modes, for the acting seat only |
| **MODEL**: computed from the public observation | the arm's belief input (two slots of 15 kinds × P(count = 0..4)) and each slot's source; the card-counting reference; the number of legal moves; the top 5 moves by Q; the move played, its Q and its rank; the epsilon draw and whether it explored | both modes (Oracle: see below) |
| **HIDDEN**: ground truth, teaching only | all three true hands; the true hidden counts; the cross-entropy of the belief and of card counting against the truth; the leftover hands at the end | TEACHING mode only |

Rules that keep this honest:
- A record made without `--reveal` contains no HIDDEN section at all, so nothing hidden is merely
  hidden from view. A NORMAL page strips whatever record it is given before rendering, and a test
  checks that it is byte-for-byte the same with or without the hidden section.
- **The Oracle arm computes from the true hands**: its belief input, and therefore its Q-values, its
  ranking of the moves and its greedy choice. In NORMAL mode all of these are masked, in the record
  and on the page. Only the move it played stays, because that is public. The comparison's "identical
  rankings" verdict covers the four public arms only.
- NORMAL is a teaching view, not a secret. A record keeps its seed, so the code can always rebuild
  the deal. Asking for `reveal` on a NORMAL record is refused all the same, so the two modes never mix.
- The bottom cards: DouZero removes a bottom card from `three_landlord_cards` when the landlord plays
  a card of that rank. Ranks are interchangeable, so the remaining ones are cards the landlord must
  still hold. That is a public deduction about a hidden hand, yet DouZero's observation does not
  include it, so neither the arms nor card counting use it.

## 4. The tools and how to use them

Run them from `DouZero/`. Each is one process and takes a few seconds.

```
# one game of M2c with every hand revealed (teaching mode)
../.venv/bin/python render_episode_replay.py --hand_model derived_both --seed 1 --reveal
# the same game as the players see it (normal mode)
../.venv/bin/python render_episode_replay.py --hand_model derived_both --seed 1
# decision t = 13 of that game (the 14th; pages count decisions from 1), as all five arms see and score it
../.venv/bin/python render_state_comparison.py --record debug/replays/replay.json --step 13 --reveal
```

Then open the `.html` file in a browser. Useful options:
- `--hand_model` (any arm, or `baseline`);
- `--predictor_dir`, and for the comparison `--k1_predictor_dir` and `--k2_predictor_dir`: trained
  predictors, once Stage 4 has made them;
- `--checkpoint_dir` (the Q-network warm start);
- `--exp_epsilon`;
- `--out`.

**Samples** (seed 1, M2c, 31 decisions, one bomb, the landlord wins), in `docs/visualization/samples/`:
- `replay_M2c_seed1_teaching.html` (+ `.json`): every hand revealed;
- `replay_M2c_seed1_normal.html` (+ `.json`): only what each acting player knows;
- `state_comparison_seed1_t13_teaching.html` and `state_comparison_seed1_t13_normal.html` (+ `.json`):
  decision t = 13 (landlord_down leads freely, 11 legal moves), five arms, in each mode.

The samples use the real DouZero-ADP warm start for the Q-network and **untrained** predictors,
because Stage 4 has not trained any yet. Their beliefs are therefore noise, and the pages say so in red.

## 5. How to read the replay

- **Header.** The arm and its two belief slots. Each slot has a coloured source chip:
  - *zero*: the slot is all zeros;
  - *predicted*: the K=1 predictor's output;
  - *derived by conservation*: other = unseen − next;
  - *learned*: the K=2 predictor's output;
  - *oracle*: the true hand.

  The header also gives the Q-network source, the predictor source (red when untrained), epsilon and
  the mode (NORMAL or TEACHING). One arm plays all three seats (self-play).
- **Navigation.** Prev/next, a slider, the arrow keys, and one tick per decision coloured by seat:
  red for the landlord, blue for landlord_down, green for landlord_up. A ringed tick marks an
  exploration move. "Show every step" displays the whole game at once.
- **The table.** landlord_up sits on top, the Landlord on the left, landlord_down on the right and the
  public table in the middle. Play goes landlord → landlord_down → landlord_up. Each seat shows:
  - its role;
  - its cards left;
  - its hand: green chips for the acting player's own hand, card backs for a hidden hand in NORMAL
    mode, dashed orange chips for the true hand in TEACHING mode;
  - the cards it has played (blue chips: public);
  - for the two hidden seats, a summary of what the actor believes about that hand.

  The acting seat has a green border.
- **The public table** shows:
  - who acts and how many legal moves it has;
  - the move to beat and the rule for beating it (or a free lead);
  - bombs so far and the stake 2^bombs;
  - the bottom cards still in the landlord's hand;
  - the unseen-card counts from the actor's point of view (its own knowledge, not public);
  - the recent moves.
- **Belief input table.** It has one column per card kind (3 … A, 2, BJ = black joker, RJ = red joker).
  For each hidden hand it shows the unseen copies (the actor's view), then:
  - the card-counting reference;
  - the arm's belief;
  - the true count (TEACHING only).

  Each cell is the expected number of copies that hand holds; hover for the full distribution.
  TEACHING mode adds the cross-entropy against the truth, which is Stage 4's metric for one decision,
  for the belief and two baselines: card counting and uniform over the feasible counts.
- **Choice table.** The top 5 legal moves by Q. Q is the network's estimate of this seat's final
  return, in training units of ±2^bombs. The move played is highlighted, and listed below the table
  if it is not in the top 5. The exploration line shows the random draw: draw < ε means a uniformly
  random legal move, otherwise the greedy (highest-Q) move.
- **Transition line.** What the move did (beat, lead or pass; cards left before and after; whether a
  bomb doubled the stakes) and who acts next.
- **Game over.** The winner, the bombs and the return G of each seat. Every decision a seat made is
  trained toward that seat's G (a Monte Carlo target). Evaluation reports ADP in other units (±2·2^bombs).

## 6. How to read the state comparison

The page rebuilds one decision exactly: the same deal, then the recorded moves. It then shows:
1. The table, as in the replay. It shows the acting seat's hand, the other seats' counts (and their
   true hands in TEACHING mode), and the public history.
2. **Slot 1 and slot 2 tables.** One row per arm for the same state, beside card counting and the
   truth. This shows directly what information each arm has:
   - M0c has zeros;
   - M1 has the next-hand prediction and a zero second slot;
   - M2c has the same next-hand row as M1 (one shared predictor) plus its conservation reflection;
   - M2 has two learned rows of its own;
   - Oracle has the truth (masked in NORMAL mode).
3. **The Q grid.** Ranks run down and arms run across, with each arm's greedy choice and, in
   TEACHING mode, each arm's CE.

What the sample shows, and why it matters:
- **All five arms rank the moves identically.** Every arm starts from the same DouZero warm start,
  with zero belief weights, so the belief cannot change Q until RL training moves those weights.
  The page says so. Differences will appear only with trained checkpoints.
- **M2c's derived slot has exactly the same CE as its predicted slot** (0.72 / 0.72 in the sample).
  This is no accident. P(other holds k) = P(next holds u − k), so at the true counts the two slots
  give the same probabilities. The reflection adds no new information, only a different
  representation of the same information, which is the project's research question.
- **A surprise the teaching mode exposed: card counting is not a strong baseline late in the game.**
  An untrained predictor outputs a near-uniform distribution over the feasible counts. Over the
  sample game, averaged over both slots, its CE (0.730 nats) is *lower* than card counting's (0.816),
  and so is an exact uniform belief's. The numbers are the mean `cross_entropy` fields of
  `replay_M2c_seed1_teaching.json`:
  - all 31 decisions (62 slot-decisions): belief 0.730, card counting 0.816, uniform 0.726;
  - the 5 early decisions, with 26 or more unseen cards: card counting wins, 0.957 vs 1.006;
  - the 26 later decisions: card counting loses, 0.790 vs 0.672.

  A likely reason is that late-game hands are not random subsets of the unseen cards. Players keep
  pairs, triples and bombs and shed loose singles, so the true counts sit at the extremes (here the
  landlord holds all four aces), where the hypergeometric baseline puts little mass. This is a hint,
  not a result: one game, 5 early decisions, correlated samples, and a 26-card split chosen after
  looking. Even so, Stage 4 should report the uniform baseline next to card counting, and break CE
  down by game phase. Otherwise "beats card counting" may be a weaker claim than it sounds.

## 7. Training curves (designed now, built with the first real runs)

RL runs already write `metrics.csv` every 5 s: time, frames, fps, and per seat the update count,
loss, Q mean, Q std, gradient norm, clipping and mean episode return. Stage 4's predictor runs will
write their own `metrics.csv`: step or epoch, train CE, validation CE, card-counting CE,
CE_next and CE_other (K=2), expected-count MAE, learning rate and wall time.

The plan is one small script, `plot_run.py <run_dir>`. It reads a run's raw CSV and writes PNGs
plus a `plots/index.html` into `artifacts/runs/<run_id>/plots/`, keeping the CSV as the source of truth:
- **RL:**
  - Q-loss per seat;
  - Q mean ± std;
  - gradient norm with the clipping rate;
  - fps;
  - validation ADP and WP, once evaluation is wired;
  - all against frames.
- **Predictor:**
  - train and validation CE against step, with the card-counting CE as a horizontal reference;
  - CE_next and CE_other;
  - MAE;
  - learning rate.

It will run on the laptop only on small existing logs. Heavy training never runs just to make plots.

## 8. Outputs that a later dashboard can use

Each run folder will gain two optional subfolders, so that a summary page can link curves to example games:

```
artifacts/runs/<run_id>/
  config.json  attempts.jsonl  metrics.csv  summary.md  checkpoints/   (exist today)
  plots/        PNG curves + index.html, from metrics.csv                (planned, §7)
  replays/      <name>.json + <name>.html, sample games of this run       (planned)
  comparisons/  <name>.json + <name>.html, sample states                  (planned)
```

Records carry a versioned `schema` field (`cs230.episode_replay/1`, `cs230.state_comparison/1`)
and a `game_id`, so a dashboard can index them without re-running anything.

## 9. Limitations of the prototype

- **Untrained predictors.** No trained predictors and no trained RL checkpoints exist yet, so the
  beliefs are noise and the arms' Q rankings are identical. The tools already accept trained
  predictors, and `compare_state` accepts per-arm Q-networks. Loading a trained `model.tar` per
  arm from the command line is still to add.
- **Size.** A page holds every decision's tables, about 430 KB per game. That is fine for single
  games but not for thousands.
- **Only actor 0's first game of a CPU-actor run is replayed** (by seed). Replaying an arbitrary game
  from a run's evaluation, or a GPU actor's game, is not wired yet.
- **The comparison scores moves greedily** (ε = 0) to compare rankings. The replay shows the real
  ε-greedy choice.
- **No plots yet** (§7 is a design).
- **No live training dashboard,** by design for now.

## 10. Next steps, after approval

1. `plot_run.py` for RL and predictor `metrics.csv`, tried on the existing integration-test logs.
2. A `--arm_checkpoint ARM=path/model.tar` option for the comparison, used once trained arms exist.
3. Sample replays and comparisons written into each run folder at the end of a run, from a fixed
   set of seeds, so every run summary links to the same states.
