# DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning (ICML 2021)

> Daochen Zha, Jingru Xie, Wenye Ma, Sheng Zhang, Xiangru Lian, Xia Hu, Ji Liu  
> ICML 2021 (arXiv:2106.06135)  
> **Tier A** — The foundational paper on deep RL for DouDizhu; essential for this project  
> **One-line takeaway:** Classic Monte Carlo methods, enhanced with deep networks and action encoding, solve the massive, variable action space of DouDizhu better than DQN or policy gradients, achieving strongest AI performance to date (ranked #1 on Botzone leaderboard) in days on a single GPU server.

---

## 0. TL;DR for a CS230 Student

- **What it is:** DouZero is a self-play deep reinforcement learning system for DouDizhu (a popular three-player Chinese card game) built on Deep Monte Carlo (DMC), not DQN or policy gradients.
- **Why you care:** (1) Demonstrates that classic RL algorithms matter; DQN and policy gradients fail badly here. (2) Shows how to handle massive variable action spaces through action encoding. (3) Imperfect-information multi-agent learning with both cooperation and competition. (4) Production-scale training: beats the strongest prior AI (DeltaDou, trained 2+ months) in 10 days with commodity hardware.
- **The core idea:** Use Monte Carlo returns (no bootstrapping) instead of value-function bootstrapping (Q-learning), encode actions as card matrices so the network generalizes to unseen action combinations, and parallelize data generation across 45 actors on 1 learner to reduce variance and accelerate training.
- **Key wins to steal:** (1) DMC + action encoding as a simple alternative to DQN for large discrete action spaces. (2) Feature engineering for imperfect information (LSTM on move history, separate features per role). (3) Parallel actor-learner architecture with shared buffers. (4) Clear evaluation protocol (WP and ADP metrics, 10,000+ decks, Elo rating).

---

## 1. Paper Overview

### 1a. Beginner-friendly explanation

DouDizhu is a card game roughly like the three-player variant of solitaire: one player (the Landlord) plays against a team of two (the Peasants). Your goal is simple—play all your cards first—but the strategy is complex because you must both compete with your opponents and (if you're a Peasant) help your teammate. The game has a **massive action space**: 27,472 possible moves, and which moves are legal changes constantly based on what cards others have played and what cards remain in play.

**The challenge:** Standard deep RL methods fail spectacularly here:
- **DQN** breaks down because Q-values are wildly overestimated in huge action spaces, and the computational cost of computing the max over 27,000 actions is prohibitive.
- **Policy gradient methods** (A3C, PPO) cannot share knowledge across similar but different actions—if the agent sees "3-K-K-K" (a trio of kings with a 3 kicker), it learns nothing about "3-J-J-J" even though the two are structurally identical.

**DouZero's insight:** Use **Deep Monte Carlo** (DMC)—a forgotten classic that directly estimates true returns without bootstrapping. Combined with two key tricks:
1. **Action encoding:** Represent every card combination as a 4×15 matrix (rows = quantity of each rank, columns = ranks and jokers). This lets the network generalize: "if 3-K-K-K wins, 3-Q-Q-Q probably also works."
2. **Parallel self-play:** Generate data with 45 actors playing simultaneously while 1 learner updates the Q-network. This parallelization reduces Monte Carlo variance and makes wall-clock training fast.

**The results:** Starting from scratch, DouZero:
- Beats supervised learning (trained on 226,230 human games) in 1-2 days
- Beats DeltaDou (the previous strongest AI, trained for 2+ months with human heuristics) in 10 days
- Ranked #1 on the Botzone online leaderboard (344 bots)
- Runs orders of magnitude faster at inference than DeltaDou or CQN

### 1b. Technical explanation

**Problem:** DouDizhu is an imperfect-information three-player card game with (§2, p. 2-3):
- **State space:** ~10^83 information sets, average size ~10^23 (much larger than No-limit Texas Hold'em's ~10^4 average)
- **Action space:** 27,472 legal combinations (Table 3, p. 14), where legal moves vary by game state
- **Properties:** Sparse reward (win/lose only at game end), long horizons (~15-30 turns per player), both competitive (Landlord vs Peasants) and cooperative (Peasants must help each other)

**Why existing methods fail** (§1-2, p. 2):
- **DQN:** Overestimation bias worsens with huge action spaces (Zahavy et al., 2018). Computing $\max_a Q(s,a)$ over 27,000 actions on a deep network is slow. The max operation concentrates overoptimism.
- **Policy gradients (A3C, PPO):** Cannot leverage action features to generalize. If you train on "3-K-K-K" but never see "3-J-J-J", the policy learns them as completely independent. In a space of 27,472 actions, this is catastrophic.

**DouZero's solution: Deep Monte Carlo** (§3.1-3.3, p. 3-4):

Traditional Monte Carlo (MC) methods:
1. Generate an episode (full game) using policy π
2. For each state-action pair in the episode, update Q(s,a) with the **discounted return** (actual cumulative reward from that point to game end)
3. Update policy: π(s) ← argmax_a Q(s,a) with ε-greedy exploration

[Paper] DMC replaces the Q-table with a neural network and MSE loss. [Interpretation] This is value-based learning without bootstrapping: the target is always the true return $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$, not an estimate of the next state's value.

**Why DMC beats DQN here:**
1. **No overestimation bias:** MC directly estimates true returns; no max operation.
2. **Handles long, sparse rewards:** Q-learning struggles when value estimates far in the future are unreliable. MC doesn't care—it just averages the observed returns.
3. **Action features generalize:** DMC takes $(s,a)$ as input, so action features in the card matrix can transfer across unseen combinations.
4. **Highly parallelizable:** MC is sample-inefficient (high variance) but parallelization (many actors) generates thousands of samples/second, masking the variance problem.

[Interpretation] The trade-off is that DMC requires more samples (higher variance), but with 45 parallel actors, DouZero generates so much data that wall-clock training is actually faster than Q-learning would be.

**Architecture & encoding** (§4.1, p. 4; Appendix C.1, p. 16):
- **State representation:** Multiple 4×15 card matrices (hand cards, union of others' cards, recent moves) + one-hot vectors (hand size of each player, bomb count) + LSTM-encoded history of last 15 moves. Total size differs by role (Tables 4-5, p. 16).
- **Action representation:** Single 4×15 card matrix, flattened and concatenated with state.
- **Network:** LSTM layer → 6 layers of MLP (hidden size 512) → scalar Q-value output. This is Q(s,a), not separate heads per action.

**Three separate networks:** One each for Landlord (L), player-before-L (U), and player-after-L (D), trained independently.

**Training** (§4.2, Algorithm 1-2, p. 5):
- **45 actors** generate episodes in parallel, compute cumulative returns, and store (s, a, return) tuples in shared buffers.
- **1 learner** samples batches from the buffers, computes MSE loss: $\mathcal{L} = \|Q(s,a) - \text{return}\|^2$, and updates all three global Q-networks.
- **Synchronization:** Actors periodically sync their local Q-networks with the global networks to stay up-to-date.

**Reward definition:**
- **WP (Winning Percentage):** +1 if agent wins game, -1 if loses.
- **ADP (Average Difference in Points):** Actual point differential (base=1, bombs double). [Paper] The authors observe that WP and ADP encourage different strategies: WP agents play bombs aggressively (it doesn't hurt WP if you lose a bomb round), while ADP agents are conservative (bombs risk doubling the loss, p. 8).

---

## 2. RL Formulation

| Component | Value/Definition | Citation |
|-----------|-----------------|----------|
| **Agents** | 3 independent agents (Landlord, Peasant-Up, Peasant-Down), but only Landlord learns exploitatively; Peasants must cooperate without explicit communication. | [Paper] §2, p. 3 |
| **Environment** | DouDizhu card game simulator; deterministic given random card shuffle at start. | [Paper] §2, p. 2 |
| **State (fully observable)** | Player's own hand cards, inferred info about others' hand sizes, move history. | [Paper] §4.1, p. 4; Appendix C.1, p. 16 |
| **Observation (what agent sees)** | Own hand cards (face-up), played cards of all players (public history), opponents' hand card counts (public), most recent move (public). Hidden: exact composition of opponents' hands. | [Paper] §A.1, p. 13; Figure 10, p. 9 |
| **Action** | Any legal combination: Solo, Pair, Trio, Bomb, Chain, Plane, Quad, Rocket, Pass. 27,472 possible, but varies per state. | [Paper] §2, Table 3, p. 14-15 |
| **Legal action handling** | Actors only sample from legal moves (ε-greedy: choose best Q-value with prob 1-ε, random legal move with prob ε). Network output is Q(s,a) for any (s,a) pair; only evaluated on legal actions. | [Paper] Algorithm 1, lines 6-7, p. 5; §4.1, p. 4 |
| **Reward** | Two variants: (a) WP: +1 for win, -1 for loss (sparse, only at game end); (b) ADP: point differential (sparse at game end). | [Paper] §5.1, p. 6 |
| **Return** | $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$ where $\gamma = 1$ (no discounting; early moves are critical). Computed backward from end of episode. | [Paper] Algorithm 1, lines 11-12, p. 5; §5.1 |
| **Policy** | $\pi(a\|s) = \text{argmax}_a Q(s,a)$ during play; $\epsilon$-greedy during training ($\epsilon=0.01$). | [Paper] Algorithm 1, p. 5; §5.1 |
| **Value function** | $Q(s,a)$ learned via deep neural network (no separate state-value); no target network. | [Paper] §3.1, §4.1, p. 4 |
| **Fully vs partially observable** | **Partially observable** (imperfect information). Agents know their own hand and move history; must infer opponents' hands. | [Paper] §2, p. 2; Appendix A.2, p. 14 |
| **Single vs multi-agent** | **Multi-agent** (3 agents). Landlord is adversary; Peasants are teammates. No explicit communication. | [Paper] §2, p. 2-3 |
| **Cooperative/competitive** | **Both:** Landlord vs Peasants (competitive), Peasant-Up vs Peasant-Down (cooperative). | [Paper] §2, p. 3 |
| **Self-play used?** | **Yes.** All three agents learn simultaneously. No fixed opponent pool; agents play fresh games against themselves. | [Paper] §4.2, p. 4-5; Experiments §5, p. 5-6 |

---

## 3. Model / Architecture

**Card representation** (§4.1, p. 4; Figure 2, p. 3; Appendix C.1, p. 16):

Cards are encoded as a **4 × 15 one-hot matrix**, where:
- **Rows** (4): Number of cards of a rank (0, 1, 2, 3+, since most ranks have max 4 copies).
- **Columns** (15): Ranks (3, 4, 5, ..., A, 2, Red Joker, Black Joker).
- **Why:** Captures both the rank and the count in a single tensor; suits are irrelevant.

For a concrete example (Figure 2, p. 3): A hand of {3, 3, 5, 5, 5, K, K} is encoded as a matrix with 1's in the row/column positions corresponding to these cards.

**State representation per role** (Tables 4-5, Appendix C.1, p. 16):

*Landlord (L) state features:*
- Card matrix of the action (54 dims after flattening and removing always-zero entries)
- Card matrix of hand cards (54)
- Card matrix of union of Peasants' hand cards (54)
- Card matrix of most recent move (54)
- Card matrices of Peasant-Up and Peasant-Down played cards (2 × 54)
- One-hot vectors: Peasant-Up hand size (17 dims), Peasant-Down hand size (17 dims)
- One-hot vector: number of bombs played (15 dims)
- **Concatenated matrix of last 15 moves:** 5 × 162 dims (15 moves × 3 players × 54 dims per move, but more precisely: every 3 consecutive moves → concatenate into a 162-dim vector; 5 such concatenations)

Total state size: 54 + 54 + 54 + 54 + 54 + 54 + 17 + 17 + 15 + 810 (after LSTM) ≈ large.

*Peasant state features:* Similar but omits symmetry; includes recent move by Landlord and other Peasant separately.

**Network architecture** (§4.1, p. 4; Figure 3, p. 4; Appendix C.1, p. 16):

```
Input: (concatenated state features) ++ (action matrix flattened)
  ↓
LSTM layer (encode last 15 moves) → hidden state h
  ↓
Concatenate: h ++ (non-LSTM state features) ++ (action features)
  ↓
6 layers of MLP, each with hidden size 512, ReLU activation
  ↓
Output: Q(s,a) (scalar)
```

**Key design choices:**
- **Why LSTM?** Historical moves are critical (e.g., tracking which cards have been played). LSTM captures dependencies over move history.
- **Why 6 MLP layers?** No ablation is provided, but standard practice for approximating complex value functions.
- **Why Q(s,a) input, not Q(s) with |A| outputs?** Action encoding (the card matrix) allows generalization; feeding the action as input lets the network reason about card combinations. [Interpretation] This is crucial for the large action space.
- **No separate networks per action type?** [Paper] The paper does not specify this; the network learns end-to-end. (§4.1, p. 4)
- **Three separate networks (L, U, D)?** Yes (§4.2, p. 4-5). Each role sees different state features and has different strategic goals, so separate networks are justified. [Interpretation]

**Activation function:** ReLU (§5.1, p. 6).  
**Optimizer:** RMSprop, learning rate ψ = 0.0001, smoothing constant = 0.99, ε = 10^-5 (§5.1, p. 6).

**Historical move encoding detail:**
- Most recent 15 moves are encoded.
- Concatenate every 3 consecutive moves → 5 groups of 162-dim vectors.
- Feed into LSTM; use the last hidden state as the summary of move history.
- If fewer than 15 historical moves, pad with zeros (§C.1, p. 16).

[Interpretation] This is a clever way to linearize the move sequence while preserving temporal structure. The LSTM learns to compress (move_1, move_2, move_3), (move_4, move_5, move_6), etc. into a fixed-size hidden state.

---

## 4. Training Algorithm

**Step-by-step pipeline:**

1. **Initialization:**
   - Three global Q-networks (QL, QU, QD) in the learner process.
   - 45 actor processes, each with three local Q-networks (copies of the global ones).
   - Shared buffers BL, BU, BD (one per role), each with B=50 entries of size S=100.

2. **Actor process (Algorithm 1, p. 5):**
   ```
   repeat until convergence:
     - Synchronize local Q-nets with global Q-nets
     - for t = 1 to T (e.g., T = one full game):
       * Select position (L, U, or D) based on game state
       * ε-greedy action selection:
         - With prob (1-ε): a_t = argmax_a Q(s_t, a), legal a only
         - With prob ε: a_t = random legal action
       * Execute action, observe s_{t+1}, r_t
       * Store (s_t, a_t, r_t) in local buffer D_p (p = position)
     - Backward pass to compute cumulative rewards:
       - for t = T-1 down to 1:
         * r_t ← r_t + γ * r_{t+1}  (γ = 1, so r_t = sum of all future rewards)
     - Write data to shared buffers in batches of size L (if buffer not full, wait)
   ```

3. **Learner process (Algorithm 2, p. 5):**
   ```
   repeat until convergence:
     - for each position p in {L, U, D}:
       * if shared buffer B_p has ≥ M full entries (M = batch size = 32):
         - Sample a batch of M×S instances (M=32 entries × 100 samples each)
         - Compute loss: L = MSE(Q(s,a), cumulative_return)
         - Update Q_p^g with gradient ∇L, learning rate ψ
         - Free the sampled entries from the buffer
   ```

4. **Data flow:**
   - Actors generate episodes (games) in parallel.
   - At the end of each game, cumulative returns are computed and pushed to shared buffers.
   - Learner consumes buffers asynchronously, updating global Q-networks.
   - Actors periodically pull updated global Q-networks.

5. **Training details** (§5.1, p. 6):
   - **Hardware:** 1 server with 48 Intel Xeon cores, 4 NVIDIA 1080 Ti GPUs.
   - **Parallelism:** 45 actor processes on 3 GPUs, 1 learner process on 1 GPU.
   - **Exploration:** ε = 0.01 (1% random moves during training).
   - **Discount factor:** γ = 1 (no discounting; early moves are as important as late moves because the game is sparse-reward).
   - **Buffer management:** B = 50 entries/buffer, S = 100 samples/entry, M = 32 entries/batch.
   - **Training duration:** 30 days (although results shown up to day 10 because performance plateaus).

6. **Episode termination:** An episode ends when one player has no cards left (Landlord wins or one Peasant wins).

---

## 5. Mathematics

### Equation 1: Monte Carlo Return (MUST UNDERSTAND)
$$G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$$

**Symbols:**
- $G_t$: cumulative discounted reward from timestep $t$ to episode end
- $\gamma$: discount factor (here, $\gamma = 1$, so no discounting)
- $r_k$: immediate reward at timestep $k$
- $T$: episode length (end of game)

**Intuition:** Unlike Q-learning, which bootstraps (estimates future value based on the next state's Q-value), Monte Carlo directly sums all realized rewards from this point to the end. This is the **true return** for this trajectory.

**Why chosen:** In DouDizhu, the only reward is at game end (sparse reward). Bootstrapping introduces error because the agent hasn't yet observed future returns. MC sidesteps this by waiting until the episode ends, then using the true sum.

**Relation to standard RL:** Q-learning is $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ (temporal-difference/bootstrapping). MC is $Q(s,a) \leftarrow Q(s,a) + \alpha [G_t - Q(s,a)]$ (direct return).

---

### Equation 2: MSE Loss (MUST UNDERSTAND)
$$\mathcal{L} = \|Q_\phi(s,a) - G_t\|_2^2$$

**Symbols:**
- $Q_\phi(s,a)$: neural network output (parameterized by $\phi$), a scalar
- $G_t$: Monte Carlo return (target, treated as fixed/non-differentiable)
- $\|\cdot\|_2^2$: squared L2 norm (mean-squared error)

**Intuition:** Minimize the prediction error: make $Q_\phi(s,a)$ close to the true return $G_t$. This is standard supervised regression.

**Why chosen:** Simple, stable, and standard for value function approximation. No min/max operations means no overestimation bias.

**Relation to standard RL:** DQN uses $\mathcal{L} = \|Q_\phi(s,a) - [r + \gamma \max_{a'} Q_{\phi'}(s',a')]\|_2^2$ (bootstrapping with target network). DouZero skips the max and bootstrapping entirely.

---

### Equation 3: ε-Greedy Policy (MUST UNDERSTAND)
$$a_t = \begin{cases} \arg\max_a Q(s_t, a) & \text{with prob } (1-\epsilon) \\ \text{random legal action} & \text{with prob } \epsilon \end{cases}$$

**Symbols:**
- $a_t$: action at timestep $t$
- $\epsilon$: exploration rate (here, $\epsilon = 0.01 = 1\%$)

**Intuition:** Mostly exploit (pick the best-looking action), but occasionally explore (pick a random legal move) to discover new strategies and reduce overconfidence.

**Why chosen:** Simple and effective. ε-greedy is standard in value-based RL; it guarantees that with probability → 1, each state-action pair is visited infinitely often (under certain conditions), ensuring convergence.

**Hyperparameter choice:** ε = 0.01 is quite low. [Interpretation] This suggests the learned Q-network is already fairly reliable by training time, so heavy exploitation is justified.

---

### Equation 4: Policy Improvement (MUST UNDERSTAND)
$$\pi(s) \leftarrow \arg\max_a Q(s,a)$$

**Symbols:**
- $\pi(s)$: policy at state $s$ (deterministic in this case)
- $Q(s,a)$: estimated action value

**Intuition:** At each state, pick the action with the highest estimated Q-value. This is a greedy policy; combined with ε-greedy exploration, it balances exploitation and exploration.

**Why chosen:** Optimal (if Q-values are accurate). Standard policy improvement step in value-based RL.

---

### Equation 5: Cumulative Return Calculation (Algorithm 1, MUST UNDERSTAND)
$$r_t \leftarrow r_t + \gamma \cdot r_{t+1} \quad \text{for } t = T-1, T-2, \ldots, 1$$

**Symbols:**
- $r_t$: immediate reward at timestep $t$ (initially), then cumulative return (after loop)
- $\gamma = 1$: discount factor

**Intuition:** Working backward from the end of the episode, accumulate rewards. At timestep $t$, $r_t$ becomes the sum of all rewards from $t$ to $T$.

**Why chosen:** Efficient way to compute returns in a single backward pass. No need to recompute for every state-action pair.

---

### Equation 6: Winning Percentage (WP) Metric (OPTIONAL / ADVANCED)
$$\text{WP}(A, B) = \frac{\text{# games won by } A}{\text{# games played}}$$

**Symbols:**
- $A, B$: two competing agents
- "Games won": A's team (Landlord or Peasant) has no cards left

**Intuition:** Fraction of games won. Higher is better. Used with mirroring: A plays as Landlord in half the games and Peasant in half, to account for role imbalance.

**Relation to standard RL:** Not a technical RL concept; it's a tournament metric. Similar to Elo rating in chess.

---

### Equation 7: Average Difference in Points (ADP) Metric (OPTIONAL / ADVANCED)
$$\text{ADP}(A, B) = \mathbb{E}[\text{points for } A - \text{points for } B]$$

**Symbols:**
- Points: base 1 per game; bombs (four-of-a-kind) double the score.

**Intuition:** Average point difference per game. Positive means A outscores B on average.

**Why multiple metrics?** [Paper] WP and ADP encourage different strategies. WP agents play bombs aggressively (a bomb doesn't hurt WP if you lose); ADP agents are cautious (bombs double the loss if you lose). (§5.1, p. 6; Appendix D.2, p. 19)

---

## 6. Loss Function

**Loss used:** Mean Squared Error (MSE)
$$\mathcal{L}(\phi) = \frac{1}{N} \sum_{i=1}^{N} (Q_\phi(s_i, a_i) - G_i)^2$$

**Target:** Monte Carlo return $G_i$ (computed at end of episode, offline).

**Where target comes from:** At the end of each episode, cumulative rewards are backpropagated (Algorithm 1, lines 11-13, p. 5). No bootstrap; the target is always the true observed return.

**Bootstrapping?** **No.** This is the key difference from Q-learning. Monte Carlo is non-bootstrapping: the target is the realized return, not an estimate.

**Monte Carlo?** **Yes, fully.** Every state-action pair's Q-value is trained toward the empirical return of that trajectory.

**Target network?** **No.** [Paper] The paper does not mention a target network (p. 4-5). [Interpretation] DMC typically doesn't use a target network because it's not bootstrapping; the target is fixed (the observed return) and doesn't depend on the current Q-network. This differs fundamentally from DQN, which uses a target network to stabilize bootstrapping.

**Policy loss?** None. This is a value-based method; no explicit policy loss.

**Value loss?** The MSE loss itself is the value loss.

**Auxiliary losses?** None mentioned in the main paper (§5.1, p. 6; Appendix C.1, p. 16).

**Multiple losses?** Two training objectives are used (WP vs ADP, §5.1, p. 6), but these are trained in separate runs, not jointly. Different reward signals → different loss targets.

**Why these choices:**
1. **MSE is simple and stable.** No max operations → no overestimation bias.
2. **No bootstrapping in sparse reward.** Long-horizon sparse-reward problems can suffer from bootstrapping error; MC sidesteps this.
3. **Parallelization reduces variance.** MC typically requires more samples, but 45 parallel actors generate massive batches (45 games × ~25 steps/game = ~1,000+ (s,a,return) tuples per batch), so variance is managed.

---

## 7. Self-Play

**Who plays whom:** All three agents (L, U, D) are trained simultaneously in self-play. In each game, three fresh instances of the learned Q-networks play against each other.

**Simultaneous updates?** [Paper] Yes (§4.2, p. 4-5). The learner updates global Q-networks QL, QU, QD in parallel using data from all three roles. No explicit sequential alternation.

**Parameter sharing?** No. Each role has its own Q-network. [Interpretation] This makes sense because each role has different reward signals and partially observable information.

**Separate policies?** Yes, three separate policies (one per role).

**Old checkpoints?** [Paper] The paper does not explicitly mention an opponent pool or checkpointing strategy (p. 4-5). [Interpretation] All agents play against the *current* learned networks, not older versions. This differs from some self-play approaches (e.g., AlphaZero uses a replay pool of previous opponents to stabilize training).

**Exploration:** ε-greedy during training (ε = 0.01); deterministic argmax during evaluation (§5.1, p. 6).

**Non-stationarity and how it is handled:**
- **Non-stationarity:** All three agents are learning simultaneously, so the "opponent" (the other players) is non-stationary. This is a fundamental challenge in multi-agent RL.
- **How handled:** [Paper] The paper does not explicitly address non-stationarity (p. 4-5). [Interpretation] Parallelization helps mitigate it: with 45 actors, each episode uses a different draw from the current network distribution, and the learner aggregates statistics across many games, averaging out transient swings.

**Cooperation between Peasants:** [Paper] Figure 10 (p. 9) shows a case study where Peasants learn to cooperate without explicit incentive. The authors observe that the two Peasants' Q-values align on small solos to help the teammate go out. [Interpretation] This emergence arises because both Peasants share the same reward signal (team win/loss); gradient updates reinforce actions that lead to team victory, even if they disadvantage one Peasant temporarily.

---

## 8. Imperfect Information

**Hidden information in DouDizhu:**
- Each player sees their own hand (17 cards) and all played cards (public history).
- Each player does **not** see opponents' hands (17 cards each).
- Each player infers hand composition from:
  - Bidding phase (signals about hand strength, though bidding is pre-trained and fixed in RL).
  - Played cards (cards you see rule out those from opponents' hands).
  - Remaining deck size.

**How DouZero handles hidden information** (§4.1, Appendix C.1, p. 16):

1. **Explicit card tracking (state representation):**
   - Own hand cards (observed, one card matrix)
   - Union of other two players' hands (unobserved, but size is known; represented as "union" because exact split is unknown)
   - Played cards of each opponent separately (observed via move history; one card matrix per opponent)
   - Hand size of each opponent (observed; one-hot vector: e.g., "Peasant-Up has 7 cards left")

2. **LSTM-encoded move history:**
   - Last 15 moves (who played what) are encoded by LSTM.
   - This captures temporal patterns (e.g., "opponent just played a bomb → likely has more bombs" or "opponent passed → likely doesn't have higher cards").

3. **No explicit belief-state inference:**
   [Paper] The paper does not mention Bayesian updating or explicit belief states (p. 4-5). [Interpretation] DouZero implicitly models beliefs through the LSTM and move history; the network learns to infer hand composition from patterns of play.

**Comparison to explicit approaches (e.g., DeltaDou):**
- DeltaDou (prior work, §2, p. 2) uses **Bayesian inference** to explicitly compute a posterior distribution over opponents' hands.
- DouZero uses **implicit inference** via the LSTM: the network learns end-to-end, without explicit belief states.
- [Interpretation] The implicit approach is simpler and faster; the trade-off is that the network must learn to extract and use belief-relevant information from the history, which the LSTM does via learned weights.

**Recurrent model (LSTM):** Yes, last 15 moves are encoded via LSTM (§4.1, p. 4; Figure 3, p. 4). [Interpretation] This is the primary mechanism for handling temporal dependencies and implicit belief updating.

**Card inference:** Implicit, via LSTM. The network learns patterns like "opponent passed when [state], so they probably don't have [card type]."

**Explicit distributions over hidden cards?** No. [Paper] The paper does not construct or output probability distributions over opponents' cards (p. 4-5). [Interpretation] The network uses the move history and played cards to implicitly condition on likely hand compositions, but doesn't factor a belief state.

**Two-column table: Actually observed vs Hidden**

| Actually Observed | Hidden |
|---|---|
| Own 17 hand cards | Exact composition of Peasant-Up's 17 cards |
| All 54 cards exist (identity known) | Exact composition of Peasant-Down's 17 cards |
| Played cards of all players (move history) | Whether opponent has specific card (until played) |
| Hand size of each opponent (count of cards left) | Opponent's "hand strength" or "card distribution" (inferred via bidding, but bidding is pre-trained SL, not RL) |
| Bidding outcome (who won the bid → who is Landlord) | |

[Interpretation] DouZero doesn't try to recover the hidden information; instead, it learns to make good decisions under uncertainty by conditioning on the observed history and the public structure of the game.

---

## 9. Evaluation

### Baselines

[Paper] The paper evaluates against (§5.1-5.2, p. 5-6):

1. **DeltaDou** (Jiang et al., 2019): Prior state-of-the-art. Uses Bayesian inference and MCTS. Trained for 2+ months, initialized with heuristic pre-training.
2. **CQN** (Combinational Q-Learning, You et al., 2019): Card decomposition + Deep Q-Learning. Breaks action space into subcategories. Open-sourced.
3. **SL (Supervised Learning):** Internal baseline trained on 226,230 human expert games (49,990,075 total samples) using the same state representation and network architecture as DouZero. Achieves 84% accuracy on held-out data.
4. **RHCP, RHCP-v2:** Open-sourced rule-based heuristic agents.
5. **RLCard Rule Model:** Rule-based baseline from the RLCard toolkit (Zha et al., 2019a).
6. **Random:** Uniformly samples legal moves.

### Evaluation Protocol

**Tournament structure** (§5.1, p. 5-6):
- For each pair of algorithms A and B, play 10,000 randomly sampled decks.
- **Mirroring:** To control for role imbalance (Landlord is advantaged), each deck is played twice:
  - Game 1: A=Landlord, B=Peasants
  - Game 2: A=Peasants, B=Landlord (swapped)
  - Metrics are averaged over both games.
- **Elo Rating for top 3:** To get more reliable estimates, the top 3 algorithms (DouZero, DeltaDou, SL) play 100,000 decks across 5 random seeds, and Elo ratings are computed.

**Metrics:**

1. **WP (Winning Percentage):** Games won by A / total games. [Paper] WP > 0.5 means A outperforms B (§5.1, p. 6).
2. **ADP (Average Difference in Points):** Mean(points_A - points_B) per game. Base=1 point/game; bombs double. ADP > 0 means A outperforms B.

**Results** (§5.2, Table 1, p. 6-7):
- DouZero WP vs DeltaDou: 0.586 (58.6% win rate) and ADP: 0.258 points/game.
- DouZero WP vs SL: 0.659 and ADP: 0.700.
- DouZero WP vs CQN: 0.810 and ADP: 1.685.
- DouZero WP vs RHCP-v2: 0.757 and ADP: 1.662.
- DouZero WP vs Random: 0.989 and ADP: 3.036.

[Interpretation] WP of 0.586 vs DeltaDou is a modest-looking margin, but given the high variance in card-game outcomes (luck of the deal), it's substantial. DouZero also achieves faster wall-clock training.

**Decks and roles:**
- Decks generated: Randomly shuffled 54-card deck.
- Roles: Each algorithm plays both Landlord and Peasants (mirrored).
- Number of games: 10,000 decks per pair (RQ1), 100,000 decks for Elo rating (top 3 algorithms).

**Botzone Leaderboard:**
- DouZero ranked #1 out of 344 bots on the Botzone online platform.
- Elo rating: 1625.11 (October 30, 2020).

**Bidding phase (§5.3, Table 2, p. 7):**
- A pre-trained supervised bidding network (Appendix C.3, p. 18) is used for all algorithms for fairness.
- DouZero still dominates (WP: 0.580 vs DeltaDou, 0.461 vs SL).

**Statistical rigor:**
- **Seeds:** 5 random seeds for Elo rating experiments.
- **Confidence intervals:** Means and standard deviations reported for Elo (Figure 4, p. 7).
- **Sample size:** 10,000 decks is standard for this domain; 100,000 for top-3 comparison.
- **Missing:** No formal significance tests (e.g., t-tests, p-values). [Interpretation] Given the large sample sizes, differences are likely statistically significant, but the paper does not report formal tests.

### Critical Evaluation

**Strengths:**

1. **Comprehensive baselines:** Includes rule-based, supervised learning, prior SOTA (DeltaDou), alternative RL methods (CQN).
2. **Fair comparison:** Mirrored role evaluation, same network architecture between DouZero and SL.
3. **Large-scale evaluation:** 10,000+ games per pair; Elo rating on 100,000 games.
4. **Online validation:** Botzone leaderboard provides independent third-party validation.
5. **Metric diversity:** WP and ADP capture different aspects of strategy (risk-taking, conservatism).

**Weaknesses:**

1. **No statistical tests:** No reported p-values, confidence intervals, or t-tests to quantify certainty of superiority.
2. **Single-seed training for main results:** Figure 5 (learning curves) shows a single training run, though top-3 Elo uses 5 seeds. [Interpretation] Variance in training trajectories could be large; a single run may not be representative.
3. **Unfair baseline comparisons?** DeltaDou is initialized with supervised learning (heuristic pre-training); DouZero starts from scratch. [Paper] The paper notes this (§2, p. 2) but does this give SL an unfair advantage? Not clear. [Interpretation] Starting from scratch is harder and more impressive, but also means DouZero learns exploratory actions humans wouldn't play early on.
4. **Limited ablation on bidding:** Bidding phase (Section 5.3, p. 7) uses a supervised network trained offline, not learned end-to-end with card-playing. This is acknowledged as future work (§7, p. 9).
5. **Cooperation analysis (Section 5.8, p. 9) is anecdotal:** Only a few hand-picked case studies are shown. No systematic analysis of whether Peasants truly coordinate (e.g., measuring how often they pass to help each other).

### What I Would Improve for a CS230 Project

1. **Add ablations:**
   - DMC vs. TD (SARSA) vs. Actor-Critic (done in Section 5.5, p. 8, but only one run each; repeat with seeds).
   - Effect of LSTM depth, MLP layers, hidden size.
   - Effect of number of actors (done, Figure 6, p. 7, showing 15/30/45 actors).
   - Effect of ε (exploration rate).

2. **Formal evaluation:**
   - Compute 95% confidence intervals for WP and ADP via bootstrapping (resample decks).
   - Significance tests (Wilcoxon signed-rank test for pairwise comparisons).

3. **Cooperation analysis:**
   - Quantify: fraction of games where a Peasant passes when they could have played (to let teammate play).
   - Count: how often Peasant plays a small solo to "sacrifice" their high cards for the teammate's win.
   - Compare: Peasant cooperation in trained DouZero vs. SL baseline vs. rule-based agents.

4. **Scalability study:**
   - How do results change with fewer GPUs? Can you train on a single GPU?
   - What's the sample efficiency? (Games/wall-clock-time vs. final performance.)

5. **Generalization:**
   - Train on one RL objective (WP), test on another metric (ADP). How much does strategy transfer?
   - Train on "unbalanced" decks (skewed card distribution), evaluate on balanced decks and vice versa.

---

## 10. Experiments and Ablations

### Experiment 1: Head-to-Head Comparison with Baselines (RQ1)

**What changed:** DouZero's Q-networks (trained with DMC, action encoding, parallel actors) vs. baselines (DeltaDou, CQN, SL, rule-based, random).

**What held constant:** Same state representation and network architecture between DouZero and SL; fair role mirroring for all pairs.

**Metric:** WP, ADP (Table 1, p. 6-7; Appendix D.1, p. 22).

**Results:**
- DouZero beats all baselines in both WP and ADP.
- vs. DeltaDou: WP = 0.586, ADP = 0.258 (statistically significant gap).
- vs. CQN: WP = 0.810, ADP = 1.685 (large gap).
- vs. SL: WP = 0.659, ADP = 0.700 (shows RL beats supervised learning despite starting from scratch).

**Authors' conclusion:** [Paper] DMC + action encoding is more effective than DQN (CQN) or supervised learning in DouDizhu (§5.2, p. 6-7).

**Critical note:** WP differences are modest (58.6% vs 50%); high variance in card games makes small win-rate differences meaningful but not overwhelming.

---

### Experiment 2: Bidding Phase (RQ2)

**What changed:** Added pre-trained bidding network (trained on human expert data via supervised learning, Appendix C.3, p. 18) to DouZero, DeltaDou, SL.

**What held constant:** Card-playing networks unchanged; only bidding layer added.

**Metric:** WP, ADP (Table 2, p. 7).

**Results:**
- DouZero WP vs DeltaDou: 0.580 (slight decrease from 0.586 without bidding).
- DouZero WP vs SL: 0.461 (change from 0.659; SL improves more with bidding?).
- [Interpretation] The bidding network introduces a new source of variance; results are still DouZero > others, but the margin is tighter.

**Authors' conclusion:** DouZero is effective even with the added complexity of bidding.

**Critical note:** Bidding network is not learned end-to-end with card-playing (it's pre-trained SL). Future work should integrate bidding into the RL loop (acknowledged, §7, p. 9).

---

### Experiment 3: Learning Progress Over Time (RQ3)

**What changed:** Training time (0 to 30 days).

**What held constant:** Hyperparameters (ε, γ, network architecture), hardware.

**Metric:** WP, ADP vs. SL and DeltaDou as opponents (Figure 5, p. 7).

**Results:**
- vs. SL: DouZero exceeds WP threshold (0.5) in ~1 day, ADP threshold (0) in ~2 days.
- vs. DeltaDou: DouZero exceeds thresholds in ~3 days (WP) and ~10 days (ADP).
- Curves flatten after 10 days, suggesting convergence or diminishing returns.

**Authors' conclusion:** DouZero is extremely sample-efficient; it matches human-expert SL in days (despite starting from scratch) and beats 2-month-trained DeltaDou in 10 days.

**Critical note:** Single training run (no error bars). Curves show high smoothness, suggesting averaging over many games or rolling window, but uncertainty is not quantified.

---

### Experiment 4: Comparison with SARSA and Actor-Critic (RQ4)

**What changed:** Algorithm variant:
- SARSA: Replace Monte Carlo return with TD target $r + \gamma \max_{a'} Q(s',a')$ (one-step lookahead).
- Actor-Critic: Separate policy network (actor) with action mask + Q-network (critic) with action features.

**What held constant:** Network architecture, hyperparameters.

**Metric:** WP, learning curve (Figure 7, p. 8).

**Results:**
- DMC learns slightly faster than SARSA (wall-clock and samples).
- [Paper] Actor-Critic fails (§5.5, p. 8). [Interpretation] The nature of the failure (divergence, poor performance, etc.) is not detailed in the paper. Authors attribute this to difficulty incorporating action features into actor-critic framework.

**Authors' conclusion:** DMC is superior to TD-based learning (SARSA) in this domain, and naive actor-critic doesn't work. [Interpretation] SARSA's bootstrap error is harmful; actor-critic's classifier-like actor cannot leverage action feature generalization.

**Critical note:** Single run, no error bars. Actor-Critic's failure is dramatic but unexplained; a deeper analysis (e.g., gradient norms, value divergence) would help.

---

### Experiment 5: Learned Strategies vs. Human Expert Data (RQ5)

**What changed:** Training time (0 to 30 days).

**What held constant:** Human expert dataset (226,230 games, fixed).

**Metric:** Accuracy on human data (Figure 8, p. 8).

**Results:**
- First 5 days: accuracy increases from 0% to ~45% (Landlord role shown), suggesting DouZero learns strategies aligned with human play.
- After 5 days: accuracy drops (to ~40% at day 30), suggesting DouZero discovers novel strategies beyond human knowledge.
- ADP vs. SL still improves after day 5 (Figure 5d, p. 7), confirming that divergence from human play is effective.

**Authors' conclusion:** [Paper] DouZero initially aligns with human expert strategies (first 5 days), then discovers novel strategies beyond human knowledge (after 5 days; still improving in game performance).

**Critical note:** Anecdotal insight into learning dynamics; no case studies shown for the "novel strategies" (though Figure 10, p. 9, shows one cooperation example). A systematic analysis (e.g., action types that diverge) would strengthen this.

---

### Experiment 6: Inference Speed (RQ6)

**What changed:** Algorithm (DouZero, DeltaDou, CQN, RHCP, RHCP-v2).

**What held constant:** Hardware (same CPU for all), fair comparison.

**Metric:** Average inference time per move (seconds, Figure 9, p. 8).

**Results:**
- DouZero: ~10^-2 seconds per move.
- DeltaDou: ~10^-1 seconds (10× slower).
- CQN, RHCP: 10^0 - 10^-1 seconds (100-10× slower).

**Authors' conclusion:** DouZero is orders of magnitude faster at inference, enabling real-world deployment.

**Critical note:** Inference speed is secondary to sample efficiency and final performance, but it matters for deployment and real-time environments.

---

### Experiment 7: Peasant Cooperation (RQ7)

**What changed:** Game scenario (shown in Figure 10, p. 9).

**What held constant:** Trained DouZero networks.

**Metric:** Qualitative case study (predicted Q-values for top-3 actions).

**Results:**
- Peasant-Down chooses to play a small solo (action 3, Q=0.971) to help Peasant-Right go out, rather than playing a higher card (action 4, Q=0.808).
- Interpretation: Peasant-Down's Q-network predicts higher value for the cooperative move, suggesting learned coordination.

**Authors' conclusion:** Peasants learn to cooperate without explicit incentive.

**Critical note:** Anecdotal; one hand-picked example. No systematic quantification. [Interpretation] Emergent cooperation is plausible because both Peasants share the team reward (Peasant team wins = both get +1 reward), so gradient updates naturally reinforce mutual-help strategies.

---

### Experiment 8: Effect of Number of Actors (Implicit in Learning Progress)

**What changed:** 15, 30, or 45 actors.

**What held constant:** Network, hyperparameters, hardware allocation.

**Metric:** Learning curve (Figure 6, p. 7): WP (left), timesteps (right).

**Results:**
- More actors → faster wall-clock learning (left) but similar sample efficiency (right).
- 45 actors: ~50 hours to beat SL on wall-clock.
- 30 actors: ~60 hours.
- 15 actors: ~80 hours.

**Authors' conclusion:** Parallelism directly improves wall-clock time without harming sample efficiency.

**Critical note:** Expected result; demonstrates linear scaling up to 45 actors on the available hardware.

---

### Experiments Reproducible at Small Scale for CS230

1. **Comparison with SL baseline:** Train DouZero on a smaller RL dataset (e.g., fewer actors, shorter training). Should still beat SL trained on the same expert data.
   - Reduce to: 10-15 actors, 1 GPU, 2-3 days training.

2. **SARSA vs. DMC ablation:** Implement SARSA (one-step lookahead) variant and compare learning curves to DMC.
   - Feasible on 1 GPU, 1-2 days per method, both converging to similar performance if samples are equal.

3. **Effect of exploration (ε):** Train with different ε values (0, 0.01, 0.05, 0.1) and measure learning speed and final performance.
   - Small scale: 2-4 days per ε, 1 GPU.

4. **Action encoding ablation:** Disable action encoding (or concatenate less state info) and measure performance degradation.
   - Feasible on 1 GPU, 1-2 days.

5. **Network architecture ablation:** Reduce LSTM depth, MLP layers, or hidden size; measure accuracy/speed trade-off.
   - Feasible, 1-2 days per variant.

6. **Hand-tuned strategy comparison:** Compare DouZero to a simple rule-based agent (e.g., "play the smallest card that beats the opponent's move") to quantify the value of learning.
   - Easy to implement, instant evaluation.

**To shrink reproduction:**
- Fewer actors (15 instead of 45) → 3× longer wall-clock time, but ~same sample efficiency.
- Fewer GPUs (1 instead of 4) → run sequentially or reduce batch size.
- Shorter training (3-5 days instead of 30) → good enough to see convergence vs. baselines.
- Smaller test set (1,000 decks instead of 10,000 per pair) → faster evaluation, higher variance.

---

## 11. Compute and Engineering

**Hardware** (§5.1, p. 6):
- 1 server with 48 Intel Xeon cores @ 2.40 GHz.
- 4 NVIDIA 1080 Ti GPUs (11 GB VRAM each).
- Total: [Interpretation] ~$8k-10k system cost (as of 2021); paper does not specify cost.

**Training duration:** 30 days continuous.

**Parallelism:**
- 45 actor processes (threads) across 3 GPUs (15 actors per GPU).
- 1 learner process on 1 GPU.
- Shared memory buffers (3 buffers: BL, BU, BD) for communication.

**Number of games / Timesteps:**
- [Paper] The paper does not specify a game count. It states: SL surpassed at ≈5×10⁸ timesteps, DeltaDou within 5×10⁹ timesteps (Fig. 16 caption, Appendix D.3, p. 21).
- [Interpretation] Rough game-count estimate: 30 days × 45 actors × ~25 moves/game × unknown game rate ≈ uncertain (game generation rate not specified in paper).

**Sampling:**
- Each actor generates one episode (game) per iteration, computes cumulative returns, pushes to buffer.
- Learner processes batches of 32 × 100 = 3,200 (s, a, return) tuples per update.

**Memory:**
- [Paper] The paper does not specify total GPU/CPU memory usage (p. 5-6). [Interpretation] 4 × 11 GB = 44 GB GPU; CPU memory for buffers and actor state is likely <100 GB.

**Distributed training:** No; single-server design. Parallelism is via multi-threading on a single machine.

**Implementation tricks** (§5.1, p. 6; Appendix C.1, p. 16):
- **Shared buffers with entries:** Buffers are divided into 50 entries (chunks), each holding 100 samples. This allows asynchronous producer-consumer communication without locking on every sample.
- **LSTM on move history:** Encodes move history efficiently (5 × 162-dim vectors) rather than concatenating all 15 moves (210 × 54 = 11,340 dims).
- **Epsilon-greedy exploration:** Simple exploration strategy; no complex exploration bonuses.
- **Implementation framework:** TorchBeast (PyTorch-based distributed RL framework, Küttler et al., 2019, cited p. 6).

**Optimization details** (§5.1, p. 6):
- Optimizer: RMSprop (learning rate 0.0001, smoothing constant 0.99, ε = 10^-5).
- Activation: ReLU.
- Batch size: M = 32 entries = 3,200 samples per learner update.

---

### Reproduction Difficulty

**For Full Reproduction: HARD / UNREALISTIC for a quarter.**

**Justification:**
1. **30-day continuous training:** CS230 is 10 weeks; a single full training run = 3 weeks of the quarter. Ablations would require 3-5 more full runs.
2. **4-GPU system:** Commodity hardware, but not all students have access. Cloud GPU rental (AWS, Google Colab) is feasible (~$5-10/day for 4 GPUs × 30 days ≈ $150-300).
3. **30 million games:** Manageable sample-wise, but requires careful implementation to avoid bugs.

**For Scaled-Down Reproduction: MODERATE.**

**What a quarter-scale reproduction looks like:**
- **Hardware:** 1 GPU (Google Colab Pro or similar, ~$10/month) + 8-16 CPU cores.
- **Training duration:** 5-7 days instead of 30 (aim for 2-3M games instead of 32M).
- **Actors:** 15-20 instead of 45 (sample efficiency stays similar; wall-clock time increases).
- **Network:** Same architecture (no need to shrink).
- **Evaluation:** 1,000 decks per comparison (instead of 10,000; higher variance but still meaningful).
- **Expected results:** Should beat rule-based agents and SL trained on the same expert data; likely beat CQN and RHCP. May not beat DeltaDou (which is very strong) but should show clear RL signal.
- **Wall-clock time:** ~7 days GPU + ~4 days evaluation = ~11 days total (feasible in a week for a CS230 milestone).

**Key bottleneck:** Game simulation is CPU-bound (move sampling, action legality checking). If parallelism is poorly implemented, actors may block on buffers.

---

## 12. Limitations

### A. Explicitly Mentioned by the Authors

1. **Bidding phase not learned end-to-end** (§7, p. 9): Bidding is pre-trained via supervised learning on human data (Appendix C.3, p. 18). The RL agent then plays card-playing while bidding is fixed. Authors list "involving bidding in the loop for reinforcement learning" as future work.

2. **Monte Carlo variance** (§3.2, p. 4): MC methods are known to have high variance. DouZero mitigates this via parallelization (45 actors) but doesn't eliminate it. Comparison with importance-weighted off-policy RL (e.g., experience replay, §7, p. 9) is listed as future work.

3. **No search at training or test time** (§7, p. 9): DouZero is model-free. Combining with search (like DeltaDou's MCTS) is listed as future work. This limits performance ceiling.

4. **Explicit collaboration not modeled** (§7, p. 9): Peasants cooperate implicitly (shared reward signal), but there's no explicit communication or coordination mechanism. Authors list "explicitly modeling collaboration" as future work.

5. **Limited architecture exploration** (§7, p. 9): Only LSTMs + MLPs tested. CNNs, ResNets, and Transformers are mentioned as future work.

### B. Inferred from Methodology

1. **Non-stationarity in self-play:** All three agents learn simultaneously; no replay buffer of past opponents. This means the "opponent pool" is always the current agents, which can lead to oscillating strategies or local minima. [Interpretation] No historical agent checkpointing is mentioned, unlike some AlphaZero variants.

2. **No target network:** Unlike DQN, DouZero has no target network (§3.1, p. 4). While MC avoids bootstrapping bias, target networks are also used for training stability. Lack of a target network could lead to divergence if MC variance is high. [Interpretation] The paper doesn't analyze this risk empirically.

3. **Imperfect information only handled implicitly:** No explicit belief-state inference or opponent modeling (§8, this document). The LSTM learns implicit beliefs, but this may be suboptimal compared to structured inference. [Interpretation] This is addressed in DouZero+ (2022), which adds opponent modeling.

4. **Role imbalance:** Landlord and Peasant are not equally strong roles. Evaluation uses mirroring (each algorithm plays both roles) to average out role strength, but this may mask unfairness in the learning algorithm itself (e.g., one role may be undertrained). [Interpretation] No analysis of per-role convergence speed or final performance is given (though Appendix D.1, p. 22, breaks down results by role).

5. **Fixed reward signal:** WP (win/loss) and ADP (point difference) are hand-tuned. Different games (tournaments, casual play) may have different scoring rules. DouZero must be retrained for each reward scheme. [Paper] The authors acknowledge this (§5.1, p. 6) and train separate agents for WP vs. ADP. [Interpretation] This limits the generality of a single trained model.

6. **Evaluation variance:** Confidence intervals are reported only for the top-3 Elo experiment (5 seeds, 100k decks, Figure 4, p. 7). Main results (10k decks) use single training runs. [Interpretation] Uncertainty bounds are not tight.

7. **No human player comparison:** Evaluation is against other AI agents. The paper claims DeltaDou is "on par with top human players" (§5.1, p. 5), but no direct human vs. DouZero data is provided. [Interpretation] Claims of "superhuman" performance are indirect.

---

## 13. Future Work

### Explicitly Listed by Authors (§7, p. 9)

1. **Other neural architectures** (e.g., CNNs, ResNets, Transformers)
   - **Classification:** Directly usable as CS230 project idea.
   - **Rationale:** Different architectures may better capture spatial structure in card grids or temporal patterns in move history.
   - **Addressed in later papers?** [From public code repo, not the paper] DouZero's code provides the base; variants could be trained and compared via a sweep. DouZero+ (2022) mentions improvements but doesn't extensively explore architectures.

2. **Bidding in the RL loop** (integrate bidding phase into reinforcement learning, not just supervised learning)
   - **Classification:** Directly usable as CS230 project idea.
   - **Rationale:** Currently, bidding is fixed (pre-trained SL network). Joint optimization could improve both bidding and card-playing strategies.
   - **Addressed in later papers?** [Paper note] DouZero+ (Zhao et al., 2022) does not appear to address bidding in the RL loop; it focuses on opponent modeling and coach-guided learning.
   - **Difficulty:** Moderate. Requires modifying the reward function to include bidding decisions and game outcome. Main challenge: bidding is discrete, and many bid→fold early, reducing variance signal. [Interpretation]

3. **Combining RL with search** (e.g., RL + MCTS like DeltaDou or AlphaZero)
   - **Classification:** Possible but difficult for CS230.
   - **Rationale:** Search can refine learned value functions. DeltaDou uses MCTS but is slower; a hybrid might balance speed and strength.
   - **Addressed in later papers?** No evidence in DouZero+ text.
   - **Difficulty:** Hard. Requires integrating a MCTS solver with the RL learner, managing computational trade-offs, and ensuring stability.

4. **Off-policy learning and experience replay** (to improve sample efficiency)
   - **Classification:** Possible but difficult for CS230.
   - **Rationale:** MC methods are sample-inefficient. Experience replay (storing and reusing past trajectories) could reduce variance or enable off-policy learning (e.g., importance-weighted updates). Authors note this directly (§3.2, p. 4).
   - **Addressed in later papers?** No explicit mention in DouZero+.
   - **Difficulty:** Hard. Experience replay in multi-agent settings with non-stationary opponents is tricky; importance-weighting introduces high variance.

5. **Explicit modeling of Peasant collaboration** (e.g., emergent communication, shared parameters, or joint policy)
   - **Classification:** Possible but difficult / mostly theoretical for CS230.
   - **Rationale:** Peasants currently cooperate implicitly (shared reward). Explicit mechanisms (shared policy, communication layer, attention to teammate state) could improve coordination.
   - **Addressed in later papers?** [From DouZero+ text] Opponent modeling (DouZero+) touches on modeling other players but doesn't explicitly address cooperative policy learning.
   - **Difficulty:** Moderate-to-Hard. Requires multi-agent RL theory; would need new evaluation metrics to measure cooperation quality.

6. **Scalable frameworks** (e.g., SEED RL, Espeholt et al., 2019)
   - **Classification:** Engineering-heavy / not directly applicable as a research idea for CS230.
   - **Rationale:** Current implementation uses TorchBeast on a single server. Scaling to multiple servers could accelerate training and enable larger experiments.
   - **Addressed in later papers?** Not mentioned in DouZero+ or other papers in the set.
   - **Difficulty:** Hard. Requires distributed systems expertise.

7. **Testing Monte Carlo methods on other tasks** (beyond DouDizhu)
   - **Classification:** Possible but outside DouDizhu scope (may not align with CS230 project goals).
   - **Rationale:** DouZero shows MC methods can be strong in large action spaces. Applying to other games (e.g., Mahjong, other card games) could validate the approach.
   - **Addressed in later papers?** Not in DouZero+; the follow-up focuses on DouDizhu improvements.

### Implied Future Work [Interpretation]

8. **Opponent modeling and diversity** (implied by cooperation analysis, §5.8, p. 9):
   - The paper observes that Peasants cooperate, but doesn't quantify how well different RL seeds converge to similar cooperation strategies. Training against diverse opponents (opponent pool, like in some AlphaZero variants) could improve robustness.
   - **Classification:** Possible but difficult.
   - **Addressed in later papers?** [Paper note] **Yes.** DouZero+ (Zhao et al., 2022) directly addresses opponent modeling, adding a "prediction model" that infers opponents' hands and a "coach network" to filter imbalanced initial hands for faster learning. This is a substantial follow-up to the implicit cooperation in original DouZero.

9. **Cross-domain transfer** (implied by the "novel strategies" finding, §5.6, Figure 8, p. 8):
   - After day 5, DouZero diverges from human play (accuracy drops). Understanding what makes these strategies novel (e.g., risk-taking patterns, bomb usage) could enable transfer learning to other card games or human-guided learning.
   - **Classification:** Theoretical / not straightforward for CS230.

---

## 14. Connection to Other Papers

### Prior Work DouZero Builds On

1. **NFSP (Heinrich & Silver, 2016):** Deep RL from self-play in imperfect-information games. Pioneered self-play + deep networks for poker. [Interpretation] DouZero uses self-play like NFSP but with Monte Carlo instead of Q-learning.

2. **Deep CFR (Brown et al., 2019):** Counterfactual regret minimization with deep nets. Dominant in heads-up poker. [Interpretation] Deep CFR is theoretically grounded but computationally expensive; DouZero opts for simplicity and sampling.

3. **DeltaDou (Jiang et al., 2019):** Prior SOTA for DouDizhu. Uses Bayesian inference + MCTS. [Interpretation] DouZero replaces this complex pipeline with a simpler, faster MC approach.

4. **RLCard (Zha et al., 2020a):** Toolkit for RL in card games. Provides DouDizhu environment, rule-based baselines. [Interpretation] DouZero uses RLCard's environment for experiments; RLCard was co-authored by DouZero's lead author (Zha).

### Later Work Inspired by DouZero

5. **DouZero+ (Zhao et al., 2022):** Adds opponent modeling + coach-guided learning to DouZero. [Paper note] The paper does not cite DouZero+ (which was not published when DouZero was submitted), but DouZero+ directly addresses two of DouZero's future-work items: (1) opponent modeling (prediction models that infer hands), (2) faster training (coach network to filter imbalanced deals). This shows DouZero's conceptual foundation remains solid; improvements stack on top.

### Conceptual Lineage [Interpretation]

```
NFSP (2016)
  Self-play + deep networks for imperfect-info games
  ↓ (addresses game-specific challenges)
DeltaDou (2019)
  + Bayesian inference (model hidden info)
  + MCTS (search-based planning)
  ↓ (complex, slow)
DouZero (2021)
  - Bayesian inference (implicit via LSTM)
  - MCTS (use pure RL sampling)
  + Action encoding (generalization in large action space)
  + Parallelization (scale MC to reduce variance)
  = Simpler, faster, end-to-end learnable
  ↓ (strong baseline; room for refinement)
DouZero+ (2022)
  + Opponent modeling (explicit prediction of hands)
  + Coach network (smarter data sampling)
  ↓ (faster training, better robustness)
```

### Evolution Chain (Detailed)

1. **NFSP → DeltaDou limitation:** NFSP works in simple imperfect-info games (poker); DouDizhu's huge action space and cooperation requirements break NFSP-style policy gradient methods. DeltaDou addresses this with domain-specific knowledge (Bayesian inference, heuristic kicker networks) and search.
   
2. **DeltaDou → DouZero limitation:** DeltaDou is slow (2+ months training), relies on human heuristics (kicker networks), and requires careful tuning (Bayesian inference). DouZero replaces heuristics with end-to-end learning and search with pure sampling, making training much faster (10 days).

3. **DouZero → DouZero+ limitation:** DouZero learns strategies through trial-and-error but doesn't explicitly model opponents or curate training data. DouZero+ adds opponent models (to predict hands better) and coach networks (to focus training on balanced deals), accelerating learning further.

[Interpretation] Each step trades complexity/interpretability for speed/learnability. DouZero's key insight—that classic MC methods + action encoding are sufficient—is validated by DouZero+'s success in building on the foundation.

---

## 15. Project Ideas for Me

### Project Idea 1: Bidding in the RL Loop

**Research question:** Can end-to-end RL (bidding + card-playing jointly) outperform the current two-stage approach (SL bidding + RL card-playing)?

**Baseline:** DouZero with supervised bidding network (as in the paper).

**Modification:** Add bidding as a first action in each episode. Reward function includes both bidding accuracy (e.g., penalties for bidding on weak hands) and game outcome. Train with the same DMC approach but with a 4th network (bidding network) that also outputs a value estimate.

**Hypothesis:** Joint training will improve performance because bidding and card-playing are coupled (a weak bid can lead to bad card-playing situations, and vice versa). Expected improvement: +3-5% WP over the SL bidding baseline.

**Experiment:**
- Train DouZero-End-to-End (E2E) for 3-5 days on 1 GPU.
- Compare WP and ADP vs. DouZero-SL-Bidding baseline on 1,000 decks (faster evaluation).
- Analyze: Does the bidding network learn different thresholds? How does bidding strategy change during training?

**Metrics:** WP, ADP, accuracy on bidding decisions vs. human data, hand strength distribution of chosen bids.

**Difficulty:** Moderate. Implementation is straightforward (add one network + modify reward logic). Main challenge: managing training stability for the bidding network (which may have sparse reward signals early on).

**Compute:** Medium (1-5 days on 1 GPU for training + evaluation).

**Novelty:** Directly addresses a known limitation of DouZero (§7, p. 9). Shows whether joint optimization of loosely coupled sub-tasks improves RL.

---

### Project Idea 2: Opponent Modeling via Hand Inference

**Research question:** Does explicit opponent modeling (predicting opponents' hands) improve performance faster than implicit learning (LSTM alone)?

**Baseline:** DouZero (implicit inference via LSTM).

**Modification:** Add an opponent hand inference module (separate network that predicts the hand distribution of each opponent given the observed game state). Concatenate predicted hands with state features before feeding to the main Q-network.

**Hypothesis:** Explicit inference will accelerate learning because the network doesn't have to learn to parse history; it gets a structured belief state. Predicted improvement: learning 20-30% faster (wall-clock) to reach the same performance.

**Experiment:**
- Implement opponent hand prediction network (similar to DouZero+ but simpler; don't need coach network).
- Train two variants: (1) DouZero + hand inference, (2) DouZero baseline (no prediction).
- Compare learning curves (Figure 5 style) vs. SL baseline.
- Analyze: Do predicted hands become accurate over time? How much does inference accuracy correlate with Q-network performance?

**Metrics:** Wall-clock time to beat SL/DeltaDou, hand prediction accuracy, WP after N games, ADP.

**Difficulty:** Moderate-to-Hard. Requires implementing a multi-head classification network + integrating predictions into the main Q-network. Challenge: ensuring hand predictions are accurate enough to help (garbage predictions would hurt).

**Compute:** Medium-to-High (5-7 days on 1 GPU for one method × 2 = 10-14 days total).

**Novelty:** Pre-dated by DouZero+ (which does opponent modeling), but the idea is valuable even if you don't know about DouZero+. Shows the trade-off between explicit belief states and end-to-end learning.

---

### Project Idea 3: Action Encoding Ablations

**Research question:** How important is action encoding (card matrix representation) for generalization in large action spaces? What is the minimal encoding required?

**Baseline:** DouZero with full 4×15 card matrix for actions.

**Modifications:**
1. Dense action encoding (flatten 4×15 matrix to 60-dim vector).
2. Minimal encoding (one-hot over action types, e.g., "Solo" vs. "Pair" vs. "Bomb"; 13-dim).
3. No encoding (action index as a scalar, 1-dim; forces network to learn structure from scratch).

**Hypothesis:** Full card matrix encoding enables generalization to unseen actions; minimal encoding forces rote memorization. Prediction: performance drops as encoding shrinks (model 1 ≈ model 0 > model 2 > model 3).

**Experiment:**
- Train 4 agents (full encoding + 3 ablations) for 3-5 days each on 1 GPU.
- Compare: learning speed, final WP/ADP, inference speed, model size.
- Analyze: Do less-encoded models memorize (high accuracy on frequent actions, low on rare)? Do they fail on unseen combinations?

**Metrics:** WP, ADP, per-action accuracy (for frequent vs. rare actions), parameter count, training time.

**Difficulty:** Easy. Mostly mechanical changes (swap action encoding format). Good for understanding the paper's design choices.

**Compute:** Low (10-15 days total for 4 methods on 1 GPU, done in parallel over a week).

**Novelty:** Ablation-style; not groundbreaking, but provides empirical evidence for a key DouZero design decision.

---

### Project Idea 4: Self-Play Non-Stationarity and Opponent Pools

**Research question:** Does training against a pool of older checkpoints (like AlphaZero) improve training stability and final performance compared to always playing against the current policy?

**Baseline:** DouZero (current policy only).

**Modification:** Maintain a pool of 5-10 previous checkpoints saved every training day. Actors randomly sample an opponent checkpoint (prob 0.3) or use the current policy (prob 0.7). This forces agents to adapt to a diverse set of opponents.

**Hypothesis:** Opponent pools will reduce oscillations (non-stationary effects) and lead to more robust strategies. Expected improvement: smoother learning curves, +2-3% WP over baselines, better generalization to unseen opponent strategies.

**Experiment:**
- Train two agents: (1) with opponent pool, (2) baseline (no pool).
- Compare learning curves, final performance (WP/ADP vs. SL, DeltaDou).
- Analyze: Are curves smoother? How often does the pool agent play vs. current? Does it hurt wall-clock training time?

**Metrics:** WP, ADP, learning curve smoothness (e.g., rolling-window variance), wall-clock time, pool utilization.

**Difficulty:** Moderate. Requires managing a checkpoint pool and modifying the actor-learner communication to select opponents.

**Compute:** Medium (5-7 days per method × 2 = 10-14 days, can parallelize on 2 GPUs).

**Novelty:** Explores a known technique (AlphaZero-style pools) in a new domain (DouDizhu). Good research question even if the result is incremental.

---

## 16. Important Implementation Components

### MUST UNDERSTAND AND IMPLEMENT/MODIFY

1. **Deep Monte Carlo Training Loop (Algorithm 1-2):**
   - Actors must compute cumulative returns at episode end and store (s, a, return) tuples.
   - Learner must sample batches and compute MSE loss: $\|Q(s,a) - \text{return}\|^2$.
   - No target network or bootstrapping.
   - [Why] Core algorithm; cannot skip.

2. **Action Encoding (4×15 Card Matrices):**
   - Encode every action and state as card matrices; flatten and concatenate.
   - Must handle variable-size histories (pad with zeros if <15 moves).
   - [Why] Critical for generalization to unseen actions.

3. **Parallel Actor-Learner Architecture:**
   - Actors: Generate episodes, compute returns, push to buffers.
   - Learner: Sample from buffers, update networks, release buffers.
   - Synchronization: Actors periodically sync local networks with global networks.
   - [Why] Enables scale (45 actors); reduces MC variance.

4. **Epsilon-Greedy Exploration:**
   - During training, sample: argmax Q(s, a_legal) with prob (1-ε), random legal action with prob ε.
   - During eval, use greedy (no randomness).
   - [Why] Balances exploration and exploitation.

5. **Role-Specific Networks and States:**
   - Train three separate networks (Landlord, Peasant-Up, Peasant-Down).
   - State features differ by role (Tables 4-5, Appendix C.1, p. 16).
   - [Why] Roles have different observation and reward structures.

6. **Game Environment and Legality Checking:**
   - Implement (or reuse) a DouDizhu simulator with move legality checking.
   - Actors must filter actions to legal moves only.
   - [Why] Core to the problem; cannot use illegal actions.

### MUST UNDERSTAND CONCEPTUALLY

1. **Monte Carlo vs. Q-Learning:**
   - MC: Directly estimate value from observed returns (no bootstrapping).
   - Q-learning: Estimate value from next state's Q-value (bootstrapping).
   - [Why] Understanding this trade-off is critical to the paper's contribution.

2. **Overestimation Bias in Q-Learning:**
   - With large action spaces, max Q(s', a') is prone to overestimation.
   - DMC avoids this by not taking a max.
   - [Why] Explains why DQN fails in DouDizhu.

3. **Action Feature Generalization:**
   - Policy gradients cannot use action features naturally (e.g., "if 3-K-K-K works, 3-J-J-J should too").
   - DMC (Q-network with action input) naturally handles this.
   - [Why] Key reason DMC beats policy gradients.

4. **Imperfect Information Handling:**
   - Agents don't see opponents' hands; must infer from history and hand sizes.
   - LSTM encodes move history implicitly (vs. explicit Bayesian inference in DeltaDou).
   - [Why] Understanding the trade-off between implicit and explicit belief states informs your own design choices.

5. **Sparse Reward and Long Horizons:**
   - Reward is only at game end (binary win/loss or point differential).
   - RL must learn value estimates from a single noisy signal per episode.
   - [Why] Explains why high variance (MC) is okay if data is abundant (parallelization).

### CAN REUSE AS INFRASTRUCTURE

1. **RLCard Toolkit (Zha et al., 2020a):** Pre-built DouDizhu environment, rule-based agents, evaluation utilities. [From public code repo, not the paper] Available at github.com/datamllab/rlcard.

2. **DouZero Source Code (Kwai Inc.): ** Full implementation of DouZero (networks, training loops, evaluation). [From public code repo, not the paper] Available at github.com/kwai/DouZero. Includes:
   - Pre-trained models for baseline comparisons.
   - Training scripts (parallelized actor-learner).
   - Evaluation utilities (Elo rating, match generation).
   - Game environment integration.

3. **TorchBeast Framework (Küttler et al., 2019):** Distributed RL framework used by DouZero for parallelization. Simplifies actor-learner architecture. [From public code repo, not the paper]

4. **Move History Encoding (LSTM on 15 Moves):** Template for encoding temporal sequences in card games. Reusable for other card games (Mahjong, Poker variants).

5. **Card Matrix Representation (4×15):** Flexible encoding for card combinations. Adaptable to other games with different card counts/ranks.

---

## 17. Questions I Should Ask My CS230 Staff

1. **Scope & Timing:**
   - If I want to reproduce DouZero at scale (30 million games, 30 days), should I plan for AWS/Google Cloud cost or is the CS230 cluster available?
   - Is a scaled-down version (5-7 days, 2-3 million games on 1 GPU) acceptable as a "reproduction," or do you expect full-scale results?

2. **Novelty Bar:**
   - DouZero+ (2022) improves on DouZero with opponent modeling and coach networks. If I implement one of these, is that novel enough for a project, or should I target something completely new (e.g., combining RL with search)?
   - Do you expect ablation studies (e.g., "action encoding is important") to count as novelty, or should the project introduce a new method/domain?

3. **Evaluation Statistical Rigor:**
   - The main paper uses 10k decks per pair (single training run). For my project, what is the minimum evaluation to show a statistically significant win? (e.g., 1k decks, 3 seeds? 5k decks, 1 seed?)
   - Should I report confidence intervals or just means?

4. **Imperfect-Information Challenge:**
   - I notice DouZero doesn't explicitly model beliefs (unlike DeltaDou's Bayesian inference). Is implicit belief learning (via LSTM) a limitation I should address, or is it acceptable for a CS230 project?
   - Would you recommend starting with a simpler, fully-observable game (e.g., Go, Chess, a simplified DouDizhu) and adding imperfect information as an extension?

5. **Self-Play Stability:**
   - DouZero uses no opponent pool; agents always play against the current policy. I'm considering adding a pool of old checkpoints (like AlphaZero) to stabilize training. Is this a meaningful direction, or is it an engineering detail that won't count as a research contribution?

---

## 18. Beginner Concepts I Need to Learn

| Concept | Plain-Language Explanation | Where It Appears in DouZero | Priority |
|---------|--------------------------|---------------------------|----------|
| **Reinforcement Learning (RL)** | Learn by trial and error: an agent takes actions, observes rewards, and adjusts its strategy to maximize cumulative reward. | Entire paper (§1-7) | NOW |
| **Markov Decision Process (MDP)** | A model of a sequential decision problem: states, actions, transitions, rewards. RL algorithms solve MDPs by finding the best policy. | §2, implicit throughout | NOW |
| **Policy** | A function (deterministic or stochastic) that maps states to actions: "in state s, do action a." π(a\|s) = probability of action a in state s. | §3.1, Eq. 4 (p. 4) | NOW |
| **Value Function / Q-Function** | V(s) = expected cumulative reward starting from state s. Q(s,a) = expected cumulative reward for doing action a in state s, then following the policy. | §3.1, p. 4 | NOW |
| **Return / Cumulative Reward** | G_t = sum of all future rewards from time t onward: r_t + r_{t+1} + r_{t+2} + ... (often discounted). | Eq. 1 (p. 5); Algorithm 1, lines 11-13 | NOW |
| **Monte Carlo (MC) Methods** | Learn by averaging over complete observed trajectories (episodes). Estimate value by averaging returns from many games, not by bootstrapping. | §3.1-3.3, p. 4-5 | SOON |
| **Temporal Difference (TD) Learning / Bootstrapping** | Learn by combining immediate reward + estimated value of the next state (one-step lookahead). Updates based on a mix of observation and estimate. | §3.3, p. 4 | SOON |
| **Deep Q-Learning (DQN)** | Use a neural network to approximate Q(s,a). Update via TD: [r + γ max Q(s',a') - Q(s,a)]^2. Classic algorithm that fails in DouDizhu due to overestimation bias. | §3.3, p. 4 | SOON |
| **Overestimation Bias** | When max Q(s',a') is estimated (not observed), it tends to be overestimated, leading to inflated value estimates and suboptimal policies. | §3.3, p. 4 | SOON |
| **Policy Gradient Methods (A3C, PPO)** | Learn by directly optimizing the policy (not the value function). Use gradient ascent to maximize expected reward. Cannot easily generalize across action features. | §3.2, p. 4 | SOON |
| **Epsilon-Greedy Exploration** | Explore random actions with probability ε; exploit best known action with probability (1-ε). Simple, effective way to balance exploration and exploitation. | §3.1, Eq. 3 (p. 5); Algorithm 1 | NOW |
| **Self-Play** | Agents learn by playing against each other (or against earlier versions of themselves). Used in games (AlphaGo, AlphaZero) to generate strong opponents. | §1, §7 | NOW |
| **Multi-Agent RL** | Multiple agents learn simultaneously, each optimizing for their own reward. Agents' policies are non-stationary (changing) because other agents are learning. | §1-2, p. 2 | NOW |
| **Imperfect Information / Partially Observable** | Agents don't see the full state (e.g., hidden opponent hands in poker, DouDizhu). Must infer from observations and history. | §2, §8, p. 2 | SOON |
| **Action Space** | Set of possible actions an agent can take. "Large" (e.g., 27,472 in DouDizhu) or "small" (e.g., 6 in Tic-Tac-Toe). Larger spaces are harder for RL. | §2, Table 3 (p. 15) | NOW |
| **Sparse Reward** | Reward signal is rare (e.g., only at episode end). Hard for RL because agents must learn from few noisy reward samples. | §3.3, p. 4 | SOON |
| **Supervised Learning (SL)** | Learn from labeled examples: "in state s, the best action is a." RL learns from unlabeled experience (trial and error); SL learns from labeled data. | §5.2, p. 6 | NOW |
| **Function Approximation** | Use a neural network (instead of a table) to represent the value function or policy. Allows generalization to unseen states. | §3.1, p. 4 | NOW |
| **Gradient Descent / Backprop** | Method to minimize a loss function by computing gradients and updating parameters. Standard in deep learning. | §6, implicit | OPTIONAL (pre-req) |
| **LSTM (Long Short-Term Memory)** | Recurrent neural network that learns temporal dependencies. Used here to encode move history. | §3.1, Figure 3 (p. 4) | OPTIONAL |
| **Elo Rating** | Chess/game rating system: a player's strength is estimated by their tournament results against other rated players. Higher Elo = stronger. | §5.2, Figure 4 (p. 7) | OPTIONAL |

---

## 19. Paper Scorecard

> **Cross-paper calibration:** this scorecard looks at this paper alone. Ranked against the whole pack for this Dou Dizhu project, the recommended verdict is **READ DEEPLY**. See `00_SYNTHESIS_AND_RECOMMENDATION.md` §20.9.

| Dimension | Score (1-5) | Justification |
|-----------|-----------|----------------|
| **Importance for my project** | 5 | DouZero is the foundational RL solution for DouDizhu. For DouDizhu RL work, this paper is essential. It defines the state-of-the-art baseline. |
| **Difficulty to understand** | 3 | Concepts (Monte Carlo, Q-learning, self-play) are standard RL, but the combination and engineering (parallel actors, action encoding) requires careful reading. The paper is well-written but dense (10 pages + 8 appendix pages). |
| **Difficulty to reproduce** | 4 | Full reproduction requires 30 days + 4 GPUs + careful implementation. Scaled-down version (5-7 days, 1 GPU) is Moderate. A student should be able to debug and extend the approach, but reproducing top performance is challenging. |
| **Relevance to Dou Dizhu** | 5 | Directly solves DouDizhu. Strongest AI to date (ranked #1 on Botzone leaderboard among 344 bots; Elo 1625.11). Comparison with top human players is indirect (via prior work). Cannot be more relevant. |
| **Relevance to self-play** | 5 | Self-play is central: all three agents learn simultaneously. The paper demonstrates that self-play with deep MC works in a complex multi-agent game. |
| **Relevance to imperfect information** | 4 | DouDizhu is partially observable; DouZero handles it implicitly via LSTM on move history. Not a deep treatment of belief states or opponent modeling (NFSP, Deep CFR handle this more explicitly), but practical and effective. |
| **Potential for project ideas** | 5 | Many open directions: bidding in the RL loop, opponent modeling, action encoding ablations, opponent pools, other architectures. Good for projects that extend or improve DouZero. |
| **Importance to read deeply** | 5 | If doing a DouDizhu project, read the full paper (especially §3, §4, §5). If doing a different game or general RL, skim §1-3 (algorithm) and glance at §4-5 (engineering). |

### Verdict

**READ DEEPLY**

### Sections/Pages to Read First (if selective, though full read is recommended):

1. **§1 Introduction (p. 1-2):** Motivation for why DQN/A3C fail in DouDizhu.
2. **§3 Deep Monte-Carlo (p. 4-5):** Core algorithm; comparison with DQN and policy gradients.
3. **§4 DouZero System (p. 4-5):** Architecture and parallel training.
4. **§5.1-5.2 Experiments & Results (p. 5-7):** Main results vs. baselines.
5. **Appendix C.1 (p. 16):** Detailed feature and architecture tables.
6. **Appendix A (p. 13-15):** Game rules and action space (skim if familiar with DouDizhu).

---

## Appendix: Synthesis Facts (for the cross-paper comparison table)

**Paper:** DouZero: Mastering DouDizhu with Self-Play Deep RL (ICML 2021)

**Year:** 2021

**Environment(s):** DouDizhu (three-player card game); standard 54-card deck with bidding and card-playing phases.

**Main algorithm:** Deep Monte Carlo (DMC): MC returns (no bootstrapping) + deep neural networks for value function approximation + action encoding (card matrices) + parallel actors (45) + learner (1).

**State representation:** For each role (L, U, D): card matrices (hand cards, union of others' cards, recent moves), one-hot vectors (hand sizes, bomb count), LSTM-encoded history of last 15 moves. Total: ~400-500 dims per role (varies; see Tables 4-5, Appendix C.1).

**Action representation:** 4×15 one-hot card matrix (same format as state cards); flattened and concatenated with state.

**Self-play?** Yes; all three roles trained simultaneously. No opponent pool; agents play current policy only.

**Opponent modeling?** No explicit modeling (no Bayesian inference or hand prediction). Implicit via LSTM on move history. [Follow-up: DouZero+ (2022) adds explicit opponent modeling.]

**Main loss:** MSE on Monte Carlo return: |Q(s,a) - G_t|^2. No bootstrapping; no target network.

**Main contribution:** Shows that classic MC methods (enhanced with deep networks, action encoding, parallelization) can beat state-of-the-art RL in large, complex, imperfect-info multi-agent games. Challenges the dominance of DQN and policy gradients. Ranked #1 on Botzone leaderboard (Elo 1625.11) in days vs. prior state-of-the-art (DeltaDou) trained 2+ months.

**Evaluation method:** Tournament-style with 10,000-100,000 randomly sampled decks; role mirroring (each algorithm plays both Landlord and Peasant); two metrics (WP = win rate, ADP = point differential); Elo rating for top 3 algorithms; Botzone online leaderboard; comparison with DeltaDou (prior SOTA), CQN, SL, rule-based agents.

**Compute:** 1 server (48 CPUs, 4×1080 Ti GPUs), 30 days continuous training, 45 actors. [Paper] States: SL surpassed at ≈5×10⁸ timesteps, DeltaDou within 5×10⁹ timesteps (§5.4 & Appendix D.3, p. 8/21). Number of games not specified.

**Main limitation:** (1) Bidding pre-trained (not learned end-to-end); (2) No search (model-free only); (3) MC variance mitigated by parallelization but still a factor; (4) No opponent pool (non-stationarity); (5) Implicit belief learning (vs. explicit inference like DeltaDou).

**Best idea I could borrow:** (1) **Action encoding as general solution for large discrete action spaces:** Represent actions as dense vectors (card matrices here) and use Q(s,a) architecture to let the network generalize across similar actions. Applicable to any action space with structure (e.g., combinations, sequences). (2) **Parallel actor-learner with shared buffers:** Efficient way to collect data and train on large batches without full distributed systems. (3) **Deep Monte Carlo as simpler alternative to DQN in sparse-reward, variable-action-space settings:** When bootstrapping is unstable and action space is huge, try MC with parallelization first.

