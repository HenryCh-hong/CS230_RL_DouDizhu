# Suphx: Mastering Mahjong with Deep Reinforcement Learning (2020)
> Li et al. · Microsoft Research Asia · arXiv 2003.13590 · **Tier C** — Multi-player imperfect-information deep RL with three novel components (global reward prediction, oracle guiding via perfect-information dropout, parametric Monte-Carlo policy adaptation) on a highly complex tile game.

---

## 0. TL;DR for a CS230 student

- **What it is**: A deep RL system that beats top human players at Japanese Mahjong (Riichi Mahjong), a 4-player game with hidden information, complex multi-round scoring, and irregular game rules that break MCTS.
- **Why you care**: (1) Shows how to design *task-specific* reward signals (global reward prediction) when round scores mislead. (2) Demonstrates a clever way to exploit privileged information during training (oracle guiding with feature dropout) — directly applicable to Dou Dizhu. (3) Handles imperfect information and multi-agent coordination without game tree search.
- **Three ideas to steal**:
  1. **Global reward prediction**: Train a GRU to predict end-of-game reward from partial state; use predicted reward differences as training targets instead of noisy round scores. Fixes credit assignment when final reward depends on multi-round play.
  2. **Oracle guiding**: Train an oracle agent with *perfect information* (opponents' hidden cards, deck), then gradually drop perfect features (Bernoulli dropout matrix with decay schedule) until it becomes a normal agent. Faster learning than supervised + RL alone.
  3. **Parametric Monte-Carlo policy adaptation (pMCPA)**: At round start, generate K random trajectories with current hand, finetune policy on them via importance sampling. Adapts offline policy to specific hand without full MCTS. (Not deployed in online system due to time; potential gain of ~4% win rate locally.)

---

## 1. Paper Overview

### 1a. Beginner-friendly explanation

Imagine you're teaching an AI to play poker, but (1) you have private cards others can't see, (2) the game has 8–12 rounds and your final reward depends on your total score across *all* rounds (so losing one round may be strategically correct), (3) there are ~10^48 hidden card combinations, and (4) you can't use tree search because the game flow is irregular (other players can interrupt your turn). Welcome to Japanese Mahjong.

**Suphx** tackles this by:
1. **Learning from humans first** (supervised learning on expert logs from Tenhou, an online platform with 350K+ players).
2. **Boosting with self-play** using a policy gradient algorithm, but with three innovations:
   - A "global reward predictor" (GRU neural network) that learns to estimate your final game score after 2–3 more rounds. This lets you get a better training signal than raw round scores, which can be misleading.
   - An "oracle agent" that plays with perfect information (can see all hidden cards). During training, you gradually hide perfect information from the oracle, so it slowly turns into a normal agent. This warm-starts learning without requiring the normal agent to learn everything from scratch.
   - A runtime adaptation step: when you're dealt a hand at the start of a round, you run 100K quick simulations and finetune your policy just for this hand, then play.
3. **Achieving superhuman performance**: Reaches 10 dan (top rank) on Tenhou and surpasses 99.99% of human players.

### 1b. Technical explanation

**Problem solved**: Build an agent for 4-player Japanese Mahjong that surpasses top human players, handling: (1) complex, context-dependent scoring (89 meld types × 34 pair types × multiple yaku combinations); (2) ~10^48 hidden information per decision point; (3) irregular game tree (actions interrupt other players' turns); (4) multi-round, sparse terminal reward (reward only at game end, not per round).

**Why difficult**:
- *Imperfect information at scale*: On average, >10^48 hidden states indistinguishable to a player at any information set (§1, p. 2). Texas hold'em has ~10^13 information sets; Mahjong has far more. Card inference alone is insufficient.
- *Complex reward structure*: Game contains 8–12 rounds. Each round yields a round score; final game reward depends on accumulated rank across all rounds. A player losing round N might be playing optimally (e.g., conceding to preserve rank-1 status). Round scores and game rewards are misaligned for RL training.
- *Irregular game tree*: Melds (Pong, Kong, Chow), discards, and winning declarations can interrupt the regular turn order. Cannot build a fixed game tree. MCTS (which requires a tree) is blocked.
- *Multi-player non-stationarity*: Four agents learning simultaneously; opponent skill changes during training.

**Previous work**: Earlier Mahjong AIs (Bakuuchi, NAGA, Gao et al. 2018–2019) relied on Monte Carlo sampling or supervised learning alone. Best reached ~6.6 stable rank (vs. Suphx's 8.74).

**Main contributions**:
1. **Global Reward Prediction (GRP)**: A GRU predictor trained on human logs that estimates final game reward from current + prior round info. During RL, use $$\Phi(x_k) - \Phi(x_{k-1})$$ as per-round reward instead of raw round score.
2. **Oracle Guiding**: Train an oracle agent with perfect features (opponents' hands, deck); gradually dropout perfect features (Bernoulli mask with decay γ_t: 1 → 0) until oracle becomes a normal agent. Combines privileged information distillation with parameter sharing.
3. **Parametric Monte-Carlo Policy Adaptation (pMCPA)**: At round start, sample K trajectories with fixed hand, finetune policy via policy gradient. Adapts offline policy to specific hand without full MCTS. Tested offline only; 66% win rate vs. non-adapted version on hundreds of rounds.

**Related work**: Builds on supervised learning from human games (Gao et al. 2018) and self-play RL (policy gradient methods, importance sampling). Similar to RL in imperfect-information games (Deep CFR, etc.) but tailored to Mahjong's specific challenges.

---

## 2. RL Formulation

| Component | Definition | Citation / Notes |
|-----------|-----------|------------------|
| **Agents** | Single player (position in game); four players per game | §2 (Decision Flow, p. 4) |
| **Environment** | Tenhou platform simulator; 4-player Japanese Mahjong game engine | §3.1, §5 (Tenhou); Appendix A |
| **State** | Observable: private tiles (4 channels × 34), public discards (sequence), open melds, round score, dealer, riichi bets; Hidden: opponents' hands, wall tiles | §2.2 (Features, p. 6–7); §8 (Imperfect Info) |
| **Observation** | Same as observable state | §2.2 |
| **Action space** | Discrete; context-dependent. Discard (34 tiles), Riichi (binary), Chow/Pong/Kong (binary or binary per meld type). Winning decided by rule-based model. | §2.1 (Decision Flow, p. 4–5); Table 1 |
| **Legal action masking** | Yes. Kong model only applies if private tiles + drawn tile form Kong. Riichi only if hand is one tile from completion. Discard only from hand. | §2.1; Figure 1 |
| **Reward** | Per-round: Round score (winner gets positive, others get 0 or negative); Per-game: Rank-based points (1st place: +60 pts @ expert room, 4th place: -165 pts). However, RL uses *predicted* reward from GRP. | Appendix A (Scoring); §3.2 (GRP); Table 6 |
| **Return** | Accumulated reward over one game (8–12 rounds), weighted by win/loss/rank. Game ends when a player's points reach 0 or certain condition triggers. | Appendix A; §3.2 |
| **Policy** | πθ(a\|s): neural network (CNN) outputs action distribution over legal actions (or confidence scores for binary decisions). | §2.2; §3.1 |
| **Value function** | The paper does not specify this clearly. [Interpretation: GRP predictor Φ acts as a *reward predictor* (not value function), estimating $V^π(s) ≈ \text{(final game reward from state s)}$. No explicit value network for bootstrapping.] | §3.2 |
| **Fully vs partially observable** | Partially observable (imperfect information). Agent observes own hand, public discards, dealer/round info; cannot see opponents' hands or deck tiles. | §2.2; §8 |
| **Single vs multi-agent** | Multi-agent (4 players). | §1 (Abstract, p. 1) |
| **Cooperative/competitive/mixed** | Competitive (win/loss is zero-sum per round). However, implicit cooperation within a player's multi-round strategy (e.g., conceding round N to secure rank-1 overall). | §3.2 (GRP motivation) |
| **Self-play used?** | Yes. Every game in RL training is Suphx (position 0) vs. three copies of Suphx (positions 1–3). All players use the same policy (parameter sharing). Policy updated via replay buffer. | §3.1 (Distributed RL System); p. 9; §4.2 (RL experiments) |
| **Opponent pool** | No explicit opponent pool during RL training. Three opponents are clones of the training policy (parameter sharing). During online eval, opponents are human players on Tenhou. | §4.2; §5 (Online Eval) |

---

## 3. Model / Architecture

### Overall Design
Suphx uses five separate CNN models for different decision types (discard, Riichi, Chow, Pong, Kong), plus a rule-based winning model. This modularization reflects Mahjong's complex decision flow (Figure 1, p. 5) rather than a single monolithic policy.

### Input Representation (Feature Encoding)

**Tile representation**: Japanese Mahjong has 34 unique tile types (3 suits: Bamboo, Character, Dot, each 1–9; 7 Honour tiles). Suphx encodes all state into `D × 34 × 1` feature maps (D varies per model).

**Channel design** (§2.2, Figure 3, p. 6–7):
- **Private tiles**: 4 channels × 34 columns. Each channel c ∈ {0,1,2,3} = count. Entry (c, tile_id) = 1 if hand has exactly (c+1) of tile_id; 0 otherwise. Encodes hand composition (up to 4 copies per tile).
- **Open melds** (Pong, Kong, Chow from previous discards): Similar 4-channel encoding per meld.
- **Doras** (bonus tiles): 1–2 channels depending on dora count.
- **Discard sequence**: One channel per discarded tile (in order), showing which tiles are publicly known.
- **Categorical features** (round ID, dealer position, repeat dealer counter, Riichi bet count): Each binary-encoded into separate channels (all 0s or all 1s).
- **Integer features** (accumulated round scores of 4 players, tiles left in wall): Bucketed into ranges, each bucket a binary channel.

**Look-ahead features** (§2.2, p. 7):
- Beyond observable state, Suphx computes 100+ "look-ahead" features that predict, for each possible discard, the probability and score of reaching a winning hand after drawing replacement tiles.
- Computed via depth-first search (DFS) over possible hands, ignoring opponent behavior. Accounts for 89 meld types × 34 pair types = huge search space, so simplified.
- Each feature is a 34-dimensional vector (one entry per tile type).
- Example: "If we discard tile X and draw from wall, can we form a 12,000-point hand by replacing 3 tiles?"
- Purpose: Reduce RL exploration time by encoding human domain knowledge.

**Total input dimension**:
- Discard & Riichi models: 34 × 838 (§2.2, Table 2, p. 8)
- Chow, Pong, Kong models: 34 × 958 (includes info about candidate melds)

### CNN Architecture

**Structure** (Figures 4–5, p. 7–8):
- **Discard model** (Figure 4):
  - Input: D × 34 × 1
  - 50 residual blocks, each: [3×1 conv 256, 3×1 conv 256] + skip connection
  - 1×1 conv → 1 output (logits for each of 34 tiles)
  - No pooling (to preserve per-tile semantic meaning)
- **Riichi, Chow, Pong, Kong models** (Figure 5):
  - Input: D × 34 × 1
  - Similar residual stack: 50 repeats of [3×1 conv, 3×1 conv] + skip
  - 2 layers of fully-connected (1024, 256)
  - 2 output neurons (binary decision: yes/no for action)

**Rationale for 1×1 convolutions in spatial dimension**:
- Mahjong state is naturally 1D over 34 tile types (not 2D image like Go).
- 3×1 convs allow local tile context (tile i depends on tiles i±1).
- No pooling because each column (tile) has fixed semantic meaning; pooling = information loss.

**Parameter sharing**: All five models share *architecture* but *not parameters*. Each has its own weights.

**Output interpretation**:
- Discard model: 34-dim logits → softmax → probability distribution over tiles to discard.
- Riichi/Chow/Pong/Kong models: 2-dim logits → softmax → P(action=yes) and P(action=no). Decision flow uses confidence score (max of the two).

---

## 4. Training Algorithm

### Step-by-Step Pipeline

**Phase 1: Supervised Learning** (§4.1, Table 3, p. 13)
1. Collect (state, action) pairs from top human professionals on Tenhou.
2. For each of 5 models:
   - Discard model: 15M training samples; 10K validation, 50K test → 76.7% accuracy.
   - Riichi model: 5M training samples → 85.7% accuracy.
   - Chow model: 10M training samples → 95.0% accuracy.
   - Pong model: 10M training samples → 91.9% accuracy.
   - Kong model: 4M training samples → 94.0% accuracy.
3. Train via cross-entropy loss: minimize − log π_θ(a | s) where a is human action.
4. Output: Supervised discard/Riichi/Chow/Pong/Kong models.

**Phase 2: Self-Play RL** (§3 + §4.2, p. 8–15)

**High-level loop**:
```
Initialize policy π_θ with SL models (discard trained; others frozen at SL)
Repeat until convergence:
  1. Generate trajectories via self-play (Figure 6, p. 9)
     - Spin up multiple self-play workers, each runs M Mahjong simulators
     - Each simulator plays 1 game with Suphx (all 4 positions) vs. current π_θ
     - Collect (s, a, r, s') tuples for discard model training
  2. Compute rewards and advantages:
     - For each round in trajectory, use GRP predictor Φ to estimate reward
       r_k ← Φ(x_k) - Φ(x_{k-1})  [reward for round k]
     - Compute advantage A(s,a) ≈ r_k + γ V(s') - V(s) [Generalized Advantage Estimation]
       [The paper uses π-based advantages; no explicit value network described]
  3. Update discard policy (§3.1, Eq. 1–3):
     - Use importance sampling to correct for stale trajectories
     - Loss: L(θ) = E[ (π_θ(a|s) / π_{θ0}(a|s)) * A(s,a) + α * ∇H(π_θ) ]
     - Dynamically adjust entropy coefficient α to maintain target entropy
  4. Update parameter server, inference engines fetch new π_θ every ~1000 mini-batches

Separately:
  - Train GRP predictor Φ on human logs (one-time; not updated during RL)
  - Gradually transition from oracle agent to normal agent via feature dropout (§3.3)
```

**Phase 2a: Global Reward Prediction (GRP) Training** (§3.2, Eq. 4, p. 10–11)
- **GRP model**: 2-layer GRU (recurrent) + 2 fully-connected layers (Figure 7, p. 11).
- **Input at round k**: x_k = (round score at k, accumulated score, dealer pos, repeat dealer count, Riichi bet count).
- **Target**: R_i = final game reward of game i (determined by final rank).
- **Loss**: $\min_Φ \frac{1}{N} \sum_{i=1}^{N} \frac{1}{K_i} \sum_{j=1}^{K_i} (Φ(x^i_1, ..., x^i_j) - R_i)^2$
  - Trained on logs of top human players; frozen during RL.
  - Predicts: given rounds 1…k and current accumulated score, what is my likely final rank reward?

**Phase 2b: Oracle Guiding** (§3.3, Eq. 5, p. 11–12)
- **Oracle agent**: Trained oracle has access to:
  - Normal features: own tiles, public discards, dealer, scores.
  - Perfect features: opponents' private hands, wall deck.
- **Training oracle**: RL training on normal + perfect features → oracle becomes superhuman.
- **Transitioning to normal agent**:
  ```
  t ← 0; γ_0 ← 1; (initialization)
  While γ_t > ε:
    δ_t ← Bernoulli random matrix, P(δ_t(i,j) = 1) = γ_t
    x(s) ← [x_n(s), δ_t ⊙ x_o(s)]  [concatenate normal + dropped-out perfect features]
    Update π_θ with policy gradient using x(s)
    γ_t ← γ_t * (decay factor)  [gradually decrease γ]
  
  After γ_t = 0:
    # Fine-tune on normal features only
    Decay learning rate by 10×
    Reject state-action pairs with importance weight > threshold
    Continue training for N iterations
  ```
  - **Intuition**: Oracle learns to exploit perfect info. As dropout increases, it learns to compensate using normal features. Eventually, it becomes a normal agent that learned from oracle's experience.
  - **Why not direct distillation?** Simple KL-divergence distillation failed; normal agent too weak to copy oracle's policies. Gradual feature dropout allows incremental transfer.

**Phase 2c: Parametric Monte-Carlo Policy Adaptation (pMCPA)** (§3.4, Eq. 6, p. 12–13)
[Tested offline; not deployed online due to time constraints.]
- **At round start**: Given fixed private hand h (dealt to our agent).
- **Procedure**:
  1. **Simulation**: Randomly sample opponent hands + wall tiles K times (excluding our hand). Use offline π_θo to roll out to round end. Collect K trajectories τ ∼ π_θo.
  2. **Adaptation**: Finetune π via:
     $$θ_a = \arg\max_θ \sum_{τ ∈ T(h)} \frac{p(τ; θ)}{p(τ; θ_o)} R(τ)$$
     - Importance-sampled likelihood ratio for trajectories starting with hand h.
  3. **Inference**: Use finetuned π_a to play this round.
- **Why effective**: Policy is parametric (not a tree or table). K simulations update policy weights, which generalize to unseen states in the round.
- **Performance**: Tested on 100s of initial rounds; adapted policy wins 66% vs. non-adapted.

### Distributed Training System (Figure 6, p. 9)

**Architecture**:
- **Self-play workers**: M workers, each with:
  - CPU-based Mahjong simulators (one per game instance).
  - GPU-based inference engines (forward passes for π_θ).
- **Parameter server**: GPU cluster; maintains π_θ and replay buffer. Updates π_θ every ~1000 mini-batches based on collected trajectories.
- **Update loop**: Simulators query inference engines → engines fetch latest π_θ from parameter server → generate trajectories → store in replay buffer → parameter server updates π_θ.
- **Asynchrony**: Trajectories are stale by ~1000 mini-batches; corrected via importance sampling.

---

## 5. Mathematics

### Equation 1: Importance-Sampled Policy Gradient (§3.1, p. 8)

$$L(θ) = \mathbb{E}_{s,a ∼ π_{θ_0}} \frac{π_θ(a|s)}{π_{θ_0}(a|s)} A^{π_θ}(s, a)$$

**Symbols**:
- $θ_0$: parameters of the *old* policy (used to generate trajectories; stale by ~1000 steps).
- $θ$: parameters of the *latest* policy to update.
- $π_θ(a|s)$: probability of action a under policy θ at state s.
- $A^{π_θ}(s, a)$: advantage function; estimated as $r_t + \gamma V(s_{t+1}) - V(s_t)$ or $r_t - b(s)$ (baseline).
- $\mathbb{E}_{s,a ∼ π_{θ_0}}$: expectation over trajectories generated by old policy.

**Intuition**: Standard policy gradient scales gradients by advantage. Importance sampling (ratio $π_θ / π_{θ_0}$) corrects for the fact that trajectories were generated by old policy but are used to train new policy. In distributed systems, this ratio prevents instability when θ drifts far from θ_0.

**Why needed**: Asynchronous distributed training means gradient updates happen before new trajectories are collected. Importance sampling is the standard fix (used in IMPALA, other distributed RL).

**Label**: **MUST UNDERSTAND** — core policy gradient method; see RL textbooks (Sutton & Barto, Spinning Up) for derivation.

---

### Equation 2: Policy Gradient with Entropy Regularization (§3.1, p. 9)

$$\nabla_θ J(π_θ) = \mathbb{E}_{s,a ∼ π_{θ_0}} \frac{π_θ(s,a)}{π_{θ_0}(s,a)} \nabla_θ \log π_θ(a|s) A^{π_θ}(s,a) + α \nabla_θ H(π_θ)$$

**Symbols**:
- $H(π_θ)$: entropy of policy, $H(π_θ) = -\sum_a π_θ(a) \log π_θ(a)$.
- $α > 0$: coefficient (trade-off between exploitation and exploration).
- $\nabla_θ H(π_θ)$: gradient of entropy w.r.t. θ (pushes policy toward higher entropy, i.e., less committed).

**Intuition**: 
- First term: standard policy gradient (maximize expected return).
- Second term: entropy bonus. Encourages the policy to remain stochastic; prevents collapse to deterministic policy too early.
- Why needed in Mahjong: Exploration is crucial; imperfect info → must explore to learn hidden card distributions. If entropy too low, RL converges without improving. If entropy too high, noisy actions and instability.

**Label**: **MUST UNDERSTAND** — entropy regularization is standard in RL (A3C, PPO, etc.).

---

### Equation 3: Adaptive Entropy Coefficient (§3.1, p. 9)

$$α ← α + β (H_{target} - \bar{H}(π_θ))$$

**Symbols**:
- $\bar{H}(π_θ)$: empirical entropy of trajectories in recent period (e.g., last batch).
- $H_{target}$: target entropy (hyperparameter).
- $β > 0$: small step size (e.g., 0.01).

**Intuition**: Dynamically adjust α to keep entropy near target. If empirical entropy drops below target, increase α to encourage exploration. Opposite if too high.

**Why needed**: Fixed α = hard to set. Adaptive α responds to actual policy entropy, ensures stable training.

**Label**: **MUST UNDERSTAND** — standard technique (used in SAC, etc.).

---

### Equation 4: Global Reward Prediction Loss (§3.2, p. 10)

$$\min_Φ \frac{1}{N} \sum_{i=1}^{N} \frac{1}{K_i} \sum_{j=1}^{K_i} (Φ(x_1^i, ..., x_j^i) - R_i)^2$$

**Symbols**:
- $N$: number of games in training data (human logs).
- $K_i$: number of rounds in game i.
- $x_j^i$: feature vector of round j in game i (round score, accumulated score, dealer, etc.).
- $R_i$: final game reward (rank-based: 1st place = +60, 4th place = -165, etc.).
- $Φ$: GRU + 2-layer FC network (reward predictor).

**Intuition**: Train Φ to predict final game reward from partial game state (rounds 1…j). Given rounds 1…j and accumulated score, what is my expected final reward?

**Why needed**: Mahjong games have 8–12 rounds. Round score at round j may not reflect skill (tactical losses happen). But final reward depends on all rounds. Φ enables per-round reward signal: $\hat{r}_j = Φ(x_1, ..., x_j) - Φ(x_1, ..., x_{j-1})$.

**Bootstrapping**: No. Φ is trained on supervised data (human logs); not bootstrapped during RL. It is deterministic, frozen after SL training.

**Label**: **MUST UNDERSTAND** — key innovation for credit assignment in multi-round MDPs with imperfect immediate rewards.

---

### Equation 5: Oracle Guiding Loss with Feature Dropout (§3.3, p. 11–12)

$$L(θ) = \mathbb{E}_{s,a ∼ π_{θ_0}} \frac{π_θ(a|[x_n(s), δ_t ⊙ x_o(s)])}{π_{θ_0}(a|[x_n(s), δ_t ⊙ x_o(s)])} A([x_n(s), δ_t ⊙ x_o(s)], a)$$

where $δ_t$ is a dropout matrix with $P(δ_t(i,j) = 1) = γ_t$, and $γ_t$ decays from 1 to 0.

**Symbols**:
- $x_n(s)$: normal features (observable: own hand, public discards, scores).
- $x_o(s)$: oracle features (hidden: opponents' hands, wall tiles).
- $δ_t ⊙ x_o(s)$: element-wise product; masks out oracle features according to Bernoulli matrix.
- $[x_n(s), δ_t ⊙ x_o(s)]$: concatenation of normal and masked oracle features.
- $γ_t$: dropout probability (keep rate) at iteration t. $γ_0 = 1$ (start with perfect info), $γ_∞ = 0$ (end with no perfect info).

**Intuition**: 
1. Train oracle from iteration 0 with γ_0 = 1 (all perfect features available). Oracle learns to exploit perfect info.
2. At iteration t, corrupt oracle features via dropout mask δ_t. Oracle must learn to compensate with normal features.
3. Gradually decay γ_t (more dropout, fewer perfect features). Eventually, oracle becomes a normal agent that learned from oracle's knowledge.
4. After γ_t = 0, continue training with learning rate decay and importance-weight thresholds for stability.

**Why this works**: 
- Gradual transition is smoother than abrupt removal (would cause catastrophic forgetting).
- Oracle's learned representations (parameters) encode both perfect and normal feature usage; as dropout removes perfect features, normal features are "promoted" to carry the same information.
- More efficient than standard distillation: oracle agent and normal agent share parameters (no separate student/teacher overhead).

**Comparison to knowledge distillation**: Direct KL distillation (minimize KL(π_oracle || π_normal)) failed because π_normal cannot match π_oracle's actions without perfect info. Gradual feature dropout allows incremental learning.

**Label**: **MUST UNDERSTAND** — novel technique in this paper; generalization of privileged information distillation. Key innovation for Dou Dizhu applications.

---

### Equation 6: Parametric Monte-Carlo Policy Adaptation (pMCPA) (§3.4, p. 12)

$$θ_a = \arg\max_θ \sum_{τ ∼ θ_o} \frac{p(τ; θ)}{p(τ; θ_o)} R(τ)$$

where $τ ∈ T(h)$ (set of trajectories with prefix h, our private hand).

**Symbols**:
- $θ_o$: offline-trained policy (frozen).
- $θ_a$: adapted policy parameters (optimized for current hand h).
- $T(h)$: set of K trajectories, each starting with our private hand h, rest sampled.
- $τ = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T, r_T)$: trajectory (sequence of states, actions, rewards).
- $p(τ; θ)$: probability of generating trajectory τ under policy θ. $p(τ; θ) = \prod_{t=0}^{T} π_θ(a_t | s_t)$.
- $R(τ)$: return (cumulative reward of trajectory).
- Importance ratio: $p(τ; θ) / p(τ; θ_o)$.

**Intuition**: 
- Generate K simulated rollouts of the round, using our fixed hand h and random opponent hands/wall.
- Maximize importance-weighted return: trajectories more likely under θ than θ_o are rewarded.
- This is a one-step policy gradient or likelihood-ratio update.

**Why parametric?** 
- Policy θ is parametric (neural network weights). One-step gradient on K simulations updates weights, which generalize to unseen states in the round.
- Contrast: tabular or tree-based methods would need to visit each state multiple times to get robust estimates.

**Computational cost**: 100K trajectories × forward passes + gradient step per round. Too expensive for online play on Tenhou (time-constrained). Tested offline on 100s of hands; 66% win rate improvement.

**Label**: **OPTIONAL / ADVANCED** — clever idea, but not deployed in final system. Useful for understanding policy adaptation without tree search.

---

## 6. Loss Function

### Supervised Learning Phase

**Loss**: Cross-entropy (standard classification)
$$L_{SL} = -\sum_{i} \log π_θ(a_i | s_i)$$
where $(s_i, a_i)$ are (state, human action) pairs from Tenhou logs.

**Target**: Human action a_i (one-hot label).

**Why**: Each model (discard, Riichi, etc.) is trained as a classifier: given state, predict action probability. Human action is ground truth.

---

### Reinforcement Learning Phase

**Loss**: Policy gradient with entropy regularization (Eq. 2)
$$L_{RL} = -\mathbb{E}_{s,a ∼ π_{θ_0}} \left[ \frac{π_θ(s,a)}{π_{θ_0}(s,a)} A(s,a) - α H(π_θ) \right]$$

**Components**:
1. **Main term**: Importance-sampled policy gradient. Maximizes expected advantage-weighted log probability.
2. **Entropy term**: Encourages stochasticity. Coefficient α is adaptive (Eq. 3).

**Target**: No explicit target; reward-based. Per-round reward is $\hat{r}_j = Φ(x_1, ..., x_j) - Φ(x_1, ..., x_{j-1})$ from GRP.

**Advantage estimation**: The paper does not specify the exact bootstrapping scheme. [**Interpretation**: Likely GAE (Generalized Advantage Estimation) or n-step returns, common in policy gradient methods. Could use round reward as immediate reward, discount future rounds.]

**Bootstrapping?** No explicit value network mentioned. Advantage likely estimated from trajectory returns (Monte Carlo) or with baseline subtraction (state value estimated from GRP predictor? unclear from paper).

**Multiple losses?**
1. **Supervised learning loss** (phase 1).
2. **RL policy loss** (phase 2, Eq. 2).
3. **GRP loss** (Eq. 4, trained once on human logs, not updated).
4. **Entropy regularization** (Eq. 3, adaptive).
5. **Importance weight clipping** (only during oracle-to-normal transition, for stability).

**Why these losses?**
- **Supervised**: Warm-start from human behavior; avoids random initialization.
- **RL policy gradient**: Maximizes expected return; standard in policy gradient.
- **Entropy**: Maintain exploration; prevent premature convergence.
- **GRP**: Provides per-round reward signals instead of misleading round scores.
- **Feature dropout**: Gradually transfers oracle knowledge to normal agent.

---

## 7. Self-Play

### Who Plays Whom

In each self-play game:
- Player 0 (our agent): π_θ (latest training policy).
- Players 1, 2, 3 (opponents): π_θ (three clones of our agent).
- All four positions share the same policy network (parameter sharing). However, each player has distinct observations (own hand, position relative to dealer, etc.).

### Simultaneous vs. Sequential Updates

**Sequential updates**:
- One policy π_θ is maintained on a parameter server.
- All players use the same π_θ.
- After ~1000 mini-batches, parameter server updates π_θ; inference engines fetch new π_θ.
- Trajectory generation and parameter updates are *decoupled* (asynchronous). No simultaneous updates of all players' policies mid-game.

### Parameter Sharing

**Full parameter sharing**: All four positions (dealer, discard order, etc.) use the same network weights. Position is encoded in the state (features include dealer position, round ID, etc.), but network weights are shared.

**Why**:
- Reduces training variance (more diverse experience per update).
- Mirrors real self-play: each copy of the policy improves together.
- Symmetry: all positions have similar game structure; network can learn generalizable features.

### Checkpoints / Old Policies

The paper does not explicitly mention old policy checkpoints or opponent pools. [**Interpretation**: All opponents are clones of π_θ, not fixed old checkpoints. This differs from AlphaGo, which maintains an opponent pool. Non-stationarity is handled via importance sampling (Eq. 1) and entropy regularization (Eq. 2–3).]

### Exploration

**Policy exploration** (stochasticity):
- Entropy regularization (Eq. 2–3) keeps policy stochastic; encourages visiting diverse actions.
- Adaptive α prevents entropy collapse.

**Environment exploration**:
- Self-play opponent randomness: three clones of π_θ are slightly stochastic (due to entropy), so different games see different opponent behaviors.
- Randomization of deck/hand dealt (natural randomness in Mahjong).

### Non-Stationarity

**Challenge**: As π_θ improves, opponents (three clones of π_θ) also improve. This is the standard non-stationarity problem in self-play.

**Handling**:
- **Importance sampling** (Eq. 1): Corrects for trajectory staleness. Trajectories generated by π_{θ_0} are used to update π_θ; ratio π_θ / π_{θ_0} normalizes the mismatch.
- **Entropy regularization**: Keeps policy stochastic; adapts to changing opponent strategies.
- **Replay buffer**: Mixes old and new trajectories, smoothing updates.

[**Note**: The paper does not explicitly discuss self-play non-stationarity or convergence guarantees. This is a limitation discussed in section 12B.]

---

## 8. Imperfect Information

### Exactly How Hidden Information is Handled

**Actually Observed by Agent** (Observable State):
1. Own private tiles (13 per round).
2. Open melds (Pong, Kong, Chow from discarded tiles) — all players' open melds.
3. Doras (bonus tiles drawn at round start) — public.
4. Sequence of all discarded tiles (by all players, in order) — fully public.
5. Accumulated round scores of all 4 players (public after each round).
6. Dealer position, repeat dealer counter, Riichi bets (public).
7. Number of tiles left in wall.
8. Round ID.

**Hidden** (Not Observable):
1. Other players' private tiles (initially 13, shrink as melds/discards happen). **Crucial**: Even after melds, private tiles remain hidden (only melds are exposed).
2. Wall tiles (70 tiles; 14 in dead wall never seen). Tiles drawn become observable via discard, but undrawn tiles are unknown.
3. Dora tiles under dora indicator tile (drawn only if Kong happens).

### Information Set and Hidden State Scale

**Information set (decision point)**: At any state where our agent acts, there are **>10^48 hidden states** indistinguishable from our perspective (§1, p. 2). This is the product of:
- Opponent 1's possible hands: C(21, 13) (13 tiles from 21 known undrawn tiles)
- Opponent 2's possible hands: C(8, 13) (after removing opp1's tiles)
- Opponent 3's possible hands: C(8, 13)
- Wall tiles: 70! / (composition of remaining tiles)

Scale is **vastly larger** than Texas hold'em (~10^13 information sets).

### Belief States and Card Inference

The paper does not use explicit belief state representations (e.g., Bayesian update over opponent hands). [**Interpretation**: Agent implicitly learns to infer opponent hands from:
- Discard sequence (tiles discarded = tiles opp does NOT have, or had and chose not to keep).
- Meld patterns (open Pong of 5M means opp discarded 5M or drew it; has another 5M or two).
- Riichi declarations (opp is one tile from winning; constrains hand structure).
- Tile counts (if 3 of 4 copies of a tile have been discarded, one opp likely has the 4th, etc.)

The CNN learns these inferences implicitly from (state, action) pairs. No explicit probabilistic model; end-to-end learning.]

### Recurrent Models

The paper does not mention LSTM or GRU for history encoding **in the policy network**. [**Interpretation**: CNN takes concatenated state features (current hand, public discards, scores, etc.); no temporal recurrence. However, the GRP predictor Φ uses GRU (Eq. 4, §3.2) to aggregate multi-round info for reward prediction. This is separate from the policy network.]

### Action History and Sequences

State features include:
- **Discard sequence** (all tiles discarded so far, in order).
- **Meld history** (open melds, exposed to all).
- **Accumulated scores** (updated each round).

These are encoded into the feature maps (e.g., one channel per historical discard). CNN processes this via convolutions, which aggregate nearby tiles.

### Explicit Hidden Card Inference

The paper does not mention explicit opponent hand distributions (e.g., belief state vector). [**Interpretation**: Inferences are implicit in the learned representations. Oracle guiding (Eq. 5) explicitly provides perfect info during training, which likely accelerates learning of these inferences. Oracle agent "sees" opponent hands directly; as feature dropout increases, it learns to infer from clues.]

---

### Imperfect Information Handling: Two-Column Table

| **Actually Observed by Agent** | **Hidden from Agent** |
|---|---|
| Own private tiles (13) | Opponents' private tiles (13 each) |
| Open melds by all players (Pong, Kong, Chow) | Private melds (concealed from other players) |
| Dora tile (drawn at round start) | Tiles under dora indicator (only drawn if Kong) |
| Discard sequence (all tiles, all players) | Undrawn wall tiles (70) |
| Accumulated round scores (all players) | Hidden strategicintent of opponents |
| Dealer position, counters, Riichi bets | True hand strength / winning probability of opponents |
| Round ID, tiles left in wall | Precise composition of remaining deck |

**How agent bridges the gap**:
- CNN learns implicit card inference from (state, history) pairs.
- Oracle guiding (training with perfect info, then dropping it) accelerates learning of inferences.
- Look-ahead features (computed hand probabilities) provide domain knowledge.
- Importance sampling (Eq. 1) and entropy regularization allow exploration to discover patterns in hidden info.

---

## 9. Evaluation

### Baselines

| **Baseline** | **Type** | **Details** | **Performance** |
|---|---|---|---|
| **SL (Supervised Learning)** | Agent | Discard/Riichi/Chow/Pong/Kong models trained on human logs only (no RL). | Baseline |
| **SL-weak** | Agent | Under-trained version of SL; used as opponent in offline eval to reduce variance. | Weaker opponent |
| **RL-basic** | Agent | Discard model improved via RL with round score as reward; others frozen at SL. | Intermediate |
| **RL-1** | Agent | RL-basic + GRP (global reward prediction). | Intermediate |
| **RL-2** | Agent | RL-1 + Oracle guiding. | Best offline |
| **Bakuuchi** | AI | MCTS + opponent modeling (University of Tokyo, 2015). No deep RL. | Baseline AI (~6.6 dan) |
| **NAGA** | AI | Deep CNN (Dwango Media Village). No RL. | Baseline AI (~6.6 dan) |
| **Top human players** | Human | 10 dan record rank, playing in Tenhou expert room. | Human baseline (7.46 dan stable rank) |
| **Suphx (online)** | Agent | RL-2 trained with ~2.5M games. Deployed on Tenhou. | **8.74 stable rank** |

### Evaluation Metrics and Methods

#### Offline Evaluation (§4.2, p. 14–16)

**Metric: Stable Rank**
- Computed from game outcomes (final rank: 1st, 2nd, 3rd, 4th) over many games.
- Formula (Appendix C, Eq. 7, p. 28): 
  $$\text{Stable Rank} = 5 × n_1 + 2 × n_2 - 2 × (n_4 / \text{total games})$$
  where n_1 = # of 1st places, n_2 = # of 2nd places, n_4 = # of 4th places.
- Rationale: 1st place = high performance. 4th place = poor performance. Punishes bad finishes heavily.
- **Why stable rank, not win rate?** Mahjong is not purely zero-sum; final rank depends on cumulative score across rounds. Win rate (single-round wins) is not a good metric for overall strength.

**Experiment 1: Offline Ablation (Figure 8, p. 15)**
- **Agents tested**: SL, RL-basic, RL-1 (RL-basic + GRP), RL-2 (RL-1 + oracle guiding).
- **Opponent**: SL-weak (fixed under-trained agent; reduces variance).
- **Setup**: 1 million games (random seeds, random deck shuffles).
- **Metric**: Stable rank, reported as interquartile range (IQR) box plot over 1000 samplings of 800K games each.
- **Results** (from Figure 8, p. 15):
  - SL: ~5.5 stable rank (IQR ≈ 0.3).
  - RL-basic: ~5.9 stable rank (improvement over SL).
  - RL-1 (+ GRP): ~6.3 stable rank (further improvement).
  - RL-2 (+ oracle guiding): ~6.7 stable rank (best).
  - **Conclusion**: Each component (RL, GRP, oracle guiding) adds value. Oracle guiding provides biggest marginal gain.

**Training cost**: 1.5M games, 44 GPUs (4 Titan XP for parameter server + 40 Tesla K80 for workers), 2 days per agent.

#### Online Evaluation (§5, p. 17–20)

**Platform**: Tenhou.net (most popular Japanese Mahjong platform, 350K+ active users, competitive ranking).

**Experiment 2: Suphx on Tenhou Expert Room**
- **Setup**: Suphx deployed in expert room (4-dan and above only).
- **Games**: 5,000+ games played by Suphx.
- **Achievements**:
  - Record rank: 10 dan (highest rank; first AI to achieve this).
  - Stable rank: 8.74 dan (§5, Table 4, p. 19).

**Experiment 3: Comparison with Other AIs and Top Humans (Table 4, p. 19)**

| **Player** | **Type** | **# Games** | **Record Rank** | **Stable Rank** |
|---|---|---|---|---|
| Bakuuchi | AI | 30,516 | 9 dan | 6.59 dan |
| NAGA | AI | 9,649 | 8 dan | 6.64 dan |
| Top human | Human | 8,031 | 10 dan | 7.46 dan |
| **Suphx** | **AI** | **5,760** | **10 dan** | **8.74 dan** |

**Interpretation**:
- Suphx record rank (10 dan) = same as top humans, but stable rank (8.74) > human stable rank (7.46). Stable rank is more reliable long-term measure.
- Suphx ~1.28 dan stronger than top humans in stable rank.
- Suphx ~2.1 dan stronger than best prior AI (Bakuuchi/NAGA).

**Experiment 4: Statistical Significance Testing (Figure 12, p. 20)**
- **Method**: For each AI/human player, randomly sample K = 2000 games from Tenhou logs, compute stable rank. Repeat N = 5000 times to get distribution of stable ranks.
- **Result**: Suphx's mean stable rank and confidence interval (interquartile range) clearly above top humans and other AIs (Figure 12).
- **Conclusion**: Difference is statistically significant (no overlap of IQRs).

**Experiment 5: Playing Style Analysis (Table 5, p. 21)**

| **Metric** | **Bakuuchi** | **NAGA** | **Top Human** | **Suphx** |
|---|---|---|---|---|
| 1st rank rate | 28.0% | 25.6% | 28.0% | **29.3%** |
| 2nd rank rate | 26.2% | 27.2% | 26.8% | 27.5% |
| 3rd rank rate | 23.2% | 25.9% | 24.7% | 24.4% |
| 4th rank rate | 22.4% | 21.1% | 20.5% | **18.7%** |
| Win rate | 23.07% | 22.69% | 22.83% | — |
| Deal-in rate | 12.16% | 11.42% | 10.06% | — |

**Interpretation**:
- Suphx has the **lowest 4th-place rate** (18.7% vs. ~20-22% for others). This is the key to high stable rank; Tenhou scoring heavily punishes 4th place.
- Suphx has **highest 1st-place rate** (29.3%), tied or better than top humans.
- Deal-in rate (losing by discard) not reported for Suphx online, but Tenhou players note it is very low.
- **Conclusion**: Suphx plays **defensively** (minimizes 4th place risk) and **safely** (low deal-in rate), which maximizes stable rank.

#### Critical Assessment of Evaluation

**Strengths**:
1. **Large sample sizes**: Offline experiments: 1M games. Online: 5K+ games (large for an AI benchmark). Stable rank computed from 800K games per snapshot (1000 snapshots).
2. **Stable rank is appropriate metric**: Reflects long-term skill in Tenhou's multi-round setting. Not just win rate.
3. **Multiple comparisons**: Suphx vs. two other AIs, vs. top human cohort (aggregated to reduce variance), vs. SL-weak baseline, vs. components (ablation).
4. **Statistical rigor**: IQR box plots (Figure 12) show distributions, not just point estimates. No claim of statistical significance without evidence.
5. **Cross-role evaluation**: N/A (all games played by same 4-player configuration). But Tenhou randomly assigns players to positions, so symmetry is implicit.

**Weaknesses**:
1. **Small Suphx online sample**: Only 5,760 games (vs. 30K+ for Bakuuchi, 8K+ for top humans). Stable rank from 5.76K games has higher variance than from 30K. Authors mitigate via IQR analysis (Figure 12), but uncertainty remains.
2. **Top human cohort is aggregated**: Cannot compare Suphx directly to a single human player (all top 10-dan players have different numbers of games; unstable rank). Aggregating all top humans may average out individual strengths/weaknesses. [**Interpretation**: Reasonable given data scarcity, but it's a limitation.]
3. **No ablation of pMCPA in online setting**: pMCPA is mentioned as *not deployed* due to time constraints (§5, footnote 8, p. 17). Authors claim it would improve performance ("integration will further improve"), but no online evidence.
4. **Opponent pool not varied**: All online opponents are human players of 4-dan and above. Performance on weaker opponents not tested online (only offline with SL-weak). Possible Suphx is overfit to expert-level play.
5. **No confidence intervals reported for stable rank of online agents**: Figure 12 shows distributions, but exact mean ± CI not stated. Unclear if Suphx's improvement over humans is robust to different game samples.
6. **No statistical test of significance**: IQR plots are visual; no formal hypothesis test (e.g., Welch's t-test, Mann-Whitney U) is reported. (Authors state Suphx "surpasses with a big margin", but this is qualitative.)

**What I would improve for a CS230 project**:
1. **Ensemble evaluation**: Train multiple Suphx variants (different random seeds, hyperparameters) and report mean ± std of stable rank.
2. **Cross-play analysis**: Evaluate Suphx against top humans in separate games; analyze win/loss records head-to-head.
3. **Ablation of online pMCPA**: Deploy pMCPA and measure improvement.
4. **Opponent diversity**: Test Suphx against weak (1-dan), intermediate (5-dan), and expert (10-dan) opponents (if accessible).
5. **Detailed loss functions**: Analyze training curves of SL → RL-basic → RL-1 → RL-2 to identify which component drives the biggest improvement.

---

## 10. Experiments and Ablations

### Experiment 1: Supervised Learning (§4.1, Table 3, p. 13–14)

| **Model** | **Training Data Size** | **Test Accuracy** | **Previous Work** | **Notes** |
|---|---|---|---|---|
| Discard | 15M samples | 76.7% | 68.8% (Gao et al. 2018) | 34-class classification. Hardest task (lowest accuracy). |
| Riichi | 5M samples | 85.7% | 90.4% (Gao et al. 2018) | Binary decision (declare Riichi or not). Higher accuracy than prior work (Suphx's larger model?). |
| Chow | 10M samples | 95.0% | 88.2% (Gao et al. 2018) | Binary decision (make Chow or not). |
| Pong | 10M samples | 91.9% | — | Binary decision (make Pong or not). |
| Kong | 4M samples | 94.0% | — | Binary decision (make Kong or not). |

**What changed**: Each model trained independently via cross-entropy loss on human (state, action) pairs.

**What held constant**: Architecture (CNN with residual blocks), training/validation/test split (validation: 10K, test: 50K samples).

**Metric**: Accuracy (% of predictions matching human action).

**Result**: All models achieve high accuracy (76.7%–95%). Discard has lowest accuracy because it's a 34-class problem; others are binary.

**Conclusion**: Supervised learning is a strong warm-start. Agent learns from human behavior without RL.

---

### Experiment 2: RL Ablation (§4.2, Figure 8, p. 14–16)

| **Agent** | **Components** | **Stable Rank** | **Training** | **Notes** |
|---|---|---|---|---|
| SL | Supervised learning only. | ~5.5 ± 0.3 IQR | — | Baseline; warm-start only. |
| RL-basic | SL discard + RL with round score reward + entropy regularization. | ~5.9 ± 0.2 IQR | 1.5M games, 44 GPUs, 2 days | Discard model trained via RL; Riichi/Chow/Pong/Kong frozen at SL. Modest improvement. |
| RL-1 | RL-basic + GRP (global reward prediction). | ~6.3 ± 0.2 IQR | Same | GRP trained on human logs; rewards computed as Φ(x_k) - Φ(x_{k-1}). Larger improvement. |
| RL-2 | RL-1 + Oracle guiding. | ~6.7 ± 0.2 IQR | Same | Oracle trained with perfect features, gradually dropout to normal. Best offline performance. |

**What changed per step**: One component added each time.

**What held constant**: Opponent (SL-weak), number of training games (1.5M), training resources (44 GPUs, 2 days), seeds (fixed).

**Metric**: Stable rank (distribution over 1000 samplings of 800K games each). Reported as IQR (25th to 75th percentile).

**Results** (reading from Figure 8):
- SL → RL-basic: **+0.4 dan** improvement.
- RL-basic → RL-1 (+ GRP): **+0.4 dan** improvement.
- RL-1 → RL-2 (+ oracle guiding): **+0.4 dan** improvement.
- Total: SL to RL-2 = **+1.2 dan** (5.5 → 6.7).

**Conclusion**: Each component is valuable. GRP and oracle guiding are equally effective (both +0.4 dan).

**Authors' interpretation** (§4.2, p. 15): GRP enables the agent to play conservatively in the last round (Figure 9, p. 16), avoiding risky moves when leading. Without GRP, RL-basic tries to win every round (round-score-myopic), leading to worse overall rank.

---

### Experiment 3: Global Reward Prediction Validation (§4.2, Figure 9, p. 16)

**Example game**: Last round, agent (south player) has big lead. Winning this round gives marginal reward; losing gives big punishment.

**RL-basic behavior**: Tries to win the round (discard red-boxed tile, which is risky because few other discards were made).

**RL-1 + RL-2 behavior**: Play defensively (discard blue-boxed tile, which was just discarded by west, hence safe). Secure rank-1 overall.

**Outcome**: Conservative play → rank 1. Aggressive play → rank 2–3.

**Conclusion**: GRP enables rank-aware play (optimize game score, not just round score).

---

### Experiment 4: Oracle Guiding Ablation (implicit in RL-1 vs RL-2)

**What changed**: RL-2 = RL-1 + oracle guiding (gradual feature dropout from γ_t = 1 to γ_t = 0).

**Metric**: Stable rank on same 1M games.

**Result**: RL-1 (6.3 dan) → RL-2 (6.7 dan), **+0.4 dan** (6% improvement).

**Why is oracle guiding effective?**
- Oracle agent learns to exploit perfect info (opponents' hands). Gradual dropout forces it to learn how to infer hidden hands from public info.
- Compared to RL-1 (RL training from scratch on normal features), oracle guiding is a form of **transfer learning** or **curriculum learning** (easier task → harder task).
- Knowledge is baked into network parameters; feature dropout enables transfer without explicit distillation loss.

**Subcomponent**: Authors also mention (§6, Conclusion, p. 21) that alternative approaches exist:
- Simultaneous training of oracle + normal agent with KL constraint.
- Oracle critic (value function with perfect info) providing feedback to normal policy.
Neither was tested in this paper.

---

### Experiment 5: Run-Time Policy Adaptation (pMCPA) (§4.3, p. 16–17)

**Setup**:
- **Phase 1 (Simulation)**: Given fixed hand h, generate 100K random trajectories (opponent hands, wall tiles sampled; other players use offline π_θo).
- **Phase 2 (Adaptation)**: Finetune policy via policy gradient on these 100K trajectories (importance-sampled likelihood ratio, Eq. 6).
- **Phase 3 (Test)**: Play on 10K test games with same hand h, using adapted policy π_a.

**Metric**: Win rate of adapted policy vs. non-adapted policy.

**Result**: 66% win rate for adapted policy.

**Interpretation**: On average, adapting to initial hand improves performance. 66% win rate indicates ~33% improvement margin (not huge, but non-trivial given high baseline performance).

**Why not deployed online?**
- **Cost**: 100K simulations + gradient updates per round. Tenhou has strict time limits per action (~5 seconds?). pMCPA requires compute not available in live play.
- **Potential online gain**: Authors speculate adapting after each tile discard (not just at round start) could further improve; but even more expensive.

**Conclusion**: pMCPA is a valuable technique, but impractical for live deployment given time constraints. Tested offline on limited data (hundreds of rounds).

---

### Ablation Summary

**Key findings**:
1. **SL is necessary but insufficient**: 5.5 dan (SL) is weaker than 8.74 dan (RL-2 online). Human play requires more than supervised imitation.
2. **RL significantly improves**: +0.4 dan just from RL (RL-basic).
3. **GRP and oracle guiding are equally valuable**: Each +0.4 dan. Together, +0.8 dan from RL-1 to RL-2.
4. **pMCPA improves offline but not tested online**: 66% win rate improvement on limited sample.

---

### Experiments Reproducible at Small Scale for CS230

**Simplified version**:
1. **Supervised learning**: Train discard model on 1M human games (subset of 15M), test on 10K. Measure accuracy. (1 GPU, 1 hour.)
2. **RL-basic**: Initialize discard model from SL; RL with round score reward. Train for 100K games (vs. 1.5M). Compare stable rank vs. SL. (2 GPUs, 1–2 days.)
3. **RL with GRP**: Train GRP predictor on 100K human games. Use predicted reward in RL. (Same compute as RL-basic, but GRP training is one-time.)
4. **Simplified oracle guiding**: Train oracle with perfect features on 100K games, then dropout γ_t from 1 to 0 over 50K games. Measure stable rank. (2 GPUs, 2 days.)
5. **Evaluation**: Play 10K games vs. fixed SL-weak opponent. Measure stable rank and 1st/4th-place rates. Collect statistics via 100 samplings.

**Estimate**: 1–2 weeks on 1 GPU (sequential experiments) or 3–5 days on 2–4 GPUs (parallel). Much smaller than 1.5M games / 44 GPUs / 2 days.

---

## 11. Compute and Engineering

### Hardware

- **Distributed RL system** (§3.1, Figure 6, p. 9):
  - **Self-play workers**: CPU-based Mahjong simulators (count not specified; likely 10–50 per GPU node).
  - **Inference engines**: GPU-based forward passes. 40 Tesla K80 (training).
  - **Parameter server**: 4 Titan XP GPUs (for policy updates / gradient computation).
  - **Total**: 44 GPUs (40 K80 + 4 Titan XP).

- **Offline evaluation** (§4.2, p. 14):
  - 20 Tesla K80 GPUs for 2 days to evaluate one agent on 1M games.

### Training Duration and Scale

- **Supervised learning phase**: Not explicitly timed. Likely <1 day on 1 GPU (15M training samples / 5M for others, standard cross-entropy loss).
- **RL training phase** (per agent: RL-basic, RL-1, RL-2): 1.5 million games, 44 GPUs, 2 days. (§4.2, p. 14).
- **GRP training**: Trained once on human logs (~millions of games from Tenhou); timing not reported. Likely 1–2 hours on 1 GPU (supervised regression).
- **Oracle guiding**: Included in the 1.5M games / 2 days budget. Gradual feature dropout runs during training; no separate cost mentioned.

### Total Compute Budget

- **Phase 1 (SL)**: ~1 GPU-day.
- **Phase 2 (RL)**: ~44 GPU-days (44 GPUs × 2 days for training per agent variant; 4 variants trained, but likely sequential or with some parallelization).
- **Phase 3 (GRP)**: ~1 GPU-day (rough estimate; trained once on human logs, not updated during RL).
- **Evaluation (offline)**: The paper states "the evaluation of one agent took 20 Tesla K80 GPUs for two days" (§4.2, p. 14). With 4 agent variants evaluated (SL, RL-basic, RL-1, RL-2), this is potentially ~160 GPU-days total, but the paper does not specify whether evaluations were sequential or parallelized.
- **Evaluation (online)**: Tenhou deployment (compute cost: server-side; not included in paper).

**Total research compute**: Rough estimate. Training (1 + 44 + 1 = ~46) + offline evaluation (40–160, depending on parallelization) = **~90–210 GPU-days**. The range reflects uncertainty about offline evaluation parallelization. *Not* an exact figure from the paper.

### Distributed Training Engineering

**Key tricks** (§3.1, Figure 6, p. 9):
1. **Decoupled trajectory generation and parameter updates**: Self-play workers generate trajectories (CPU + inference GPU); parameter server updates policy on GPU (no blocking). Asynchronous reduces total wall-clock time.
2. **Importance sampling** (Eq. 1): Corrects for stale trajectories (generated by π_{θ_0}, used to train π_θ). Standard in distributed RL.
3. **Frequent policy fetches**: Inference engines pull new π_θ from parameter server every ~1000 mini-batches. Keeps trajectories ~1000 steps stale (good balance between staleness and freshness).
4. **Replay buffer**: Collects trajectories from workers; parameter server samples mini-batches. Allows offline-online RL (trajectories don't need to be processed immediately).

**Entropy regularization and feature dropout** (§3.1, §3.3):
- Adaptive α (Eq. 3): Stabilizes exploration during training. Prevents entropy collapse (which would break exploration).
- Feature dropout decay (Eq. 5): Gradual transition oracle → normal agent. No explicit trick needed; just Bernoulli mask + decay schedule.
- Importance-weight clipping during oracle transition: Reject (s,a) pairs with very large importance weights to prevent gradient explosion.

---

### Reproduction Difficulty: Moderate / Hard for CS230

**Why Moderate (could reproduce core algorithm)**:
1. Supervised learning from human logs is standard.
2. Policy gradient + importance sampling is textbook (covered in RL courses).
3. GRP is a simple supervised GRU (easy to implement).
4. Feature dropout is straightforward to code.
5. RLCard or DouZero codebase provides game environment (don't need to build Mahjong engine from scratch).

**Why Hard (reproducing full system)**:
1. **Distributed training complexity**: Figure 6 requires careful engineering of data pipelines, GPU/CPU coordination, parameter servers. Off-the-shelf libraries (Ray, Kubernetes, etc.) help but require expertise.
2. **Hyperparameter tuning**: Learning rate, entropy coefficient α, β, γ_t decay schedule, GRP architecture, etc. Paper does not specify all values. Likely requires extensive ablation.
3. **Compute**: 44 GPUs and 2 days is significant. Most CS230 students have access to 1–4 GPUs max. Scaling down to 1–2 GPUs would require adapting batch sizes, trajectory collection rates, etc.
4. **Data**: 1.5M games for RL training. Generating games via self-play is doable, but for comparison with humans, need access to Tenhou logs (likely not public; paper uses proprietary data from Tenhou developers).
5. **Evaluation on live platform**: Online evaluation on Tenhou requires account, player time, and results depend on human opponent skill. Not reproducible for a CS230 project.

**What a scaled-down reproduction would look like**:
- **Goal**: Reproduce core idea (SL → RL → GRP + oracle guiding) on simplified environment (e.g., 2-player Dou Dizhu or simplified Mahjong).
- **Timeline**: 1–2 weeks on 2–4 GPUs.
- **Data**: Use RLCard/DouZero self-play data (or generate via open-source simulators).
- **Metrics**: Stable rank on fixed opponent pool (not live platform).
- **Simplifications**:
  - Train on 100K games (vs. 1.5M).
  - Use 2–4 GPUs (vs. 44).
  - 5–10 days wall-clock (vs. 2 days).
  - Evaluate on self-play opponents or public benchmarks.

---

## 12. Limitations

### A. Explicitly Mentioned by Authors

1. **Global Reward Prediction** (§6, p. 21–22):
   - GRP currently uses limited input features (round score, accumulated score, dealer, counters, Riichi bets).
   - *Improvement idea*: Leverage perfect information (e.g., compare initial hand strengths of all players) to measure game difficulty. Reward difficult wins more than easy wins.
   - *Status*: Acknowledged as future work; not yet implemented.
   - *Citation*: "More information will lead to better reward signal." (§6, p. 21)

2. **Oracle Guiding** (§6, p. 21):
   - Current approach: Gradual feature dropout from oracle to normal agent.
   - *Alternative ideas*: (1) Simultaneously train oracle + normal agent with KL constraint. (2) Oracle critic (value function with perfect info) providing state-level feedback. (3) Leverage perfect information for more effective reward signals.
   - *Status*: Acknowledged; preliminary experiments suggest alternatives work "quite well," but not tested in final system.
   - *Citation*: "There could be other approaches to leverage the perfect information." (§6, p. 21)

3. **Run-Time Policy Adaptation (pMCPA)** (§6, p. 22–23):
   - Current approach: Adapt policy only at round start (one-time per round).
   - *Improvement idea*: Continue adaptation after each tile is discarded (more frequent updates). As game progresses, more information becomes observable; gradual adaptation can exploit this. Amortize computational cost over the round.
   - *Status*: Not implemented due to computational constraints and time limits on Tenhou.
   - *Citation*: "We did not integrate [pMCPA] into Suphx while testing on Tenhou.net since it is time consuming." (footnote 8, p. 17)

4. **General** (§6, p. 22):
   - Authors emphasize that despite strong performance, "there is still a lot of space for further improvement." (§6, p. 22)
   - Key challenges remain: multi-round scoring complexity, hidden information scale (>10^48 states), and irregular game tree.

---

### B. Inferred Limitations (from Methodology)

1. **Parameter Sharing in Self-Play**:
   - All four players use the same policy π_θ. This means the agent is learning a "symmetric" strategy that works from any position.
   - *Limitation*: Real Mahjong has position asymmetry (dealer has advantages: draws first, draws extra tile if Kong happens, higher scoring). By parameter sharing, Suphx may not fully exploit position-specific strategies.
   - *Impact*: Could slightly underperform if trained with position-specific policies.
   - *Citation*: §3.1, Figure 6 (not explicitly stated as limitation, but implicit).

2. **Opponent Pool is Not Diverse During RL**:
   - During self-play training, three opponents are clones of π_θ. No opponent pool of diverse strategies.
   - *Limitation*: Agent may overfit to exploiting a single strategy (its own old version). Real players have diverse styles; generalization to diverse opponents is not tested during training.
   - *Impact*: Potential for brittleness against novel strategies. However, online eval on Tenhou (diverse human players) provides some evidence of generalization.
   - *Citation*: §3.1, Figure 6 (implicit).

3. **Non-Stationarity Not Explicitly Addressed**:
   - As π_θ improves, self-play opponents (clones of π_θ) also improve. This is a moving target (non-stationary game).
   - *Limitation*: Paper does not discuss convergence guarantees or Nash equilibrium. Importance sampling (Eq. 1) helps, but theoretical guarantees are absent.
   - *Impact*: Unclear if RL training converges to a stable strategy or cycles.
   - *Citation*: §3.1 (importance sampling mentioned, but non-stationarity not explicitly discussed).

4. **Evaluation on Small Suphx Sample**:
   - Online evaluation: only 5,760 games. Stable rank from <6K games is less reliable than from 30K+ games (cf. Bakuuchi).
   - *Limitation*: High variance in stable rank. Statistical significance claims are based on IQR visual inspection, not formal tests.
   - *Impact*: Suphx's claimed superiority over humans may be overstated if Suphx had more games.
   - *Citation*: §5, Table 4, p. 19 (number of games listed).

5. **Oracle Guiding Requires Privileged Information at Training Time**:
   - Oracle guiding assumes access to perfect information (opponents' hands, wall) during training.
   - *Limitation*: In some domains, perfect information may not be available during training (e.g., real-world applications where you can't observe the true state). This limits transfer to other problems.
   - *Impact*: For Dou Dizhu, assuming access to opponents' hands during self-play training is reasonable (simulator can compute this). But for other games without this, oracle guiding may not apply.
   - *Citation*: §3.3 (oracle has access to private tiles, wall tiles).

6. **Look-Ahead Features Are Hand-Crafted**:
   - Look-ahead features (100+ features; probability/score of winning) are computed via domain-specific heuristics (DFS search).
   - *Limitation*: Not learned end-to-end. Requires domain expertise to design. Transferability to other games is limited.
   - *Impact*: If generalizing to other Mahjong variants or different games, these features would need to be redesigned.
   - *Citation*: §2.2, p. 6–7 (100+ look-ahead features via DFS).

7. **GRP Trained on Human Data, Not Updated During RL**:
   - GRP predictor Φ is trained once on human logs and frozen. It is not fine-tuned during RL.
   - *Limitation*: As RL policy diverges from human behavior, GRP may become less accurate for the RL agent's states. GRP is only as good as its training data (human games).
   - *Impact*: Potential distribution mismatch: human games vs. RL games. Reward signal may degrade as RL policy diverges.
   - *Citation*: §3.2, p. 11 (GRP trained on human logs; frozen).

---

## 13. Future Work

### A. Directly Mentioned by Authors

1. **Enhance Global Reward Prediction** (§6, Conclusion, p. 21–22):
   - *Idea*: Measure game difficulty (compare initial hand strength of all players) and weight rewards accordingly. Winning a difficult round should be rewarded more than winning an easy round.
   - *Rationale*: Current reward predictor ignores hand difficulty; could be more informative if hand difficulty is accounted for.
   - *Status*: "We are investing how to leverage prefect information..." (p. 21); ongoing research.
   - **Classification for CS230**: **Possible but difficult**. Requires careful design of hand strength metric; could introduce new biases or overfitting. Suitable for a focused project, but would need significant experimentation.

2. **Alternative Oracle Guiding Approaches** (§6, Conclusion, p. 21):
   - *Idea 1*: Simultaneously train oracle + normal agent with KL constraint (oracle distills to normal; normal learns from oracle).
   - *Idea 2*: Oracle critic — value function that has access to perfect information, provides state-level feedback to normal policy.
   - *Rationale*: Current approach (gradual feature dropout) is effective, but alternatives may be more elegant or efficient.
   - *Status*: "According to our preliminary experiments, this approach also works quite well." (p. 21); not fully developed.
   - **Classification for CS230**: **Directly usable as a project idea**. Comparing three oracle guiding approaches (dropout vs. KL-distillation vs. oracle critic) is a solid ablation study. Moderate difficulty (requires understanding of KL regularization, critic networks).

3. **Improve Run-Time Policy Adaptation (pMCPA)** (§6, Conclusion, p. 22–23):
   - *Idea 1*: Adapt policy after each tile discard (not just at round start). Gradual adaptation as more information becomes observable.
   - *Idea 2*: Amortize computational cost over the entire round (incremental updates per step) rather than one-time update at round start.
   - *Rationale*: As game progresses, hidden information decreases (more tiles are discarded, wall shrinks). Continuous adaptation can exploit this. One-time adaptation at round start only uses initial information.
   - *Status*: Not implemented due to time constraints on Tenhou.
   - **Classification for CS230**: **Possible but difficult**. Requires careful management of computational budgets. Online deployment would be tricky (time-limited moves). Good for a project if you have access to a local game environment.

4. **State-Level Reward Signal from Oracle Critic** (§6, Conclusion, p. 21–22):
   - *Idea*: Train an oracle critic V_o(s; perfect info) that provides state-level feedback, replacing or augmenting round-level rewards from GRP.
   - *Rationale*: More fine-grained reward signal; instant feedback per state, not per round.
   - *Status*: Mentioned as alternative; not yet tested in depth.
   - **Classification for CS230**: **Too ambitious** (would require retraining entire RL system with new reward structure; high risk).

---

### B. Implied Future Work [**Interpretation**]

1. **Opponent Modeling and Diversity** [**Interpretation**]:
   - Current RL uses only clones of π_θ as opponents. A more sophisticated approach would maintain an opponent pool of diverse strategies (old checkpoints, human-like agents, etc.).
   - *Why*: Reduce overfitting to a single strategy. Improve robustness to diverse opponents.
   - *Feasibility*: Well-established in multi-agent RL (e.g., AlphaGo's opponent pool). Would require storage and sampling overhead.
   - **Classification for CS230**: **Directly usable**. Implementing an opponent pool (keep 5–10 old policy checkpoints, sample uniformly during self-play) is a practical modification. Moderate difficulty.

2. **Position-Specific Policies** [**Interpretation**]:
   - Current architecture uses parameter sharing (all positions share weights). Could train separate policies per position (dealer vs. non-dealer) or per seat relative to dealer.
   - *Why*: Positions have asymmetric advantages (dealer draws first, higher scoring). Position-specific policies could be more effective.
   - *Feasibility*: Straightforward; just remove parameter sharing. Trade-off: more parameters to learn, more variance.
   - **Classification for CS230**: **Directly usable**. Simple modification with clear motivation. Ablation would show whether position matters.

3. **Real-Time Policy Distillation (during self-play)** [**Interpretation**]:
   - Current oracle guiding uses feature dropout (qualitative transition oracle → normal). Could also have an explicit distillation step: normal agent tries to mimic oracle's actions on subsets of features.
   - *Why*: Combine two training signals (RL reward + imitation of oracle) for potentially faster learning.
   - **Classification for CS230**: **Possible but difficult**. Requires balancing two loss functions; risk of one dominating the other.

4. **Evaluation Across Multiple Game Variants** [**Interpretation**]:
   - Paper focuses on Japanese Mahjong (Riichi). Could extend to other Mahjong variants (Cantonese, Chinese, etc.) or related games (Dou Dizhu).
   - *Why*: Demonstrates generality of techniques; applicability beyond Mahjong.
   - **Classification for CS230**: **Directly usable**. If you implement on Dou Dizhu, you'd naturally do this comparison.

---

### C. Connection to Dou Dizhu Papers in This Set

**Question**: Have any Dou Dizhu papers (DouZero, DouZeroPlus, DeltaDou) adopted oracle guiding, privileged information distillation, or similar ideas?

**Search results**:
- **DouZero (01)**: No mention of oracle guiding, privileged information, or perfect-information dropout. Uses self-play with parameter sharing (all three players share weights) and policy gradient (similar to Suphx), but no oracle guiding innovation.
- **DouZeroPlus (02)**: No mention of oracle guiding. Focuses on opponent modeling (training multiple agent types) and coach-guided learning (separate coach agent). Not a perfect-information dropout approach.
- **DeltaDou (04)**: No mention of oracle guiding. Focuses on expert-level play via self-play, but no privileged information distillation.

**Conclusion**: **Oracle guiding does not appear to be adopted by Dou Dizhu papers in this set.** [**Interpretation**: Suphx's oracle guiding technique is novel and has not yet influenced Dou Dizhu research (at least not in papers up to 2022). This is a potential research opportunity: applying oracle guiding to Dou Dizhu could be a direct contribution.

---

## 14. Connection to Other Papers

### Which Earlier Papers Suphx Builds On

**In this paper set**:
- **NFSP (05, 2016)** — Heinrich & Silver, Deep RL from Self-Play in Imperfect-Information Games:
  - Suphx uses similar self-play RL framework (policy gradient, parameter sharing, importance sampling for stale trajectories).
  - Suphx does not use CFR (counterfactual regret minimization) like NFSP does; focuses on policy gradient instead.
  - Connection: **Methodological foundation** (self-play in imperfect-info games).

- **Deep CFR (06, 2019)** — Brown et al., Deep Counterfactual Regret Minimization:
  - Suphx acknowledges Deep CFR as prior work but does not adopt it. Suphx avoids game tree search and CFR due to Mahjong's irregular game tree.
  - Connection: **Problem domain** (imperfect-information games, but Suphx uses different approach).

- **RLCard (07, 2020)** — Zha et al., A Toolkit for RL in Card Games:
  - RLCard provides infrastructure for imperfect-information card games. Suphx could potentially use RLCard (though paper was concurrent, so Suphx likely built custom environment).
  - Connection: **Tools and benchmarks** (though Suphx uses proprietary Tenhou logs, not RLCard).

**Outside this set**:
- **AlphaGo / AlphaZero** (Silver et al., etc.):
  - Tree search + deep learning. Suphx does not use MCTS (irregular game tree blocks it). Cites AlphaGo as related game AI work.
  - Connection: **Inspiration** (superhuman game AI) but different approach.

- **A3C, PPO, SAC** (distributed policy gradient, entropy regularization):
  - Suphx directly uses policy gradient (Eq. 1–2) with importance sampling and entropy regularization (Eq. 3). These are standard techniques.
  - Connection: **Core algorithm** (policy gradient RL).

---

### What Suphx Keeps and What It Changes

| **Component** | **Kept From** | **Changed / Added** | **Why** |
|---|---|---|---|
| **Self-play RL** | NFSP, standard RL | Same core (policy gradient + importance sampling) | Works for imperfect-info games; standard approach. |
| **Parameter sharing** | NFSP | Kept (all players share weights) | Reduces variance; symmetric strategies. |
| **Game tree search (MCTS)** | AlphaGo | **Removed** | Mahjong's irregular turn order blocks MCTS. Mahjong has ~10^48 states; tree search infeasible. |
| **Reward signal** | Round score (standard) | **Changed to global reward prediction (GRP)** | Round scores mislead (tactical losses). GRP predicts final game reward. |
| **Imitation learning (SL)** | Standard practice | **Kept** (warm-start from human logs) | Efficient initialization; avoids random start. |
| **Distillation** | Knowledge distillation (standard) | **Changed to oracle guiding with feature dropout** | Standard distillation failed (oracle too strong). Feature dropout provides curriculum-like transition. |
| **Runtime adaptation** | None | **Added pMCPA** | Adapt policy to specific hand without MCTS. Compensates for inability to use tree search. |

---

### Evolution Chain

$$\text{NFSP (self-play RL)} \xrightarrow{\text{imperfect-info games}} \text{Standard RL in card games}$$
$$\xrightarrow{\text{irregular game tree, multi-round scoring}} \text{Suphx innovations:}$$
$$\text{GRP (handle multi-round reward)} + \text{Oracle guiding (fast learning from privileged info)}$$
$$+ \text{pMCPA (runtime adaptation without MCTS)}$$

**Suphx's specific contributions**:
1. **Global Reward Prediction**: Addresses the challenge of misaligned round-level and game-level rewards. Novel approach to credit assignment in multi-round MDPs. [**Interpretation**: Inspired by real-world RL challenges (e.g., goal-conditioned tasks where intermediate rewards are sparse or misleading), but instantiated specifically for Mahjong's scoring structure.]

2. **Oracle Guiding with Feature Dropout**: Addresses the challenge of learning from hidden information without explicit game tree search. Novel approach to knowledge transfer when direct distillation fails. [**Interpretation**: Generalization of privileged information distillation (Hinton et al., Lopez-Paz et al.); applied to imperfect-information games. Not known to be used in prior game AI work.]

3. **Parametric Monte-Carlo Policy Adaptation (pMCPA)**: Addresses the challenge of runtime adaptation when tree search is infeasible. Novel approach to policy adaptation for specific game instances. [**Interpretation**: Relates to likelihood-ratio policy optimization (LRPO) and importance sampling, but applied to per-round adaptation in an imperfect-info game. Not known to be used in prior work.]

**Did Suphx's innovations improve performance?**
- **GRP**: Yes, +0.4 dan stable rank (RL-basic → RL-1, Figure 8, p. 15).
- **Oracle guiding**: Yes, +0.4 dan stable rank (RL-1 → RL-2, Figure 8, p. 15).
- **pMCPA**: Locally yes (66% win rate adapted vs. non-adapted), but not tested online.

**What limitations remain after Suphx?**
- Oracle guiding requires perfect information at training time (may not generalize to all domains).
- pMCPA is computationally expensive (not deployed online).
- Parameter sharing may underutilize position-specific strategies.
- Non-stationarity in self-play not explicitly addressed (no theoretical convergence guarantees).

---

## 15. Project Ideas for Me

### Project Idea 1: Oracle Guiding for Dou Dizhu

**Research Question**: Can oracle guiding (training with opponents' cards visible, then gradually hiding them via feature dropout) accelerate learning in Dou Dizhu compared to standard self-play RL?

**Baseline**: DouZero (standard self-play policy gradient without oracle guiding).

**Modification**: Implement oracle guiding:
1. Train oracle agent with perfect features (all three players' cards visible during training).
2. Gradually dropout perfect features using Bernoulli mask (γ_t: 1 → 0) over training iterations.
3. After dropout complete (γ_t = 0), fine-tune on normal features only.
4. Compare convergence speed (stable rank vs. training games) with RL baseline.

**Hypothesis**: Oracle guiding will reach higher stable rank faster (e.g., 80% of max performance in 50% fewer games). Why: Oracle agent learns from privileged information; gradual dropout transfers knowledge to normal agent without catastrophic forgetting.

**Experiment**:
1. Train DouZero-like baseline on 100K games, measure stable rank.
2. Train oracle guiding (same budget), measure stable rank.
3. Measure convergence curves: stable rank vs. training games (plot both).
4. Ablation: compare dropout decay schedules (linear vs. exponential vs. step decay).

**Metrics**:
- Sample efficiency: games to reach 90% of max stable rank.
- Final stable rank (after 100K games).
- Ablation: effect of decay schedule on final performance.

**Difficulty**: Moderate. Requires understanding of feature dropout (matrix masking + decay schedule), but implementation is straightforward (2–3 lines of code per forward pass).

**Compute**: Medium (100K games × 2 seeds × 2–3 weeks ≈ 2–4 GPUs for 1–2 weeks).

**Novelty**: Directly applies Suphx's oracle guiding to Dou Dizhu. Novel in the sense that oracle guiding has not been used in Dou Dizhu papers (in this set), but technique is from Suphx. **Moderate novelty**: incremental application of existing technique to new domain. Adds value if shows measurable improvement.

---

### Project Idea 2: Global Reward Prediction for Dou Dizhu

**Research Question**: Can a separate GRU-based reward predictor improve RL training in Dou Dizhu by providing per-round reward signals that reflect final game outcome, instead of using raw win/loss scores?

**Baseline**: DouZero (uses round scores / win-loss signals).

**Modification**: Implement GRP:
1. Train GRU predictor on human Dou Dizhu game logs (if available) or RL-generated games.
2. GRU input: [accumulated points of three players, round number, ...]
3. GRU output: predicted final game rank (or score difference) at game end.
4. During RL, use $\hat{r}_k = \Phi(x_1, ..., x_k) - \Phi(x_1, ..., x_{k-1})$ as per-round reward instead of raw round score.
5. Compare RL convergence (stable rank) with and without GRP.

**Hypothesis**: GRP improves learning efficiency by providing more informative per-round feedback. Raw win/loss (1/0) is sparse; GRP gradient should be smoother and richer.

**Experiment**:
1. Collect 10K games of RL agents playing against each other.
2. Train GRP predictor to predict final rank from partial game info (R^2 metric on held-out test set).
3. Train RL agent with standard round scores; train parallel RL agent with GRP.
4. Compare convergence and final stable rank.

**Metrics**:
- GRP prediction accuracy (R^2 on test games).
- RL convergence speed (stable rank vs. training games).
- Final stable rank after 100K games.

**Difficulty**: Moderate. GRU is standard; main challenge is designing input features (what game info predicts final outcome?).

**Compute**: Low to Medium (100K games + GRP training ≈ 1–2 GPUs for 1 week).

**Novelty**: Directly applies Suphx's GRP to Dou Dizhu. Similar to oracle guiding: incremental application, but measurable improvement if demonstrated. **Moderate novelty**.

---

### Project Idea 3: Multi-Agent Opponent Modeling with Diverse Opponent Pool

**Research Question**: Does training against a diverse opponent pool (random subset of old policies, not just current policy) improve generalization and robustness compared to parameter sharing?

**Baseline**: DouZero (parameter sharing; all players use same policy).

**Modification**:
1. Maintain an opponent pool: store policy checkpoints at regular intervals (every 10K games).
2. During self-play, randomly sample one opponent policy from the pool (50% current, 50% old) for each of the three opponents.
3. Train main policy (position 0) against this diverse opponent pool.
4. Compare with baseline: same parameter-sharing setup but all opponents are current policy.

**Hypothesis**: Diverse opponent pool reduces overfitting to a single strategy. Main agent learns more robust play, generalizing better to unseen opponents.

**Experiment**:
1. Train baseline (parameter sharing) on 100K games.
2. Train modified (diverse pool) on 100K games (same wall-clock time, but fewer policy updates due to evaluating old policies).
3. After training, evaluate both on:
   - Self-play (current-vs-current): measure stable rank.
   - Cross-play: have trained main policy play against multiple old checkpoints.
   - Variance across games: measure std dev of win rate.

**Metrics**:
- Stable rank (self-play).
- Win rate vs. specific past policies (cross-play).
- Robustness: variance of stable rank across games.

**Difficulty**: Easy to Moderate. Simple modification (change opponent sampling logic); main challenge is efficient storage/loading of old policies.

**Compute**: Low (100K games ≈ 1–2 weeks on 1–2 GPUs).

**Novelty**: Well-established in multi-agent RL (e.g., AlphaGo uses opponent pool). But applying to Dou Dizhu and comparing with parameter sharing is a clear contribution. **Moderate novelty**: standard technique applied to Dou Dizhu, but value shown empirically.

---

### Project Idea 4: Parametric Policy Adaptation (pMCPA) for Dou Dizhu (Offline)

**Research Question**: Can adapting the policy at the start of each Dou Dizhu round (given the landlord/peasant roles and initial hand) improve performance without full Monte-Carlo tree search?

**Baseline**: Frozen policy from RL training (no adaptation).

**Modification**: Implement pMCPA:
1. At round start, given fixed role + initial hand h, sample K = 10K random trajectories (opponent hands sampled from remaining deck).
2. Use policy gradient (or likelihood ratio) to finetune policy on these trajectories: $\theta_a = \arg\max_\theta \sum_\tau \frac{p(\tau; \theta)}{p(\tau; \theta_0)} R(\tau)$.
3. Use finetuned policy π_a to play this round.
4. Compare win rate adapted vs. non-adapted on held-out test rounds.

**Hypothesis**: Per-round adaptation improves performance (higher win rate). Effect size: ~5–10% improvement (similar to Suphx's 66% win rate for adapted version, though tested on limited rounds).

**Experiment**:
1. Train baseline RL policy on 50K games.
2. Generate test set: 100 Dou Dizhu games (Tenhou or self-play).
3. For each game, play with and without pMCPA.
4. Measure win rate (adapted vs. non-adapted) across 100 games.

**Metrics**:
- Win rate improvement (adapted - non-adapted).
- Computational cost (wall-clock time per adaptation).

**Difficulty**: Hard. Requires careful implementation of importance-sampled likelihood ratio; risk of instability if not done correctly.

**Compute**: High (K = 10K trajectories × 100 test rounds + gradient updates ≈ 1–2 GPUs for 1 week, but mostly for offline testing).

**Novelty**: Direct application of Suphx's pMCPA to Dou Dizhu. **Moderate novelty**, but useful for demonstrating that runtime adaptation works in a different game.

---

### Summary: Recommended Projects for CS230

| **Project** | **Research Question** | **Difficulty** | **Compute** | **Timeline** | **Novelty** |
|---|---|---|---|---|---|
| **1. Oracle Guiding** | Does oracle guiding accelerate RL in Dou Dizhu? | Moderate | Medium | 2 weeks | Moderate |
| **2. Global Reward Prediction** | Does GRP improve per-round reward signals in Dou Dizhu? | Moderate | Low–Medium | 1.5 weeks | Moderate |
| **3. Opponent Pool** | Does diverse opponent pool improve generalization? | Easy | Low | 1 week | Moderate |
| **4. pMCPA** | Can per-round policy adaptation improve play without MCTS? | Hard | High | 2 weeks | Moderate |

**Recommendation**: Start with **Project 2 or 3** (easier, shorter timeline). If comfortable, combine with **Project 1** (oracle guiding). **Project 4** is more ambitious; good for a deep-dive if you have extra time or interest in runtime adaptation.

---

## 16. Important Implementation Components

### MUST UNDERSTAND AND IMPLEMENT / MODIFY

1. **Feature Encoding for State Representation** (§2.2, p. 6–7):
   - Encode tiles (34 types) into multiple channels: private tiles (4 channels), open melds, doras, discard sequence, scores, etc.
   - Total: 34 × D channels (D = 838 for discard, 958 for Chow/Pong/Kong).
   - **For Dou Dizhu**: Analogous encoding: encode card types (54 unique cards in typical deck), hand composition, played cards, trump, bomb status, etc.
   - **Why**: Core representation; determines what information the CNN processes.
   - **Implementation**: Create a `State` class with a `to_feature_planes()` method. Test on a few states to verify feature dimensions and correctness.

2. **Global Reward Prediction (GRU + FC layers)** (§3.2, Figure 7, p. 11):
   - Train a GRU on sequence of (round score, accumulated score, dealer, etc.) → predict final game reward.
   - Loss: MSE between predicted and actual final reward.
   - **For Dou Dizhu**: Analogous GRU with per-round input features → predict final game score/rank.
   - **Why**: Enables per-round reward signals $\hat{r}_k = \Phi(x_k) - \Phi(x_{k-1})$ instead of round scores.
   - **Implementation**: Use PyTorch GRU layer + Linear layers. Train on 10K human/self-play games. Evaluate MSE on held-out test set.

3. **Oracle Guiding with Feature Dropout** (§3.3, Eq. 5, p. 11–12):
   - Maintain a normal feature vector (observable) and perfect feature vector (hidden).
   - At training iteration t, create Bernoulli dropout mask δ_t with P(δ_t[i,j] = 1) = γ_t.
   - Concatenate [x_n, δ_t ⊙ x_o] as input to policy network.
   - Decay γ_t from 1 to 0 over training (linear or exponential decay).
   - **For Dou Dizhu**: Create oracle features (opponent hands, deck state); normal features (own hand, played cards).
   - **Why**: Gradually transition oracle agent to normal agent; knowledge transfer without direct distillation.
   - **Implementation**: Create `dropout_matrix = torch.bernoulli(gamma * torch.ones(...))`, then `x_oracle_dropped = dropout_matrix * x_oracle`. Update γ after each epoch. Track convergence of policies.

4. **Policy Gradient Loss with Importance Sampling and Entropy** (§3.1, Eq. 1–2, p. 8–9):
   - $$L(\theta) = \mathbb{E} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_0}(a|s)} A(s,a) - \alpha H(\pi_\theta) \right]$$
   - A(s,a) is advantage (computed from rewards or GAE).
   - H(π_θ) is policy entropy; α is adaptive coefficient.
   - **Why**: Core RL objective; combines policy gradient with exploration regulation.
   - **Implementation**: Compute log-probability under current and old policies. Compute advantages from trajectory rewards. Add entropy bonus. Backprop to update θ.

5. **Advantage Estimation** (§4.2, p. 14):
   - The paper does not specify exact method. [**Interpretation**: Likely GAE (Generalized Advantage Estimation) with parameter λ, or n-step returns with value function baseline.]
   - **Implementation**: Choose one: (a) n-step returns: $A_t = \sum_{i=0}^{n-1} \gamma^i r_{t+i} - V(s_t)$; (b) GAE: $A_t = \sum_{i=0}^\infty (\gamma \lambda)^i \delta_{t+i}$ where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.

### MUST UNDERSTAND CONCEPTUALLY

1. **Self-Play in Multi-Agent RL** (§3.1, Figure 6, p. 9):
   - All agents (4 players) use the same policy (parameter sharing).
   - Trajectories collected via self-play; policy updated via replay buffer.
   - Non-stationarity handled via importance sampling.
   - **Key insight**: Self-play is both curriculum (agents improve together) and data augmentation (more diverse experience).

2. **Imperfect Information and Belief States** (§8, p. 11–12):
   - Agent observes own cards, public cards, but not opponents' cards.
   - CNN implicitly learns to infer hidden cards from public clues (discard sequence, melds, patterns).
   - No explicit Bayesian belief state; end-to-end learning.
   - **Key insight**: Neural networks can learn implicit card inference; simpler than explicit belief state updates.

3. **Reward Shaping and Credit Assignment** (§3.2, p. 10–11):
   - Multi-round games have sparse terminal rewards (only at game end).
   - Round scores are misleading (tactical losses).
   - GRP predictor solves this by predicting future reward from current state.
   - **Key insight**: Reward shaping is critical in multi-round tasks; generic per-step rewards often insufficient.

4. **Knowledge Distillation vs. Oracle Guiding** (§3.3, p. 11–12):
   - Standard distillation: minimize KL(π_oracle || π_normal). Fails because π_normal cannot mimic π_oracle without perfect info.
   - Oracle guiding: Train oracle with perfect info, then gradually hide perfect info via feature dropout. Succeeds because network learns to rely on normal features incrementally.
   - **Key insight**: Curriculum learning (easy → hard) is more effective than direct imitation when there's a capability gap.

### CAN REUSE AS INFRASTRUCTURE

1. **RLCard or DouZero Codebase** (§7 reference):
   - For Dou Dizhu: use RLCard or DouZero's environment, training loop, evaluation utilities.
   - No need to rebuild game engine; focus on algorithm modifications.
   - **What to reuse**: Game environment, self-play training loop, evaluation metrics.
   - **What to modify**: State representation (feature encoding), reward signal (add GRP), policy architecture (add oracle guiding).

2. **Distributed RL Infrastructure** (Figure 6, p. 9):
   - RL training requires parallel trajectory generation (self-play workers) and centralized policy updates (parameter server).
   - Use Ray or similar distributed framework to implement this.
   - **If starting from scratch**: RLCard or DouZero may already have distributed training; reuse if available.
   - **For CS230 scale**: May not need full distribution; single machine with multiple GPUs or multi-process on CPUs.

3. **Policy Gradient Optimization**:
   - Standard PyTorch optimizers (Adam, SGD) work for policy gradient.
   - Use existing implementations in popular RL libraries (Stable Baselines3, OpenAI Baselines, etc.) if available.
   - Key: ensure importance sampling and entropy regularization are correctly applied.

---

## 17. Questions I Should Ask My CS230 Staff

1. **Reproduction Scope for CS230**: 
   - Is it acceptable to do a "lite" version of Suphx (e.g., train on Dou Dizhu with 50K games instead of 1.5M)? Or is the expectation to reproduce the full system?
   - If lite version: what metrics should I target to convincingly show the algorithm works?

2. **Compute and Time Budget**:
   - I have access to X GPU-hours. Should I focus on novel modifications (project ideas 1–4) rather than reproducing Suphx as-is?
   - Guidance: prioritize novelty over scale; smaller experiments with clear insights are better than large-scale reproduction.

3. **Baseline and Novelty**:
   - If I implement oracle guiding for Dou Dizhu, is that a sufficient novel contribution? (I'm applying Suphx's idea to a different game.)
   - Guidance: novelty = applying existing technique to new domain + showing empirical improvement. Acceptable for CS230 final project.

4. **Evaluation Rigor**:
   - How many games should I play to get statistically significant results? (Suphx plays 5K online; I have 1 GPU, can run ~10K games in 2 weeks.)
   - Guidance: 10K games should yield stable rank with reasonable variance (IQR). Report IQR plots or confidence intervals, not just point estimates.

5. **Hyperparameter Tuning**:
   - How much time should I spend tuning hyperparameters vs. running experiments?
   - Guidance: Use values from Suphx/DouZero as defaults. Run 1–2 ablations (e.g., entropy coefficient, dropout decay schedule) to show sensitivity. Don't exhaustively search.

---

## 18. Beginner Concepts I Need to Learn

| **Concept** | **Plain-Language Explanation** | **Where It Appears in This Paper** | **Priority** |
|---|---|---|---|
| **Policy gradient** | A way to improve a policy by computing gradients of expected return w.r.t. policy parameters. E.g., if an action leads to high reward, increase its probability. Standard RL algorithm. | §3.1, Eq. 1–2 | **NOW** |
| **Importance sampling** | A technique to correct for the fact that data was generated by an old distribution (π_{θ0}) but used to train a new distribution (π_θ). Ratio π_θ / π_{θ0} rescales the data. | §3.1, Eq. 1 | **NOW** |
| **Entropy regularization** | Adding a term to the loss that encourages high entropy (randomness) in the policy. Prevents the policy from becoming too deterministic too early, which helps exploration. | §3.1, Eq. 2–3 | **NOW** |
| **Advantage function** | A(s,a) = expected return of action a - baseline (e.g., expected return of state s). Measures how much better (or worse) this action is compared to average. Used to weight policy gradient updates. | §3.1, Eq. 1; §5 (loss) | **NOW** |
| **Self-play** | Agents play against earlier versions of themselves (or clones). Used to generate data and improve the policy via learning from its own mistakes. Common in game AI. | §3.1, Figure 6; §7 | **NOW** |
| **Imperfect information** | Game state has hidden information (e.g., opponents' cards). Agents only observe partial state. Contrast with perfect information (e.g., chess, Go), where all info is public. | §1, §8 | **NOW** |
| **Recurrent neural network (RNN/GRU)** | A neural network that processes sequences. At each time step, it reads an input and maintains a "hidden state" that summarizes previous inputs. Useful for sequential data (e.g., game history). | §3.2, Figure 7 (GRU for reward prediction) | **SOON** |
| **Convolutional neural network (CNN)** | A neural network designed for spatial/grid data (e.g., images). Uses convolutions (local filters) to process data. In Suphx, CNNs process 34 × D feature planes (tiles). | §2.2, Figures 4–5 | **SOON** |
| **Residual block** | A neural network building block: skip connection that adds the input to the output of a sub-network. Helps training deep networks. | §3 (CNN architecture, 50 residual blocks) | **SOON** |
| **Reward shaping** | Modifying the reward function to provide better learning signals (e.g., using GRP to compute per-round rewards from final game reward). Not the true reward, but an auxiliary signal for learning. | §3.2 (GRP) | **SOON** |
| **Curriculum learning** | Training approach: start with easy tasks, gradually increase difficulty. Helps convergence. Oracle guiding is a form of curriculum: oracle agent → harder normal agent. | §3.3 (oracle guiding) | **SOON** |
| **Feature dropout / feature masking** | Randomly setting some input features to zero (or masking them out). Prevents overfitting; also used in oracle guiding to gradually hide perfect information. | §3.3, Eq. 5 | **SOON** |
| **Supervised learning** | Training a model to predict labels from inputs, using labeled (input, output) pairs. Opposite of RL (no labels; learn from rewards). In Suphx, SL warm-starts the policy. | §4.1 | **OPTIONAL** |
| **Cross-entropy loss** | Loss function for classification: measures how far predicted probability distribution is from true label. Used in supervised learning. | §4.1 (SL training) | **OPTIONAL** |
| **Offline RL / batch RL** | RL training from a fixed dataset (not from live environment). GRP is trained offline on human logs (not updated during RL). | §3.2, §4.1 | **OPTIONAL** |
| **Information set** | In imperfect-information games: the set of game states indistinguishable from the player's perspective. Mahjong has ~10^48 info sets per decision point. | §1, §8 | **OPTIONAL** |
| **Monte Carlo Tree Search (MCTS)** | A tree search algorithm for perfect-information games (e.g., Go, chess). Suphx cannot use MCTS because Mahjong's game tree is irregular. pMCPA is a substitute. | §1, §3.4 | **OPTIONAL** |

---

## 19. Paper Scorecard

> **Cross-paper calibration:** this scorecard looks at this paper alone. Ranked against the whole pack for this Dou Dizhu project, the recommended verdict is **READ SELECTIVELY**. See `00_SYNTHESIS_AND_RECOMMENDATION.md` §20.9.

| **Criterion** | **Score** | **Justification** |
|---|---|---|
| **Importance for my project** | 4/5 | Suphx directly addresses imperfect-information multi-player games (like Dou Dizhu). Three core innovations (GRP, oracle guiding, pMCPA) are highly relevant and applicable. |
| **Difficulty to understand** | 3/5 | Requires solid grasp of policy gradient RL, importance sampling, entropy regularization (all standard), plus three novel components (GRP is simple GRU; oracle guiding is clever but not complex; pMCPA is moderate). Main challenge is integrating concepts, not individual complexity. |
| **Difficulty to reproduce** | 4/5 | Full reproduction requires distributed RL infrastructure (44 GPUs, 1.5M games). However, core ideas (SL→RL→GRP→oracle guiding) are implementable at smaller scale on 1–2 GPUs (100K games, 1–2 weeks). Hardest part is feature engineering (look-ahead features) and hyperparameter tuning. |
| **Relevance to Dou Dizhu** | 5/5 | Mahjong and Dou Dizhu share: imperfect information, multi-player competition, multi-round games, complex rewards, irregular game flow. Suphx's solutions are directly transferable. No other paper in this set is as close to Dou Dizhu structurally. |
| **Relevance to self-play** | 5/5 | Self-play is central to Suphx (§3.1, Figure 6). Parameter sharing, importance sampling, entropy regularization—all key techniques in self-play RL. |
| **Relevance to imperfect information** | 5/5 | Core challenge addressed by Suphx. Oracle guiding is a novel technique for learning from hidden information without explicit belief states. Directly applicable to Dou Dizhu. |
| **Potential for project ideas** | 4/5 | Four concrete project ideas (oracle guiding, GRP, opponent pool, pMCPA) that are feasible for CS230. Each has clear hypothesis, metric, ablation, and difficulty level. |
| **Importance to read deeply** | 5/5 | **Key paper**. Suphx is the most directly relevant to Dou Dizhu projects in this set (after DouZero, which is Tier A). Techniques are novel and effective. |

---

### Verdict: **READ DEEPLY**

**Which sections / pages to read first (if selective)**:
1. **§1 (Introduction, p. 1–3)**: Problem motivation, why Mahjong is hard, main innovations. Essential for context.
2. **§2.1 (Decision Flow, p. 4–5) + Figure 1**: Understand Mahjong action structure. Helps with feature engineering for Dou Dizhu analogy.
3. **§3.2 (Global Reward Prediction, p. 10–11)**: Core innovation for multi-round reward structure. Most directly applicable to Dou Dizhu.
4. **§3.3 (Oracle Guiding, p. 11–12) + Eq. 5**: Second core innovation; clever technique for privileged information. Read in detail.
5. **§4.2 (RL Ablation, Figure 8, p. 14–16)**: Empirical validation. Shows that each component adds value.
6. **§5 (Online Evaluation, p. 17–20) + Table 4**: Results on Tenhou. Demonstrates real-world superhuman performance.

**Skip if short on time**:
- §2.2 (detailed feature encoding): Scan for ideas, but exact design is Mahjong-specific.
- §3.4 (pMCPA): Optional; not deployed online; more advanced.
- §4.1 (supervised learning): Standard; referenced in many papers.
- §4.3 (pMCPA evaluation): Low priority if skipping §3.4.
- Appendices A–C (Mahjong rules, Tenhou ranking): Reference only; read if unfamiliar with Mahjong.

---

## Appendix: Synthesis Facts

**Paper**: Suphx: Mastering Mahjong with Deep Reinforcement Learning

**Year**: 2020

**Environment(s)**: 4-player Japanese Mahjong (Riichi Mahjong); Tenhou online platform for evaluation.

**Main algorithm**: Policy gradient with importance sampling + entropy regularization (Eq. 1–2). Three key additions: (1) global reward prediction (GRU-based), (2) oracle guiding (feature dropout from 1 to 0), (3) parametric Monte-Carlo policy adaptation (pMCPA).

**State representation**: Observable state encoded into 34 × D feature channels: private tiles (4 channels), open melds, doras, discard sequence, scores, dealer, etc. D = 838 (discard/Riichi) or 958 (Chow/Pong/Kong). Includes 100+ hand-crafted look-ahead features (DFS-based hand strength predictions).

**Action representation**: Five decision models (discard, Riichi, Chow, Pong, Kong) + rule-based winning model. Discard outputs 34 actions (which tile to discard); others are binary (yes/no).

**Self-play?**: Yes. All four players in a game use the same policy (parameter sharing). Opponents are clones of training policy. No explicit opponent pool.

**Opponent modeling?**: No explicit opponent modeling. Implicit learning via self-play (agent learns how to exploit standard strategies).

**Main loss**: Policy gradient with importance sampling (Eq. 1) + entropy bonus (Eq. 2). GRP predictor trained via MSE loss (Eq. 4) on human logs. Oracle guiding uses same loss as RL but with dropout mask on perfect features.

**Main contribution**: Three novel techniques addressing Mahjong-specific challenges: (1) **Global Reward Prediction**: per-round reward signals from multi-round game reward, solving credit assignment in multi-round MDPs. (2) **Oracle Guiding**: privileged information distillation via gradual feature dropout, enabling fast learning without direct KL distillation. (3) **Parametric Monte-Carlo Policy Adaptation**: runtime policy adaptation via likelihood-ratio optimization, compensating for inability to use MCTS.

**Evaluation method**: Offline: ablation on 1M games vs. fixed SL-weak opponent, stable rank metric, IQR box plots. Online: 5,760 games on Tenhou expert room, stable rank = 8.74 dan, surpasses top humans (7.46 dan) and prior AIs (Bakuuchi 6.59, NAGA 6.64 dan).

**Compute**: 44 GPUs (40 Tesla K80 + 4 Titan XP), 2 days for 1.5M games of RL training per agent. Offline eval: 20 Tesla K80 GPUs, 2 days per agent.

**Main limitation**: Oracle guiding requires privileged information at training time (perfect cards visible during training). pMCPA is expensive; not deployed online. Parameter sharing may underutilize position-specific strategies. Non-stationarity in self-play not explicitly addressed.

**Best idea to borrow**: **Oracle guiding with feature dropout**. Directly applicable to Dou Dizhu (train with opponent cards visible, gradually hide via dropout). More efficient than direct distillation; clever curriculum-learning approach that doesn't require a separate student network.

---

