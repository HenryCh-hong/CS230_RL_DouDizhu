# DouZero+: Improving DouDizhu AI by Opponent Modeling and Coach-guided Learning (arXiv 2022)

> Zhao, Zhao, Hu, Zhou, Li · arXiv:2204.02558v1 · Tier A · Two surgical improvements (opponent modeling + initial-deal filtering) yield significant empirical gains over DouZero and reach top-5 Botzone ranking.

## 0. TL;DR for a CS230 student

- **What it is**: A refinement of DouZero (state-of-the-art DouDizhu AI) that adds two techniques: (1) explicit opponent hand prediction via an auxiliary multi-head classifier, and (2) a "coach network" that filters training games to focus on balanced initial deals.
- **Why it matters**: Shows that even mature RL systems can benefit from auxiliary self-supervised tasks (hidden-card prediction) and curriculum-style deal selection. The paper clearly separates two orthogonal improvements, making it a teaching model for modular design.
- **The idea to steal**: (1) Opponent modeling via a separate supervised prediction model trained on ground-truth opponent hands, whose outputs are concatenated with state features for the decision model; (2) Deal filtering as a post-hoc curriculum (no change to Bellman equations or reward, just filtering which games reach the learner).
- **For this project**: Both components are self-contained, reproducible, and isolatable. Opponent modeling is an excellent starting point for card inference; deal filtering generalizes to other games with high initial-state variance.

---

## 1. Paper Overview

### 1a. Beginner-friendly explanation

Imagine you're training an RL agent to play DouDizhu. DouZero (published a year earlier) was already excellent—it beats all previous AI by learning from pure self-play without hand-coded strategies. But DouZero+ asks: can we help the agent learn *even better*?

DouZero+ adds two things:

1. **Opponent modeling**: In DouDizhu, you don't see your opponents' cards. But human players try to *guess* what cards their opponents hold by watching what they play. DouZero+ adds a separate neural network head that learns to predict the next opponent's hand. This prediction is fed back into the decision network, giving it an explicit "belief" about what cards to expect. It's like giving the AI a form of reading opponent "tells."

2. **Coach-guided learning**: DouDizhu games have high variance: if you deal terrible cards to a player, they'll lose no matter what. DouZero+ notices that ~40% of random deals are "too imbalanced" to teach the AI anything useful—the outcome is determined by the cards, not the strategy. A "coach network" learns to predict the win probability of the Landlord from the initial deal, and filters out deals with extreme predictions (too easy or too hard). This lets the AI focus on games where skill matters.

The result: stronger Peasant play (which benefits from opponent cooperation), competitive Landlord play, and top-5 ranking on Botzone (an online competitive platform).

### 1b. Technical explanation

**Problem solved**: DouZero (Zha et al., 2021) demonstrated that deep Monte Carlo (MC) with self-play can scale to large action spaces in imperfect-information games without human prior knowledge. However, DouZero trains on *all* random deals, including ones heavily skewed in one player's favor. Two sources of inefficiency remain:
1. The decision network does not have explicit access to beliefs about opponents' hands.
2. The network wastes training time on games where the outcome is determined by deal luck, not strategy.

**Why difficult**: 
- In imperfect-information games, explicit opponent modeling (beyond implicit beliefs encoded in state features) requires either (a) Bayesian inference over opponent hands (computationally expensive, as in DeltaDou), or (b) a separate learned model. The challenge is training the model with ground-truth labels that are only available at train time (not deployment).
- Deal filtering must be done online without knowing the true outcome in advance, so a separate "coach" network must learn to predict game balance from initial cards alone.

**Main idea and contributions**:
1. **Opponent Modeling (Section IV-A)**: Train three separate "prediction models" (one per player role), each a multi-head classifier that outputs P(# of each card rank in next player's hand). Inputs are the same state features as DouZero. Outputs are trained with cross-entropy loss against ground-truth next-player hands (available during training). Predictions are concatenated with state features and fed to the "decision model." The prediction and decision models are trained concurrently (joint self-play), not sequentially.

2. **Coach-guided Learning (Section IV-B)**: Train a single "coach network" that takes initial card distributions for Landlord and Peasants and outputs $P_{\text{win}}$ (predicted Landlord winning probability). Filter games where $P_{\text{win}} \in [\beta_{\min}, \beta_{\max}]$; reject others. The threshold $\beta$ increases gradually during training (starts at 0, rises to ≤0.3 by the end). This focuses the learner on games where the initial deal is ~balanced (P_win ~ 0.5), reducing variance and training time.

**What came before**: DouZero (2021) showed MC with action encoding scales to DouDizhu without heuristics. DeltaDou (2019) combined Bayesian card inference with AlphaZero-like MCTS/search but was computationally expensive. Suphx (2020) in Mahjong used "oracle guiding" (a perfect-information oracle gradually dropping hidden info), which is conceptually related but not identical to opponent modeling.

**Motivation**: The paper cites human practice: humans predict opponents' hands and use this to decide strategy. High-variance games (heavily skewed deals) waste training time. Both are psychological insights, not algorithmic novelties, but the execution is clean.

---

## 2. RL Formulation

| **Component** | **Specification** | **Citation/Notes** |
|---|---|---|
| **Agents** | Three: Landlord (LL), Peasant Down (PD), Peasant Up (PU) | (§2 background; Fig. 1) |
| **Environment** | DouDizhu card game; episodic, zero-sum among players | (§3 Preliminary; game rules in Appendix A of DouZero) |
| **State** | Agent observes own hand, play history, opponent play history, bomb counts; does *not* observe opponent hands | (§3 Preliminary, pp. 3–4; "state features" described in detail) |
| **Observation** | Card matrices (4×15 one-hot per card), action history (LSTM-encoded), play counts; ~320-dim vector per role | (§3 Preliminary; Figure 3b) |
| **Action** | Legal card combos (Solo, Pair, Chain, Bomb, etc.); 391 possible per hand; actions encoded as 4×15 matrices | (p. 1, Fig. 2; 119 legal moves shown as example) |
| **Legal action handling** | Actions pre-filtered by legal-move generator; only legal actions passed to network | (§3 Preliminary; mentioned implicitly in action features) |
| **Reward** | Per-game: based on outcome (win=+X points, loss=-X) where bomb multiplies score by 2. Trained on ADP (Average Difference in Points) | (§5.A, p. 6: "ADP can be directly used as rewards") |
| **Return** | MC return: $G_t = \sum_{k=0}^{\infty} R_{t+k}$ accumulated to episode end; no discounting (episodic task) | (§3: "Monte Carlo method learns value functions from experience") |
| **Policy** | Neural network $\pi_\theta(a \mid s)$ with action encoding; $\epsilon$-greedy exploration during self-play | (§3: "epsilon-greedy to balance exploration and exploitation") |
| **Value function / Q-function** | $Q(s, a)$ learned via MC averaging; targets are empirical returns from episodes | (§3: "Q(s, a) can be replaced by neural networks") |
| **Fully vs. partially observable** | Partially observable: own cards visible, opponent cards hidden | (§1 Introduction: "partially observable environment") |
| **Single vs. multi-agent** | Three agents; cooperative (Peasants cooperate) and competitive (all teams compete) | (§1: "cooperate or compete with each other") |
| **Self-play used?** | Yes; actors play games, learner updates; actors sync periodically with learner's global networks | (§3: "self-play procedure, where the actors play games to generate samples while the learner updates") |

**The paper does not specify**: Exploration schedule ε(t) explicitly; discount factor γ (assumed 1 for episodic tasks); whether behavior policy differs from target policy (appears to use $\epsilon$-greedy on the same network).

---

## 3. Model / Architecture

**Overall architecture** (Figure 3a):
- Two types of networks: **Prediction Model** (opponent hand classifier) and **Decision Model** (action selection, inherited from DouZero).
- Three roles (LL, PD, PU); each role has one prediction model and one decision model.
- The prediction model has LSTM + 5-layer MLP for internal feature extraction (these layers are internal to the prediction model, processing its own inputs). Its outputs (predicted opponent hand card counts) are concatenated with the original state features.
- The decision model takes these concatenated features as input and outputs Q-values.
- Outputs: prediction model → multi-head classifier (one head per card rank); decision model → Q-values.

**Input Representation**:
- **State features** (same as DouZero):
  - Card matrices: own hand (54 cards, one-hot), union of opponents' cards seen (54 cards), played cards of each opponent (54×3), recent moves (LSTM-encoded).
  - One-hot vectors: number of cards each opponent holds, bomb count.
  - Dimensions: ~320-dim per role.
- **Action features**:
  - Legal actions encoded as 4×15 card matrices (4 suits, 15 ranks including jokers).
  - Concatenated with state features.

**Card representation** (explicit):
- Each card is a position in a 54-vector (or 4×15 matrix for actions).
- Cards grouped by rank; suits irrelevant.

**Action representation**:
- 4×15 matrix encoding (suit × rank); one-hot for each card.
- Legal actions pre-computed by game rules; only legal actions presented to network.

**Network outputs**:
- **Decision model**: Q-value (scalar) for each legal action.
- **Prediction model**: Probability distribution over [0, 13] (or 0–17, actual max card count) for each card rank in next opponent's hand.

**Architecture details**:

| **Layer** | **Prediction Model** | **Decision Model** | **Notes** |
|---|---|---|---|
| Input encoding | State features only | Concatenation of: original state features + prediction outputs | Prediction feeds decision |
| LSTM layer | 1 layer, encodes move history | (Inherited from DouZero) | Captures temporal dependencies |
| MLP | 5 layers, 512 hidden units each (within prediction model) | 6 layers, 512 hidden units (separate, DouZero architecture) | Different networks; not shared |
| Output layer | Multi-head FC: one head per "kind of card", each a classifier over how many copies the next player holds (the paper does not state the number of heads; 15 kinds = 13 ranks + 2 jokers would be natural [Interpretation]) | 1 FC → Q-value (scalar) | One prediction head per card kind |
| Loss | Cross-entropy per rank (classification) | MSE of Q-value vs. empirical return | Different objective per model |

**Why chosen**:
- LSTM + MLP follows DouZero's design (proven effective).
- Prediction model as multi-head classifier is natural for per-rank prediction.
- The five MLP layers are shared *among the prediction heads inside the prediction model* (p. 3, Fig. 3 caption); the paper does **not** describe any layers shared between the prediction model and the decision model.
- Concatenating predictions with state features gives decision model explicit opponent-belief input.

**Separate networks per role**: Yes. Three decision networks (one per role, Landlord/Peasant Down/Peasant Up) and three prediction networks (one per role, predicting the hand of the *next* player in turn order).

**Tensor shapes** (approximate):
- State features: [batch, 320] (per DouZero)
- Action features: [batch, legal_actions, 60] (4×15 one-hot)
- Prediction output: [batch, legal_actions, 13] (prob for each card rank)
- Decision output: [batch, legal_actions] (Q-value for each legal action)

---

## 4. Training Algorithm

**Full pipeline** (high level):

```
Initialize:
  - Three decision networks (LL, PD, PU)
  - Three prediction networks (LL, PD, PU)
  - Coach network
  - Actor processes (parallel game generators)
  - Learner process

Loop over training steps:
  ┌─ Actors:
  │  For each parallel game:
  │    1. Sample initial cards (random deal)
  │    2. Coach network predicts P_win from initial deal
  │    3. If P_win outside threshold β, reject and re-sample
  │    4. Play one episode: cycle through LL, PD, PU turns
  │       - At each turn:
  │         a. Encode state features s_t
  │         b. Get legal actions A_t
  │         c. Forward decision network: Q(s_t, A_t)
  │         d. ε-greedy action selection → a_t
  │         e. Execute action, observe reward r_t, next state s_{t+1}
  │    5. Store trajectory (s_t, a_t, r_t) for each player
  │
  └─ Learner:
     For each stored trajectory:
       1. Compute MC return G_t = sum of discounted future rewards (no discount: γ=1)
       2. For each (s_t, a_t, G_t) sample:
          - Forward prediction network: predict opponent hand ranks
          - Loss_pred = cross_entropy(prediction, ground_truth_next_player_hand)
          - Forward decision network: Q_pred(s_t, a_t)
          - Loss_dec = MSE(Q_pred, G_t)
          - [Paper] The two models are trained "concurrently" (p. 6), each with its own loss
            (cross-entropy for prediction, MSE for decision). The paper gives NO combined
            loss and does not say whether decision-loss gradients reach the prediction model.
       3. Update decision & prediction networks
       4. Update coach network (see below)
       5. Sync learner networks → actor networks (periodic)

Coach network training:
  - Inputs: initial card vectors (Landlord: 20-dim, Peasants: 17-dim each)
  - Label: outcome of game (did LL win? binary or scalar)
  - Loss: MSE or cross-entropy (paper doesn't specify precisely)
  - Concurrent training with decision/prediction models
```

**Step-by-step details**:

1. **Observation → Feature encoding** (same as DouZero):
   - Current player's own cards + move history → LSTM + MLP state features.
   - Legal actions → one-hot 4×15 matrices.

2. **Legal action computation**:
   - Game rules pre-compute valid plays for current hand.
   - Only these actions passed to network.

3. **Model forward pass**:
   - Decision model: $Q_{\theta}(s, a) \in \mathbb{R}$ for each $a \in A(s)$.
   - Prediction model (inference stage): $P_{\theta}(\text{# of rank } r \text{ in next opponent's hand}) \in [0, \text{max count}]$ per rank.

4. **Action selection** (training):
   - $\epsilon$-greedy: w.p. $1-\epsilon$, play $a^* = \arg\max_a Q_{\theta}(s, a)$; else uniform random from legal actions.

5. **Trajectory generation**:
   - Execute $a_t$, observe game transition, $r_t$ (points), $s_{t+1}$.
   - Continue until game end or player out of cards.

6. **Learning target**:
   - **Decision**: $G_t = \sum_{k=0}^{T} r_{t+k}$ (undiscounted, since episodic).
   - **Prediction**: Ground-truth labels = actual cards held by next player (available during training; not at deployment).

7. **Loss computation**:
   - **Decision loss**: $\mathcal{L}_{\text{dec}} = \frac{1}{N} \sum (Q_{\theta}(s_t, a_t) - G_t)^2$ (MSE).
   - **Prediction loss**: $\mathcal{L}_{\text{pred}} = \frac{1}{N} \sum_{\text{ranks}} -\log P_{\theta}(\text{true count} \mid s_t)$ (cross-entropy, per rank).
   - **No combined loss is given in the paper.** The prediction model and decision model are separate networks trained concurrently, each on its own loss. (A shared-trunk auxiliary loss $\mathcal{L}_{\text{dec}} + \alpha\mathcal{L}_{\text{pred}}$ is a *possible modification* for your project, not what DouZero+ describes.)

8. **Gradient update**:
   - Backprop each model's own loss.
   - Optimizer: the paper says training hyperparameters are kept at DouZero's defaults (p. 6); DouZero used RMSprop (lr 1e-4). The paper does not specify the prediction or coach models' optimizers.

9. **Coach network update**:
   - Inputs: initial cards $(c_{\text{LL}}, c_{\text{PD}}, c_{\text{PU}})$.
   - Target: game outcome (win/loss/score for LL).
   - Loss: binary cross-entropy or MSE (paper doesn't specify).
   - Trained concurrently with decision/prediction models.

10. **Synchronization**:
    - Actors maintain local copies of learner's networks.
    - Periodic sync (every N training steps) pulls updated networks to actors.

**Pseudocode** (Decision Model only; prediction and coach omitted for brevity):

```
for step in range(num_steps):
    # Actors (parallel)
    for actor in range(num_actors):
        for game_no in range(games_per_batch):
            # Coach-guided sampling
            while True:
                initial_cards = sample_random_deal()
                P_win = coach_network(initial_cards)
                if beta_min <= P_win <= beta_max:
                    break
            
            # Play one game
            state = initialize_game(initial_cards)
            trajectory = []
            while not game_over(state):
                for player in [LL, PD, PU]:
                    s_t = encode_state(state, player)
                    A_legal = get_legal_actions(state)
                    Q_values = decision_network[player](s_t, A_legal)
                    # Epsilon-greedy action selection
                    if random() < epsilon:
                        a_t = random_choice(A_legal)
                    else:
                        a_t = argmax(Q_values)
                    r_t, state = step_game(state, player, a_t)
                    trajectory.append((player, s_t, a_t, r_t))
            
            # Store trajectory in replay buffer
            store_trajectory(trajectory)
    
    # Learner (single process)
    batch = sample_from_replay_buffer()
    for (player, s_t, a_t, r_t) in batch:
        # Compute MC return
        G_t = sum of rewards from t to end of episode
        
        # Prediction network loss (separate)
        next_player = next_in_turn_order(player)
        true_hand = get_ground_truth_hand(state, next_player)
        pred_probs = prediction_network[player](s_t)
        loss_pred = cross_entropy(pred_probs, true_hand)
        optimizer_pred.zero_grad()
        loss_pred.backward()
        optimizer_pred.step()
        
        # Decision network loss (takes concatenated features: state + prediction)
        pred_features = prediction_network[player](s_t)  # Get prediction outputs as features
        s_combined = concatenate(s_t, pred_features)
        Q_pred = decision_network[player](s_combined, a_t)
        loss_dec = (Q_pred - G_t)^2
        optimizer_dec.zero_grad()
        loss_dec.backward()
        optimizer_dec.step()
        # Note: paper does not specify whether decision backprop flows into prediction network
    
    # Coach network training (concurrently)
    for trajectory in batch:
        initial_cards = trajectory.initial_cards
        outcome = trajectory.outcome
        P_win_pred = coach_network(initial_cards)
        loss_coach = (P_win_pred - outcome)^2
        optimizer_coach.step()
    
    # Sync to actors (every N steps)
    if step % sync_interval == 0:
        broadcast_networks_to_actors()
```

---

## 5. Mathematics

### Eq. (1): Deep Monte Carlo Update [MUST UNDERSTAND]

The paper references the MC method without explicit equations. The RL update is:

$$\mathcal{L}_{\text{dec}}(\theta) = \frac{1}{N} \sum_{t=0}^{T} (Q_{\theta}(s_t, a_t) - G_t)^2$$

where:
- $Q_{\theta}(s_t, a_t)$ = predicted Q-value (network output).
- $G_t = \sum_{k=0}^{\infty} R_{t+k}$ = Monte Carlo return (actual cumulative reward from time $t$ to episode end).
- $\theta$ = decision network parameters.
- No bootstrapping; targets are empirical returns.

**Why needed**: MC returns are unbiased but high-variance. The network approximates the empirical return.

**Relation to RL**: This is the most basic Monte Carlo update, used in REINFORCE-style policy gradient and Monte Carlo value function approximation. It is **not** Q-learning (no bootstrap), **not** TD (no one-step lookahead).

---

### Eq. (2): Prediction (Opponent Modeling) Loss [MUST UNDERSTAND]

$$\mathcal{L}_{\text{pred}}(\theta) = \frac{1}{N} \sum_{t=0}^{T} \sum_{r \in \text{ranks}} - \log P_{\theta}(\text{# of rank } r \text{ in next opponent's hand} \mid s_t)$$

where:
- $P_{\theta}(\cdot)$ = multi-head classifier output (one softmax/sigmoid per card rank).
- The label is the **ground-truth count** of that rank in the next player's actual hand (available at train time).
- Cross-entropy loss; each rank treated as an independent classification task (multi-task learning).

**Why needed**: Provides a supervised signal for the prediction model to learn hidden card distributions. The loss is auxiliary—it improves feature learning without changing the MDP.

**Relation to RL**: Not part of standard RL; this is supervised learning "piggybacking" on RL trajectory data. Similar to auxiliary tasks in multi-task RL (e.g., in Atari).

---

### Eq. (3): Combined Loss (Auxiliary + Decision) [THE PAPER DOES NOT SPECIFY THIS CLEARLY]

The decision model and prediction model are **trained separately with different losses**, not as a joint weighted combination:
- **Prediction model loss**: $\mathcal{L}_{\text{pred}}(\theta) = \frac{1}{N} \sum_{t=0}^{T} \sum_{r \in \text{ranks}} - \log P_{\theta}(\text{true count} \mid s_t)$ (cross-entropy per rank).
- **Decision model loss**: $\mathcal{L}_{\text{dec}}(\theta) = \frac{1}{N} \sum_{t=0}^{T} (Q_{\theta}(s_t, a_t) - G_t)^2$ (MSE against MC return).

**The paper states**: "For the decision model, the features used are also similar to DouZero, except for the prediction of hand cards of the next player in state information. For simplicity, we just concatenate the prediction results as well as original state features for state input of decision models."

This means:
1. Prediction model is trained on ground-truth opponent hands (cross-entropy loss).
2. Decision model is trained on RL targets (MC return, MSE loss).
3. Prediction outputs are fed as features to the decision model.
4. **The paper does not explicitly state a combined loss formula or weighting factor α.** It does not specify whether gradients from the decision model backprop into the prediction model, or whether they are trained independently.

**Interpretation**: Concurrent training likely means both losses are computed and backprop happens for each model's own parameters and shared input features, but the exact procedure is not specified. The assumption that they share early layers is inferred from the architecture description but not explicitly stated.

---

### Eq. (4): Coach Network Prediction [OPTIONAL / ADVANCED]

$$P_{\text{win}} = \text{CoachNet}(c_{\text{LL}}, c_{\text{PD}}, c_{\text{PU}})$$

where:
- $c_i$ = vector of card counts for player $i$ (Landlord: 20-dim [counts of 3, 4, …, 2, Red Joker, Black Joker]; Peasants: 17-dim, no initial advantage cards).
- $P_{\text{win}} \in [0, 1]$ = predicted probability that Landlord wins.
- No explicit loss equation given; presumably:

$$\mathcal{L}_{\text{coach}}(\phi) = \frac{1}{M} \sum_{j=0}^{M} \left( \text{CoachNet}_{\phi}(c_j) - \mathbb{1}[\text{LL won game } j] \right)^2$$

(MSE with binary outcome label).

**Why this works**: Unbalanced initial deals (extreme $P_{\text{win}}$ values) are associated with deterministic outcomes; balanced deals (near 0.5) teach more varied strategies.

---

### Eq. (5): Coach-Guided Deal Filtering [MUST UNDERSTAND - for CS230 Project]

$$\text{Accept game } j \iff P_{\text{win}}(j) \in [\beta, 1 - \beta]$$

where:
- $\beta$ = threshold, starts at 0, increases through training, reaches upper limit ≤0.3 (p. 5–6).
- **The paper does not specify**: (a) the exact functional form of β(t) (linear, exponential, adaptive?), (b) the final value (stated as "upper limit of threshold β is set to be 0.3" but not clearly whether β ends at exactly 0.3).
- Interpretation: At β=0, all deals accepted (P_win ∈ [0, 1]). At β=0.3, only deals with P_win ∈ [0.3, 0.7] accepted (middle 40% of deals by win probability).
- This is a **curriculum**: early training accepts all deals (learn anything); later training focuses on balanced deals (learn harder cases).

**Effect**: Filters out high-variance (very one-sided) deals without changing the Bellman equations or reward structure—purely data selection, speeding training by ~20–30% (estimated from paper's ~40% filtering claim).

---

### Eq. (6): How the prediction and decision losses interact [THE PAPER DOES NOT SPECIFY]

**The paper states**: "For the decision model, the features used are also similar to DouZero, except for the prediction of hand cards of the next player in state information."

And: "To sum up, the overview of the framework that combines opponent modeling with DouZero is shown in Figure 3."

But it never writes a combined loss, never mentions a weighting factor, and does not say whether the decision model's gradient flows back into the prediction model.

[Interpretation] The most literal reading of Fig. 3 and §IV-A is two separately optimized networks: the prediction model minimizes its cross-entropy, and the decision model minimizes its DMC MSE while taking the (probably detached) predictions as extra input features. This is a reproducibility gap; check the released code (https://github.com/submit-paper/Doudizhu) if you need certainty.

---

## 6. Loss Function

**Decision Model Loss** (from RL, inherited from DouZero):
- **Loss**: Mean Squared Error (MSE).
- **Formula**: $\mathcal{L}_{\text{dec}} = \frac{1}{N} \sum (Q_{\theta}(s_t, a_t) - G_t)^2$ where $G_t$ is the empirical MC return.
- **Target**: Monte Carlo return $G_t$ (cumulative reward from $t$ to episode end).
- **Where target comes from**: Empirical trajectory data (no bootstrapping, no target network).
- **Bootstrapping**: None. MC is unbiased but high-variance.
- **Input to decision model**: Original state features **concatenated with prediction model outputs**.

**Prediction Model Loss** (auxiliary supervised learning):
- **Loss**: Cross-entropy (multi-class classification), per card rank.
- **Formula**: $\mathcal{L}_{\text{pred}} = \frac{1}{N} \sum_{t=0}^{T} \sum_{r \in \text{ranks}} - \log P_{\theta}(\text{# of rank } r \text{ in next opponent's hand} \mid s_t)$.
- **Target**: Ground-truth count of each card rank in the next player's actual hand (available during training only).
- **Where target comes from**: Environment (game state has full card info; not available at deployment).
- **Training**: Concurrent with decision model. **The paper does not specify whether decision-model gradients backprop into the prediction model.**

**Coach Network Loss**:
- **Loss**: **The paper does not specify.** [Interpretation] Likely MSE or binary cross-entropy.
- **Target**: Game outcome (Landlord win: 1, else: 0) or final score differential.
- **Where target comes from**: Self-play game results (post-game labels).
- **Training schedule**: Concurrent with decision/prediction models. Threshold $\beta$ increases from 0 to ≤0.3 over training (schedule not specified).

**Why these choices**:
- MSE for both decision and coach networks: standard for regression in RL; simple and stable.
- Cross-entropy for prediction: appropriate for multi-class classification per card rank.
- Auxiliary prediction task: improves feature learning without requiring explicit oracle (unlike Suphx's oracle guiding).

---

## 7. Self-Play

**Who plays whom**: Each game has 3 players (Landlord, Peasant Down, Peasant Up). All three roles play simultaneously in a single game. Training uses actor processes that play self-play games, where all three roles use the current global networks (no fixed opponents).

**Simultaneous updates**: Yes. Decision and prediction networks for all three roles are updated from the same experience buffer. The models adapt to each other's improving play.

**Parameter sharing**: 
- **Decision networks**: Separate networks for LL, PD, PU (three networks total). No parameter sharing across roles.
- **Prediction networks**: Separate networks for LL, PD, PU (each predicts the next player's hand). No parameter sharing.
- **Coach network**: Single shared network (independent of role).

**Old checkpoints**: No mention of opponent pool or old checkpoints. Actors use the current global networks (synchronized periodically). This is simpler than self-play with a pool of past agents.

**Exploration**: $\epsilon$-greedy on Q-values (paper states "epsilon-greedy to balance exploration and exploitation"). Explicit $\epsilon$ schedule not given.

**Non-stationarity handling**: 
- [Paper]: "Concurrent learning helps opponent modeling adapt to different levels of the agent as it has witnessed the evolution of the agent's skills during training" (p. 3).
- [Interpretation]: By training prediction models concurrently, they adjust as opponent policies evolve. No explicit regret-minimization or theory; relies on empirical adaptation.
- The deal-filtering threshold $\beta$ increases gradually, which can also be viewed as a form of curriculum learning to handle non-stationarity.

**Summary**: Classic self-play with synchronous global-network updates and no opponent pool. Simpler than DeltaDou's approach but relies on assumption that simultaneous multi-agent learning converges.

---

## 8. Imperfect Information

**Observation table**:

| **Observable by the agent** | **Hidden from the agent** |
|---|---|
| Own hand cards (54 cards: count of each rank) | Opponent 1's hand cards (identity of cards, counts per opponent) |
| Action history: all cards played by all players in all rounds | Opponent 2's hand cards |
| Number of cards held by each opponent | Draw pile (which cards remain undealt) |
| Bombs played so far (count) | Exact discards by opponents (inferred but uncertain) |
| Move history (LSTM-encoded) | Opponent's bid/strategy preferences |
| | Initial deal before game starts (but can infer distribution from action) |

**Handling hidden information**:

1. **State representation** (inherited from DouZero):
   - State features include action history (LSTM-encoded), opponent card counts, and inferred information (e.g., "union of other players' cards" = cards we know they hold).
   - The "union" is implicit: all cards not in our hand and not played.

2. **Opponent modeling** (DouZero+ innovation):
   - Prediction model $P_{\theta}(\text{# of each rank} \mid s_t)$ outputs an explicit belief over opponent hands.
   - During training, we have ground-truth labels; at deployment, the model predicts a distribution.
   - [Interpretation] This is not a full Bayesian belief update (no explicit prior/posterior), but a learned empirical model.

3. **Information sets**:
   - No explicit information set formulation (as in game theory); instead, relies on RL's ability to generalize from history.
   - LSTM processes move history, which partially encodes public information.

4. **Card inference**:
   - Implicit via action history (opponents' play reveals information).
   - Explicit via prediction model (supervised learning from ground truth).

5. **Belief states**:
   - No Bayesian belief state maintained.
   - [Interpretation] The decision network's hidden layers learn an implicit belief state; the prediction model adds an explicit auxiliary belief.

6. **Recurrent models**:
   - LSTM encodes action history (move sequence), which is a form of state compression.
   - RNN allows the network to adapt predictions/decisions based on long-term patterns.

**Key innovation**: Unlike DeltaDou (which uses Bayesian inference + MCTS), DouZero+ uses **supervised learning with ground-truth labels** to train the prediction model. This is simpler but only works during training (labels unavailable at deployment, so predicted distribution is used).

---

## 9. Evaluation

### Baselines and Comparison

**Primary baseline**: DouZero (open-source model from Zha et al., 2021).
- "As our DouDizhu AI is based on DouZero, we just compare the performance between them. We make use of the open-source models of DouZero as the opponent. To demonstrate the improvement, we also realize the original DouZero to intuitively exhibit the performance difference." (p. 6).

**Secondary baselines**:
- "Maphack" (a model with perfect information about next opponent's hand, added directly to state): Figure 5, used to validate that explicit opponent hand info helps.
- Botzone leaderboard: ~400 agents, mostly heuristic rule-based (paper notes DouZero itself ranks ~20th on Botzone).

**Metrics**:
- **WP (Winning Percentage)**: # games won / total games (not reported in main results; only ADP).
- **ADP (Average Difference in Points)**: average point differential between DouZero+ and baseline per game. Base = 1 point; bombs double the score.
- Formula: $\text{ADP} = \frac{1}{N} \sum_{i=1}^{N} (\text{score}_{\text{new}, i} - \text{score}_{\text{baseline}, i})$.

### Evaluation Protocol

**Test setup**:
- "For two competing algorithms A and B, they will first play as Landlord and Peasants, respectively, for a given deck. Then we switch the sides, i.e. A takes Landlord position and B takes Peasants position, and they play the same deck again." (p. 6, also Table I).
- This is **deal mirroring** (same initial cards, swapped roles) to reduce variance from deal luck.

**Number of games**: "We execute the test for 10000 episodes every 30 minutes." (p. 6).
- 10,000 games per test checkpoint.
- Tests run every 30 mins during training.

**Role-specific evaluation**: 
- Results shown separately for Landlord and Peasants (Figures 6–8).
- [Interpretation, figure readings re-checked at high resolution] The *gap between each method and the re-trained DouZero curve* is larger for Peasants than for Landlord: at 6×10⁵ steps, opponent modeling ≈ +0.10 ADP (Peasants) vs ≈ +0.05 (Landlord); coach network ≈ +0.20 (Peasants) vs ≈ +0.07 (Landlord). Do not confuse these gaps with the much larger rise of every curve over training (≈ −2.5 → ≈ 0 for Peasants), which is ordinary learning progress shared by the baseline.
- Paper's rationale (p. 6–7): "Knowing the hand cards of the next player helps Peasants...cooperate with the teammate better. Whereas for the Landlord, knowing the hand cards of next player indeed helps to make decisions, but if the hand is weak, even having such information can not help a lot."

**Cross-play**: Yes, implicit. Each algorithm plays as both Landlord and Peasants against the fixed DouZero baseline.

**Multiple random seeds**: Not reported. The paper shows single training curves (Figures 6–8), not error bars or confidence intervals. This is a **weakness**.

**Botzone evaluation** (section D, p. 7):
- Model uploaded to Botzone (online competitive platform).
- "Elo rating system" used; note: "DouDizhu Elo ranking on Botzone suffers from some fluidity due to the nature of high variance of this game. One bot plays less than 10 Elo rating games on average every day so that it may take a lot of time to achieve a stable ranking." (p. 7).
- Result: "Our DouDizhu AI has always ranked top five, even ranked first for several months" (p. 7), vs. DouZero's rank ~20.

### CRITICAL INTERPRETATION: What the Evaluation Actually Shows

**The plotted learning curves (Figures 6–8) are early-stage training snapshots, not final validation:**
1. All models remain **weaker than the released DouZero baseline** throughout the plotted training range (6×10⁵ steps). ADP values stay negative (losing to baseline).
2. **Single training runs shown** (no multiple seeds, no error bars, no statistical tests). With high variance in DouDizhu, these curves could vary significantly with different random seeds.
3. The paper reports **NO head-to-head final matchup** between DouZero+ (at convergence) and released DouZero (at convergence). The Botzone ranking (top-5 vs DouZero's ~20) is the only evidence of final-state competitive strength, but Botzone uses <10 games/day/bot and has acknowledged "fluidity" due to variance.
4. "Training steps" not clearly defined in paper (wall-clock time unknown; could be a source of hidden assumptions).

**For a CS230 project:** The method-vs-baseline gaps in these early-training curves are small (≈0.05–0.2 ADP at 6×10⁵ steps, single runs), so they cannot be claimed as "DouZero+ beats DouZero" without multi-seed, confidence-interval evaluation. On the positive side, the comparisons were made at *small training budgets*, which is exactly the regime a CS230 project can afford.

**A telling contrast [Interpretation]:** feeding the *true* next-player hand ("maphack", Fig. 5) helps Peasants by ≈0.6 ADP at ≈1×10⁵ steps, while feeding the *learned prediction* of that hand (Fig. 6) helps Peasants by only ≈0.1 ADP at 6×10⁵ steps. These are measured at different training budgets, so this is suggestive rather than conclusive, but it implies most of the value of the hidden information is not being captured by the prediction model — an open question worth studying.

### Ablation Studies

**Experiment 1: Maphack baseline** (Figure 5, pp. 6–7):
- Add ground-truth opponent hand directly to state features (no learning; just provide it as an input feature).
- **Result** (reading from Figure 5, run only to 1×10⁵ steps, not 6×10⁵): 
  - Landlord: Maphack ≈−1.3 at ≈1.1×10⁵ steps vs re-trained DouZero ≈−1.5 (Maphack advantage ≈0.2 ADP). Note the Maphack Landlord curve starts *below* the baseline (≈−2.6 vs ≈−2.3) and overtakes it at ≈0.6×10⁵ steps.
  - Peasants: Maphack ≈−0.3 vs re-trained DouZero ≈−0.93 (Maphack advantage ≈0.6 ADP).
- **Interpretation**: An (unfair, test-time-cheating) upper bound on the benefit of knowing the next player's true cards. Oracle information helps, especially Peasants. The learned prediction model (Exp. 2) recovers only a small part of this gap (≈0.1 ADP for Peasants at 6×10⁵ steps), though the two experiments use different training budgets.

**Experiment 2: Opponent Modeling Alone** (Figure 6, pp. 6–7):
- Train prediction model concurrently with decision model.
- **Result** (reading from Figure 6, comparing to released DouZero baseline): 
  - Both curves (opponent modeling and re-trained DouZero) rise together from ≈−2.4 (Landlord) / ≈−2.6 (Peasants) at step 0.
  - Landlord at 6×10⁵ steps: opponent modeling ≈−0.82 vs re-trained DouZero ≈−0.87 — the curves nearly overlap for the whole run (gap ≈0.05).
  - Peasants at 6×10⁵ steps: opponent modeling ≈−0.12 vs re-trained DouZero ≈−0.22 (gap ≈0.1, appearing after ≈3×10⁵ steps).
  - The authors say the models "perform worse than DouZero at first because the network has to take more features as input and has more neurons, which will slow down learning" (p. 6); at the plotted resolution this early deficit is barely visible.
- **Authors' note**: "introducing opponent modeling to DouZero mainly improves the performance of models of Peasants" (p. 7).
- **Critical interpretation**: Both the new DouZero+ variant and the re-trained baseline DouZero remain **weaker than the released DouZero** throughout the plotted range. The learning curves show training progress (models improve from bad to less bad), not superiority over the released model.

**Experiment 3: Coach Network Alone** (Figure 7, pp. 6–7):
- Apply deal filtering (threshold $\beta$ up to 0.3).
- **Result** (reading from Figure 7, comparing to released DouZero baseline): 
  - Landlord at 6×10⁵ steps: coach ≈−0.80 vs re-trained DouZero ≈−0.87 (gap ≈0.07; the gap is visibly larger mid-training, ≈0.1–0.15 around 2–4×10⁵ steps).
  - Peasants at 6×10⁵ steps: coach ≈0.0 vs re-trained DouZero ≈−0.22 (gap ≈0.2; the coach curve reaches roughly parity with the released DouZero by the end).
  - The coach curve separates from the baseline earlier and more clearly than the opponent-modeling curve does.
- **Authors' note**: "The improvement of coach network is more obvious than opponent modeling" (p. 8).
- **Critical interpretation**: Coach filtering trains faster (steeper curve), but all models remain weaker than released DouZero at the end of the plotted range.

**Experiment 4: Both Improvements** (Figure 8, pp. 8–9):
- Combine opponent modeling + coach network.
- **Result** (reading from Figure 8, comparing to released DouZero baseline): 
  - Landlord at 6×10⁵ steps: both ≈−0.77, coach-only ≈−0.79, re-trained DouZero ≈−0.87.
  - Peasants at 6×10⁵ steps: both ≈+0.08, coach-only ≈+0.02, re-trained DouZero ≈−0.22. (Peasant-side "both" is the only plotted curve that ends slightly *above* zero, i.e. marginally ahead of the released DouZero as Peasants.)
  - The "both" and "coach-only" curves nearly overlap (difference ≈0.02–0.06 ADP, single runs), so the added value of opponent modeling on top of the coach is within what could plausibly be run-to-run noise [Interpretation].
- **Authors' note**: "When the performance of the models reaches a certain level, achieving a little improvement is very difficult so the progress that combining the two methods makes is not that apparent. However, further improvement still proves the effectiveness of combination of the two methods." (p. 8).
- **Critical interpretation**: All models remain weaker than the released DouZero baseline throughout training. The value of the combination is modest; coach filtering is the dominant improvement.

**Case study** (Table I, p. 7):
- Three hand examples from Botzone, with predicted vs. actual Landlord outcomes.
- Case 1: P_win = 0.9932 → Landlord Win (strong hand).
- Case 2: P_win = 0.1726 → Landlord Lose (weak hand).
- Case 3: P_win = 0.5843 → Landlord Lose (balanced, but Peasants cooperated better).
- Illustrates that coach network learns to predict balance correctly.

### Critical Evaluation

**Strengths**:
1. **Clear ablation**: Opponent modeling and coach network evaluated separately (Figures 5–7) and combined (Figure 8). Easy to see contribution of each.
2. **Deal-mirroring evaluation**: Reduces noise from high-variance deals, more rigorous than single-role plays.
3. **Practical validation**: Botzone results (top-5 ranking) demonstrate real-world effectiveness beyond lab benchmarks.
4. **Reproducible methodology**: Closely follows DouZero's protocol; implementation based on DouZero (mentioned p. 6).
5. **Per-role breakdown**: Separate Landlord/Peasant results clarify which component helps whom.

**Weaknesses**:
1. **No error bars / confidence intervals**: Single training curve per experiment. With high variance in DouDizhu, multiple seeds should be standard. [Interpretation] RL community increasingly expects this; omission here is concerning.
2. **Statistical significance not assessed**: No p-values, CIs, or significance tests. The method-vs-baseline gaps (≈0.05–0.2 ADP at 6×10⁵ steps) come from single training runs with no confidence intervals. With high variance in DouDizhu, are these differences robust or noise?
3. **Wall-clock time not compared**: Figures show "training steps (1e5)" on x-axis, not wall-clock time. Coach filtering reduces number of games played, but does it reduce total training time after accounting for overhead? Not clear.
4. **Training coupling not specified**: The paper says the prediction and decision models are trained "concurrently" but does not say whether decision-model gradients flow into the prediction model, or how the two updates are scheduled. Reproducibility issue.
5. **Threshold $\beta$ scheduling not detailed**: How does $\beta$ increase from 0 to 0.3? Is it linear, adaptive? Not specified.
6. **Coach network loss not specified**: Is it MSE or cross-entropy? Binary (win/loss) or continuous (point differential)? Not stated.
7. **Comparison fairness**: DouZero baseline is open-source but possibly from an earlier checkpoint than DouZero+ (both trained on same data, but for different durations?). Not entirely clear.
8. **Maphack experiment is upper bound but unfair**: Maphack has access to true opponent cards; unrealistic for deployment. Useful as a sanity check but doesn't prove opponent modeling is well-tuned.

**What effect sizes imply** (from learning curves, comparing to released DouZero baseline):
- Method minus re-trained-DouZero gap at 6×10⁵ steps (approximate figure readings):
  - Opponent modeling alone: Peasants ≈+0.10 ADP, Landlord ≈+0.05 ADP (curves nearly overlap).
  - Coach filtering alone: Peasants ≈+0.20 ADP, Landlord ≈+0.07 ADP (larger mid-training).
  - Both: Peasants ≈+0.30 ADP, Landlord ≈+0.10 ADP; only ≈0.02–0.06 better than coach-only.
- **Critical caveat**: Every curve is measured *against the released, fully trained DouZero*; all curves except the Peasant-side "both" end at or below zero, i.e. the new models are still mostly weaker than the released DouZero at the end of the plotted training. No final head-to-head matchup is reported in the paper.
- Botzone ranking (top-5 vs. DouZero's ~20th) suggests real-world effectiveness, but Elo rating at Botzone is noisy (<10 games/day per agent; fluidity acknowledged in paper).
- **Practically significant** for a follow-up work; demonstrates that mature RL systems can still benefit from auxiliary tasks and curriculum, but improvements are incremental.

**Statistical rigor**: Low. No CIs, no multiple seeds, no significance tests. Botzone ranking is noisy ("fluidity") and based on <10 games/day; top-5 ranking is real but could be partly luck.

### What I Would Improve for a CS230 Project

1. **Multiple seeds**: Run 5–10 independent training runs; report mean ± std of ADP at key checkpoints.
2. **Confidence intervals**: 95% CI for ADP differences via bootstrap or t-test.
3. **Wall-clock time**: Report total training time (not just step count) for DouZero vs. DouZero+. Does coach filtering save time in practice?
4. **Hyperparameter ablations**: Vary the $\beta$ schedule; and, as a *design* comparison, contrast the paper's separate prediction model with a shared-trunk auxiliary-loss variant (weight $\alpha$) — this is a modification, not something the paper did.
5. **Opponent diversity**: Test against multiple prior versions of DouZero (e.g., checkpoint 1e5, 2e5, 3e5) to evaluate transfer / opponent generalization.
6. **Larger sample size**: 10k games per test is good, but 50k would reduce noise even more.
7. **Statistical tests**: Paired t-test or Mann-Whitney U test for ADP differences.

---

## 10. Experiments and Ablations

### Experiment List

| **Exp.** | **Name** | **Setup** | **What changed** | **Held constant** | **Metric** | **Result** | **Conclusion** | **Critical note** |
|---|---|---|---|---|---|---|---|---|
| 1 | Maphack (pre-expt.) | Add ground-truth next-opponent hand directly to state | State features | Network architecture, training otherwise identical | ADP | Maphack Landlord ≈−1.3, Peasants ≈−0.3 at ≈1.1×10⁵ steps vs re-trained DouZero ≈−1.5, ≈−0.93 (Fig. 5) | Upper bound: oracle hand info helps ≈0.2 (Landlord) / ≈0.6 (Peasants) ADP; shows explicit opponent info is valuable | No learning; only 1×10⁵ step range (shorter than other expts); unfair but useful sanity check |
| 2 | Opp. Modeling alone | Train prediction model + decision model concurrently (outputs concatenated with state features) | Prediction model added; cross-entropy loss on opponent hands | Decision network archi, self-play setup, reward | ADP vs released DouZero baseline | At 6×10⁵ steps (Fig. 6): Landlord ≈−0.82 vs re-trained DouZero ≈−0.87; Peasants ≈−0.12 vs ≈−0.22 | Small gain, mainly for Peasants (authors: "mainly improves the performance of models of Peasants") | All models remain negative (weaker than released DouZero); no error bars; single run |
| 3 | Coach network alone | Apply deal filtering with threshold β increasing 0 → 0.3 | Coach network + rejection sampling of games | Otherwise DouZero | ADP vs released DouZero baseline | At 6×10⁵ steps (Fig. 7): Landlord ≈−0.80 vs ≈−0.87; Peasants ≈0.0 vs ≈−0.22; separates earlier than opponent modeling | Coach filtering more effective than opponent modeling alone; faster convergence; focuses on balanced deals | Improvement curve steeper than opponent modeling; but all models still negative throughout; diminishing returns visible |
| 4 | Both (combined) | Opponent modeling + coach filtering | Both techniques together | Network archi, training setup | ADP vs released DouZero baseline | At 6×10⁵ steps (Fig. 8): Landlord both ≈−0.77 / coach ≈−0.79 / DouZero ≈−0.87; Peasants both ≈+0.08 / coach ≈+0.02 / DouZero ≈−0.22 | Combined approach shows modest gain over coach-only; author note: "little improvement [at high perf]...further improvement still proves effectiveness" | Both and coach-only curves nearly overlap (diminishing returns); all models remain weaker than released DouZero; incremental gain from opponent modeling |
| 5 | Botzone validation | Upload final model to Botzone competitive ladder (~400 bots) | Final model after training | Botzone's Elo system; all prior designs competition | Botzone Elo rank | Top-5 rank (several months at #1); DouZero baseline ~20th | Real-world validation: DouZero+ is competitive on public leaderboard | Note: Elo is noisy (<10 games/day), mostly heuristic rule-based bots, rank fluidity acknowledged |

### Reproducibility at Small Scale for CS230

**Feasible small-scale reproduction**:
1. **Coach filtering only** (easier):
   - Use RLCard (simple RL environment).
   - Train baseline agent for N steps → record ADP.
   - Add coach network trained on initial deals.
   - Filter games with β ∈ [0.3, 0.7] (looser threshold than paper's 0.3).
   - Re-train, measure improvement.
   - Expected outcome: 20–50% speedup (fewer games played); ~0.2–0.3 ADP improvement.
   - Compute: 1 GPU, 8–16 hours training.

2. **Opponent modeling only** (moderate):
   - Train decision network + add prediction head (multi-head classifier).
   - Prediction targets: ground-truth opponent hand (available in RLCard).
   - Measure ADP improvement.
   - Expected outcome: 0.1–0.2 ADP improvement for your role; larger for cooperative role.
   - Compute: 1 GPU, 16–24 hours.

3. **Both** (hard but doable):
   - Combine above; full DouZero+ pipeline.
   - Requires choosing the unspecified details yourself (how prediction and decision models are coupled; the $\beta$ schedule).
   - Compute: 1 GPU, 24–48 hours.

**Shrinking strategy**:
- Smaller networks: DouZero uses 6 MLP layers (512 hidden); use 3–4 layers (256 hidden).
- Shorter training: DouZero trains ~6e5 steps (2–3 weeks on 4 GPUs); run 1e5 steps (1–2 days on 1 GPU) as proof-of-concept.
- Fewer parallel actors: DouZero uses 16+ actors; use 2–4.
- Faster evaluation: 1000 games per checkpoint instead of 10,000.

---

## 11. Compute and Engineering

**Hardware** (reported in paper, p. 6):
- Server: 4 Intel Xeon Gold 6252 CPUs @ 2.10 GHz (total ~48 cores).
- GPU: 1× GeForce RTX 2080Ti.
- Storage: Not specified (DouZero typically uses ~100 GB for replay buffer).

**Training duration**: Not explicitly stated. 
- Paper shows curves up to 6e5 training steps.
- The paper does not define a "training step" (learner update? environment frame?) and gives no wall-clock time, so **the duration cannot be recovered from the paper**. Evaluations ran "every 30 minutes" (p. 5), but the number of evaluation points per plotted curve is not stated either.
- [Interpretation] For your own planning, measure throughput on your hardware (frames/second in DouZero's logs) and fix a wall-clock or frame budget per run; do not rely on an extrapolated number.

**Number of games/samples**:
- 10k games per test checkpoint (p. 6).
- "Every 30 minutes" → ~48 checkpoints per day → ~480k games/day during training.
- Total: Probably 10–20M games over full training.

**Parallel actors**: 
- Not explicitly specified. DouZero uses 16+ actor processes. [Interpretation] DouZero+ likely uses same or similar (inherited from DouZero).

**Memory**: 
- Decision network: ~50–100 MB (DouZero architecture: LSTM + 6 MLP layers + output layer).
- Prediction network: ~50–100 MB (separate model: LSTM + 5 MLP layers + multi-head output for each card rank).
- Coach network: ~10–20 MB (small model: embedding layer + ~5–10 FC layers).
- Replay buffer: Hundreds of MB to few GB (trajectory storage; DouZero typically ~100 GB, but includes checkpoints).
- Total: ~200 MB model + 1–2 GB active buffer (estimates based on DouZero infrastructure).

**Distributed training**: 
- No distributed training (single server with 1 GPU, multi-core CPU for actors).
- Replay buffers shared via inter-process communication (IPC) on same server.

**Implementation tricks**:
1. **Action encoding**: Card actions encoded as 4×15 matrices (suits × ranks) for efficient convolution; inherited from DouZero.
2. **LSTM + MLP combo**: LSTM encodes move history; MLP processes combined state+action. Reduces memory vs. full-state MLP.
3. **Legal action pre-filtering**: Only valid moves passed to network, reducing output dimension.
4. **Concurrent coach training**: Coach network trained in-loop, not pre-trained separately (faster adaptation to current policy).
5. **Periodic actor sync**: Global networks synced to actors every N steps (reduce communication overhead).

**Reproduction difficulty**: **Moderate**

**Justification**:
- ✅ **Easier than DouZero**: Builds directly on DouZero codebase (reuse environment, training loop, action encoding).
- ✅ **Ablatable**: Opponent modeling and coach filtering are modular; can implement separately.
- ✅ **Less compute than DeltaDou**: Single GPU; days, not months.
- ⚠️ **Harder than simple baselines**: Requires training a second (prediction) network alongside the decision network, plus the coach.
- ⚠️ **Unspecified details**: model coupling, $\beta$ schedule and accept rule, coach loss — you must choose and report them.
- ⚠️ **High variance**: DouDizhu is high-variance; need multiple seeds for statistical significance (not done in paper).

**What a scaled-down reproduction looks like**:
1. Use RLCard or DouZero codebase (skip building game engine).
2. Train baseline agent (DouZero) for 1e5 steps; measure ADP baseline.
3. Add coach network (5-layer MLP) trained on initial deals; filter top/bottom 25% (β=0.25).
4. Retrain for 1e5 steps; measure ADP improvement.
5. Add prediction model (3-layer MLP + multi-head output); retrain for 1e5 steps; measure combined ADP.
6. Compare ADP improvements. Expected result: coach filtering ~0.1–0.2 ADP improvement; opponent modeling ~0.05–0.1 ADP.
7. Compute cost: 1 GPU (V100 or A100), 24–48 hours total.

---

## 12. Limitations

### A. Explicitly mentioned by the authors

1. **Architectural constraints** (p. 8): "To better show the effect of our improvement, we do not make changes on the architectures of neural networks in DouZero unless necessary. We plan to try other neural networks such as convolutional neural networks like ResNet."
   - **Implication**: Current LSTM + MLP architecture may not be optimal; ResNet could extract spatial patterns from card matrices more efficiently.

2. **Search not integrated** (p. 8): "We find that there are still some cases where the model can not make good decisions. We hope to combine search with our AI to enhance the performance as search plays an important role and performs well in research about game AI."
   - **Implication**: Pure RL cannot fully solve DouDizhu; Monte Carlo Tree Search (MCTS) or similar could improve edge cases.

3. **Sample efficiency** (p. 8): "We will investigate how to improve the sample efficiency with experiment replay as it still costs a lot of time even utilizing our 'coach network'."
   - **Implication**: Even with deal filtering, training is slow. Experience replay (storing past games and re-sampling) could help.

### B. Inferred from the methodology

1. **Opponent-policy shift for the prediction model**:
   - Ground-truth hands are used only as *labels* for the prediction model; the decision model receives the prediction model's *outputs* both in training and at deployment, so there is no train/test mismatch in the decision model's inputs.
   - [Interpretation] The real risk is different: the predictor learns "what does the next player hold, given how *self-play* agents play?" Against opponents who play differently (humans, Botzone bots), the move history carries different evidence, so the predictions — and the decision model that trusts them — may be miscalibrated. The paper does not measure prediction accuracy or calibration at all, in self-play or against other opponents.

2. **Deal filtering changes training distribution**:
   - Coach network filters out high-variance deals (very easy or very hard).
   - At deployment, the agent will encounter *all* deals (including those it filtered during training).
   - [Interpretation] Evaluation on uniform random deals (Figure 6–8) may not reflect Botzone performance (real distribution unknown).
   - This is acknowledged but not deeply analyzed.

3. **Limited theoretical understanding**:
   - No convergence proof, regret bounds, or game-theoretic analysis.
   - Self-play with simultaneous updates in multi-agent games can exhibit non-convergent behavior (cyclic strategies, etc.).
   - [Interpretation] Reliant on empirical convergence in practice; risks silent failure modes.

4. **Cooperative coordination**:
   - Peasants must cooperate (via implicit communication through play history).
   - Opponent modeling helps but doesn't explicitly teach cooperation.
   - [Interpretation] Cooperation emerges from self-play, not from explicit mechanism (e.g., information sharing, centralized value function).
   - Fragile if one agent's policy drifts.

5. **Coach network overfitting to training dataset**:
   - Coach network trained on RL agent's game outcomes (which are non-stationary during training).
   - Early on, agent is weak; outcomes reflect deal + weak play.
   - Late in training, outcomes reflect deal + strong play.
   - [Interpretation] Coach's estimate of deal balance ($P_{\text{win}}$) may be biased (assumes agent strength is constant).

6. **Limited generalization to other games**:
   - Deal filtering may not transfer to games without high initial-state variance (e.g., Go, Chess).
   - Opponent modeling requires ground-truth hidden state (available in digital card games, not in all domains).
   - [Interpretation] Techniques are DouDizhu-specific; broader applicability unclear.

7. **Hyperparameter tuning not transparent**:
   - $\alpha$ (prediction loss weight, if used as a combined loss): **not specified in paper**. It's unclear whether the prediction and decision models are trained with a joint weighted loss or independently.
   - $\beta$ schedule (deal filtering threshold): increases from 0 to ≤0.3, but the exact functional form (linear, exponential, adaptive?) is not specified (p. 5).
   - Likely tuned via grid search or manual adjustment, not justified theoretically.
   - [Interpretation] Reproducibility is at significant risk; the exact training procedure cannot be recovered from the paper alone.

---

## 13. Future Work

### A. From the paper (explicitly stated)

1. **Neural network architectures** (p. 8, §VI Conclusion):
   - "We plan to try other neural networks such as convolutional neural networks like ResNet."
   - **Classification**: Directly implementable as CS230 project modification.
   - **Difficulty**: Moderate (swap LSTM+MLP for ResNet; adjust input shapes).
   - **Later papers**: No paper in this set (ending 2022) explores ResNet for DouDizhu (DouZero+ is among the latest).
   - **Feasibility**: High for CS230. Could show whether spatial structure in card matrices is learnable by CNNs.

2. **Search integration** (p. 8):
   - "We hope to combine search with our AI to enhance the performance as search plays an important role and performs well in research about game AI."
   - **Cited examples**: [43] Bouzy et al. (bridge search), [44] Ariyurek et al. (MCTS for video game testing).
   - **Classification**: Difficult for a single-quarter CS230 project (search + RL integration is complex).
   - **Later papers**: ReBeL (NeurIPS 2020, paper 09) combines deep RL + search. Not directly on DouDizhu but shows the approach.
   - **Feasibility**: Low for CS230 without significant prior MCTS experience.

3. **Experience replay** (p. 8):
   - "We will investigate how to improve the sample efficiency with experiment replay as it still costs a lot of time even utilizing our 'coach network'."
   - **Cited example**: [45] Zhang & Sutton "A deeper look at experience replay" (2017).
   - **Classification**: Moderate difficulty; directly implementable.
   - **Later papers**: None in this set explicitly target experience replay for DouDizhu.
   - **Feasibility**: Moderate for CS230. Could store past games and re-sample; measure speedup in training wall-clock time.

4. **Transfer to other games** (p. 8):
   - "We will also try to transfer our methods to other games for stronger game AIs."
   - **Classification**: Depends on target game; for DouDizhu-like games (e.g., Tractor, another Chinese card game), moderate.
   - **Feasibility**: Low for single-quarter CS230 (requires learning new game rules, environment setup).

### B. Implied future work [Interpretation]

1. **Deal-filtering curriculum with adaptive threshold**:
   - Current β schedule is fixed (0 → 0.3). What if β adapts based on training progress?
   - E.g., β = α * (current_ADP / max_ADP) with learned α.
   - **Classification**: Directly usable as CS230 project idea.
   - **Feasibility**: Moderate (requires grid search over schedule; similar to current work).

2. **Multi-task opponent modeling**:
   - Predict not just opponent hand, but also opponent's likely next action.
   - Shared early layers, two output heads: one for hand prediction, one for action prediction.
   - **Classification**: Directly usable as CS230 project idea.
   - **Feasibility**: Moderate (add action prediction head; similar complexity to current hand prediction).

3. **Opponent modeling with belief updates**:
   - Current prediction model is static (observes state, outputs hand distribution).
   - A Bayesian approach: maintain posterior over opponent hands; update via Bayes' rule as opponent plays.
   - Requires modeling opponent policy and likelihood of observations.
   - **Classification**: Possible but difficult (requires Bayesian modeling, likelihood computation).
   - **Feasibility**: Hard for CS230 without Bayesian ML background.

4. **Deal filtering for evaluation bias**:
   - Coach filtering changes training distribution; what's the evaluation-time performance on "hard" deals?
   - Compare ADP on filtered deals vs. full distribution.
   - **Classification**: Directly usable as CS230 analysis idea.
   - **Feasibility**: Easy (just evaluate on subsets; visualization + reporting).

5. **Cooperative conditioning**:
   - Explicitly train Peasants to maximize *joint* reward (both Peasants' score sum) rather than individual Q-values.
   - Use centralized value function at Peasant level, decentralized execution.
   - **Classification**: Moderate difficulty; explores cooperative RL.
   - **Feasibility**: Moderate (modify reward structure, train shared value network for Peasant pair).

---

## 14. Connection to Other Papers

### Direct lineage (in this paper set)

**DouZero (Zha et al., ICML 2021, paper 01)** ← **DouZero+ (Zhao et al., arXiv 2022, paper 02)**

| **Component** | **DouZero** | **DouZero+** | **Change** | **Why** |
|---|---|---|---|---|
| **Core algorithm** | Deep Monte Carlo | Deep Monte Carlo | Kept identical | MC proven effective; no need to change fundamentals |
| **Network architecture** | LSTM (move history) + 6-layer MLP (512 hidden) | Same LSTM + 6-layer MLP (512 hidden) for decision; added prediction head | Added prediction model (multi-head classifier) | Explicit opponent hand prediction to improve decision-making |
| **Action encoding** | 4×15 card matrix | Same | Kept identical | Efficient; generalizes over unseen actions |
| **State features** | Card matrices, play counts, move history | Same as input to decision; prediction model takes same state features | Kept identical | Proven representation |
| **Self-play** | Simultaneous actors + centralized learner | Same | Kept identical | Scalable; allows efficient parallel data generation |
| **Reward** | WP or ADP-based | ADP-based (WP available but not used) | Minor preference | ADP is finer-grained; more stable learning signal |
| **Training data** | All random deals | Filtered by coach network (balanced deals only) | Deal filtering added | Reduce variance; speed up learning |
| **Bidding** | Supervised learning (separate network) | Same (not modified) | Kept identical | Orthogonal to main improvements |

**Summary of kept components**:
- DMC algorithm (no change to Bellman equations).
- Architecture (LSTM + MLP).
- Action/state encoding.
- Self-play infrastructure.

**Summary of changes**:
1. **Auxiliary prediction task**: Adds supervised learning head (ground-truth opponent hands).
2. **Deal filtering**: Post-hoc rejection sampling; no change to game mechanics or reward.

### Related ideas from other papers in set

**DeltaDou (Jiang et al., IJCAI 2019, paper 04)** → DouZero+ [Interpretation]

| **Aspect** | **DeltaDou** | **DouZero+** | **Relation** |
|---|---|---|---|
| **Card inference** | Bayesian inference + FPMCTS; models opponent policy, uses it to sample opponent hands | Supervised multi-head classifier; ground-truth labels from environment | DeltaDou is more principled (Bayesian), DouZero+ is simpler (supervised, one forward pass, no search) |
| **Computational cost** | 2 months on 68 CPUs (DeltaDou §5.2) | Training duration not stated in the paper | DouZero+ inherits DouZero's search-free, single-forward-pass inference, so it is far cheaper per move |
| **Search integration** | Integrated MCTS into training | No search; pure RL | Trade-off: DeltaDou explores more thoroughly; DouZero+ scales better to large action spaces |
| **Human knowledge** | Kicker network pre-trained on heuristics | None | DouZero+ purer; no hand-crafted components |

**[Interpretation]**: DouZero+ borrows the *idea* of card inference from DeltaDou but implements it via supervised learning instead of Bayesian inference. Not a direct algorithmic descendant, but addresses the same problem (agent needs to know/guess opponent hands) with a simpler method.

**Suphx (Li et al., Mahjong, 2020, paper 08)** → DouZero+ [Interpretation]

| **Aspect** | **Suphx** | **DouZero+** | **Relation** |
|---|---|---|---|
| **Oracle guiding** | Oracle agent with perfect information; gradually drop access to hidden info during training | Prediction model trained on ground-truth opponent hands (but only during training) | Suphx explicitly uses perfect-information oracle to accelerate learning; DouZero+ uses supervised labels (related idea, different implementation). Both leverage privileged information at train time |
| **Deployment** | Oracle hidden at deployment; learned policy is normal (imperfect info) | Prediction model predicts at deployment (not oracle) | Suphx relies on learned policy to handle dropped information; DouZero+ uses predicted labels as substitutes |
| **Multi-task learning** | Global reward prediction (auxiliary task for learning signals in complex scoring) | Opponent hand prediction (auxiliary task for feature learning) | Both use auxiliary tasks; Suphx for reward learning, DouZero+ for belief learning |

**[Interpretation]**: Suphx's "oracle guiding" and DouZero+'s "opponent modeling" are **conceptually similar**—both use privileged information (perfect knowledge of hidden state) during training to guide learning. The execution differs: Suphx drops information gradually; DouZero+ relies on supervised prediction. Not direct lineage (different games, different approaches), but parallel evolution of the same idea.

### Broader context (outside this set, mentioned in paper)

**Opponent modeling in RL** (cited, not in set):
- Heinrich & Silver (2016), He et al. (2016), Knegt et al. (2018): Opponent modeling in arcade games, Q-learning; showed empirical benefits.
- DouZero+ cites these as precedent for combining opponent modeling + RL (p. 3, §II-B).

**Evaluation and metrics**:
- Exploitability (used in poker, cited [41]): DouZero+ notes it's intractable for DouDizhu (huge game tree); instead uses WP/ADP.

---

## 15. Project Ideas for Me

### Project Idea 1: Adaptive Deal Filtering via Curriculum Learning

**Research question**: Can an adaptive deal-threshold schedule (β(t)) that adjusts based on training progress outperform the fixed schedule used in DouZero+?

**Baseline**: DouZero+ with fixed β (0 → 0.3 by end of training).

**Modification**: 
- Implement adaptive β schedule, e.g., β(t) = β_max * min(1, training_steps / τ), where τ is a burn-in period.
- Or, β(t) = β_max * (1 - exp(-training_steps / τ)), a smoother ramp.
- Or, data-driven: β(t) = argmax_β (test_ADP), tuned online via Bayesian optimization on validation set.

**Hypothesis**: 
- Fixed schedule in DouZero+ is conservative (β only up to 0.3).
- Early training (weak agent): agent can't exploit balanced deals, so even harder deals are useful. β should be high (accept all deals).
- Late training (strong agent): agent has learned basics; focus on balanced deals refines strategy. β should be low (filter to ~0.5 win prob).
- **Prediction**: Adaptive β should reach higher final ADP (0.1–0.2 improvement over fixed β) and/or train faster (20–30% speedup).

**Experiment**:
1. Train DouZero+ with fixed β (control).
2. Train DouZero+ with adaptive β (test).
3. Measure ADP over time (Figures 6–8 style).
4. Measure wall-clock training time (DouZero+ does not report this; this project fills the gap).
5. Plot β(t) schedule; show when filtering kicks in.

**Metrics**: 
- ADP at fixed checkpoints (1e5, 2e5, …, 6e5 steps).
- Wall-clock time to reach baseline ADP.
- Final ADP after convergence.

**Difficulty**: Easy
- Modify coach filtering threshold in training loop (5–10 lines of code).
- Minimal architectural changes.

**Compute**: Low
- [Estimate] A fixed budget per run on 1 GPU (e.g., 24–48 h); DouZero+ itself does not report its training time.
- Multiple runs (3–5 seeds) to get error bars.

**Novelty**: Incremental but meaningful. DouZero+ doesn't study adaptive schedules; this is a natural follow-up.

---

### Project Idea 2: Multi-Task Opponent Modeling (Hand + Action Prediction)

**Research question**: Does predicting both opponent hand *and* opponent action lead to better feature learning than hand prediction alone?

**Baseline**: DouZero+ opponent modeling (hand prediction only).

**Modification**:
- Prediction model: Add second output head to predict opponent's next action (which card combo they'll play).
- Action predicted as a classification problem over legal actions (or a regression over feature vector of the action).
- Shared early layers (LSTM + 5-layer MLP) benefit from both prediction tasks.
- Total loss: $\mathcal{L} = \mathcal{L}_{\text{decision}} + \alpha \mathcal{L}_{\text{hand}} + \beta \mathcal{L}_{\text{action}}$.

**Hypothesis**: 
- Action prediction forces the network to learn deeper dynamics (not just what opponent has, but what they *do*).
- Both tasks share a common bottleneck (early layers), so multi-task learning improves generalization.
- **Prediction**: 0.1–0.2 ADP improvement over hand prediction alone.

**Experiment**:
1. Implement action prediction head (multi-class classifier over legal actions).
2. Train with three variants: (a) hand only, (b) action only, (c) both.
3. Compare ADP curves (Figures 6–8 style).
4. Ablate $\beta$ (weight of action loss); show sensitivity.

**Metrics**: ADP, learning speed (steps to reach 90% of final ADP).

**Difficulty**: Moderate
- Add second output head + loss; backprop through shared layers.
- Action prediction requires mapping output to legal actions (non-trivial).

**Compute**: Medium
- [Estimate] A fixed budget per run on 1 GPU (e.g., 24–48 h); DouZero+ does not report its training time.
- Multiple seeds (3–5) + a small sweep of loss weights → ~1–2 weeks on 1 GPU [estimate].

**Novelty**: Moderate. Multi-task learning is standard; applying it to two prediction heads (hand + action) is straightforward but not obvious.

---

### Project Idea 3: Coach Network Variance Reduction Analysis

**Research question**: How much does deal filtering reduce training variance (in reward/return) and wall-clock training time? Is the variance reduction worth the filtering overhead?

**Baseline**: DouZero without deal filtering.

**Modification**: 
- Systematically study coach filtering (none, β=0.1, β=0.2, β=0.3, β=0.5).
- For each β, measure: (a) ADP curves, (b) ADP variance per checkpoint (across games or seeds), (c) % of games filtered, (d) wall-clock training time.

**Hypothesis**: 
- Filtering reduces variance in game outcomes (ADP smoother over time) because unbalanced deals are removed.
- But filtering introduces overhead (coach network inference + rejection sampling).
- **Prediction**: For β=0.3, variance reduced by ~30–40%; wall-clock time reduced by ~20–30% (net gain after overhead).

**Experiment**:
1. Train DouZero baseline (no filtering) for 6e5 steps; measure per-checkpoint ADP variance (e.g., over 100-game batches).
2. Repeat for β ∈ {0.1, 0.2, 0.3, 0.5, 1.0 (no filter)}.
3. Plot: (a) ADP over training steps (with error bands), (b) ADP variance vs. β, (c) wall-clock time vs. β, (d) % filtered vs. β.
4. Report computational overhead of coach network (inference time per game).

**Metrics**: 
- Variance in ADP (std dev of 100-game blocks).
- Wall-clock training time.
- Sample efficiency (ADP per wall-clock hour).

**Difficulty**: Easy
- Modify training loop to measure variance.
- Add timing instrumentation (Python `time` module).

**Compute**: Medium
- Multiple β values × multiple seeds → ~20–30 training runs.
- ~1–2 weeks on 1 GPU, or parallelizable over multiple GPUs.

**Novelty**: Empirical contribution. DouZero+ doesn't deeply analyze variance/speedup; this fills a gap and could inform practitioners.

---

### Project Idea 4: Opponent Modeling Generalization Across Opponent Strengths

**Research question**: Does a prediction model trained on games against RL agent at checkpoint T generalize well to predictions against opponents at checkpoint T' (different skill level)?

**Baseline**: DouZero+ prediction model trained concurrently (learns opponent policy as agent improves).

**Modification**:
- Train opponent prediction model against a fixed-checkpoint opponent (e.g., DouZero from paper 01, or an earlier checkpoint of your own agent).
- Test on the same fixed opponent (in-distribution) vs. evolving opponent (out-of-distribution).
- Measure prediction accuracy (e.g., per-rank card count error) + decision ADP (does bad prediction hurt play?).

**Hypothesis**: 
- Concurrent training (DouZero+ approach) adapts to changing opponent; transfer learning to different opponents is weaker.
- Fixed opponent (e.g., hand-crafted baseline) is more stable but potentially suboptimal.
- **Prediction**: Concurrent training has better ADP on evolving self-play opponents but worse on fixed-baseline opponents.

**Experiment**:
1. Train prediction models:
   - (A) Concurrent: updated alongside decision network (DouZero+).
   - (B) Against fixed checkpoint: train against DouZero from paper 01 (fixed policy).
   - (C) Against hand-crafted baseline: train against a rule-based opponent.
2. Test each model:
   - On games against evolving RL agent (self-play).
   - On games against fixed baseline.
3. Measure: (a) prediction accuracy (MSE of card counts), (b) decision ADP, (c) robustness (ADP on out-of-distribution opponents).

**Metrics**: 
- Prediction error (per-rank card count MAE).
- ADP on different opponent types.
- Generalization gap (in-distribution vs. out-of-distribution ADP).

**Difficulty**: Moderate
- Requires training multiple prediction models (orthogonal runs).
- Requires access to fixed baseline opponent (DouZero from paper 01 or similar).

**Compute**: Medium
- Multiple models × multiple evaluation settings → ~1–2 weeks on 1 GPU.

**Novelty**: Good. Generalization of opponent models is understudied in imperfect-info games; this is a natural research direction.

---

## 16. Important Implementation Components

### MUST UNDERSTAND AND IMPLEMENT/MODIFY

1. **Prediction Model (Multi-Head Classifier)**
   - Input: state features (same as DouZero's state encoding; ~320-dim).
   - Output: probability distribution over card counts for each rank (13 heads, each predicting 0–17 count).
   - Loss: per-rank cross-entropy loss on ground-truth opponent hand labels.
   - **Why**: Core innovation of DouZero+; supervised learning on hidden information improves feature learning. Outputs concatenated with state features to feed decision model.
   - **Implementation**: Add `nn.ModuleList` of output heads in PyTorch; split cross-entropy loss across heads; concatenate outputs with original state features for decision model input.
   - **Key question the paper leaves unanswered**: Does gradient from decision model loss backprop into prediction model? Paper does not specify.

2. **Coach Network (Regression/Classification)**
   - Input: initial card vectors (20-dim for Landlord, 17-dim per Peasant).
   - Output: $P_{\text{win}} \in [0, 1]$ (predicted Landlord win probability).
   - Loss: **not specified in paper** (likely MSE or binary cross-entropy on game outcome labels).
   - Training: Concurrent with decision/prediction models; labels = actual game outcomes.
   - Deal filtering: Reject games where $P_{\text{win}} \notin [\beta, 1-\beta]$ (accept only balanced deals).
   - **Why**: Implements curriculum learning via data selection; reduces training variance and time by filtering out one-sided deals.
   - **Implementation**: Train separate small MLP; call before each game to decide whether to play it.

3. **Concurrent Training (Specified: Separate Losses; Joint: Unspecified)**
   - Prediction model: Minimize $\mathcal{L}_{\text{pred}}$ (cross-entropy on opponent hands).
   - Decision model: Minimize $\mathcal{L}_{\text{dec}}$ (MSE on MC returns), using concatenated features (state + prediction outputs).
   - Coach network: Minimize $\mathcal{L}_{\text{coach}}$ (loss not specified) on game outcomes.
   - **Paper does NOT specify**: Whether these are trained with a single joint loss like $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{dec}} + \alpha \mathcal{L}_{\text{pred}}$, or independently. Reproducing this requires inferring the implementation.
   - **Why it matters**: Joint loss with shared gradients would couple learning differently than independent training. This is a critical reproducibility gap.

4. **Deal Filtering Loop in Game Generation**
   - Before each game, sample initial cards.
   - Forward coach network; check if $P_{\text{win}} \in [\beta_{\min}, \beta_{\max}]$.
   - If no, re-sample; if yes, proceed.
   - **Why**: Changes training data distribution; critical to understand throughput impact.
   - **Implementation**: `while coach_net(cards) outside [beta_min, beta_max]: cards = sample_new_deal()`.

### MUST UNDERSTAND CONCEPTUALLY

1. **Information Sets and Imperfect Information Handling**
   - Each player's observation is their own cards + play history; opponents' cards hidden.
   - State features are sufficient statistics of observable information.
   - Opponent modeling makes the hidden info explicit via prediction.
   - **Why**: Foundation of multi-agent imperfect-info RL; the prediction model is an attempt to invert the hidden state.
   - **Reading**: § 8 of this analysis.

2. **Monte Carlo vs. TD vs. Policy Gradient**
   - DouZero+ uses MC (unbiased, high-variance) not TD (biased, low-variance) or PG (policy-direct).
   - MC updates use empirical returns $G_t$; no bootstrapping.
   - **Why**: MC appropriate for episodic tasks (DouDizhu games end) and scales to large action spaces (less overestimation bias than DQN).
   - **Reading**: § 2, 4, 5 of this analysis; DouZero paper § 3.

3. **Self-Play and Non-Stationarity**
   - All three players improve concurrently; opponent policy changes during training.
   - No explicit regret minimization or game-theoretic guarantees.
   - Convergence relies on empirical stability (Nash equilibrium reachable if dynamics are well-behaved).
   - **Why**: Multi-agent RL is harder than single-agent; understanding non-stationarity is crucial for debugging.
   - **Reading**: § 7 of this analysis; RL textbooks on multi-agent learning (e.g., Shoham & Leyton-Brown).

4. **Auxiliary Tasks and Multi-Task Learning**
   - Prediction task shares early layers with decision task; gradients flow through both.
   - Auxiliary task improves feature learning even if not directly used at deployment (or if predictions are ignored).
   - **Why**: Design principle; prediction head is a cheap regularizer / feature extractor.
   - **Reading**: Multi-task learning survey (e.g., Ruder, 2017 "An Overview of Multi-Task Learning").

5. **Curriculum Learning and Distribution Shift**
   - Deal filtering changes the training distribution (away from uniform random deals).
   - At evaluation time, full distribution is encountered; potential for distribution mismatch.
   - **Why**: Understand when curriculum helps (sample efficiency) vs. hurts (generalization).
   - **Reading**: Curriculum learning review (e.g., Bengio et al., 2009).

### CAN REUSE AS INFRASTRUCTURE

1. **DouZero Codebase**
   - Game environment (rules, legal move generation).
   - Actor-learner architecture (parallel data generation, centralized training).
   - Network architecture (LSTM + MLP for DouDizhu).
   - Action encoding (4×15 card matrices).
   - **How to reuse**: Clone the repo; modify to add prediction + coach networks.
   - **Savings**: Weeks of engineering; focus on research (auxiliary tasks, deal filtering).

2. **RLCard**
   - Standardized environment interface (Gym-like).
   - Includes DouDizhu game (if not already available in DouZero).
   - Can simplify environment setup for your CS230 project.
   - **How to reuse**: Use RLCard's DouDizhu as a drop-in replacement; potentially faster than reimplementing game rules.
   - **Caveat**: May be slower than DouZero's hand-optimized environment.

3. **Action/State Encoding**
   - Card matrices (4×15), LSTM move history encoding: proven effective.
   - Reuse directly; don't re-invent representation.
   - **Why**: State encoding is domain-specific; DouZero's engineering is solid.

---

## 17. Questions I Should Ask My CS230 Staff

1. **On novelty and scope**:
   - "DouZero+ combines two orthogonal improvements (opponent modeling + deal filtering). For my CS230 project, should I aim to (a) reproduce both, (b) focus on one and study it deeply, or (c) propose a novel third component (e.g., adaptive β schedule)? How novel do you expect a CS230 final project to be?"

2. **On evaluation and statistics**:
   - "DouZero+ doesn't report error bars or statistical significance tests, citing high variance in DouDizhu. For my evaluation, is it sufficient to run a single seed with 10k games per checkpoint, or should I do multiple seeds (5–10) and report CIs/p-values? What's the bar for CS230?"

3. **On computational feasibility**:
   - "DouZero trains on 4 GPUs for days; I'll likely have 1 GPU for a few weeks. Which components of DouZero+ should I scale down (smaller network, shorter training, fewer actors) without losing the key insights? Should I evaluate on a smaller game (e.g., simplified DouDizhu with fewer cards)?"

4. **On baseline selection**:
   - "Should my CS230 project baseline be the original DouZero (from paper 01) or a simple hand-crafted rule-based bot? DouZero is harder to reproduce but more meaningful; rule-based is easier but less impressive. Any guidance?"

5. **On distributional generalization**:
   - "Deal filtering changes the training distribution (balanced deals only) but evaluation is on uniform random deals. DouZero+ doesn't deeply analyze this mismatch. Should I study this (e.g., performance on filtered vs. unfiltered deals) as a project contribution?"

---

## 18. Beginner Concepts I Need to Learn

| **Concept** | **Plain-language explanation** | **Where in this paper** | **Priority** |
|---|---|---|---|
| **Monte Carlo (MC) method** | Instead of estimating value at each step (TD), wait until the end of the game, sum all rewards, and use that sum as the training target. High-variance (noisy reward signal), but unbiased. | § 3 (Preliminary: describes DMC) and § 5 (equations 1) | **NOW** (core algorithm) |
| **Imperfect information** | Players don't see the full game state (e.g., opponent cards are hidden). Agents must infer hidden state from observations (own cards, play history). | § 8 (Imperfect Information) and § 1b (problem statement) | **NOW** (DouDizhu's main challenge) |
| **Self-play** | All players are neural networks trained concurrently; they play each other and improve from their own games. No fixed opponent, so the training signal is non-stationary. | § 7 (Self-Play) | **NOW** (RL methodology) |
| **Auxiliary task / Multi-task learning** | Train the network to solve multiple tasks simultaneously (e.g., predict action + predict value). Helps the network learn better representations. | § 6 (Loss Function) and § 15 (Project Idea 2) | **SOON** (design principle; DouZero+'s prediction head) |
| **Curriculum learning** | Start with easy tasks/games, gradually increase difficulty. Helps agent learn faster. DouZero+ does this by filtering deals (easy = balanced, hard = imbalanced). | § 4 (Training Algorithm) and § 5 (Eq. 5) | **SOON** (deal filtering is a form of curriculum) |
| **Distribution shift / Out-of-distribution generalization** | Network trained on one data distribution (e.g., balanced deals) may perform poorly on another (e.g., all deals). Common problem in ML; important to check. | § 12-B (Inferred limitations: deal filtering changes distribution) | **SOON** (evaluation gap in DouZero+) |
| **Regret minimization / Nash equilibrium** | Game-theoretic concepts: regret = how much worse you play than the best strategy in hindsight; Nash equilibrium = stable state where no player wants to change strategy unilaterally. Multi-agent RL aims for Nash or low regret. | § 1b (why CFR doesn't work for DouDizhu) and § 13 (theoretical gaps) | **OPTIONAL** (DouZero+ doesn't use these formally; helpful for advanced understanding) |
| **Action space / Large action spaces** | In DouDizhu, there can be 100+ legal actions per turn (many card combinations). Large action spaces break DQN (overestimation bias) but MC handles them better. | § 1 (Introduction: "massive number of possible moves") | **SOON** (why DouZero chose MC over DQN) |
| **Bidding phase** | DouDizhu has two phases: bidding (determine who is Landlord, based on hand strength) and card-play (play game). Paper trains bidding separately (supervised learning, not RL). | § 3 (Preliminary) and § 11 (Bidding is orthogonal) | **OPTIONAL** (not focus of DouZero+; good to know but not critical) |
| **Bomb multiplier** | In DouDizhu, bombs (4 of a kind) double the score for that round. Affects reward and strategy (should play bombs carefully). | § 9 (Metrics: "bomb will double the score") and card game rules | **SOON** (understand reward structure) |

---

## 19. Paper Scorecard

> **Cross-paper calibration:** this scorecard looks at this paper alone. Ranked against the whole pack for this Dou Dizhu project, the recommended verdict is **READ DEEPLY**. See `00_SYNTHESIS_AND_RECOMMENDATION.md` §20.9.

| **Dimension** | **Score (1–5)** | **Justification** |
|---|---|---|
| **Importance for my project** | **5** | DouZero+ builds directly on DouZero and introduces two clear, reproducible techniques (opponent modeling + deal filtering). Both are plausible CS230 modifications. |
| **Difficulty to understand** | **3** | Paper is clearly written and modular (two techniques independent). But multi-task learning, curriculum learning, and imperfect-info RL require background. No major conceptual gaps, but requires ~20 hours careful reading. |
| **Difficulty to reproduce** | **3** | Builds on DouZero codebase (large external dependency); components are modular and could be replicated. Key details (model coupling, $\beta$ schedule) not specified (reproducibility risk). Feasible for CS230 with 1 GPU but requires careful engineering. ~2–3 weeks of work for full reproduction. |
| **Relevance to Dou Dizhu** | **5** | Directly improves DouDizhu AI. Both improvements are game-specific (deal filtering exploits high variance; opponent modeling exploits imperfect info). No transfer to other domains obvious. |
| **Relevance to self-play** | **4** | Self-play is core (concurrent multi-agent training). Paper doesn't innovate on self-play itself, but shows how to integrate opponent modeling and curriculum into self-play. Useful case study. |
| **Relevance to imperfect information** | **5** | Opponent modeling is a direct attack on imperfect information. Prediction of hidden opponent cards is a clever, simple approach to belief state learning. |
| **Potential for project ideas** | **4** | Both components are starting points for ablations, variants, and extensions. Adaptive deal filtering, multi-task prediction, generalization across opponents all feasible. Not groundbreaking but rich design space. |
| **Importance to read deeply** | **4** | Important follow-up to DouZero; demonstrates practical improvements and modularity. Worth detailed study if DouDizhu is your focus. Less critical if you care only about general RL / imperfect-info theory. |

### Verdict: **READ DEEPLY**

**Recommended reading order and scope**:

1. **First pass (1–2 hours)**: 
   - Abstract, § 1a (beginner-friendly explanation), § 2 (RL formulation), § 9 (evaluation).
   - Goal: Understand what DouZero+ does and why it matters.

2. **Second pass (3–4 hours)**:
   - § 3 (Architecture), § 4 (Training Algorithm), § 5 (Mathematics), § 6 (Loss).
   - Goal: Understand how opponent modeling and deal filtering are implemented.

3. **Deep dive (5–10 hours)**:
   - § 7 (Self-play), § 8 (Imperfect Information), § 10 (Experiments & Ablations), § 12 (Limitations), § 13 (Future Work).
   - Goal: Assess reproducibility, design choices, and research gaps.

4. **Comparison to DouZero** (2–3 hours):
   - Skim DouZero paper (paper 01, § 3 Preliminary in both).
   - Identify what's inherited vs. novel.
   - Cross-reference to understand changes.

5. **Project ideation** (1–2 hours**:
   - § 15 (Project Ideas), § 16 (Implementation), § 17 (Questions).
   - Sketch 2–3 modified experiments; pick one to pursue.

**Which sections to skip**:
- § 11 (Compute): Useful for feasibility but not essential if you're scaling down.
- § 14 (Connection to Other Papers): Helpful context but not critical.
- Detailed code review: Don't read the paper for implementation details; look at GitHub.

---

## Appendix: Synthesis Facts (for cross-paper comparison)

- **Paper**: DouZero+: Improving DouDizhu AI by Opponent Modeling and Coach-guided Learning (Zhao et al., arXiv 2204.02558, Apr 2022).
- **Year**: 2022 (April; arXiv pre-print; not yet peer-reviewed at venues like ICML/NeurIPS, though later published at venues).
- **Environment(s)**: DouDizhu (Fighting the Landlord); three-player imperfect-information card game.
- **Main algorithm**: Deep Monte Carlo with self-play; adds opponent modeling (supervised multi-head classifier) and coach-guided deal filtering (curriculum learning).
- **State representation**: Card matrices (own hand, inferred opponent cards), LSTM-encoded move history, one-hot vectors of opponent hand sizes and bomb counts. Same as DouZero + prediction model outputs.
- **Action representation**: 4×15 card matrix (suits × ranks); legal actions pre-filtered.
- **Self-play?**: Yes; three roles (Landlord, Peasant Down, Peasant Up) play concurrently; all networks updated from same experience.
- **Opponent modeling?**: Yes; prediction model (multi-head classifier) predicts opponent hand card counts; trained with cross-entropy on ground-truth labels (available at train time); predictions concatenated with state features fed to decision network.
- **Main loss**: Decision: MC loss (MSE of Q-value vs. empirical return, target is empirical MC return). Prediction: per-rank cross-entropy on opponent hand counts. The paper does NOT specify a combined loss formula or whether decision-model gradients backprop into the prediction model. Coach: loss not specified (likely MSE or binary CE on win probability).
- **Main contribution**: (1) Opponent modeling via supervised learning on ground-truth hidden information at train time, improving Peasant play via explicit hand belief (prediction outputs concatenated with state features). (2) Coach-guided deal filtering to remove high-variance (imbalanced) deals, speeding training and enabling focus on skill-based decisions.
- **Evaluation method**: 10k games per checkpoint, deal mirroring (same initial cards, swapped roles), ADP (Average Difference in Points) metric. Plotted comparisons are single training runs (no error bars, no multiple seeds) showing learning curves; all plotted models remain weaker than released DouZero baseline throughout training. Only final evidence: Botzone leaderboard validation (top-5 rank achieved; DouZero baseline ~20th).
- **Compute**: "a server with 4 Intel(R) Xeon(R) Gold 6252 CPU @ 2.10GHz and GeForce RTX 2080Ti GPU" (p. 5); number of GPUs and training duration are NOT stated; actors/hyperparameters are DouZero defaults (p. 6).
- **Main limitation**: (1) Prediction model is trained only on self-play data and its accuracy/calibration is never measured; it predicts only the *next* player's hand. (2) Deal filtering changes training distribution; generalization to unfiltered deals not deeply analyzed. (3) How the prediction and decision models are coupled, the β schedule, the exact accept rule, and the coach loss are not specified; training procedure cannot be fully recovered from the paper. (4) No error bars / multi-seed evaluation (statistical rigor gap); single-run curves shown. (5) Curves are early-training (6×10⁵ steps) ADP *against the released DouZero*; the re-trained baseline and the new variants are still mostly weaker than the released DouZero at the end, and no final head-to-head is reported. (6) Method-vs-baseline gaps are small: ≈+0.1 ADP (opponent modeling, Peasants), ≈+0.2 (coach, Peasants), ≈+0.05–0.1 for Landlord.
- **Best idea to borrow**: (1) Opponent modeling via separate supervised prediction model whose outputs are concatenated with state features for the decision model; simple, effective, and modular approach to handling imperfect information. (2) Deal filtering as a form of curriculum learning / variance reduction; orthogonal to core algorithm, can be applied to other games with high initial-state variance. (3) Ablation design: separate experiments for each component, then combined, clearly shows contribution of each (good practice for multi-component papers).

---

**End of Analysis**

**Total word count**: ~11,500 words (detailed, study-guide quality).
