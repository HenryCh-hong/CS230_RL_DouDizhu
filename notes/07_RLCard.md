# RLCard: A Toolkit for Reinforcement Learning in Card Games (IJCAI 2020)
> Zha, Lai, Cao, Huang, Wei, Guo, Hu · arXiv:1910.04376 / IJCAI 2020 · **Tier C (Infrastructure) — EXTREMELY IMPORTANT for this project** · An open-source toolkit providing standardized environments for RL in imperfect-information card games, including Dou Dizhu, with reproducible, configurable interfaces and baseline algorithm implementations

---

## 0. TL;DR for a CS230 Student

- **What it is**: RLCard is a game *environment toolkit*, not an algorithm paper. It wraps seven card games (Blackjack, Leduc, Limit Hold'em, Dou Dizhu, Mahjong, No-limit Hold'em, UNO) with a unified Python interface so you can focus on algorithm development instead of game engine implementation.

- **Why you care**: Dou Dizhu has an enormous action space (up to 10⁴ legal card combinations per turn) and imperfect information (hidden opponent cards). RLCard provides a working game simulator and pre-built evaluation harness—this entire infrastructure can be reused for this CS230 project rather than writing a game from scratch.

- **Games complexity**: Dou Dizhu is extreme: ~10⁵³–10⁸³ information sets, ~10²³ average info-set size (what the agent observes could correspond to 10²³ underlying game states). For comparison, Limit Hold'em is "only" 10¹⁴ info-sets. Action space (more than 3×10⁴ raw actions) is reduced to 309 via *action abstraction* (kicker selection is handled by rules, not the learner).

- **Environment API you'll use**: 
  - `env.run()` (quick): Generate trajectories, get payoffs.
  - `env.step(action)` and `env.step_back()` (advanced): Tree traversal for search algorithms.
  - Legal action masking: Always provided, so you never sample invalid moves.
  - Multi-agent: Handles 3 players (Dou Dizhu) or 2 (poker), manages role-specific rewards.

- **Baselines and their failure**: DQN, NFSP, and CFR are included. **Problem**: They make only tiny progress on Dou Dizhu. Learning curves in Figure 2d show near-flat improvement over 700k timesteps. **Why**: action space is too large, rewards are sparse. This is **your opportunity**: DouZero (same first author, 2021 ICML) replaced DQN with Deep Monte-Carlo and beat RLCard's baselines decisively. Could you experiment with RL algorithms that handle large action spaces better?

- **Design principles you'll recognize in the code**: (1) **Reproducibility**: fixed random seeds guarantee same results. (2) **Accessibility**: state/action/reward all customizable via subclassing. (3) **Scalability**: multi-process data generation (×4, ×8, ×16 speedup). Use these when building your own projects.

- **One idea to steal**: **Action encoding and abstraction**. Dou Dizhu's raw action space is combinatorially explosive (any 3-5 cards can be paired with kickers). RLCard encodes actions as integer indices and uses *rule-based kicker selection* (e.g., "always play the smallest valid kicker"). This reduces 30k actions to 309. For a project with a large action space, steal this: abstract the action space intelligently, then use rules or a separate module to fill in the details.

---

## 1. Paper Overview

### 1a. Beginner-friendly explanation

Imagine you want to build an AI that learns to play Dou Dizhu (Chinese Poker / Fighting the Landlord). You have two choices:

1. **Build everything from scratch**: Write game rules, validate legal moves, track state, compute payoffs, log training data. This takes weeks and bugs are everywhere.
2. **Use a toolkit**: Download RLCard, call `env = rlcard.make('doudizhu')`, and focus on your learning algorithm. The game logic is proven, evaluation is built in, and you can run experiments on your laptop.

RLCard is choice #2. It's a **library of card game environments** for training RL agents. Think of it like OpenAI Gym (for single-agent Atari games) or PettingZoo (for multi-agent environments), but specifically for card games—which are important because they have imperfect information, multiple agents, huge action spaces, and sparse rewards. These are exactly the hard problems RL hasn't solved yet.

The authors—Daochen Zha and colleagues—built RLCard to **bridge the gap** between game theory researchers (who study perfect play using algorithms like CFR) and deep RL researchers (who usually work on simpler domains). RLCard lets both communities run experiments on standardized benchmarks and share results.

### 1b. Technical explanation

**Problem solved**: Imperfect-information card games pose multiple challenges for RL:
- **Multiple agents**: Environments must track turns, roles (landlord vs peasant in Dou Dizhu), rewards per player, and handle simultaneous observations only after all players move (delayed state observation).
- **Huge action spaces**: Dou Dizhu has ~10⁴ legal moves per turn due to card combination explosions (e.g., a "bomb" (quad) can be paired with any single card or pair as "kicker").
- **Imperfect information**: Agents don't see opponent cards—only infer them from action history.
- **Sparse rewards**: In Mahjong, winning is rare (1 every 500 games playing randomly); in betting games, outcomes depend on long-term strategy.

**Why it is difficult**: 
- Most RL toolkits (Gym, PettingZoo) are generic; they don't capture the specific quirks of card games (e.g., delayed next-state observation in extensive-form games where you play an action, then wait for opponents, *then* see the result).
- Action spaces are too large for DQN-style tabular Q-functions; the legal actions change drastically per turn, making policy gradients harder to train.
- Implementing a game correctly is error-prone—one bug in rule checking and your trained agent exploits the bug instead of learning strategy.

**Main idea**: Provide a **standardized, configurable, reproducible environment toolkit** with:
1. Seven games (Blackjack, Leduc, Limit/No-limit Hold'em, Dou Dizhu, Mahjong, UNO).
2. **Two interface styles**: 
   - **Basic interface** (`run()`) for sampling-based RL (DQN, policy gradients).
   - **Advanced interface** (`step`/`step_back`) for tree search and regret-minimizing algorithms (CFR, MCCFR).
3. **Action abstraction** to make large action spaces tractable.
4. **Multi-process support** to parallelize data generation.
5. **Customizable state/action/reward** for experimentation.

**Most important contributions**:
- **Game implementations**: Seven games correctly implemented, tested, and released open-source.
- **Unified API**: Researchers can switch games and algorithms without rewriting boilerplate.
- **Benchmarking baseline results**: DQN, NFSP, CFR evaluated on all games; reveals which algorithms struggle (DQN/NFSP on large games; CFR only feasible on small ones).
- **Reproducibility infrastructure**: Random seed control, parallel generation, hyperparameter logging.

**What came before**: 
- OpenAI Gym (Brockman et al., 2016) provided single-agent environments for RL. RLCard extends this to multi-agent, imperfect-information games.
- OpenSpiel (Lanctot et al., 2019) is contemporary and includes simple card games; RLCard is card-game-focused with more sophisticated interfaces.
- Hand-rolled game engines existed (e.g., Jiang et al., 2019 for DeltaDou; You et al., 2019 for Combinatorial Q-Learning), but no unified toolkit.

**What limitation motivated it**: Previous work on Dou Dizhu (You et al., 2019; Jiang et al., 2019) used bespoke game implementations, making reproduction hard and algorithm comparisons unfair. Researchers couldn't easily try new algorithms because the environment interface and action space abstraction differed per project. RLCard unifies this.

---

## 2. RL Formulation

This section applies standard RL concepts to the card games (especially Dou Dizhu). The paper does not formally define an MDP per game; instead, it describes the game rules and then shows code examples. I synthesize the formulation below.

| Component | Description | Citation/Notes |
|-----------|-------------|-----------------|
| **Agents** | 2 or 3 players per game (e.g., 3 in Dou Dizhu: 1 Landlord, 2 Peasants). Multi-agent, turn-based. | (§ Overview, p. 2) |
| **Environment** | Card game with formal rules (detailed in Appendix A for Dou Dizhu). Dealer shuffles and distributes cards. Judger enforces rules and computes payoffs. | (Appendix, Game Design, p. 6) |
| **State** | All information accessible to one player at a specific timestep: player's own hand, observed action history (cards played by others), and inferred knowledge about opponent hands (hidden). Represented as a dictionary with `{"legal_actions": list, "observation": matrix}`. | (Appendix, State Representation, p. 6) |
| **Observation** | For Dou Dizhu: 6 card planes = (1) current hand, (2) union of other two players' hands, (3) recent 3 actions (one plane per action), (4) union of all played cards. Shape: [6, 5, 15] (6 planes, 5 rows representing card ranks, 15 columns for suits/masks). For other games: varies (e.g., Blackjack: two scalars for player/dealer score). | (Appendix, State Representation, p. 6–7; paper does not give exact shapes) |
| **Action** | Card combinations (in Dou Dizhu: single card, pair, trio, quad, straight, plane, bomb, etc.) or betting decisions (in Hold'em: fold, call, raise). Encoded as integer indices 0, 1, ..., action_space_size - 1. | (Appendix, Action Encoding, p. 7) |
| **Legal Actions** | Always provided to the agent as a list of valid action indices. Prevents sampling invalid moves. Examples: Dou Dizhu only allows passes if no valid beats; Hold'em disallows raise if all-in. | (Appendix, p. 7; Figure 2d caption) |
| **Reward** | **Immediate** (per turn, per hand): Betting games (Hold'em, Leduc) reward in big blinds won (+1 for win, 0 for fold, etc.). Non-betting games reward in outcomes (Dou Dizhu: +1 for Landlord win, -1 for Peasant win, or 0.5 for Peasant; Mahjong: 1 for win, 0 otherwise). Sparse. | (Evaluation, Results, p. 4; Training Agents, p. 3) |
| **Return** | Cumulative reward over a full game (one game = many turns). Not explicitly formulated. | (Implicit in game ending condition) |
| **Policy** | π(a \| s): probability of action a given state s. In RLCard, typically represented by a neural network (MLP, CNN, or RNN) or rule-based module. | (Training Agents, p. 3–4) |
| **Value function** | V(s) or Q(s, a): estimated future discounted reward. Used by DQN and NFSP (see §6). Not explicitly discussed as a general concept. | (Implicit in algorithm descriptions) |
| **Fully/partially observable** | **Partially observable** (imperfect information). Each agent sees own hand, action history, and inferred opponent hands (via Bayesian updates or learning). Hidden: opponent current hands. | (Introduction, p. 1–2) |
| **Single vs multi-agent** | **Multi-agent**: 2 or 3 players. Turn-based (sequential). Note: The state you observe now depends on *what all other players will play*—so "next state" is delayed until after all players' moves. | (Overview, step/step_back explanation, p. 2) |
| **Cooperative/competitive/both** | **Mixed**: In Dou Dizhu, Peasants cooperate (both win/lose together) while competing with the Landlord. In poker, pure competition. | (Introduction, p. 1) |
| **Self-play** | Not explicitly required by RLCard's API, but supported. DQN trains against fixed random agents (§ Training, p. 3). NFSP self-plays (see §7). | (Training Agents, p. 3) |

**Interpretation note** [Interpretation]: The paper formalizes card games as *extensive-form games with imperfect information*, a standard game-theory model. It does not derive MDPs per game (that would require abstracting away game-specific details), choosing instead to provide a flexible interface and let researchers specify reward/observation designs.

---

## 3. Model / Architecture

**This section is not applicable as such** to a toolkit paper—RLCard does not propose a new model architecture. Instead, I cover:
1. **How states/actions are encoded** in the codebase (the "architecture" of representations).
2. **How agents are represented** (network types used in baseline algorithms).
3. **Game-specific design for Dou Dizhu** (the focus game).

### State Encoding for Dou Dizhu

**State representation** (§ Appendix, State Representation, p. 6):

For Dou Dizhu, the observation is encoded as a **matrix of 6 card planes**:

1. **Plane 1**: Current hand of the player (cards held by the observing agent).
2. **Plane 2**: Union of the other two players' hands (inferred, based on play history).
3. **Planes 3–5**: Recent three actions (one plane per action, showing what cards were played).
4. **Plane 6**: Union of all cards played so far in the game.

Each plane encodes cards as a binary matrix (5 rows = card ranks, 15 columns = card types/suits × multiplicities or suit masks). [Paper does not specify exact tensor shape; likely [6, 5, 15] but inferred.]

**Why this encoding?** 
- Card planes are **hierarchical**: distinguishing hand vs. played vs. inferred allows the network to learn different strategies (e.g., "don't play cards that break straights in my hand").
- Action history is **localized**: recent actions are more relevant than old ones, so only the last 3 are encoded.
- Inferred opponent hands reflect **information-set structure** without explicitly computing all possible hands (which would be exponential).

### Action Encoding for Dou Dizhu

**Raw action space**: Dou Dizhu has more than 3×10⁴ legal card combinations (any single card, pair, trio, quad, straight, plane, bomb, and combinations with kickers).

**Action abstraction** (§ Appendix, Action Encoding, p. 7):
- **Without abstraction**: ~30,000 actions.
- **With abstraction**: **309 actions**.
- **How**: Encode only the "major part" of a combination (e.g., the trio in a "trio with kicker"), then use **rule-based kicker selection** to choose the kicker deterministically.
  - Example: If playing a trio of 5s, the action index specifies "trio of 5s," and a separate rule module picks the smallest valid kicker (e.g., 2 of hearts) to avoid breaking other card patterns.

**Paper's statement** (p. 7): "To reduce the action space, we only encode the major part of a combination and use rules to decide the kicker. In this way, the action space of Dou Dizhu is reduced to 309."

**Critical note** [Interpretation]: This is a form of **reward shaping** or **action space pruning** that trades off generality for tractability. The learner never sees kicker selection—it's fully determined by rules. This could be a limitation if optimal strategy requires non-greedy kicker selection (e.g., playing a worse kicker to mislead opponents), but in practice it works. Later work (DouZero, 2021) avoids this by using a different action encoding entirely.

### Network Architecture for Baseline Algorithms

**DQN** (§ Training, p. 3–4):
- **Type**: Multilayer perceptron (MLP).
- **Sizes**: Chosen based on state/action space size: "10-10," "128-128," "512-512," or "512-1024-2048-1024-512."
  - Smaller games (Blackjack, Leduc): smaller networks.
  - Larger games (Dou Dizhu, UNO, Mahjong): larger networks (up to 4 hidden layers, 2048 units).
- **Input**: Vectorized observation (flattened card planes or game state).
- **Output**: Q-values for each action (scalar per action).
- **Why MLP?** Simple, general-purpose, and the baseline—not state-of-the-art, but sufficient for a toolkit paper to show that RL can run on these games.

**NFSP** (§ Training, p. 3–4):
- Two networks:
  - **Policy network**: MLP, same architecture as DQN's Q-network, outputs action probabilities.
  - **Value/Best-response network**: MLP, outputs Q-values for supervised learning (mimicking opponent behavior).
- Both trained jointly; see §7 (Self-Play) for details.

**CFR**:
- No neural network; pure game-theoretic algorithm (regret-based strategy updates). Only tested on Leduc Hold'em (computationally expensive).

**Paper's statement** (p. 3): "the network structure is MLP with size 10-10 128-128, 512-512 or 512-1024-2048-1024-512 based on the size of the state and action space."

**Why no CNN or RNN?** The paper pre-dates widespread use of transformers in games. CNNs could capture spatial card patterns; RNNs could model action sequences. These are left as future work.

---

## 4. Training Algorithm

This section describes the full pipeline for training agents on RLCard environments. The paper does not provide a single unified algorithm; instead, it shows how to use the toolkit with existing RL algorithms (DQN, NFSP, CFR).

### General Training Pipeline

**Step 1: Initialize environment and agent(s)**
```python
env = rlcard.make('doudizhu')
agent = DQNAgent(state_space_size, action_space_size)
env.set_agents([agent, RandomAgent(), RandomAgent()])
```

**Step 2: Sampling (via `run()` interface)**
```python
trajectories, payoffs = env.run()
# trajectories: list of (state, action, reward, next_state, done) tuples per player
# payoffs: final rewards [landlord_reward, peasant1_reward, peasant2_reward]
```

**Step 3: Training**
```python
for transition in trajectories[agent_id]:
    agent.update(transition)  # Update Q-network, policy, etc.
```

**Step 4: Evaluation**
- Play agent against random agents, count wins.
- Run tournament: agent vs. agent with different checkpoints, measure average payoff.

**Step 5: Repeat** (loop Steps 2–4 for N games).

### DQN-Specific Pipeline (as used in the paper)

**Assumption**: Other players are fixed as random agents (simplifies to single-agent RL for the training agent).

1. **Collect trajectory**: Run game; agent chooses actions via ε-greedy(Q_network). Random agents act randomly.
2. **Extract experience**: For agent's turns only, store (state, action, reward, next_state, done).
3. **Replay**: Sample a batch from replay buffer; compute TD target (reward + γ·max_a Q(s', a)).
4. **Gradient step**: MSE loss between Q_network(s, a) and target; Adam optimizer, learning rate 0.00005.
5. **Soft update** (not mentioned but typical): θ_target ← τ·θ_network + (1−τ)·θ_target.

**Hyperparameters** (§ Training, p. 3):
- Memory size (replay buffer): {2000, 100000} (tuned).
- Discount factor γ: 0.99.
- Optimizer: Adam, lr = 0.00005.
- Network: MLP, size based on game.
- ε-decay: (not specified; paper says "lightly tuned").

### NFSP-Specific Pipeline

(See §7 for details. NFSP alternates between RL updates (playing) and supervised learning (mimicking the aggregated policy).)

### Multi-Process Training Pipeline

(§ Appendix, Parallel Training, p. 7):
1. **Main process** initializes an agent and environment.
2. **Worker processes** (4, 8, or 16):
   - Each copies model parameters from main.
   - Generates game data in a duplicate environment.
   - Sends trajectories back to main.
3. **Main process** aggregates trajectories, trains on CPU or GPU, broadcasts new parameters.

**Speedup**: ~4×, ~8×, ~16× (Table 2). Per-timestep cost stays roughly constant (sublinear speedup due to communication overhead, but acceptable).

### Advanced Tree-Search Pipeline (for CFR)

(§ Advanced Interfaces, p. 2–3):
- Use `step()` to move forward; `step_back()` to traverse backward in the game tree.
- **Delayed state observation**: After you call `step(action)`, you don't immediately see the next state—only after all other players have acted. This design mirrors extensive-form game solving.

```python
# Traverse forward (e.g., for MCCFR sampling)
env.step(action)  # Your action; other players' states updated
...
env.step_back()   # Rewind
```

**Why this matters**: CFR-based algorithms (Lanctot et al., 2009; Zinkevich et al., 2008) require sampling paths through the tree or computing counterfactual values. The `step_back()` interface enables external sampling MCCFR—efficient than full tree traversal for large games.

---

## 5. Mathematics

RLCard is a toolkit, not an algorithm paper, so it does not derive new loss functions or theoretical results. However, it implements three algorithms (DQN, NFSP, CFR), each with key equations. I cover the essential math for understanding what's happening during training.

### DQN Loss (Deep Q-Network)

**MUST UNDERSTAND**

The DQN agent minimizes:

$$L = \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}} \left[ \left( Q(s, a) - (r + \gamma \max_{a'} Q_{\text{target}}(s', a')) \right)^2 \right]$$

**Symbols**:
- $Q(s, a)$: predicted Q-value (output of the online network).
- $r$: immediate reward.
- $\gamma$: discount factor (0.99 in the paper).
- $\max_{a'} Q_{\text{target}}(s', a')$: bootstrapped estimate of best future value (target network, updated periodically).
- $\mathcal{B}$: replay buffer (experience tuples).

**Intuition**: We're training the network to predict cumulative future reward. The target is "one-step lookahead": immediate reward plus the agent's estimate of the best value in the next state. The network is penalized if its prediction is far from this target.

**Why needed**: Q-learning with function approximation (neural networks) is unstable. The loss formalizes the TD (temporal difference) update, and the target network (a copy of the online network updated every N steps) helps stabilize training.

**Where in the paper**: Implicit in the training procedure (§ Training, p. 3); not derived, only mentioned as "Deep Q-Network (DQN) (Silver et al., 2016)."

---

### NFSP Loss (Neural Fictitious Self-Play)

**MUST UNDERSTAND**

NFSP combines RL and supervised learning. Two losses:

$$L_{\text{RL}} = \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}_{\text{RL}}} \left[ (r + \gamma V(s') - Q_{\text{policy}}(s, a))^2 \right]$$

$$L_{\text{SL}} = \mathbb{E}_{(s,a) \sim \mathcal{B}_{\text{SL}}} \left[ -\log \pi_{\text{avg}}(a | s) \right]$$

**Symbols**:
- $Q_{\text{policy}}(s, a)$: Q-function (RL part).
- $V(s')$: estimated value of next state.
- $r + \gamma V(s')$: RL target (one-step lookahead).
- $\pi_{\text{avg}}(a | s)$: average policy over all past iterations (supervisory signal).
- $\mathcal{B}_{\text{RL}}$: RL trajectory buffer.
- $\mathcal{B}_{\text{SL}}$: supervised learning buffer (past policies).

**Intuition**: 
- **RL loss**: Learn to maximize reward against the average of all past opponent policies (moving target).
- **SL loss**: The policy network is regularized to stay close to the empirical average policy (prevent overfitting to exploiting the current opponent).

**Where in the paper**: Mentioned at a high level (§ Training, p. 3: "Neural Fictitious Self-Play (NFSP) (Heinrich and Silver 2016)"). Full equations are in the original NFSP paper (Heinrich & Silver, 2016), not derived here.

---

### CFR Loss / Value Update (Counterfactual Regret Minimization)

**OPTIONAL / ADVANCED**

CFR does not use a neural network loss in the traditional sense. Instead, it iteratively updates regrets and strategies:

$$\text{strategy}(a | s) = \frac{\max(0, R^{\text{cum}}_t(s, a))}{\sum_{a'} \max(0, R^{\text{cum}}_t(s, a'))}$$

**Symbols**:
- $R^{\text{cum}}_t(s, a)$: cumulative regret for action $a$ in state $s$ at iteration $t$.
- Regret is the difference between what you would have earned (if you'd always played $a$) and what you actually earned (mixed strategy).

**Intuition**: Actions with positive cumulative regret are played more often; actions with negative regret are played less often. Over many iterations, the strategy converges to a Nash equilibrium (or approximate equilibrium).

**Where in the paper**: Briefly mentioned (§ Training, p. 3: "Counterfactual Regret Minimization (CFR) (Zinkevich et al., 2008)"). Full algorithm is in the original CFR paper. RLCard does not combine CFR with neural networks (Deep CFR requires a separate architecture); it uses tabular CFR.

**Why CFR only on Leduc?** CFR requires complete game tree traversal, which is infeasible for large games (Dou Dizhu, Mahjong). Leduc Hold'em has ~10² information sets; tractable. Dou Dizhu has ~10⁵³; CFR would take years.

---

### Key Equations Summary

| Equation | Label | Concept |
|----------|-------|---------|
| $L = \mathbb{E}[(Q(s,a) - (r + \gamma \max_a Q_{\text{tgt}}(s',a')))^2]$ | **MUST UNDERSTAND** | DQN temporal difference loss |
| $L_{\text{RL}} = \mathbb{E}[(r + \gamma V(s') - Q(s,a))^2]$ | **MUST UNDERSTAND** | NFSP RL component |
| $L_{\text{SL}} = \mathbb{E}[-\log \pi_{\text{avg}}(a \| s)]$ | **MUST UNDERSTAND** | NFSP supervised learning (policy regularization) |
| $\text{strategy}(a \| s) \propto \max(0, R^{\text{cum}}(s,a))$ | **OPTIONAL / ADVANCED** | CFR strategy update |

---

## 6. Loss Function

**This section is not applicable as such** to a toolkit paper. RLCard does not propose a new loss; it implements existing algorithms. However, I summarize the losses of the three baseline algorithms used in evaluation.

### DQN Loss

**Loss**: Mean squared error (MSE) between predicted Q-value and bootstrapped TD target.

$$L = \frac{1}{|\mathcal{B}|} \sum_{(s,a,r,s') \in \mathcal{B}} \left( Q_{\text{online}}(s, a) - (r + \gamma \max_{a'} Q_{\text{target}}(s', a')) \right)^2$$

**Target**: $r + \gamma \max_{a'} Q_{\text{target}}(s', a')$ (one-step Monte-Carlo lookahead with bootstrapping).

**Where target comes from**: The target network is a copy of the online network, updated every N steps (typically N=1000). This stabilizes training by preventing the network from chasing a moving target.

**Bootstrapping?** Yes (value of next state is estimated by the network, not by rollout).

**Monte Carlo?** No (TD, not MC).

**Policy loss?** No (value-based, not policy-gradient).

**Value loss?** Yes (Q-values are value estimates).

**Auxiliary loss?** No.

**Why chosen**: DQN is the standard baseline for single-agent RL. Fast to train, simple to implement. The paper uses it as a baseline to show how RL can be applied to card games, not because it's the best (in fact, Figure 2d shows it's quite weak on large games like Dou Dizhu).

---

### NFSP Loss

**Losses**: Two-part (RL + SL).

$$L_{\text{total}} = L_{\text{RL}} + \alpha L_{\text{SL}}$$

where

$$L_{\text{RL}} = \frac{1}{|\mathcal{B}_{\text{RL}}|} \sum_{(s,a,r,s') \in \mathcal{B}_{\text{RL}}} (r + \gamma V(s') - Q(s, a))^2$$

$$L_{\text{SL}} = -\frac{1}{|\mathcal{B}_{\text{SL}}|} \sum_{(s,a) \in \mathcal{B}_{\text{SL}}} \log \pi_{\text{avg}}(a | s)$$

**Targets**: 
- RL target: $r + \gamma V(s')$ (value of next state, estimated by a value network).
- SL target: $\pi_{\text{avg}}(a | s)$ (empirical distribution of past actions in state $s$, weighted by past episode outcomes).

**Bootstrapping?** Yes, for RL part.

**Why chosen**: NFSP is designed for self-play in imperfect-information games. It decouples best-response learning (RL: exploit opponent) from strategy stabilization (SL: average). This leads to convergence to Nash equilibrium (or approximation thereof). It's theoretically motivated and empirically strong on poker.

**Paper's hyperparameters** (§ Training, p. 3):
- Anticipatory parameter: {0.1, 0.5} (controls the weight of SL loss).
- Memory for supervised learning: $10^6$ transitions.
- Memory for RL: $3 \times 10^4$ transitions (much smaller; RL is online, SL is offline averages).

---

### CFR Loss (Regret Minimization)

**Not a traditional loss.** CFR iterates regrets, not gradients.

Regret at state $s$ for action $a$ after $T$ iterations:

$$R_T(s, a) = \sum_{t=1}^{T} u_t(s, a) - u_t(s, \sigma^t)$$

where $u_t(s, a)$ is utility if you played $a$ and $u_t(s, \sigma^t)$ is actual utility under mixed strategy $\sigma^t$.

**Strategy update**:

$$\sigma^{T+1}(a | s) = \frac{\max(0, R_T(s, a))}{\sum_{a'} \max(0, R_T(s, a'))}$$

**Why chosen**: CFR is the gold standard for solving two-player zero-sum imperfect-information games to near-equilibrium (e.g., Texas Hold'em). It's game-theoretic, not learning-based, so it doesn't use neural networks directly. It guarantees convergence to Nash equilibrium.

**Limitation**: Requires full game-tree traversal, infeasible for large games (Dou Dizhu ~10⁵³ info-sets). RLCard only tests CFR on Leduc Hold'em.

---

## 7. Self-Play

**This section is not directly applicable**: RLCard's API does *not require* self-play. Different algorithms use it differently.

### DQN (No Self-Play)

**Setup** (§ Training, p. 3):
- Training agent: DQN.
- Opponent agents: Fixed as random agents (do not learn).
- Simplification: Reduces multi-agent learning to single-agent RL (training agent vs. stationary random opponents).

**Implication**: No non-stationarity (opponent policy is constant). The agent converges faster but to a lower-quality policy (exploits random agents, not robust to good opponents).

**Evaluation against random agents** (Figure 2d, Dou Dizhu):
- Reward axis: expected payoff per game.
- Curve flattens quickly (~100k timesteps), suggesting DQN is overfitting to random play.

---

### NFSP (Self-Play with Aggregation)

**Setup** (Heinrich & Silver, 2016; RLCard implements this):
1. **RL phase**: Train a best-response network against the average past policy (replay buffer of old policies).
2. **SL phase**: Update policy network to stay close to the aggregate of all past best-response networks (moving average, weighted by episode outcomes).

**Who plays whom**:
- Agent A (RL network) plays against Agent A's past selves (aggregated policy from the SL buffer).
- Non-stationary: The opponent policy shifts as the SL buffer is updated.

**Parameter sharing?** No (RL network and SL network are separate, though they share the state encoder).

**Old checkpoints?** Yes, implicitly—the SL buffer contains past rollouts, which encode past policy snapshots.

**Opponent pool?** Implicit: the aggregate policy in the SL buffer is a mixture of many past policies.

**Exploration**: ε-greedy in RL phase; SL phase is deterministic (supervised, no exploration noise).

**Non-stationarity handling**: By design. The SL buffer stabilizes the training target, preventing overfitting to a single opponent.

**Results** (Table 3, Dou Dizhu with NFSP landlord): NFSP achieves 0.7049 payoff vs. DQN's 0.2951 (tournament, 10k games). NFSP is stronger, suggesting self-play helps.

---

### CFR (No Self-Play, Pure Game Theory)

**Setup**: 
- CFR solves for strategy (mixed policy) without learning from an opponent.
- Deterministic regret accumulation (no randomness in strategy updates, only in sampling).

**Non-stationarity**: None. The algorithm converges to fixed strategy (Nash equilibrium for 2-player zero-sum).

**Only tested on Leduc Hold'em** (computationally expensive). Results show CFR beats both DQN and NFSP (0.0776 vs NFSP, 1.2493 vs DQN), confirming game-theoretic solvers are gold standard for small games.

---

## 8. Imperfect Information

This is a **critical section** for Dou Dizhu and RLCard's value. Imperfect information is the defining challenge.

### Information-Set Structure

An **information set** is the set of game states that look identical to a player (same observation, different hidden states for opponents).

**Dou Dizhu**: ~10⁵³–10⁸³ information sets (Table 1, p. 2). **Average size**: ~10²³ (a single observation could correspond to 10²³ possible hidden states for opponent hands).

**Why huge?** 
- 54 cards split among 3 players: ~$\binom{54}{17,17,20}$ possible initial deals.
- Imperfect play history: same cards can be played in different orders → different future game states → same current observation.

**Comparison**:
- Limit Hold'em: 10¹⁴ info-sets, ~10³ average size (more tractable).
- Texas Hold'em (no-limit): 10¹⁶² info-sets (enormous).
- UNO: 10¹⁶³ info-sets (largest in RLCard).

---

### How Hidden Information is Handled

**Key design principle** (§ Appendix, State Representation, p. 6):

The state dictionary provided to the agent includes:

1. **Observed by agent**:
   - Own hand (cards physically held by the agent).
   - Action history (all cards played and by whom).
   - Legal actions (computed by the environment).

2. **Hidden (but inferred)**:
   - Exact opponent hands (known only to opponents).
   - Inferred via:
     - **Card elimination**: If opponent played a 5, they no longer have a 5 in hand.
     - **Action history**: Strong bids suggest strong hands (Bayesian inference, not explicit in RLCard's encoding).

**State encoding** (6 card planes, Appendix p. 6–7):
- Plane 1: Own hand (fully observed).
- Plane 2: Union of opponent hands = All cards - Own hand - Discarded cards (deterministic, no inference).
- Planes 3–5: Recent actions (observed).
- Plane 6: All played cards (observed).

**What the agent **does NOT** receive**:
- Explicit probabilities over hidden hands (e.g., "opponent A has 70% chance of 5♠").
- Particle filters or belief states.
- Counterfactual value estimates (CFR computes these; RL agents learn them implicitly).

**Inference mechanism**: 
- Implicit: The neural network learns to infer opponent hands from action history and the pattern of which cards are still in play.
- No explicit Bayesian inference (unlike DeltaDou, which uses it). This is a simplification.

---

### Handling Legal Actions

**Paper** (§ Appendix, Action Encoding, p. 7; Basic Interface, p. 2):

```python
env.reset()
state = env.current_state()  # {"legal_actions": [0, 2, 5, ...], "observation": card_planes}
# Agent picks action from legal_actions only.
```

**Legal actions are always provided** to prevent invalid moves. The environment computes which actions are legal by:
1. Checking current game rules (e.g., "can only play a quad if it's a bomb").
2. Verifying the action beats the previous play (if required to beat).
3. Checking for valid card combinations.

**Advantage**: Agents never sample invalid moves; simplifies policy learning (no need for post-hoc filtering or penalty).

**Network output**: The network outputs Q-values for *all* possible actions (or unnormalized policy logits). At inference, the agent masks illegal actions to -∞ (DQN) or 0 (policy gradient) before taking argmax/softmax.

---

### Imperfect Information: Two-Column Table

| **Actually Observed by the Agent** | **Hidden (Not Directly Observed)** |
|---|---|
| Own hand cards (100% certainty) | Exact opponent hands |
| Cards played by opponents (history) | Opponent cards still held |
| Actions taken by opponents | Opponent intentions / bidding strength |
| Current game phase (bidding vs. play) | Probability distribution over hidden hands (implicit) |
| Legal actions (computed by rules) | Cards opponents will play (future) |

**Paper's assumption**: Agents learn to infer hidden cards from history (learned implicitly by the network). No explicit Bayesian filtering. This is simpler than DeltaDou's approach but potentially less sample-efficient.

---

## 9. Evaluation ← ONE OF THE MOST IMPORTANT SECTIONS

This section covers how RLCard evaluates agents and critically assesses the methodology. It then proposes a statistically sound protocol for CS230 projects on 3-player games.

### Evaluation Baselines and Methodology

**[Paper]** Two evaluation methods (§ Evaluation, p. 3–4):

1. **Winning rates against random agents** (Figure 2):
   - Agent trained for N timesteps; tested against 3 random agents.
   - Reward = average payoff per game (winning = +1 for betting games, win-rate for others).
   - Learning curve: plot reward over training timesteps.

2. **Tournament play** (Table 3):
   - Each agent plays 10,000 games against every other agent.
   - Evaluate each agent in each role (landlord, peasant 1, peasant 2 in Dou Dizhu).
   - Report average payoff (e.g., NFSP: 0.7049, DQN: 0.2951 in Dou Dizhu tournament).
   - Compare CFR vs NFSP vs DQN on Leduc Hold'em: CFR wins decisively.

**Hyperparameter tuning** (§ Training, p. 3):
- "Lightly tuned": Memory size, discount factor, learning rate chosen from small grids.
- No description of tuning procedure (grid search? random search?).

---

### Baseline Results

**Learning curves vs random agents** (Figure 2, p. 3):
- **Blackjack**: Both DQN and NFSP converge quickly (expected; simple game with ~10³ info-sets).
- **Leduc Hold'em**: Both converge; NFSP slightly better (RL problem reduced by imperfect information; RL-friendly game).
- **Limit Texas Hold'em**: Similar; both reach ~0.5 reward (outperform random, but modest improvement).
- **Dou Dizhu** (Figure 2d): **Both DQN and NFSP plateau quickly** (~0.1 reward, tiny improvement after 100k timesteps). **This is a major finding**: standard RL algorithms fail on large games.
- **Mahjong, No-limit Hold'em, UNO**: Similar flat curves; sparse reward and large action space make learning very hard.

**Tournament results** (Table 3, 10k games):
- **Leduc Hold'em**: NFSP (1.0691) >> DQN (-1.0691). CFR (0.0776) > NFSP. CFR is gold standard.
- **Dou Dizhu**:
  - NFSP (landlord role): 0.7049 payoff.
  - DQN (landlord role): 0.2951 payoff.
  - **NFSP is ~2.4× stronger**. But absolute payoffs are low, suggesting both agents are weak.
- **UNO**: NFSP (−0.0428) ≈ DQN (0.0428); both agents are nearly random on this massive game (10¹⁶³ info-sets).

**Key observation** [Paper, Discussion, p. 4–5]:

> "DQN agents play very aggressively in betting games... this naive strategy works well when the opponent is a random agent since the random agent may easily choose 'fold'... However, DQN policy may be highly exploitable. Thus, performance against random agents can only be a way to get a sense of whether the agent is improving on the environment, but is NOT enough to be used to evaluate algorithms."

**Paper's conclusion**: Evaluation against random agents is **unreliable**. Agents learn to exploit specific weaknesses (fold too much in random play), not to learn robust strategies. Proper evaluation requires:
1. **Tournament against other trained agents** (fixes random baseline to something plausible).
2. **Rule-based or pre-trained baseline models** (to be added in future work; not in initial release).

---

### Critical Evaluation of RLCard's Methodology

**Strengths**:
1. **Multi-method evaluation**: Both "learning curves vs random" and "tournament" provide complementary info.
2. **Role-specific evaluation**: Dou Dizhu results separate landlord and peasant roles, showing performance is role-dependent.
3. **Reproducibility**: Fixed seeds and open-source code allow reproduction.
4. **Multiple games**: Shows RL difficulty across diverse games, not cherry-picked winners.

**Weaknesses**:
1. **Small tournament size**: 10,000 games is modest. For Dou Dizhu (3-player), to get 95% confidence intervals on a ~50% win rate, you'd need ~1,300 games per agent (binomial CI). 10k games is sufficient, but marginal. No confidence intervals reported.
2. **No statistical significance testing**: Payoffs in Table 3 are point estimates. NFSP (0.7049) vs DQN (0.2951): is this difference significant given sampling noise? Not tested.
3. **Unfair DQN comparison**: DQN is trained against random agents (single-agent RL). NFSP self-plays, adapting to improve opponents. **Different training regimes make comparison unfair**. [Interpretation]
4. **Limited opponent diversity**: Only two agents tested head-to-head. Tournament should include ≥3 competitors to avoid spurious results (e.g., rock-paper-scissors dynamics).
5. **No baseline models yet**: Paper promises "rule-based agents and pre-trained models" but doesn't release them initially. Makes evaluation weaker (relies on random opponents).
6. **Role rotation not clearly specified**: For Dou Dizhu, do agents play all roles equally? Table 3 separates NFSP landlord vs DQN landlord, but doesn't specify if they trade roles in the tournament.

---

### What I Would Improve for a CS230 Project [Interpretation]

**For your own evaluation protocol on a 3-player game** (e.g., Dou Dizhu variant or new game), I recommend the following statistically sound evaluation practices:

#### 1. **Seat Rotation and Deal Mirroring**
**Why**: In a 3-player game, **seats matter** (landlord vs peasant win rates are different). To fairly compare two agents, you need:
- **Seat rotation**: Each agent plays each role (landlord, peasant 1, peasant 2) equally often. In a tournament of M games:
  - First M/3 games: Agent A as landlord, B/C as peasants.
  - Next M/3 games: Agent A as peasant 1, B as landlord, C as peasant 2.
  - Last M/3 games: Agent A as peasant 2, B as peasant 1, C as landlord.
- **Deal mirroring**: If a specific deal (card distribution) is used, same deal is played with agents in different seats. This isolates skill from luck.

**Example**: Test all 6 permutations of seats across the same set of N dealt hands.

#### 2. **Fixed Deal Sets for Variance Reduction**
**Why**: Randomness in card distribution adds noise. With a fixed deal set:
- Generate 1,000 random deals, seed them to a file.
- All agents play all 1,000 deals.
- Compute average payoff per agent (average over deals, not random variation).

**Benefit**: Reduces variance by 10–100×, making statistically significant differences easier to detect.

#### 3. **Number of Games for Desired Confidence Interval**
**Statistical power**: For a binomial win-rate (agent A wins at rate $p$, opponent at rate $1-p$):
- Sample size $n$ for 95% CI of width ±0.05: $n ≈ 1,537$ (rule of thumb: $n = 1.96^2 \cdot p(1-p) / w^2$, max at $p=0.5$).
- For 3-player games, "win" is undefined; instead, track **payoff difference**. Variance is higher; need ~2,000–5,000 games for tight CI.

**Recommendation for CS230**:
- **Minimal (quick experiment)**: 1,000 games per agent pair, across 3 roles.
- **Standard (publishable)**: 3,000–5,000 games per agent pair, fixed deal set, 95% CI width ≤ ±0.10.
- **Thorough**: 10,000+ games, cross-validation across deal sets.

#### 4. **Cross-Play Matrices**
**What**: Report a matrix $M[i, j]$ = average payoff of agent $i$ vs agent $j$.
- Include trained agents, random agent, rule-based baselines.
- Report both absolute payoffs and relative rankings.

**Example (Dou Dizhu)**:

| Agent | Landlord (vs Random) | Peasant (vs Random) | vs Other Agents | Notes |
|-------|---|---|---|---|
| DQN (10k steps) | 0.15 | -0.05 | 0.295 vs NFSP | Weak |
| NFSP (10k steps) | 0.50 | 0.20 | 0.705 vs DQN | Stronger |
| Rule-based v1 | 0.60 | 0.30 | — | Baseline |
| Random | 0 | 0 | — | Baseline |

#### 5. **Statistical Testing**
**Hypothesis test**: Is agent A significantly better than agent B?
- **Null**: $H_0$: $\mu_A = \mu_B$ (equal average payoff).
- **Test**: Paired t-test (if using same deals) or independent t-test (if different deals).
- **Report**: p-value, 95% CI on difference, effect size (Cohen's $d$).

**Tool**: Python `scipy.stats.ttest_ind()` or `scipy.stats.ttest_rel()`.

#### 6. **Ablation Experiments**
**Example**: "Does action abstraction help?"
- Train agent A with full action space (no abstraction).
- Train agent B with Dou Dizhu's action abstraction (309 actions).
- Compare payoffs over same deal set.

---

## 10. Experiments and Ablations

RLCard is a toolkit paper, not an algorithm paper, so experiments focus on **evaluating the toolkit** (do environments work? how fast? how do baselines compare?), not on ablating algorithm components.

### Major Experiments

#### Experiment 1: Learning Curves on All Environments (Figure 2)
**What changed**: Algorithm (DQN vs NFSP vs CFR), game (Blackjack → UNO).
**Held constant**: Hyperparameters (lightly tuned), random agent opponents, training time (timesteps shown on x-axis).
**Metric**: Average reward against random agents.
**Results**:
- Simple games (Blackjack, Leduc): Both DQN/NFSP converge quickly.
- Large games (Dou Dizhu, Mahjong, UNO): Both plateau quickly, minimal improvement after ~100k timesteps (Figure 2d, 2e, 2g).

**Authors' conclusion** [Paper, Discussion, p. 4–5]:
> "NFSP and DQN are highly unstable in large games... There is a lot of room for improvement."

**Your critical note** [Interpretation]: This is the **most important result in the paper**. It shows that standard RL algorithms **fail on Dou Dizhu**. This motivates future work (e.g., DouZero, 2021) to develop new algorithms. For this CS230 project, this result suggests that simply applying DQN or policy gradients to Dou Dizhu is futile—you need to innovate.

---

#### Experiment 2: Tournament Evaluation (Table 3)
**What changed**: Matchup (which agents play each other).
**Held constant**: Number of games (10,000), role assignment (detailed in caption).
**Metric**: Average payoff per game.
**Results**:
- NFSP beats DQN on most games (Leduc, Dou Dizhu, No-limit Hold'em).
- DQN wins on Limit Texas Hold'em and UNO (marginal differences).
- CFR crushes both on Leduc Hold'em (0.0776 NFSP, 1.2493 DQN).

**Authors' conclusion**: NFSP > DQN overall; CFR is best (but only on small games).

**Your critical note** [Interpretation]: NFSP's superiority is expected (it's designed for self-play in imperfect information games). DQN's weakness is expected (designed for single-agent RL, here fixed to single-agent by freezing opponents as random). The comparison is somewhat unfair.

---

#### Experiment 3: Computational Efficiency (Table 2)
**What changed**: Number of processes (1, 4, 8, 16).
**Held constant**: Number of games (1,000,000), random agents, game selection (all games).
**Metric**: Wall-clock time (seconds) and throughput (per-step time).
**Results**:
- Total time decreases sublinearly: 156.1s (×1) → 45.6s (×4) → 23.0s (×8) → 12.8s (×16) for Blackjack.
- Per-step time stays roughly constant (~1e-4 to 1e-5 seconds).
- **Dou Dizhu**: Much slower per game (87,270.8s single-process) due to long sequences (~20 turns per game vs. ~5 for Blackjack).

**Authors' conclusion**: Parallel generation scales reasonably; Dou Dizhu is computationally demanding but feasible on modest hardware.

**Your critical note** [Interpretation]: These numbers are from 2019 hardware (24 Intel Xeons @ 2.1 GHz). Modern GPUs would be faster, but Dou Dizhu still has overhead (tree traversal, legal action computation). For CS230, plan on ~1 second per game if using your laptop CPU; GPUs help if you optimize the environment.

---

### Experiments Reproducible at Small Scale for CS230

**Question**: Which RLCard experiments can you reproduce in a quarter on a laptop or 1 GPU?

#### Feasible (Easy):
1. **Learning curves on Blackjack** (Figure 2a):
   - 25k timesteps in ~100s.
   - DQN + NFSP both trainable.
   - Reproduce exactly.

2. **Learning curves on Leduc Hold'em** (Figure 2b):
   - 250k timesteps in ~10 min.
   - Slightly slower but fast.
   - Reproduce exactly.

3. **Tournament on small subset**:
   - Run 1,000 games (instead of 10,000) between DQN/NFSP.
   - Trade statistical power for speed; use seat rotation to reduce variance.
   - Rough reproduction.

#### Moderate (Doable with care):
4. **Learning curves on Limit Hold'em** (Figure 2c):
   - 300k timesteps in ~10 min (larger state/action space).
   - Trainable but requires patience.
   - Reproduce exactly.

5. **Dou Dizhu learning curves** (Figure 2d):
   - 600k timesteps in ~1 hour on CPU, ~10 min on GPU.
   - Slow but achievable.
   - Feasible to reproduce if you optimize; see note below.

#### Hard (Unrealistic for CS230):
6. **Efficiency experiments** (Table 2):
   - Requires ×4, ×8, ×16 process launches (need multi-core machine).
   - Worth doing if you're measuring your own algorithm's speedup.

7. **All environments** (comprehensive evaluation):
   - 7 games × 2 algorithms × 250k–700k timesteps ≈ many GPU-hours.
   - Infeasible for a quarter-long project.

---

### Optimization Tips for Small-Scale Reproduction

**To speed up Dou Dizhu experiments**:

1. **Use the smaller Dou Dizhu variant** (paper mentions p. 2): "RLCard also implements a smaller version of Dou Dizhu, where we only keep cards 8, 9, 10, J, Q, K, and A."
   - Reduces state space dramatically.
   - Keeps game structure but faster to play.
   - Rough reproduction of learning curve shape.

2. **Reduce number of games**:
   - Instead of 700k timesteps, train for 100k (rough curve shape).
   - Still shows plateau vs. improvement.

3. **Reduce evaluation frequency**:
   - Original paper evaluates every ~50k timesteps.
   - Evaluate every 100k timesteps to save wall-clock time.

4. **Use GPU**:
   - RLCard supports TensorFlow/PyTorch on GPU.
   - [From public code repo, not the paper] Modern RLCard uses PyTorch and can run on GPU if available.
   - 10× speedup typical.

---

## 11. Compute and Engineering

### Hardware

**RLCard's original experiments** (§ Running Time Analysis, p. 4):
- 24 Intel Xeon @ 2.1 GHz CPUs.
- 64 GB RAM.
- No GPUs mentioned (paper is from 2019; GPUs were not standard in toolkit work yet).

**What you need**:
- **Laptop (CPU-only)**: Can run Blackjack, Leduc, Limit Hold'em comfortably. Dou Dizhu will be slow (~1s per game).
- **1 GPU (e.g., NVIDIA Tesla T4 or RTX)**: ~10–30× faster for neural network inference/training.
- **Multi-core CPU (8+ cores)**: Useful for parallel data generation (Table 2 shows ×4, ×8, ×16 speedups).

---

### Training Duration

**Experiments in the paper**:
- Figure 2: 20k–700k timesteps per game × 2 algorithms = substantial training time.
- Table 3: 10,000 games per pair (tournament evaluation) at ~1s per game = 2.7 hours of gameplay per pair.
- Inference only: Fast (~100 games/sec on CPU; ~1000/sec on GPU).

**Dou Dizhu specifics** (Table 2):
- Single-process: 87,270.8 seconds for 1M games = **~24 hours on a laptop CPU**.
- With ×16 parallelization: 7,275.9 seconds = **~2 hours on 24 cores**.
- Inference is faster; bottleneck is environment simulation (legal action checking, state encoding).

---

### Number of Samples

**DQN hyperparameters** (§ Training, p. 3):
- Memory size (replay buffer): 2,000 or 100,000 transitions.
- Training: Adam optimizer, ~1 gradient step per collected transition (rough estimate).
- Convergence: Figure 2d shows plateau at ~700k timesteps (~700k transitions collected).

**NFSP hyperparameters** (§ Training, p. 3):
- RL buffer: $3 \times 10^4$ transitions.
- SL buffer: $10^6$ transitions.
- More data-hungry due to offline SL phase.

---

### Parallel Actors

**§ Appendix, Parallel Training, p. 7**:
- Duplicated environments in workers.
- Each worker generates data independently, sends to main.
- Main aggregates and trains on CPU/GPU.

**Speedup** (Table 2):
- ×4 cores: ~3.5× speedup (not perfect, due to communication).
- ×8 cores: ~4× speedup (better parallelization).
- ×16 cores: ~4× speedup (diminishing returns; communication overhead).

**Amdahl's law**: If only part of training is parallelizable, speedup plateaus. For RLCard, the environment (environment simulator, legal action checking) is the bottleneck, and it parallelizes well.

---

### Important Implementation Tricks

**[From public code repo, not the paper]**:
1. **Action masking**: Legal actions are masked (set to -∞ or 0 probability). Prevents invalid move sampling.
2. **Card planes encoding**: Efficient integer-to-one-hot conversion; avoids dense representations.
3. **Batch inference**: Collect a batch of games, then run inference once on GPU for all agents.

---

## Reproduction Difficulty: **Easy to Moderate (depending on game)**

### Easy (Blackjack, Leduc Hold'em)
**Why**: Small state/action spaces, fast environment simulation (~1k games/sec). Can reproduce Figure 2a, 2b in <1 hour on laptop.

**What you need**:
- Laptop with Python.
- RLCard + PyTorch or TensorFlow.
- ~1 hour.

### Moderate (Limit Hold'em, Dou Dizhu)
**Why**: Larger state/action spaces, slower simulation (~100 games/sec on CPU). Figure 2c replicable in ~30 min; Figure 2d in ~1 hour on GPU or ~12 hours on CPU.

**What you need**:
- Laptop or cloud GPU (Colab free tier, AWS, GCP).
- RLCard + PyTorch.
- ~1–12 hours (depending on GPU access).

**Gotcha**: Hyperparameter tuning took the authors time; you'd likely just use their defaults.

### Hard (Full reproduction of all experiments)
**Why**: Table 2 requires parallel hardware; Table 3 requires careful experimental protocol (10k games × multiple agent pairs).

**Realistic for CS230?** **Scaled-down version only**:
- Use small Dou Dizhu variant.
- Train DQN/NFSP for 100k timesteps (instead of 700k).
- Evaluate on 1,000 games (instead of 10,000).
- Estimate would take ~4–8 GPU-hours.

---

### Scaled-Down Reproduction Plan

**Feasible CS230 reproduction** (1–2 weeks of work, including debugging):

1. **Week 1: Setup and Baseline**
   - Install RLCard, PyTorch.
   - Run pre-trained models on Dou Dizhu (to verify setup).
   - Train DQN on small Dou Dizhu variant for 50k timesteps.
   - Reproduce learning curve shape (not exact numbers).

2. **Week 2: Comparison and Analysis**
   - Train NFSP for 50k timesteps.
   - Run tournament: 1,000 games, DQN vs. NFSP.
   - Compute payoff matrix (rough version of Table 3).
   - Report results, discuss why DQN/NFSP plateau (sparse reward? large action space?).

**Output**: A working reproduction that shows (1) RLCard works, (2) standard RL fails on Dou Dizhu, (3) NFSP > DQN. This positions you well to propose improvements.

---

## 12. Limitations

### A. Explicitly Mentioned by the Authors

1. **Standard RL algorithms fail on large games** [Paper, Discussion, p. 4–5]:
   > "NFSP and DQN are highly unstable in large games. Specifically, they only achieve minor improvements during the learning process on UNO, Mahjong and Dou Dizhu. These games are challenging due to their large state/action space and sparse reward."
   
   **Implication**: The toolkit reveals a limitation, not of RLCard, but of the algorithms. DQN/NFSP need improvements for large games.

2. **Evaluation against random agents is insufficient** [Paper, Discussion, p. 4–5]:
   > "For large games, we recommend evaluating algorithms by playing against existing models. To benchmark the evaluation, we will develop rule-based agents and stronger pre-trained models in the future."
   
   **Implication**: Initial RLCard lacks rule-based baseline agents. Evaluation is only against random opponents, which is weak. (Later versions [From public code repo, not the paper] include rule-based models.)

3. **CFR only on Leduc** [Paper, Training, p. 3–4]:
   > "We only test CFR on Leduc Hold'em since it is computationally expensive, requiring complete traversal of the game tree."
   
   **Implication**: CFR cannot scale to large games (Dou Dizhu). Deep CFR (neural network variant) is mentioned in related work but not implemented.

4. **No visualization or analysis tools** [Paper, Conclusions, p. 5]:
   > "We plan to develop visualization and analysis tools for the environments."
   
   **Implication**: Initial release lacks debugging/introspection tools (what is the agent learning? why does it fail?).

### B. Inferred Limitations (Not Explicitly Stated)

**[Interpretation]**:

1. **Action abstraction for Dou Dizhu is ad-hoc**: Reducing 30k actions to 309 by rule-based kicker selection assumes kicker choice is not strategic. But optimal play might require context-dependent kicker selection (e.g., playing a weak kicker to mislead opponents). This assumption is not validated empirically.

2. **No opponent modeling**: Agents infer opponent hands from action history, but don't explicitly model opponent beliefs or strategies. In poker, opponent modeling (e.g., estimating opponent hand distribution) is crucial. RLCard's implicit learning is simpler but potentially less efficient.

3. **Imperfect information handling is simplistic**: The 6 card planes are heuristic. More sophisticated representations (e.g., neural belief states, particle filters) are not explored. DouZero (2021) uses a different approach entirely.

4. **Multi-player evaluation is limited**: Dou Dizhu has cooperative peasants (vs. competitive poker). RLCard's evaluation does not deeply explore how cooperation/competition affects learning. Do agents learn to coordinate? Barely studied.

5. **Reward design is game-specific**: No principled approach to reward shaping. Each game gets a bespoke reward (winning big blinds for poker; win/loss for Dou Dizhu). A general RL framework would have a unified reward model.

6. **No domain randomization or sim-to-real**: RLCard is purely simulated. Transfer to human play, or robustness to rule variations, is not discussed.

---

## 13. Future Work

### Future Work Explicitly Mentioned by the Authors

**[Paper]** (Conclusions, p. 5):

1. **"Design rule-based agents and provide more pre-trained models for evaluation."**
   - **Motivation**: Evaluation against random agents is weak (Discussion, p. 4–5).
   - **Difficulty for CS230**: Directly usable / Engineering-heavy.
   - **Status in later work**: [From public code repo, not the paper] ✓ Implemented. RLCard now includes rule-based models for Leduc, Limit Hold'em, Dou Dizhu, UNO.
   - **Your opportunity**: Build a rule-based agent for a new game or variant; integrate into RLCard.

2. **"Develop visualization and analysis tools for the environments."**
   - **Motivation**: Debugging and introspection (what is the agent learning?).
   - **Difficulty for CS230**: Moderate (depends on scope).
   - **Status in later work**: Partial (Botzone leaderboard visualization for Dou Dizhu agents; not general RLCard tools).
   - **Your opportunity**: Build a visualization dashboard (learning curves, action heatmaps, game replays).

3. **"Further accelerate the environments with more efficient implementations."**
   - **Motivation**: Dou Dizhu simulation is slow (~100 games/sec on CPU).
   - **Difficulty for CS230**: Hard (systems/optimization work).
   - **Status in later work**: [From public code repo, not the paper] Some optimization; Cython wrappers for hot loops mentioned in DouZero.
   - **Your opportunity**: Profile and optimize the environment (e.g., C++ card plane encoding).

4. **"Include more interesting games and more algorithms to enrich the toolkit."**
   - **Motivation**: Broaden scope.
   - **Difficulty for CS230**: Hard (requires game implementation).
   - **Status in later work**: ✓ Implemented. Gin Rummy, Bridge added [From public code repo, not the paper].
   - **Your opportunity**: Implement a variant of Dou Dizhu (e.g., 4-player mode) or a new shedding game.

---

### Implied Future Work (Not Explicitly Stated)

**[Interpretation]**:

1. **Better algorithms for large action spaces**:
   - **Motivation**: DQN/NFSP plateau on large games (Figure 2d, 2e, 2g).
   - **Difficulty for CS230**: Moderate to Hard (algorithm research).
   - **Status in later work**: ✓ Addressed by DouZero (2021), which uses Deep Monte-Carlo. DQN has overestimation bias; policy gradient methods can't generalize; DMC avoids both.
   - **Your opportunity**: Compare action abstractions (e.g., learning-based vs. rule-based kicker selection) or implement a different algorithm (e.g., A3C, PPO) and analyze why it fails/succeeds.

2. **Neural network architectures for imperfect information**:
   - **Motivation**: MLPs are generic; specialized architectures (RNN for sequence modeling, Transformer for card patterns) might help.
   - **Difficulty for CS230**: Moderate (deep learning, not RL).
   - **Status in later work**: Transformer-based agents explored in recent work (not RLCard-specific).
   - **Your opportunity**: Experiment with RNN/CNN/Transformer for state encoding in Dou Dizhu; compare to MLP baseline.

3. **Deep CFR or other game-theoretic RL hybrids**:
   - **Motivation**: CFR solves games exactly (on small games) but doesn't scale. Deep CFR uses neural networks as function approximators.
   - **Difficulty for CS230**: Hard (combines deep learning + game theory).
   - **Status in later work**: Deep CFR paper exists (Brown et al., 2019, ICML); not integrated into RLCard initially.
   - **Your opportunity**: Implement and benchmark Deep CFR on Leduc Hold'em (vs. CFR and DQN).

4. **Opponent modeling and diversity**:
   - **Motivation**: Agents should adapt to different opponents (e.g., aggressive, conservative, random). Multi-agent curriculum learning.
   - **Difficulty for CS230**: Moderate to Hard (multi-agent learning).
   - **Status in later work**: ✓ Addressed by DouZero+ (2022, opponent modeling and coach-guided learning) and other work.
   - **Your opportunity**: Train an agent to learn opponent models (e.g., hand strength distribution) and adapt policy accordingly.

---

### Connection to Later Papers in This Set

**How does DouZero (Zha et al., ICML 2021) address RLCard's limitations?** [From cross-paper comparison]

| Limitation (from RLCard) | DouZero's Solution | Status |
|---|---|---|
| DQN/NFSP fail on Dou Dizhu (large action space, sparse reward) | Replace with Deep Monte-Carlo (DMC) + action encoding | ✓ Achieved expert-level play (ranked #1 on Botzone) |
| Action abstraction is ad-hoc (rule-based kicker) | Use action encoding as learned representation; kickers encoded as card matrices | ✓ Allows generalization to unseen actions |
| Evaluation weak (vs. random agents) | Evaluate against strong baselines (CQN, DeltaDou, rule-based agents) and on Botzone leaderboard | ✓ Transparent, high-bar evaluation |
| Offline/inference slow on CPU | Optimize with parallel actors, GPU inference, Cython hotspots | ✓ 10-day training on 4 GPUs (vs. DeltaDou's 2 months) |
| No opponent modeling | Implicit: self-play explores opponent diversity via MCTS sampling | ✓ Learns to handle diverse opponents |

**Summary**: DouZero is the direct follow-up, using many of RLCard's ideas (environment, evaluation, parallel training) but replacing the algorithm core. It addresses the central limitation (standard RL fails) with a novel algorithm (DMC).

---

## 14. Connection to Other Papers

### How RLCard Relates to Earlier Work

**[Paper]** (Related Work, p. 4–5):

1. **OpenAI Gym** (Brockman et al., 2016):
   - **What Gym does**: Single-agent Atari games, continuous control, simple benchmarks.
   - **What RLCard does**: Multi-agent, imperfect-information card games.
   - **Relationship**: RLCard extends Gym's interface design (environment wrapper, step/reset methods) to card games.
   - **Why the change**: Card games need (a) multi-agent turn-taking, (b) legal action masking, (c) role-specific rewards (landlord vs. peasant), (d) delayed state observation (extensive-form games).

2. **OpenSpiel** (Lanctot et al., 2019, contemporary):
   - **What OpenSpiel does**: General game-theory framework; includes Leduc Hold'em, Tic-Tac-Toe, Chess, etc.
   - **What RLCard does**: Card games specifically, with simpler/more intuitive RL interface.
   - **Relationship**: Both provide standardized game environments. RLCard is more RL-friendly (run() interface for sampling-based learning); OpenSpiel is more general-purpose (supports MCTS, CFR).
   - **Trade-off**: RLCard is shallower (fewer games, simpler algorithms) but easier for RL researchers to use.

3. **Previous Dou Dizhu work** (You et al., 2019; Jiang et al., 2019):
   - **What CQN (You et al.) does**: Hand-rolled game engine + Combinatorial Q-Learning (decomposes action space).
   - **What DeltaDou (Jiang et al.) does**: Hand-rolled engine + AlphaZero-like tree search + Bayesian opponent modeling.
   - **What RLCard does**: Unified environment, multiple algorithms (DQN, NFSP, CFR), easy API.
   - **Relationship**: RLCard standardizes the Dou Dizhu benchmark, enabling fair algorithm comparison. CQN and DeltaDou are one-off implementations, hard to reproduce.
   - **Why it matters**: RLCard democratizes Dou Dizhu research (lowers barrier to entry).

### Evolution Chain in This Paper Set

**[Interpretation]** (based on your paper set):

```
Previous Work (You et al., AIIDE 2020; Jiang et al., IJCAI 2019)
  ↓ (limitation: bespoke game engines, hard reproduction)
RLCard (Zha et al., IJCAI 2020) 
  - Standardized environments, but weak algorithms (DQN/NFSP plateau)
  ↓ (limitation: standard RL fails on Dou Dizhu large action space)
DouZero (Zha et al., ICML 2021)
  - Deep Monte-Carlo + action encoding, reaches expert play
  ↓ (limitation: no opponent modeling)
DouZero+ (Zhao et al., 2022)
  - Opponent modeling + coach-guided learning
```

**Key insight**: Each paper solves a problem posed by the previous one.
- RLCard unifies the benchmark; DouZero replaces the algorithm; DouZero+ adds opponent modeling.

### What RLCard Keeps from Prior Work

1. **Extensive-form game representation** from game theory (Zinkevich et al., 2008; Von Stengel, 2010).
2. **CFR algorithm** for benchmarking (Zinkevich et al., 2008).
3. **Deep RL baselines** (DQN from Mnih et al., 2015; NFSP from Heinrich & Silver, 2016).
4. **Gym-like API** for RL environments (Brockman et al., 2016).

### What RLCard Changes

1. **Multi-agent turn-taking**: Delayed state observation; players move sequentially, but agent sees result only after all move.
2. **Imperfect information handling**: Card planes instead of full observability.
3. **Legal action masking**: Environment enforces action validity (unlike Gym, which returns out-of-bounds errors).
4. **Action abstraction**: Rule-based kicker selection (novel for Dou Dizhu).

### Why Changes Improve Performance

[Interpretation]:
- **Multi-agent API**: Makes it natural to train agents against multiple opponents (not just single-agent MDP).
- **Legal action masking**: Prevents agents from wasting learning on invalid moves.
- **Imperfect information design**: 6 card planes strike a balance between full observability (too much info, unrealistic) and belief states (too complex).
- **Action abstraction**: Reduces action space from 30k to 309; makes learning tractable (though potentially suboptimal).

### New Limitation Introduced by RLCard

[Interpretation]:
- Action abstraction assumes kicker choice is not strategic (may be wrong).
- No opponent modeling (agents don't learn to recognize opponent types).

---

## 15. Project Ideas for Me

These are realistic CS230 project ideas **inspired by RLCard**, leveraging the toolkit as infrastructure, and building something novel.

### Idea 1: Learning-Based Action Abstraction for Dou Dizhu

**Research question**: Can we learn better action abstractions than RLCard's rule-based kicker selection?

**Baseline**: RLCard's current abstraction (309 actions; kicker selected greedily by rules).

**Modification**: Implement a learned kicker selector:
- Train a separate neural network that, given state and "major" card combination, predicts the best kicker.
- Use supervised learning (label with human-level play or strong agent play).
- Integrate into RLCard's action encoding.

**Hypothesis**: A learned kicker module will outperform rules because it can exploit strategy-dependent kicker selection (e.g., play a weak kicker to hide strength).

**Experiment**:
1. Train baseline DQN with RLCard's rule-based abstraction (50k steps, small Dou Dizhu).
2. Train DQN with learned kicker abstraction (same setup).
3. Tournament: 1,000 games, baseline vs. learned.
4. Hypothesis: Learned wins by ≥5% payoff.

**Metrics**:
- Win rate vs. random agents (learning curve).
- Payoff vs. baseline agent (tournament).
- Kicker selection patterns (do they make sense?).

**Difficulty**: Moderate (data collection + training overhead).

**Compute**: Low–Medium (single GPU, ~2 hours).

**Novelty**: Yes; learning action abstractions is underexplored in card games. Addresses RLCard's acknowledged limitation (action abstraction is ad-hoc).

---

### Idea 2: Opponent Modeling via Self-Play Checkpoints

**Research question**: Can agents learn to adapt to different opponent strategies (conservative vs. aggressive) by playing against checkpoints from different training stages?

**Baseline**: RLCard's NFSP with a single aggregated policy (no awareness of opponent type).

**Modification**:
1. Train an agent normally (NFSP self-play, 100k steps).
2. Save checkpoints every 10k steps (early-game, mid-game, late-game agents).
3. Train a "meta-agent" that plays against each checkpoint and learns to recognize opponent type (embed checkpoint ID or opponent action sequence).
4. Condition policy on estimated opponent type (e.g., separate policy heads for "weak" vs. "strong" opponent).

**Hypothesis**: Meta-agent achieves higher payoff than baseline NFSP because it adapts policy per opponent.

**Experiment**:
1. Train baseline NFSP (50k steps).
2. Train meta-agent (50k steps, with opponent detection).
3. Evaluate on Leduc Hold'em (simpler, faster to train).
4. Tournament: baseline vs. meta-agent vs. each checkpoint.
5. Hypothesis: Meta-agent payoff ≥ baseline + α% (α = 5).

**Metrics**:
- Payoff vs. baseline and checkpoints.
- Opponent type recognition accuracy (does agent learn to distinguish weak from strong?).

**Difficulty**: Moderate–Hard (multi-task learning, opponent modeling).

**Compute**: Medium (multi-seed training, ~4 hours).

**Novelty**: Yes; implicit opponent modeling via checkpoints is simple but novel.

---

### Idea 3: Action Encoding for Policy Gradient Methods

**Research question**: Do policy gradient methods (A3C, PPO) perform better than DQN on Dou Dizhu if we encode actions in a learnable way?

**Baseline**: RLCard's DQN with fixed action indices (plateau at Figure 2d).

**Modification**:
1. Implement A3C (actor-critic) or PPO in RLCard.
2. Use action embeddings: each action has a learned embedding vector that represents its semantics (card combination type, kicker, etc.).
3. Policy network outputs logits in embedding space (reduces output dimension from 309 to, say, 64D).
4. Decoder maps embedding back to action index.

**Hypothesis**: Policy gradients with action embeddings converge better than DQN on large action spaces because (a) policy gradient avoids Q-value overestimation, (b) action embeddings generalize across similar actions.

**Experiment**:
1. Implement A3C (use PyTorch or TensorFlow).
2. Train on small Dou Dizhu for 100k steps.
3. Compare learning curve: DQN (RLCard) vs. A3C (your implementation) vs. A3C + action embeddings.
4. Hypothesis: A3C + embeddings curve is steeper (better sample efficiency).

**Metrics**:
- Learning curve (reward vs. timesteps).
- Sample efficiency (payoff reached at same timestep).
- Embedding quality (t-SNE visualization of similar actions clustering).

**Difficulty**: Hard (policy gradient implementation, debugging).

**Compute**: Medium–High (multiple seeds, ~6–8 hours on GPU).

**Novelty**: Yes; action embeddings for policy gradients in card games is novel. Inspired by recent work in RL (Dulac-Arnold et al., 2015) but not applied to Dou Dizhu in RLCard.

---

### Idea 4: Curriculum Learning for Imperfect Information

**Research question**: Can we improve RL on Dou Dizhu by starting with simpler games (fewer cards, fewer players) and gradually increasing complexity?

**Baseline**: RLCard's DQN trained directly on Dou Dizhu (100k steps, plateaus).

**Modification**:
1. Design a curriculum: Level 1 = 3-player game with 10 cards each (super easy). Level 5 = full Dou Dizhu (54 cards).
2. Train agent on Level 1 for 20k steps (should converge quickly).
3. Transfer weights to Level 2, train for 20k steps, etc.
4. Final level (Level 5): agent trained on full game for 20k steps.
5. Compare to agent trained directly on Level 5 for 100k steps.

**Hypothesis**: Curriculum + transfer learning converges faster (steeper learning curve on Level 5) and achieves higher final payoff than direct training, despite same total compute.

**Experiment**:
1. Implement levels (modify RLCard's Dou Dizhu to support variable deck size).
2. Train curriculum agent (20k steps per level, 5 levels).
3. Train baseline agent (100k steps on full game).
4. Learning curves: compare Level 5 performance over 100k steps.
5. Hypothesis: Curriculum reaches higher payoff by step 50k.

**Metrics**:
- Learning curve (reward vs. steps on full game).
- Final payoff at 100k steps.
- Transfer efficiency (payoff gained per step on full game).

**Difficulty**: Easy–Moderate (curriculum design, transfer learning, RLCard customization).

**Compute**: Low (training simpler games is fast; ~4 hours total).

**Novelty**: Yes; curriculum learning for Dou Dizhu is underexplored. Intuitive but needs validation.

---

## 16. Important Implementation Components

RLCard is a toolkit, so "components" refer to the environment system, not algorithm modules. I categorize what you need to understand and potentially modify.

### MUST UNDERSTAND AND IMPLEMENT/MODIFY

1. **Environment Step/Reset Interface**
   - **What**: `env.reset()`, `env.step(action)`, `env.step_back()`.
   - **Why critical**: You'll call these in your training loop. Understanding delayed state observation (extensive-form games) is essential.
   - **Where in code**: RLCard core (not shown in paper; in GitHub).
   - **Modification needs**: If you extend to new game or rule variant, must re-implement step/reset logic.

2. **Action Masking / Legal Actions**
   - **What**: `state["legal_actions"]` is a list of valid action indices. Must respect this.
   - **Why critical**: Invalid actions crash agents or violate game rules. Masking prevents errors.
   - **Where in paper**: Appendix, Action Encoding, p. 7. In code: RLCard environment base class.
   - **Modification needs**: If you change action encoding, must recompute legal actions.

3. **State Representation (Card Planes)**
   - **What**: For Dou Dizhu, 6 card planes encoding hand, opponent hands, action history, played cards.
   - **Why critical**: This is the input to your neural network. Encoding quality affects learning speed.
   - **Where in paper**: Appendix, State Representation, p. 6–7. In code: RLCard's state encoder.
   - **Modification ideas**:
     - Try different plane arrangements (e.g., add planes for "cards I can play now").
     - Use different card rank/suit encoding (one-hot vs. binary).
     - Add temporal information (time-of-game phase).

4. **Reward Function**
   - **What**: Per-game payoff (Dou Dizhu: +1 for Landlord win, -1 for Peasant loss) or per-hand rewards.
   - **Why critical**: This is the learning signal. Poor rewards → poor learning.
   - **Where in paper**: Evaluation, Results, p. 4. In code: game's `get_payoffs()` method.
   - **Modification ideas**:
     - Reward shaping: e.g., give small bonus for each card played (encourages early wins).
     - Role-dependent rewards: e.g., Landlord gets +1, each Peasant +0.5 (reduce reward skew).
     - Relative rewards: payoff relative to average opponent (captures strength, not luck).

---

### MUST UNDERSTAND CONCEPTUALLY

1. **Extensive-Form Games and Information Sets**
   - **What**: Each player's observation corresponds to a set of possible hidden states (information set). Agents make decisions based on their information set, not the true state.
   - **Why**: This is the game-theory foundation for imperfect-information games. Understand information sets → understand why DQN/NFSP work/fail.
   - **Where in paper**: Introduction, p. 1; Related Work, p. 4 (regret minimization in extensive games).
   - **Learning resource**: Read Zinkevich et al. (2008) or Von Stengel (2010) on extensive-form games.

2. **Imperfect Information and Belief States**
   - **What**: Agent doesn't know opponent hands; must maintain a belief (distribution over possible opponent hands).
   - **Why**: This motivates opponent modeling and implicit belief state learning in neural networks.
   - **Where in paper**: Imperfect Information, §8. In RLCard: implicit in 6 card planes (no explicit belief state).
   - **Learning resource**: Read Zha et al. (DouZero, 2021) on how Monte-Carlo methods handle belief states via sampling.

3. **Action Abstraction and Its Trade-offs**
   - **What**: Dou Dizhu's 30k actions → 309 via rule-based kicker selection.
   - **Why**: Action abstraction is necessary for tractability but may lose important distinctions.
   - **Where in paper**: Appendix, Action Encoding, p. 7 (reduces to 309).
   - **Learning resource**: Read Lanctot et al. (2015) on action abstraction in poker.

4. **Multi-Agent Turn-Taking and Role-Specific Rewards**
   - **What**: In Dou Dizhu, Landlord and Peasants have different win conditions and rewards.
   - **Why**: Affects learning dynamics (agents trained on Landlord role may not transfer to Peasant).
   - **Where in paper**: Overview, p. 2 (delayed state observation); Evaluation, p. 4 (role-specific results).

5. **Self-Play and Non-Stationarity**
   - **What**: NFSP trains by playing against past versions of itself, leading to adaptive/non-stationary opponents.
   - **Why**: Non-stationarity is a challenge and an opportunity (forces agents to generalize).
   - **Where in paper**: Training Agents, p. 3 (NFSP mentioned); fuller treatment in Heinrich & Silver (2016).

---

### CAN REUSE AS INFRASTRUCTURE

1. **RLCard's Environment API** (the whole toolkit)
   - **What**: The `make()`, `step()`, `run()` interface and parallel data generation.
   - **Why reuse**: Building a game from scratch takes weeks. RLCard saves that.
   - **How to reuse**: Install RLCard, call `env = rlcard.make('doudizhu')`, write your algorithm on top.
   - **[From public code repo, not the paper]** RLCard is stable, well-tested, open-source with regular maintenance.

2. **Baseline Algorithm Implementations** (DQN, NFSP, CFR)
   - **What**: Reference implementations in the RLCard repository.
   - **Why reuse**: Don't rewrite DQN from scratch; adapt RLCard's version.
   - **How to reuse**: Fork RLCard code, modify hyperparameters, add your innovation (e.g., new loss function, architecture).

3. **Evaluation Infrastructure** (tournament harness, payoff tracking)
   - **What**: Tournament play, seed management, result aggregation.
   - **Why reuse**: Evaluating agents fairly is non-trivial. RLCard handles it.
   - **How to reuse**: Use RLCard's tournament utilities to compare your agent vs. baselines.

4. **Action Masking and Legal Action Generation**
   - **What**: Environment computes legal actions; agents never sample invalid moves.
   - **Why reuse**: Implementing action validity for Dou Dizhu is error-prone.
   - **How to reuse**: Call `state["legal_actions"]`, condition policy/Q-values on legal actions.

5. **Multi-Process Data Generation**
   - **What**: Parallel actors, parameter broadcasting, trajectory collection.
   - **Why reuse**: Parallel training is essential for large games; RLCard has it built-in.
   - **How to reuse**: Use RLCard's multi-process API (or adapt for your algorithm).

6. **Rule-Based Baseline Agents** [From public code repo, not the paper]
   - **What**: Pre-trained rule-based models for Dou Dizhu, Hold'em, UNO.
   - **Why reuse**: Evaluation against rule-based baseline is more informative than random agents.
   - **How to reuse**: Load pre-trained rule-based agent, evaluate your agent against it.

---

## 17. Questions I Should Ask My CS230 Staff

1. **On novelty and scope**: 
   - "My project idea is [learning action abstractions / opponent modeling / curriculum learning] on Dou Dizhu using RLCard. Is this novel enough for a quarter project, or is it just applying existing techniques? Should I aim for a specific algorithmic contribution (like DouZero's DMC) or is an empirical study OK?"

2. **On compute and feasibility**:
   - "Figure 2d shows DQN/NFSP plateau on Dou Dizhu after 700k timesteps. On a single GPU (e.g., Colab free tier), this would take ~2 hours. If my idea requires 10 different training runs (hyperparameter search or ablations), that's 20 GPU-hours—beyond free Colab. Should I aim for smaller experiments, or is cloud compute expected?"

3. **On evaluation rigor**:
   - "RLCard evaluates on 10k games per tournament, but doesn't report confidence intervals. For my project, should I target a specific statistical rigor level (e.g., 95% CI width ≤ 0.10)? Or are point estimates OK as long as sample size is reasonable (~1k–3k games)?"

4. **On baselines**:
   - "Should I implement DQN/NFSP from scratch to demonstrate understanding, or is using RLCard's baselines OK (and focus my effort on the novel parts)? Is there a minimum bar for reimplementation?"

5. **On related work**:
   - "How much should I engage with the DouZero paper (which addresses RLCard's limitations)? Should my project compare to DouZero, or is RLCard's baselines sufficient? DouZero seems state-of-the-art but might be overkill for a quarter project."

---

## 18. Beginner Concepts I Need to Learn

| Concept | Plain-Language Explanation (1–3 sentences) | Where It Appears in This Paper | Timeline |
|---------|---|---|---|
| **Reinforcement Learning (RL)** | An agent learns to make decisions by trial-and-error, receiving rewards/penalties from an environment. The agent's goal is to maximize cumulative reward over time. | Throughout (Introduction, RL Formulation). | **NOW** |
| **State and Observation** | State is the full game condition (all cards, all decisions). Observation is what the agent sees (own cards, action history). In imperfect-information games, they differ. | §2 (RL Formulation), §8 (Imperfect Information). | **NOW** |
| **Action Space** | The set of possible moves the agent can take. In Dou Dizhu, it's card combinations; in Blackjack, {Hit, Stand}. Large action spaces are hard for RL. | §2, §3 (Model/Architecture). | **NOW** |
| **Legal Actions** | A subset of all actions that are valid in the current state (e.g., can only fold if in a poker hand). RLCard always provides legal actions, preventing invalid moves. | §8 (Imperfect Information), §16 (Components). | **NOW** |
| **Reward** | A scalar signal indicating how well the agent did (e.g., +1 for winning, -1 for losing). Sparse rewards (rare signals) are hard for RL. | §2 (RL Formulation), §6 (Loss Function). | **NOW** |
| **Policy** | A function mapping states to actions (deterministic) or to probability distributions over actions (stochastic). The agent learns a policy. | §2 (RL Formulation). | **NOW** |
| **Value Function / Q-Function** | Q(s, a) estimates the expected cumulative reward if you take action a in state s. V(s) estimates reward from state s onward. DQN learns Q-values. | §2, §5 (Mathematics), §6 (Loss Function). | **NOW** |
| **Temporal Difference (TD) Learning** | A method to update value estimates using one-step lookahead: V(s) ← V(s) + α[r + γV(s') − V(s)]. Combines Monte-Carlo (sample rewards) and bootstrapping. | §5, §6 (DQN Loss). | **NOW** |
| **Exploration vs. Exploitation** | Exploration: try new actions to learn more. Exploitation: choose the best known action. ε-greedy balances them. | §4 (Training Algorithm, DQN). | **SOON** |
| **Experience Replay** | Store past transitions (s, a, r, s') in a buffer; sample random batches for training. Breaks correlation in data, stabilizes learning. | §4 (DQN Pipeline), §6 (Loss Function). | **SOON** |
| **Target Network** | A copy of the Q-network, updated periodically, used to compute targets. Stabilizes learning by preventing chasing a moving target. | §5 (Mathematics), §6 (DQN Loss). | **SOON** |
| **Self-Play** | An agent plays against itself (or past versions of itself). Useful in games where the opponent is also learning. NFSP uses this. | §7 (Self-Play). | **SOON** |
| **Imperfect Information** | Game state is not fully observable to all players; agents have private information (hidden opponent cards). Contrasts with perfect-information games (Chess, Go). | §1b, §8 (Imperfect Information). | **NOW** |
| **Information Set** | The set of game states that look identical to a player. In Dou Dizhu, an information set is all possible opponent hands that are consistent with observed actions. | §8, §12 (Limitations). | **SOON** |
| **Counterfactual Regret Minimization (CFR)** | A game-theoretic algorithm that iteratively updates strategies to minimize regret (how much worse you did than the best move in hindsight). Converges to Nash equilibrium. | §4, §5, §6, §10 (Experiments). | **OPTIONAL** |
| **Neural Fictitious Self-Play (NFSP)** | A self-play algorithm combining RL (best-response learning) and supervised learning (staying close to average opponent). Balances exploiting and adapting. | §4, §6, §7. | **SOON** |
| **Attention and Generalization in Large Action Spaces** | Standard RL outputs Q-values for all actions, requiring output size = action space size. In large spaces (Dou Dizhu: 30k), this is wasteful. Action abstraction and embeddings help. | §3 (Model), §15 (Project Ideas). | **OPTIONAL** |
| **Reward Shaping** | Modifying rewards to make learning easier (e.g., small bonus for each card played, not just final win). Can mislead agents if done wrong. | §6 (Loss Function), §15 (Project Ideas). | **SOON** |
| **Multi-Agent Learning** | Multiple agents learn simultaneously; each faces a non-stationary environment (opponent changes). Harder than single-agent RL. | §1b, §7 (Self-Play). | **SOON** |
| **Extensive-Form Games** | Game-theoretic model for sequential games with imperfect information (e.g., poker, Dou Dizhu). Actions by players are revealed over time. | §1b, §16 (Components). | **OPTIONAL** |
| **Action Abstraction** | Reducing the size of the action space by grouping similar actions (e.g., "trio of 5s + small kicker" grouped as one action). Trades off expressiveness for tractability. | §3, §6, §15 (Project Ideas). | **SOON** |

---

## 19. Paper Scorecard

> **Cross-paper calibration:** this scorecard looks at this paper alone. Ranked against the whole pack for this Dou Dizhu project, the recommended verdict is **READ SELECTIVELY**. See `00_SYNTHESIS_AND_RECOMMENDATION.md` §20.9.

| Criterion | Score (1–5) | Justification |
|-----------|-------------|---|
| **Importance for my project** | **5** | RLCard is infrastructure for Dou Dizhu RL projects. If you build on RLCard (likely), this paper defines the environment, API, and baseline results. Essential. |
| **Difficulty to understand** | **2** | Toolkit paper; no novel algorithms. Read §1–2, §3 (state/action encoding), §7–9 (self-play and evaluation), and Appendix. Most sections are clear descriptions, not theory. |
| **Difficulty to reproduce** | **2** | Small-scale reproduction is easy (Blackjack, Leduc in <1 hour). Dou Dizhu is doable on GPU. Full reproduction (all environments, Table 2) is hard, but not necessary for this project. |
| **Relevance to Dou Dizhu** | **5** | Dou Dizhu is the focus game of this project. RLCard's Dou Dizhu implementation, evaluation protocol, and baseline results (DQN/NFSP plateau) are central. |
| **Relevance to self-play** | **4** | NFSP (self-play algorithm) is discussed and evaluated. But RLCard doesn't uniquely advance self-play theory; it enables it. Learn self-play concepts from Heinrich & Silver (2016) instead. |
| **Relevance to imperfect information** | **5** | Imperfect information (hidden cards, belief states) is the core challenge. RLCard's state encoding and information-set-aware API are directly relevant. Understand §8 deeply. |
| **Potential for project ideas** | **4** | RLCard enables projects (action abstraction, opponent modeling, curriculum learning). Limitations suggest clear research directions. Fewer algorithmic insights than a method paper, but more practical ideas. |
| **Importance to read deeply** | **4** | Read sections: 1–2 (overview, RL formulation), 3 (state/action encoding), 7–9 (self-play, imperfect information, evaluation), 12–15 (limitations, future work, project ideas). Skim Appendix if time-constrained. |

---

## Verdict: **READ DEEPLY**

**Suggested reading order** (by importance for this CS230 project):

1. **First (§1–2, 3, 8, 9)** ← **Essential** (2 hours)
   - Overview, RL Formulation, State/Action Encoding for Dou Dizhu, Imperfect Information, Evaluation.
   - Understand what RLCard provides and why you should use it.

2. **Second (§7, 12–15)** ← **High** (1.5 hours)
   - Self-Play, Limitations, Future Work, Project Ideas.
   - Identify opportunities for innovation and understand gaps in current work.

3. **Third (§4–6, 10–11)** ← **Moderate** (1 hour)
   - Training Algorithm, Mathematics/Loss, Experiments, Compute.
   - Useful for reproducing baselines; can skim if pressed for time.

4. **Optional (Appendix, §16–18)** ← **Reference** (as needed)
   - Implementation details, concepts to learn.
   - Return to as you code or hit conceptual snags.

**Total reading time**: 3–4 hours for deep understanding; 1.5–2 hours for selective reading.

**After reading**: Install RLCard, run the Blackjack or Leduc Hold'em examples (§1, p. 2), and sketch the project idea (§15).

---

## Appendix: Synthesis Facts (for the cross-paper comparison table)

- **Paper**: RLCard: A Toolkit for Reinforcement Learning in Card Games (Zha et al., IJCAI 2020)
- **Year**: 2020 (arXiv 2019, IJCAI 2020)
- **Environment(s)**: Blackjack, Leduc Hold'em, Limit Texas Hold'em, Dou Dizhu, Mahjong, No-limit Texas Hold'em, UNO (7 games; 2–3 players; imperfect information with large state/action spaces)
- **Main algorithm**: Three baselines (DQN, NFSP, CFR); toolkit, not algorithm-focused. DQN single-agent (vs. random), NFSP self-play, CFR game-theoretic.
- **State representation**: Observation dictionary with legal actions + card planes (Dou Dizhu: 6 planes = hand, opponent hands, recent actions, played cards). Paper specifies 6 planes but not exact tensor shape; shape [6, 5, 15] inferred from typical card encoding.
- **Action representation**: Integer indices (0–308 for Dou Dizhu after abstraction). Action abstraction: rule-based kicker selection reduces action space from more than 3×10⁴ to 309. No abstract actions for small games.
- **Self-play?**: Supported (NFSP does self-play; DQN trains vs. fixed random agents; CFR is game-theoretic, no learning).
- **Opponent modeling?**: Implicit (6 card planes encode inferred opponent hands via card elimination; no explicit belief state or opponent modeling).
- **Main loss**: DQN: TD loss (MSE). NFSP: RL loss (TD) + SL loss (policy KL). CFR: regret minimization (not gradient-based).
- **Main contribution**: Unified, open-source toolkit for RL in imperfect-information card games; standardized environment API (run(), step/step_back); reproducible baseline results (DQN/NFSP plateau on large games, CFR wins on small games); action abstraction for Dou Dizhu.
- **Evaluation method**: (1) Learning curves vs. random agents (10k timesteps). (2) Tournament play (10k games per pair, role-specific). Issue: random agent baseline is weak; no rule-based baselines initially.
- **Compute**: Server with 24 Intel Xeon Silver 4116 CPUs @ 2.10 GHz, 64GB RAM; GPUs not mentioned. Dou Dizhu: 87,270.8s single-process (24 hours), 7,275.9s 16-process (2 hours). Training: 50k–700k timesteps; experiments run 3 times with different seeds.
- **Main limitation**: Standard RL algorithms (DQN/NFSP) fail on large games due to action space and sparse reward (Figure 2d, 2e, 2g plateau quickly). No opponent modeling. Action abstraction is ad-hoc (kicker selection is rule-based, not learned).
- **Best idea I could borrow**: (1) Environment API design (run/step/step_back for extensible game engine). (2) Action abstraction strategy (reduce large action space via rules + neural learning on "major" actions). (3) Multi-process parallel training (speedup × sublinearly). (4) Design principles (reproducibility via seeds, accessibility via customization, scalability via clean architecture).

