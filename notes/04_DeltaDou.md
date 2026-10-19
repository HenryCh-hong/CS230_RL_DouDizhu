# DeltaDou: Expert-level Doudizhu AI through Self-play (IJCAI 2019)

> Qiqi Jiang, Kuangzheng Li, Boyao Du, Hao Chen, Hai Fang (SweetCode Inc, Beijing) · [IJCAI-19](https://www.ijcai.org/) · **Tier A** · First RL system to reach human-expert level in multi-player imperfect-information card games using FPMCTS (a new search variant) + Bayesian card inference + self-play.

---

## 0. TL;DR for a CS230 student

- **What it is:** DeltaDou combines three techniques to play Dou Dizhu at human-expert level: (1) Fictitious Play MCTS (FPMCTS)—a new search tree variant where only the acting player uses tree search, other players' moves are sampled from policy networks; (2) Bayesian inference over hidden opponent hands, weighted by policy probabilities; (3) AlphaZero-style self-play with separate networks for each role (Landlord, Peasant-D, Peasant-E).
- **Why it matters:** It demonstrates that **policy-based opponent modeling** (inferring opponent hands using their own policy network) is more robust than hand-crafted features—a key insight for imperfect-information games.
- **The hard problem:** Dou Dizhu has ~10^4 possible actions, hidden opponent hands, 3-player asymmetry (Landlord vs Peasants), and cooperation dynamics. Previous MCTS variants (ISMCTS, self-play MCTS) suffered from "Strategy Fusion" (mixing multiple determinizations) and "Non-Locality" (incorrect belief updates).
- **The core innovation:** FPMCTS sidesteps these by building a separate, smaller search tree for each player (only branching over player's own actions, not all determinizations). This converges faster and avoids information-leaking bugs.
- **One big caveat:** FPMCTS + Bayesian inference + neural networks are **computationally expensive**—training takes ~2 months on 68 CPUs. Later work (DouZero, 2021) surpasses DeltaDou in ~10 days with 4 GPUs by using Monte-Carlo sampling instead of tree search.
- **What to steal for a CS230 project:** (1) **Bayesian posterior over opponent hands**, conditioned on observed play; (2) **two-phase inference** (random sampling → filtering → Bayesian update) as a practical middle ground between brute-force and intractability; (3) **role-specific networks** to handle asymmetric games; (4) **kicker heuristics** for action abstraction.

---

## 1. Paper Overview

### 1a. Beginner-friendly explanation

Imagine you are playing a card game with two friends, but you cannot see their cards. At each turn, you must choose which cards to play. The challenge is: *How do you decide what your opponents have?* And then, *how do you search forward to find your best move, when the game tree explodes because you don't know what cards are hidden?*

DeltaDou solves this in three steps:

1. **Guess your opponents' hands using math + AI.** After each move by an opponent, update your belief ("I saw them play a high card; they probably don't have many highs left"). Use a neural network trained on millions of self-play games to predict what they would play given each possible hand.

2. **Search ahead using a smart tree.** Instead of building a single tree and guessing all possible hidden hands (which would explode), build one tree per player. When it's your turn, explore moves in the tree. When it's your opponent's turn, *sample* one possible hand and one move from their learned policy. This is **much faster** than traditional search methods.

3. **Train by playing against yourself.** Play millions of games where your network learns from its own wins and losses (self-play). Every few episodes, add the self-play results to a dataset, train the network, and play again. After weeks of this, the AI becomes expert-level.

**Why is this hard?** Dou Dizhu is not just imperfect information (hidden cards). It's also:
- **Asymmetric:** One player (Landlord) plays alone; two players (Peasants) are teammates. Their goals are opposite, but teammates must cooperate without talking.
- **Huge action space:** ~10,000 possible legal moves, varying per game state.
- **Search explosion:** Normal game tree search (AlphaZero style) requires exploring all possible hidden hands, which is combinatorially huge.

DeltaDou cuts through this by using *separate trees per player* and *learned policy sampling* rather than exhaustive enumeration.

### 1b. Technical explanation

**Problem.** Dou Dizhu is a three-player sequential game with imperfect information (each player observes only their own cards and the history of played cards). The state space is large (~10^15 game states), the action space is massive (~10^4 actions), and the players are asymmetrically positioned (Landlord vs Peasants). Existing MCTS variants for imperfect-information games (ISMCTS, Self-play MCTS) suffer from:
- **Strategy Fusion:** Mixing strategies from multiple determinizations (possible hand assignments) can lead to playing strategies that are inconsistent with any real hidden state.
- **Non-Locality:** Incorrect belief updates when game trees are built over information sets.

**Why it is difficult.** Classical MCTS (AlphaGo style) works for perfect-information games by exploring the full deterministic tree. For imperfect-information games, naive MCTS requires sampling ("determinizing") opponent hands and playing out all possibilities, which explodes combinatorially. Prior solutions (ISMCTS, self-play MCTS) attempted to fix this but introduced subtle bugs (strategy fusion, information leaking).

**Main idea.** DeltaDou proposes **Fictitious Play MCTS (FPMCTS)**, which:
- Builds a separate search tree **for each player** (i.e., each player's own information state).
- At decision nodes (player's turn), uses p-UCT (policy-prior upper confidence bound tree search) to balance exploitation and exploration.
- At chance nodes (opponent's turn), **samples opponent moves from their learned policy network** (rather than exhaustively branching).
- Determinizes the game once at the root (via Bayesian inference over hidden hands), then follows the sampled play.
- Expands by two levels at a time (player's move + both opponents' consecutive moves), to keep leaf nodes as decision nodes.

This avoids strategy fusion and information leaking because each player's tree is consistent with that player's information set—no mixing of determinizations.

**Most important contributions:**
1. **FPMCTS algorithm:** A variant of MCTS for multi-player imperfect-information games that sidesteps strategy fusion and non-locality by using separate trees and policy-based sampling.
2. **Policy-based inference:** A two-phase Bayesian method to infer opponent hands:
   - **Phase 1:** Randomly generate many possible hands and filter to likely ones using a neural network trained on self-play.
   - **Phase 2:** Use Bayes' theorem to weight remaining samples by the probability they explain the observed action history.
3. **AlphaZero-like integration:** Combines FPMCTS + inference + separate role-specific policy-value networks in a self-play loop.
4. **Empirical result:** First RL system to achieve human-expert level in Dou Dizhu.

**What came before:** 
- AlphaZero (Silver et al., 2018) showed that self-play + deep neural networks can master perfect-information games (Go, Chess, Shogi).
- Neural Fictitious Self-Play (NFSP; Heinrich & Silver, 2016) combined policy gradients with game-theoretic thinking for imperfect-information games.
- Poker research (Libratus, DeepStack) used CFR and search, but CFR doesn't scale to large action spaces.
- Prior Dou Dizhu work (CQN by You et al., 2019) relied on heavy action abstraction and heuristics.

**What limitation motivated it:** AlphaZero-style tree search does not directly apply to imperfect-information games. Prior attempts (ISMCTS, self-play MCTS) were either prone to bugs or computationally expensive. DeltaDou aimed to adapt AlphaZero's self-play framework to multi-player imperfect-information games by using **policy networks to model opponents** and **Bayesian inference to handle hidden information**, rather than trying to build a single global search tree.

---

## 2. RL Formulation

| Component | Description | Citation |
|-----------|-------------|----------|
| **Agent(s)** | Three agents: Landlord (C), Peasant-D (D), Peasant-E (E). Peasants D and E are teammates (cooperative). Each has its own policy network. | §4.4, p. 6 |
| **Environment** | Dou Dizhu card game. Deterministic except for card shuffling at game start. | §3.1, p. 3–4 |
| **State** | Game state $s \in S$: full deck composition, current player's cards, all played cards, action history. Players cannot observe opponent hands. | §3.2, p. 4 |
| **Observation** | Each player $i$ observes: own cards $c_i$, action history $H = [a_0, a_1, ..., a_j]$. Information state $u^i = I^i(s)$ groups indistinguishable states. | §3.2, p. 4 |
| **Action** | Legal move at state $s$: play a hand category (single, pair, chain, trio+kicker, bomb, rocket, or pass). ~10^4 possible combinations, varying per state. | §3.1, p. 4; §4.2, p. 5 |
| **Legal action handling** | Network outputs are constrained to legal moves only (via masking or softmax over legal action buckets). Kicker network applies heuristic rules to convert policy output to final move. | §4.2, p. 5 |
| **Reward** | Terminal reward: Landlord gets +1 if they win, -1 if Peasants win. Peasants get +1 collectively if at least one has no cards. Bombs and rockets played during the game double the score. The paper does not specify immediate rewards during play. | §3.1, p. 4; §5.1, p. 6 |
| **Return** | Cumulative discounted reward over the game (or final outcome). The paper does not specify the discount factor; likely undiscounted (γ=0 or γ=1 with episodic games). | The paper does not specify this clearly. |
| **Policy** | $\pi^i(u) \to \mathcal{P}(A(u))$: each player $i$ has a learned policy mapping information state to action distribution. Represented by a neural network $f_\theta$ outputting $(p, v)$: action logits $p$ and value $v$. | §4.2, p. 5 |
| **Value function / Q-function** | $v^\theta(u)$: approximates the value (expected return) for the acting player at information state $u$. No explicit Q-function; value is scalar per state. | §4.2, p. 5 |
| **Fully vs partially observable** | Partially observable. Each player observes own cards + action history, but not opponent hands. | §3.2, p. 4 |
| **Single vs multi-agent** | Multi-agent (3 players). Each player has its own policy $\pi^i$ learned independently (but updated in mutual self-play). | §4.4, p. 6 |
| **Cooperative/competitive/both** | Mixed: Landlord (C) is competitive with Peasants (D, E); D and E are cooperative with each other (teammates). Peasants are also competitive with each other in some sense (both want to win, but only one needs to empty cards first). | §3.1, p. 4; implicitly from game rules |
| **Self-play used?** | Yes. Self-play is the core training loop. Episodes alternate roles (each player plays as C, D, E across games), and the network is updated after each episode to improve. | §4.4, p. 6; §5.1, p. 6 |

---

## 3. Model / Architecture

**Overview.** The model is a single **policy-value network $f_\theta(s) = (p, v)$** shared across all players, but separate network instances are trained for each role (Landlord, Peasant-D, Peasant-E). This asymmetry is necessary because the Landlord and Peasants have different objectives and information sets.

**Input representation:** A condensed **15×7 matrix** encoding the current information state (p. 5–6):
- Current player's cards
- Union of other players' cards (visible because played cards are public)
- Recent three actions (the last 3 moves in game history)
- Union of played cards (all cards that have been played out)

The paper notes this is a "trade-off between precision and speed"—a lower-dimensional representation than full state enumeration, chosen for computational efficiency.

**Card representation:** The paper does not explicitly state how individual cards are encoded in the matrix, but given the use of 1D-convolutional layers and emphasis on "neighboring relations between different card ranks," cards are likely encoded by rank (15 ranks: Red Joker, Black Joker, 2, A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3) and frequency, with spatial structure preserved.

**Action representation:** The paper reduces the action space from ~10,000 to a smaller set by encoding **rank and hand category** (single, pair, trio, chain, bomb, etc.) separately, without the kicker (the extra card appended to a trio or five-card combo). After the network outputs a distribution over (rank, category) pairs, a separate **kicker network** and heuristic rules convert this to the actual move with a specific kicker. This is a form of **action abstraction**—reducing the raw action space by assuming certain structure. (§4.2, p. 5; p. 6.)

**Network architecture:**
$$f_\theta(s) = (p, v)$$
- **10 residual blocks** with batch normalization and ReLU non-linearities (§4.2, p. 5).
- **1D-convolutional layers** to exploit the sequential structure of card ranks.
- Output $(p, v)$: 
  - $p$: policy logits over legal moves (classified into $n$ buckets by hand category).
  - $v$: scalar value estimate.

**Separate networks per role:** Three independent networks, $f_\theta^C$, $f_\theta^D$, $f_\theta^E$, for Landlord, Peasant-D, and Peasant-E, respectively. This captures role-specific strategies (e.g., Landlord plays more aggressively to empty cards first; Peasants play cooperatively). (§4.4, p. 6.)

**Network I/O shapes:**
- Input: $15 \times 7$ matrix.
- Output $p$: matrix of logits over legal moves, classified into hand-category buckets (exact dimensions not stated).
- Output $v$: scalar in $[-1, 1]$ (normalized).

**Kicker network:** A separate small network trained to predict the optimal kicker given the trio/5-card-combo rank and context. The paper notes that "the kicker plays an important role in DouDizhu and cannot be easily abstracted" (DouZero paper, p. 2), so the decision to use a dedicated kicker network was pragmatic. Details of the kicker network architecture are not provided in the paper.

**Why these choices:** 
- **Residual blocks + batch normalization:** Standard for deep networks; improves training stability and gradient flow.
- **1D-convolutions:** Exploit the linear structure of card ranks; neighboring ranks have similar strategic properties.
- **Separate networks per role:** Asymmetric game structure (Landlord vs Peasants) is better captured with role-specific policies.
- **Action abstraction (rank+category, then kicker):** Reduces branching factor in search from ~10^4 to ~few hundred for the main network, plus deterministic or learned kicker selection.

---

## 4. Training Algorithm

**Full pipeline (step-by-step):**

1. **Initialization:** Train a heuristic AI on hand-crafted rules (§5.1, p. 6). Generate 200,000 games of self-play using the heuristic AI. Use these games to supervised-learn initial policy-value networks for C, D, E via cross-entropy loss on actions and MSE on outcomes.

2. **Self-play episode:** (§5.1, p. 6)
   - Generate 8,000 random games in parallel (one episode = 8,000 games).
   - For each game, all three players use the latest network copies.
   - During each game, play out according to FPMCTS + Bayesian inference (see §4.1, §4.3 below).

3. **Observation & action selection in-game:**
   - Current player observes: own cards, action history, played cards → information state $u^i$.
   - Call FPMCTS(u^i, π^i) to get a distribution over moves (visit counts or action probabilities from the search tree).

4. **FPMCTS search:** (§4.1, Algorithm 1, p. 5)
   - **Setup:** At the root, infer a small set of deterministic worlds (opponent hand samples) using Bayesian inference (§4.3, p. 5). Typically 1–2 determinizations per game state.
   - **Simulation loop:** Run 400 playouts per search during self-play training (§5.1, p. 6). During evaluation, 600 simulations are used (§5.2, p. 7).
     - For each simulation:
       - Sample a deterministic world $s$ from the posterior over hidden hands (line 3, Algorithm 1).
       - Call SIMULATE(s):
         - If terminal, return reward.
         - While not at a leaf: SELECT the best action using p-UCT (line 12), update information state.
         - At leaf: evaluate via value network $v^\theta(u_i)$ or terminal reward (lines 14–19).
         - EXPANDTREE: add the current player's move to the tree; sample two consecutive opponent moves from their policy networks (lines 23–28); update the tree with the new information state and backprop value (line 20).

5. **FPMCTS details (Fictitious Play assumption):** The search assumes opponents always play according to their learned policy $\pi^j(u^j)$, without best-responding. This is called "fictitious play"—each opponent maintains a fixed strategy during the search, even if it's suboptimal in the current game state. This simplifies search and avoids recursive best-response computation.

6. **Move selection after search:** After FPMCTS completes, use the root node's visit counts or action distribution to select a move:
   - Early in training (high temperature): sample from the distribution.
   - Late in training (low temperature): take the greedy action.
   (§5.1, p. 6: "high temperature is used to reduce the overfitting problem of policy, and the temperature cools down during self-play.")

7. **Environment step:** Apply move to game state; opponent's move is sampled in FPMCTS, so the game transitions deterministically through the sampled worlds.

8. **Trajectory collection:** Store (state, search_policy, game_outcome) for each state visited during the episode. **The paper does not specify whether individual move rewards are stored or only terminal outcomes.**

9. **Learning phase (after episode):** (§4.2, §6, p. 5–6)
   - Accumulate all trajectories from the 8,000 games into a **replay buffer** (reservoir).
   - Sample mini-batches from the buffer.
   - For each state:
     - **Target policy:** MCTS visit-count distribution (from the search) or empirical action frequency.
     - **Target value:** Game outcome (win=+1, loss=-1 for the player's team).
   - Compute loss (see §6 below).
   - Gradient descent on $\theta$.

10. **Iteration:** Repeat for multiple episodes (80 episodes in the final DeltaDou; §5.2, p. 7 = 640,000 games total).

**Algorithm 1 (FPMCTS) pseudocode from paper (p. 5):**
```
function SEARCH(u_i, π):
  for i in (1, ..., simulate_num) do
    s ~ u_i                          // determinization: sample a hidden-card assignment
    SIMULATE(s)
  end for
end function

function SIMULATE(s):
  if ISTERMINAL(s) then
    return REWARD(s)
  end if
  while u_i is not leaf do
    (a, u_i) = SELECT(T^i(u_i))      // p-UCT selection
  end while
  if ISTERMINAL(u_i) then
    v = R_i(u_i)                     // terminal reward
  else
    v = v^i(u_i)                     // value function estimate
    EXPANDTREE(T^i, u_i, s, a)
  end if
  UPDATE(u_i, a, v)                  // backprop
end function

function EXPANDTREE(T^i, u_i, s, a):
  a ~ π^i(u_i)                       // player's action sampled from policy
  s = G(s, a)                        // game transition
  for i in (1..2) do                 // two opponent moves
    k = ρ(s)                         // determine whose turn
    a ~ π^k(I^k(s))                  // opponent k samples move from policy
    s = G(s, a)                      // game transition
  end for
  u_i = I^i(s)                       // new information state for player i
  update chance node with u_i
end function
```

**Key insight:** FPMCTS does not require players to best-respond during search. Instead, opponents play according to their learned policies. This is computationally cheaper than recursive best-response (which would require solving a game tree in the inner loop) and has a nice game-theoretic interpretation: the search is computing a best response to the current policy of the opponents.

---

## 5. Mathematics

### Equation 1: p-UCT (Policy-prior Upper Confidence Bound Tree search)

**MUST UNDERSTAND (for CS230)** — This is how the search tree is explored.

$$\pi_{tree}(u^i) = \arg\max_{a \in A(u^i)} \left[ Q(u^i, a) + \pi^i(u^i, a) \sqrt{\frac{N(u^i)}{1 + N(u^i, a)}} \right]$$

where:
- $Q(u^i, a)$ = average value of action $a$ taken from information state $u^i$ (averaged over all simulations that visited this edge).
- $\pi^i(u^i, a)$ = prior probability of action $a$ from the policy network $f_\theta^i$. This is the key difference from classical UCT: the exploration bonus is weighted by the neural network's prior, so actions the network thinks are good get more exploration.
- $N(u^i)$ = visit count to information state $u^i$ (total times this state was visited across all simulations).
- $N(u^i, a)$ = visit count to edge $(u^i, a)$ (total times this action was taken from this state).
- $\sqrt{\frac{N(u^i)}{1 + N(u^i, a)}}$ = exploration bonus, scaled by network prior. The paper does not specify $c_{pUCT}$ for training; during evaluation, $c_{pUCT} = 2$ is used (§5.2, p. 7).

**Intuition:** Balance exploitation (high Q-values) with exploration (high counts). The network prior $\pi^i$ helps focus search on promising actions. Without the prior, UCT explores uniformly; with it, UCT is "policy-guided."

**Relation to RL:** This is from AlphaGo/AlphaZero. The p-UCT formula is a tree search policy that uses a learned prior. In standard policy-gradient RL, the policy is learned via gradient updates. In MCTS + policy-value networks, the policy is used as a *prior* for the search, and the value function guides leaf evaluation (instead of rollouts, as in classical MCTS).

---

### Equation 2: Loss function

**MUST UNDERSTAND (for CS230)** — This is how the network is trained.

$$\ell = C_1 (z - v)^2 - \pi^* \log p + C_2 || \theta ||_2$$

where:
- $(z - v)^2$ = **value loss**: MSE between the target value $z$ (game outcome or bootstrapped value) and network output $v$. $C_1$ is a scaling constant to balance value and policy loss (the paper mentions "rescale the value error since $v$ is normalized").
- $-\pi^* \log p$ = **policy loss**: cross-entropy between the target policy distribution $\pi^*$ (from MCTS visit counts) and network output $p$ (logits). This is standard supervised learning on the search policy.
- $C_2 || \theta ||_2$ = **L2 regularization** for weight decay, controlling model complexity.

**Symbols defined:**
- $z$: target value. **The paper does not specify clearly.** Likely: game outcome (+1 for win, -1 for loss) if the game has terminated; otherwise bootstrapped from the value network at a future state. [Interpretation: In AlphaZero-style training, $z$ is typically the game outcome (Monte Carlo return), not a bootstrapped TD target, because self-play games are short and terminal rewards are well-defined.]
- $\pi^*$: target policy, derived from MCTS root visit counts: $\pi^*(a) \propto N(root, a)$. This is the distribution over actions that MCTS explored the most.
- $p$: network output (softmax over legal actions).
- $v$: network value output (scalar).
- $\theta$: network weights.

**Intuition:** The loss combines three goals:
1. Predict game outcomes accurately (value loss).
2. Match the search policy (the distribution of actions explored by MCTS).
3. Regularize to avoid overfitting.

**Relation to standard RL concepts:**
- **Bootstrapping?** No explicit temporal-difference (TD) bootstrapping. The target $z$ is likely the final game outcome (Monte Carlo return), not a TD target. This is consistent with AlphaZero, which uses episodic rollouts.
- **Monte Carlo?** Yes—the game is played to termination, and the return (cumulative reward) is used as the target.
- **Policy loss?** Yes—this is supervised learning on the MCTS policy, a form of "policy distillation." This is central to AlphaZero-style training.
- **Value loss?** Yes—standard regression on game outcomes.

---

### Equation 3: Bayesian posterior over hidden hands (for opponent card inference)

**MUST UNDERSTAND (for CS230)** — This is core to handling imperfect information.

$$p(s | H) = \frac{p(s) p(H | s)}{\sum_{s' \in S} p(s') p(H | s')}$$

where:
- $p(s | H)$ = **posterior** probability of a specific determinized game state $s$ (i.e., a specific assignment of hidden cards to opponents) given the observed action history $H$.
- $p(s)$ = **prior** probability of state $s$. Uniform if cards are dealt uniformly at random.
- $p(H | s)$ = **likelihood**: probability of observing the action history $H$ given state $s$. Factorizes as:

$$p(a_1, a_2, \ldots, a_k | s) = \prod_{j=1}^{k} \Pr(a_j | s_j)$$

where $s_j$ is the game state after $j-1$ actions, and $\Pr(a_j | s_j) = \Pr(a_j | I_{\rho(s_j)}(s_j))$ is the probability of action $a_j$ under the (acting) player's policy.

- $S$ = set of all possible game states (determinizations of hidden cards).

**Intuition:** Given what we've observed (the actions opponents took), which hidden-card assignments are most likely? Bayes' rule updates the prior (uniform) by weighting it by the likelihood (how probable is the observed action sequence under each hidden assignment?). States where opponents' actions make sense (high $p(H | s)$) get higher posterior weight.

**Example (from Figure 2, p. 5):** If peasant D sees peasant E play a high card (Ace, King), the posterior shifts away from "E has no Aces/Kings" and toward "E has other high cards but played this one." After observing multiple actions, the posterior can sharply concentrate (e.g., "E has exactly {AA, 8}") because those hands are consistent with the observed sequence.

**Practical issue (§4.3, p. 5):** Computing the full posterior over all $\binom{54}{17,17,20}$ possible hands is intractable. DeltaDou addresses this with a two-phase approximation (see §4.3 summary below).

---

### Equation 4: Two-phase inference approximation (practical Bayesian inference)

**OPTIONAL / ADVANCED** — This is the implementation detail.

**Phase 1 (filtering by neural network):**
- Generate a large number of random hands ($s_1, s_2, \ldots, s_N$).
- Train a 6-layer 1D-convolutional neural network $h_\phi(s)$ to score each hand. The scores reflect how likely a hand is, given the self-play experience.
- Keep only the top-$M$ hands by score.

**Phase 2 (Bayesian reweighting):**
- Compute the posterior $p(s_i | H)$ using the formula in Equation 3, but only over the $M$ retained hands.
- Use the network's policy $\pi$ to compute $\Pr(a_j | s_j)$ via a "temperature operator" (parameter $\tau$ that controls how sharp the policy distribution is).

**Why two phases?** 
- Phase 1 reduces from $\binom{54}{17,17,20} \approx 10^{15}$ possible hands to $M \approx 100$–$1000$ plausible hands.
- Phase 2 reweights these using the true Bayesian formula, giving a proper posterior.
- The temperature operator makes the inference robust: a temperature $\tau > 1$ softens the policy (useful if opponent plays suboptimally), while $\tau < 1$ sharpens it.

**The paper does not specify** $M$, the temperature schedule, or the training objective for $h_\phi$. [Interpretation: $M$ is likely ~100–1000 based on memory constraints; the temperature likely starts high (exploration) and cools during self-play; $h_\phi$ is likely trained as a classifier on self-play game logs.]

---

## 6. Loss Function

**Loss (from §4.2, p. 5):**
$$\ell = C_1 (z - v)^2 - \pi^* \log p + C_2 || \theta ||_2$$

**Three components:**

1. **Value loss: $C_1 (z - v)^2$**
   - Target $z$: The paper does not specify clearly. Most likely the **game outcome** (win=+1, loss=-1) since DeltaDou plays games to termination. [Interpretation: Unlike TD-learning, which bootstraps from a future value estimate, DeltaDou likely uses Monte Carlo returns (final outcome of the game), consistent with AlphaZero.]
   - Predicted value $v$: network output (scalar, normalized to $[-1, 1]$).
   - Scaling constant $C_1$: rescales value loss to account for normalization and to balance against the policy loss (which is typically larger in magnitude).

2. **Policy loss: $-\pi^* \log p$**
   - Target policy $\pi^*$: derived from MCTS visit counts at the root node: $\pi^*(a) \propto N(root, a)$. This encodes which actions the search explored most.
   - Predicted policy $p$: softmax over legal actions from the network.
   - Loss: cross-entropy, standard for policy distillation.
   - Why MCTS visit counts? After FPMCTS runs, the visit counts reflect which actions are most promising (high Q-values), so they form a better target than the raw network policy. This is a key idea in AlphaGo/AlphaZero: use search as a policy teacher.

3. **Regularization: $C_2 || \theta ||_2$**
   - L2 penalty on weights, preventing overfitting.
   - $C_2$ is a hyperparameter (the paper does not specify its value).

**Where does the training target come from?**
- **Value target $z$:** Game outcome at termination.
- **Policy target $\pi^*$:** MCTS visit distribution (computed during search).
- Both come from self-play games: play the game using FPMCTS + Bayesian inference, store the visit distribution and outcome, then update the network.

**Bootstrapping?** No. Unlike TD-learning (which bootstraps from a future value estimate $V(s')$), DeltaDou uses the terminal outcome $z$, which is a Monte Carlo return. This is more sample-efficient for episodic games (complete games per episode) but requires games to terminate.

**Multiple losses?** No. A single unified loss combining value and policy, plus L2 regularization.

**Why this choice?** This is the AlphaZero recipe: supervised learning on both the value and policy from MCTS. It is simpler than TD-learning (no bootstrapping, no target network) and works well when episodes are short and terminal outcomes are clear (as in board games and card games).

---

## 7. Self-Play

**Setup:**

- **Who plays whom:** Each game involves three agents: one Landlord, two Peasants. All three are AI (no human involvement during self-play). In each episode of 8,000 games, the role assignment is randomized or cycled so each network plays each role.
  
- **Simultaneous or sequential updates?** Sequential. After each episode (8,000 games), the experience is accumulated in a replay buffer, and the networks are updated once via gradient descent. The updated networks are then used for the next episode. This is episodic learning, not simultaneous online updating.

- **Parameter sharing?** No. Each role (Landlord C, Peasant D, Peasant E) has its own network: $f_\theta^C$, $f_\theta^D$, $f_\theta^E$. This is necessary because the roles have different objectives and information states (e.g., only the Landlord sees the extra 3 cards; peasants must cooperate but without communication).

- **Old checkpoints?** The paper does not mention keeping a pool of old policy versions. Likely, only the latest networks are used during self-play. This is simpler than AlphaZero's checkpoint pool but may lead to non-stationarity issues (the environment changes as the opponent policies change).

- **Exploration:** 
  - **During search:** p-UCT exploration bonus, guided by network priors.
  - **During action selection:** Temperature-based sampling. High temperature (early training) → more random action selection, reducing overfitting. Low temperature (late training) → greedy selection. (§5.1, p. 6.)
  - **Randomized deals:** "We use randomly generated deck in each game for self-play." (§4.4, p. 6.) This ensures diverse game states.

- **Non-stationarity and handling:** The paper does not explicitly address non-stationarity (the fact that opponent policies change over time, making the learning target non-stationary). The approach is implicit: after each episode, re-train the networks on the latest batch of experience, effectively adapting to the new policies. This is a form of "curriculum" via self-play non-stationarity—as the agent improves, it faces stronger opponents, which drives further learning.

**Bootstrapping the self-play:** To avoid learning from random play initially, DeltaDou pre-trains on heuristic data (§5.1, p. 6): "We used a hand-coded heuristic algorithm to bootstrap the training process. In the first phase, 200,000 games were self-played by the heuristic algorithm, then the game results were used to generate the initial policy-value network under supervised learning." This is crucial: without a good starting policy, random play would take much longer to find any useful signal.

---

## 8. Imperfect Information

### Two-column table: What is observed vs. what is hidden

| Aspect | Actually observed by the agent | Hidden |
|--------|--------------------------------|--------|
| **Own cards** | All of the agent's current hand (e.g., "LJJ555443" for peasant D). | — |
| **Opponents' cards** | None of the specific cards in the opponent's hand. Only the **count** of cards remaining. | Exact composition of opponent hands. |
| **Played cards** | All cards played so far (public information). | — |
| **Action history** | All moves and passes by all players: $(a_0, a_1, \ldots, a_j)$. | — |
| **Bids (bidding phase)** | The paper excludes the bidding phase (§3.1, p. 4), so this is not modeled. | — |

### How hidden information is handled

**1. Information sets and Extensive-Form Games (§3.2, p. 4):**
The paper models Dou Dizhu as an extensive-form game. Each player $i$ has an **information state** $u^i = I^i(s)$, which groups all game states $s$ that look identical to player $i$ (same own cards, same action history, same card counts for opponents). An information state $u^i$ is an **information set**—a set of indistinguishable states.

For example, if peasant D has cards {L, J, J, 5, 5, 5, 4, 4, 3} and has seen moves $[play 2, play A, play KK, ...]$, then D's information state lumps together all game states consistent with this: any distribution of the remaining cards between opponents C and E is considered the same "information state" until D observes more moves.

**2. Belief state (not explicit, but implicit):**
The agent maintains an implicit belief distribution over the hidden cards—which hidden assignments are most likely, given the action history. This is formalized via the Bayesian posterior $p(s | H)$ (Equation 3).

**3. Bayesian card inference (§4.3, p. 5):**

**Phase 1 (filtering):**
- Generate $N$ random possible hands for opponents (e.g., uniformly sample a deal of the remaining 17+17 cards to opponents C and E, given D's hand).
- Filter by a learned network $h_\phi(s)$ to keep only plausible hands.
- Result: a set $\{s_1, s_2, \ldots, s_M\}$ of $M$ plausible determinizations.

**Phase 2 (Bayesian reweighting):**
- For each retained determinization $s_i$, compute the posterior weight:
$$w_i = p(s_i | H) = \frac{\prod_{j} \Pr(a_j | s_j)}{\sum_{s'} \prod_{j} \Pr(a_j | s'_j)}$$
where $\Pr(a_j | s_j)$ is the probability of observed action $a_j$ under the opponent's learned policy at state $s_j$.

- Use temperature operator to soften the policy: $\Pr(a_j | s_j, \tau) = \frac{\exp(\log \pi(a_j|s_j) / \tau)}{\sum_{a'} \exp(\log \pi(a'|s_j) / \tau)}$. Higher temperature → flatter distribution (robust to policy misspecification). Lower temperature → sharper (more confident).

**4. Use in MCTS:**
- At the root of the FPMCTS search, sample one determinization $s$ from the posterior $p(\cdot | H)$ (line 3 of Algorithm 1).
- During search, treat this $s$ as the true state for simulation purposes (even though the real state is unknown).
- Opponent moves are then sampled from their learned policy at the inferred state.

**5. Card history (action history):**
- The information state $u^i$ includes all actions taken so far: $(a_0, a_1, \ldots, a_j)$.
- From this history, DeltaDou can infer what cards have been played (since playing a card removes it from the deck), and thus what cards remain.

**6. Explicit distributions over hidden cards:**
- The Bayesian inference gives a distribution $p(s | H)$ over possible hands for each opponent.
- Figure 2 (p. 5) shows an example: after observing certain actions by peasant E, the distribution over E's possible hands is updated from "uniform over 17-card hands" to "concentrated on {AA, 8} with 93% probability."

### Why this approach?

Traditional MCTS for imperfect-information games (e.g., ISMCTS) builds a single search tree over information sets and mixes determinizations, leading to **strategy fusion** (inconsistent play) and **information leaking** (incorrect belief updates). 

DeltaDou sidesteps this by:
1. **Sampling one determinization** at the root (via Bayesian inference), making it explicit which hidden cards are assumed.
2. **Using a separate search tree per player**, ensuring the tree is consistent with that player's information state.
3. **Sampling opponent moves from their policy** (not best-responding), which avoids need to recursively solve for equilibria.

This is simpler and more efficient than prior MCTS approaches.

---

## 9. Evaluation   ← **CRITICAL SECTION for CS230**

### Baselines, comparison methods, and metrics

**Baselines (§5.2, p. 6–7):**
1. **RHCP**: An open-source heuristics-based algorithm (hand-crafted evaluation function, no learning).
2. **XDou**: A strong in-house heuristic AI combining "sophisticated heuristics and a PIMC-based endgame solver" (p. 7). This is the authors' own reference implementation.
3. **Human experts**: Tournament winners of online Dou Dizhu competitions.

**Policy-only ablation (§5.2, p. 7):** A variant of DeltaDou that extracts the policy network greedily (no MCTS search) to isolate the contribution of search vs. the learned policy.

**How policies were compared (§5.2, §5.3, p. 6–7):**

- **Tournament format (Duplicate Bridge-style):** "Our approach is inspired by teams tournament for duplicate bridge. When two teams A and B are compared with each other, the same hand is played twice: First, Team A plays as the landlord and Team B plays as the peasants, and then Team A plays as the peasants and Team B as the landlord." This controls for deal/role variance. Reported in the paper as (Table 1, p. 7):
  - **D-D** (DeltaDou landlord vs Human peasants).
  - **D-H** (DeltaDou peasants vs Human landlord).
  - **H-D** (Human landlord vs DeltaDou peasants).

- **Metrics:**
  - **Win Rate (WP):** Fraction of games won by a team. A team is considered stronger if WP > 50%.
  - **Average Difference in Points (ADP):** Average point difference per game (accounting for bombs doubling scores). Winner gets +1, loser gets -1 per game; bombs double.
  - **Significance threshold:** "When a team won 5 more games out of 100 games against the opponent, we treat them as teams at different skill levels." (§5.3, p. 7.) This is a crude rule of thumb; no formal statistical test is stated.

### Evaluation results

**Self-play progress (Figure 4, p. 6):**
- **Episode 0 (initial heuristic):** WP ≈ 0.3–0.5 (random-ish).
- **Episode 3:** Beat the initial heuristic AI.
- **Episode 10:** Surpassed XDou (the strong heuristic baseline).
- **Episode 80:** Reached final evaluation version (640,000 games total).

**vs. Heuristic AI (Table 1, p. 7):**
| Comparison | WP | ADP |
|------------|----|----|
| XDou vs RHCP | 79% | +0.88 |
| DeltaDou vs RHCP | 75% | +0.72 |
| DeltaDou vs XDou | 59% | +0.28 |

- DeltaDou achieves **59% WP and +0.28 ADP against XDou**, indicating a statistically significant edge. (59% is > 50%, suggesting genuine superiority, though the 59% figure is far from overwhelming; a larger sample size or formal test would strengthen this claim.)

**Ablation: Policy-only vs. DeltaDou (Table 2, p. 7):**
| Setting | vs RHCP WP | vs RHCP ADP | vs XDou WP | vs XDou ADP |
|---------|------------|-------------|------------|-------------|
| Policy | 58% | +0.19 | 49% | -0.08 |
| DeltaDou | 75% | +0.72 | 59% | +0.28 |

- **Policy-only is weak:** 49% WP vs XDou (essentially tied). The network, without search, is comparable to XDou but not better.
- **MCTS+Inference boost:** DeltaDou jumps to 59% WP, showing that search and inference are critical. Separate ablation: "The agent with inference got WP 54.5% and ADP +0.145 in the heads-up game against the agent without inference" (§5.2, p. 7)—**inference alone adds +5.5% WP**, which is substantial.

**vs. Human experts (§5.3, Table 3, p. 7):**
The paper's Table 3(a) reports the **Landlord's** WP/ADP on the same 100 deals for three seatings ("X-Y" = X plays Landlord, Y plays Peasants; D = DeltaDou, H = human team). Values verified from the PDF image (p. 6):

| Seating (Landlord-Peasants) | Landlord WP | Landlord ADP | Peasants' WP (= 1 − Landlord WP) |
|---|---|---|---|
| D-D (DeltaDou vs itself) | 39% | −0.44 | 61% |
| D-H (DeltaDou Landlord vs human Peasants) | 33% | −0.61 | 67% |
| H-D (human Landlord vs DeltaDou Peasants) | 36% | −0.47 | 64% |
| **Table 3(b): DeltaDou vs Human overall** | **WP 48.5%** | **ADP −0.07** | (= average of 33% as Landlord and 64% as Peasants) |

How to read the caption's comparisons (each holds the *opponent* fixed):
- **DeltaDou is stronger as Landlord:** against the same DeltaDou Peasants, DeltaDou-as-Landlord wins 39% (D-D) vs human-as-Landlord 36% (H-D).
- **DeltaDou is weaker as Peasants:** against the same DeltaDou Landlord, DeltaDou Peasants win 61% (D-D) vs human Peasants 67% (D-H).
- **Overall:** 48.5% WP for DeltaDou, -0.07 ADP. This is **not statistically superior to human experts**—very close (within the ±5-game threshold).
- **Qualitative:** "DeltaDou won 3 less games out of the 100 games to the human team, and has achieved comparable performance as our human experts" (p. 7–8). "A drawback of DeltaDou is that sometimes it misses a straight win that is obvious for the human players. On the other side, some nice play by DeltaDou are very impressive."

**Number of games and rigor:**
- **AI vs AI:** 100 games per matchup (small; confidence intervals would be ~±10% WP).
- **AI vs Human:** 100 games on landlord, 100 on peasants (200 total games; small).
- **No formal statistical tests** (e.g., chi-squared, t-test, confidence intervals, multiple comparison correction).
- **No mention of multiple seeds / cross-validation** for the human experiments.

### Critical evaluation

**Strengths:**
1. **First RL system to reach human-expert level in multi-player imperfect-information games.** This is a significant achievement and opens a new domain for RL.
2. **Comprehensive comparisons:** Includes heuristic baselines, ablations (policy-only, inference ablation), and human experiments.
3. **Duplicate bridge tournament format:** Using the same deals twice (Landlord/Peasant swapped) reduces variance from random deals, which is smart.
4. **Clear progression:** Figure 4 shows steady self-play improvement, validating the training procedure.
5. **Honest about limitations:** "DeltaDou is weaker than human as the peasants" and "sometimes misses a straight win."

**Weaknesses:**
1. **Small sample sizes:** 100 games per AI vs AI matchup is small (~±10% confidence interval on WP). Human experiments are also 100 games each, making the "comparable to human" claim marginal.
2. **No formal statistical tests:** No χ², t-test, or confidence intervals. The "5 games = different skill levels" rule is ad-hoc.
3. **Human evaluation constraints:** Limited to "several winners of online Dou Dizhu tournaments" (n=1 team?). No mention of inter-rater agreement or whether the human team had studied DeltaDou's play style.
4. **Seat/deal mirroring:** The duplicate bridge format helps, but not all deal dependencies are controlled. The paper does not report separate results by deal/random seed.
5. **Weak vs XDou baseline:** Only 59% WP over 100 games. For a learning-based system after 2 months of training (640,000 games), one would hope for a larger margin. The ablation shows that the policy network alone is nearly as good as XDou (49% WP)—most of the gain comes from search and inference, not the learned policy.
6. **No comparison to other RL methods:** No direct comparison to NFSP or other deep RL for imperfect-information games (though NFSP may not be applicable to 3-player Dou Dizhu).
7. **Missing analysis:** No breakdown by game phase (bidding, early/mid/late endgame). Human players likely dominate different phases; DeltaDou's weakness as peasants suggests a cooperation/signaling gap.

**What I would improve for a CS230 project:**
1. **Larger sample sizes:** Run ≥500 games per matchup to reduce variance; report 95% confidence intervals.
2. **Formal significance tests:** Use binomial test or chi-squared to claim statistically significant differences at p<0.05.
3. **Ablations:** Separate ablations for (a) search depth, (b) number of inference samples, (c) value network vs. random value, (d) policy network vs. uniform policy.
4. **Multi-seed evaluation:** Train 3–5 independent seeds; report mean ± std of WP, test for statistical significance across seeds.
5. **Breakdown by role:** Analyze performance as Landlord, Peasant-D, Peasant-E separately; look for asymmetries.
6. **Phase-specific analysis:** Measure win rate in early-game (first 10 moves), mid-game, endgame (final 5 cards). Identify which phase is weak.
7. **Human ablation:** Test human vs (a) policy-only, (b) search-only (random value), (c) weak inference, (d) strong inference, to isolate human-AI gaps per component.

---

## 10. Experiments and Ablations

### List of major experiments

1. **Self-play training (Figure 3, Figure 4, §5.1, p. 6):**
   - **What changed:** Episode number (0 to 80), network weights via gradient descent.
   - **Held constant:** Game rules, FPMCTS parameters (400 playouts, simulate_num), evaluation set (100 fixed games).
   - **Metric:** WP and ADP vs XDou; ADP by position (C, D, E).
   - **Result:** WP improves from ~0.3 (episode 0) to ~0.55 (episode 80). ADP improves from ~-0.3 to ~+0.3.
   - **Authors' conclusion:** "self-play can significantly improve the performance" (p. 6). The learning curve is smooth, validating the training procedure.
   - **Critical note:** Figure 4 shows learning vs a fixed heuristic baseline (XDou). However, XDou itself is static, so the environment is non-stationary only due to changes in DeltaDou. A more informative ablation would hold the network constant and vary opponents. Figure 3 (WP by position) shows interesting dynamics: Landlord (C) reaches ~50% WP early (easier role for RL?), Peasant-E is weak initially (cooperation takes time?), then Peasant-D learns to "let E win" (Figure 3 caption mentions "D learns how to let E wins by send him the hand he wants"), demonstrating learned cooperation.

2. **Policy vs. MCTS + Inference (Table 2, §5.2, p. 7, "DeltaDou vs Policy against baseline"):**
   - **What changed:** Search (on/off), inference (on/off).
   - **Held constant:** Network architecture, same evaluation set.
   - **Metric:** WP, ADP vs RHCP and XDou.
   - **Result:** 
     - Policy (network only): 58% vs RHCP, 49% vs XDou.
     - DeltaDou (policy+MCTS+inference): 75% vs RHCP, 59% vs XDou.
   - **Authors' conclusion:** "The policy algorithm achieves comparable performance as XDou, and is still left far behind by DeltaDou, which used both looking forward search and inference" (p. 7).
   - **Critical note:** The policy-only baseline is weak, suggesting that the network itself is not the bottleneck. Most of the gain comes from search and inference, not the learned policy. This raises the question: how good is the learned policy compared to a simple heuristic? A finer ablation would separate search gain from inference gain (the paper mentions "54.5% WP" for inference-only vs no-inference, but this is heads-up, not vs XDou).

   **Ablation: Inference ablation (§5.2, p. 7):**
   - **What changed:** Inference (on/off).
   - **Metric:** WP, ADP in a head-to-head DeltaDou-vs-DeltaDou game, one with inference, one without.
   - **Result:** WP 54.5%, ADP +0.145 in favor of agent with inference.
   - **Conclusion:** Inference is valuable (~+4.5% WP over no inference).
   - **Critical note:** This is a heads-up comparison (not against XDou or humans), so the effect size is not directly comparable to other experiments. However, it isolates the inference contribution nicely.

3. **Human expert evaluation (§5.3, Table 3, p. 7):**
   - **What changed:** DeltaDou vs. human tournament winners.
   - **Held constant:** Game rules, 100 games per role.
   - **Metric:** WP, ADP.
   - **Result:** 48.5% WP overall, -0.07 ADP (not significantly different from human).
   - **Authors' conclusion:** "DeltaDou has achieved comparable performance as our human experts" (p. 8).
   - **Critical note:** The margin is very small (48.5% is essentially a tie); more games would be needed to claim statistical superiority or inferiority. The authors' own criterion ("5 games = different skill level") suggests this is a tie.

### Experiments reproducible at small scale for CS230

| Experiment | Original scale | Scaled-down version | Feasibility |
|------------|-----------------|---------------------|-------------|
| **Self-play training** | 80 episodes × 8,000 games = 640,000 games | 10 episodes × 100 games = 1,000 games on a single GPU. Expect ~1 hour. | **Easy to Moderate**. Use 1 GPU (RTX 2080 or cloud VM), skip the heuristic pre-training (train from scratch with random initial policy), and evaluate every episode against a simple heuristic. |
| **Policy-only ablation** | Full network, trained for 80 episodes. | Same network, but extract greedily at test time (no MCTS). | **Easy**. Just modify the test-time code to skip search. |
| **Inference ablation** | Full FPMCTS + inference. | Disable Phase 2 (Bayesian reweighting); just use Phase 1 (neural network filtering). Or disable both and sample uniformly. | **Moderate**. Requires implementing the inference pipeline; the code change is small (one line to skip reweighting). |
| **Search depth ablation** | 400 playouts per move. | Reduce to 50, 100, 200 playouts; measure WP vs a fixed baseline. | **Easy**. Modify a hyperparameter. |
| **AI vs AI tournament** | 100 games vs XDou per episode. | 10–20 games per episode vs a simple heuristic baseline. | **Easy**. Parallel self-play should be fast enough. |
| **AI vs Human-simulator** | Play against 1 team of humans (1 run). | Play against a simulated "human strategy" learned from a dataset of human games (if available). Or use a fixed heuristic as a proxy. | **Moderate to Hard**, depending on data availability. If no human data, use a strong heuristic (e.g., hand-evaluation function) as the baseline. |

---

## 11. Compute and Engineering

**Hardware:**
- **Training:** 68 CPUs (parallel game generation) + unspecified GPU (network inference and training). The paper mentions "modest computational power" but does not specify GPU count or type. [Interpretation: Likely 1–2 GPUs, given the 2-month training time; modern systems could do better.]
- **Evaluation:** "single 8-core computer with the average time for a move of roughly 5 to 8 seconds" (§5.3, p. 7). This is for human evaluation, not training.

**Training duration:**
- **Total:** ~2 months on 68 CPUs (§5.2, p. 7).
- **Breakdown by episode:**
  - Pre-training: 200,000 heuristic games, then supervised learning on those games.
  - Self-play: 80 episodes × 8,000 games/episode = 640,000 games.
  - ~640,000 games ÷ (68 CPUs × 2 months) ≈ 150 games/CPU/day. This is a rough estimate; actual throughput depends on game length and parallelization efficiency.

**Sample efficiency:**
- **Total samples:** 640,000 games. Average Dou Dizhu game has ~30 moves/player, so ~90 moves/game total across 3 players, or ~30 moves per training step per player. Total unique states trained on: ~640,000 games × 30 unique states per game ≈ 19.2 million unique (state, policy, value) tuples. **The paper does not specify the replay buffer size or sampling strategy.**

**Parallel actors and engineering:**
- 68 CPUs for parallel game generation ("parallel actors").
- Network training: likely on GPU (unspecified).
- **Distributed training?** The paper does not mention distributed training (e.g., parameter servers, asynchronous updates). Likely a simple single-machine setup with CPU workers generating games and GPU training the network sequentially after each episode.

**Key implementation tricks (inferred from the paper, not explicitly stated):**
1. **Action abstraction:** Reduce branching factor by decoding rank+category separately, then apply heuristic rules for kicker selection. This is critical for search efficiency.
2. **Kicker network:** A separate small network (architecture unspecified) to predict the best kicker. [Interpretation: This likely avoids the combinatorial explosion of (rank, category, kicker) tuples.]
3. **Bayesian inference with neural network filtering (Phase 1):** Drastically reduces the number of determinizations to consider (from ~10^15 to ~100–1000), making search tractable.
4. **Deterministic game simulation:** Once a determinization is sampled, the game is deterministic (no rollouts needed), so simulation is fast.

**Why MCTS is expensive:**
1. **Tree expansion:** Each simulation expands the tree by 2 levels (player move + 2 opponent moves), creating many nodes. With 400 playouts and ~20 moves per game, the tree can grow large.
2. **Neural network inference:** Every node evaluation and policy sampling requires a forward pass through the network (3 networks for the 3 players). With 400 playouts × multiple MCTS calls per game, this adds up.
3. **Bayesian inference:** Phase 1 requires scoring ~1000 hands with the filtering network; Phase 2 requires computing likelihoods for retained hands. This is done at every decision node (every turn, ~30 turns per game).
4. **Determinization sampling:** Each simulation requires sampling a possible hidden-card assignment, which involves permutation sampling and filtering.

Combined, **DeltaDou is computationally expensive**, which is why training takes 2 months.

### Reproduction difficulty: **Hard** for full reproduction; **Moderate** for scaled-down version.

**Full reproduction (640,000 games, 80 episodes):**
- **Hardware cost:** ~$500–$1000 for cloud GPU (AWS, GCP, Azure) to run 2 months, or own 68 CPUs + 2 GPUs.
- **Code complexity:** Implementing FPMCTS, Bayesian inference, policy-value networks, and self-play is non-trivial (~2000–3000 lines of code). Existing codebases (DouZero, RLCard) can be leveraged, but FPMCTS and Bayesian inference are custom.
- **Debugging:** Imperfect-information games are hard to debug; policy convergence can be slow, and self-play can get stuck in local minima.
- **Verdict:** Hard for a CS230 student without substantial coding experience. Takes weeks to months of implementation and tuning.

**Scaled-down reproduction (1,000–10,000 games, 10 episodes):**
- **Hardware cost:** Minimal; a single GPU or even CPU (RTX 2060 or cloud VM $10–$50 rental).
- **Code complexity:** Reuse RLCard's game engine and network infrastructure; implement a lightweight FPMCTS and skip the complex Bayesian inference (use simple uniform sampling). Code: ~500–1000 lines.
- **Debugging:** Faster iteration; can see results in hours to days.
- **What you lose:** Convergence to expert-level (likely plateau at intermediate strength), and detailed understanding of Bayesian inference and policy-value networks.
- **Verdict:** Moderate for a CS230 student. Feasible in 2–4 weeks with focus on core algorithms and simplified baselines.

**What a scaled-down reproduction would look like:**
- Use RLCard's Dou Dizhu environment.
- Implement FPMCTS with policy sampling (§4.1).
- Skip Bayesian inference; sample uniform random determinizations or use a simple heuristic (e.g., uniform over hands consistent with undealt cards).
- Train a small policy-value network (e.g., 3–5 residual blocks instead of 10).
- Pre-train on ~5,000 heuristic games (instead of 200,000).
- Run 10 self-play episodes of 100 games each (instead of 80 episodes of 8,000 games).
- Evaluate every episode on a fixed set of 20 games vs XDou (or another heuristic baseline).
- Expect: policy converges to 40–50% WP vs heuristic after ~3–5 days of training on 1 GPU, compared to the paper's 59% WP after 2 months.

---

## 12. Limitations

### A. Explicitly mentioned by the authors

1. **Heuristic initialization (§6, p. 8):** "In future work, we would like to remove the heuristic-based neural network for bootstrapping so that the training process can start from scratch without any human knowledge." The authors acknowledge that pre-training on heuristic games (200,000 games) introduces human knowledge into the training pipeline, potentially biasing the learned policy. Learning from scratch (random initialization) would be more general but likely requires more training data/time.

2. **Kicker abstraction (implicit in §4.2, p. 5):** The kicker network and heuristic rules for kicker selection are ad-hoc. A bad kicker selection can lose the game (e.g., breaking a Chain of Solo). The paper mentions this is a "trade-off between precision and speed" but does not discuss the failure modes.

3. **Weakness as Peasants (§5.3, p. 8):** "A drawback of DeltaDou is that sometimes it misses a straight win that is obvious for the human players." The authors note DeltaDou is weaker than the human team as Peasants: against the same DeltaDou Landlord, DeltaDou Peasants win 61% vs human Peasants 67% (Table 3, p. 6). [Interpretation] This may reflect weaker Peasant cooperation — but note the opposite pattern against XDou, where "the advantage of DeltaDou against XDou is mainly from the peasants side" (Fig. 4 caption, p. 6). With only 100 deals, the 6-point human-vs-DeltaDou Peasant gap is well within noise (a 95% interval on a win rate from 100 games is roughly ±10 points), so neither claim is statistically established.

4. **Computational cost (§5.2, p. 7):** "As neural network computation is required by both looking forward search and inference, DeltaDou is a quite heavy method." Two months of training on 68 CPUs is expensive. This limits reproducibility and practical deployment.

5. **Bidding phase not modeled (§3.1, p. 4):** "The bidding phase, which is not considered in this paper, designates one of the players as the landlord..." DeltaDou only learns the card-play phase, not the bidding phase (where players bid for the Landlord role based on card strength). This is a significant simplification; in real Dou Dizhu, bidding affects role assignment and thus strategy.

### B. Inferred from the methodology (with reasoning)

1. **Non-stationarity and convergence:** Self-play faces non-stationarity: as the agent improves, the opponents (the previous version of itself) become weaker, changing the learning signal. The paper does not discuss convergence guarantees or whether the learned policy is an equilibrium. [Interpretation: Likely the policy converges to a local Nash equilibrium within the class of policies representable by the network, but global optimality is not guaranteed.]

2. **Limited opponent diversity during training:** DeltaDou always plays against itself. If the self-play settles into a suboptimal strategy (e.g., a local minimum), it will not escape. Humans and expert systems (like XDou) may exploit weaknesses that the self-play does not discover. [Interpretation: The 59% WP vs XDou (not 70–80%) suggests this may be happening; DeltaDou has learned to beat a weak heuristic but not necessarily found strategies robust to adversarial opponents.]

3. **Bayesian inference approximation errors:** The two-phase inference (Phase 1 filtering, Phase 2 Bayesian reweighting) is an approximation. Errors in Phase 1 (keeping the wrong hands) or Phase 2 (miscalibrated likelihood) could lead to systematically wrong beliefs, which in turn cause poor search decisions. The paper does not analyze the approximation error.

4. **Value function generalization:** The value network is trained on self-play outcomes. If the distribution of game states in self-play is biased (e.g., more Landlord-winning states), the value network may not generalize well to human play with a different distribution. [Interpretation: This could explain the weak performance as Peasants, where DeltaDou may be less experienced.]

5. **Cooperation signal in Peasants:** Peasants must cooperate without communication. DeltaDou learns cooperation via self-play, but the learned signals may be implicit (e.g., "playing a low card means you have other high cards"). Humans may use different signals, causing miscommunication when DeltaDou plays with human Peasants. [Interpretation: The 61% loss rate (39% win rate) as Peasants suggests this is a real limitation.]

6. **Action abstraction ceiling:** By separating rank+category (main network) from kicker (kicker network), DeltaDou assumes the kicker choice is independent of the rank+category choice. This may be suboptimal; e.g., choosing a trio rank may depend on which kickers are available. [Interpretation: This is a design trade-off (speed vs. optimality); the kicker network is meant to mitigate but may not fully solve it.]

7. **Scalability to other games:** The Bayesian inference method is specific to Dou Dizhu (hand categories, kicker structure). Adapting to other imperfect-information games (Bridge, Skat) would require redesigning the inference algorithm. [Interpretation: The paper mentions this as future work (§6, p. 8), suggesting the current method is not modular.]

---

## 13. Future Work

### Explicitly mentioned by authors (with classification)

1. **Remove heuristic bootstrapping (§6, p. 8):** "In future work, we would like to remove the heuristic-based neural network for bootstrapping so that the training process can start from scratch without any human knowledge."
   - **Classification:** Directly usable as a CS230 project idea.
   - **Difficulty:** Moderate to Hard (requires longer training, careful initialization).
   - **Already addressed?** Partially in DouZero (2021), which trains from scratch without heuristics, but uses Monte Carlo instead of MCTS. Not directly addressed in DouZeroPlus (2022).

2. **Opponent modeling (§6, p. 8):** "We have started experimenting with opponent modeling, which may help the AI to reach super-human level strength by exploiting the weakness of opponents more and by better understanding its partner."
   - **Classification:** Directly usable as a CS230 project idea (if scaled down).
   - **Difficulty:** Moderate (requires learning separate opponent models for each opponent type).
   - **Already addressed?** Yes, in DouZeroPlus (2022), which adds opponent modeling and coach-guided learning. The paper claims "DouZero+ ranks top in the Botzone leaderboard among more than four hundred AI agents" (DouZeroPlus abstract).

3. **Adaptation to other games (§6, p. 8):** "Finally, we will try to adapt our framework to other imperfect information card games such as bridge or skat."
   - **Classification:** Possible but difficult for CS230 (requires domain knowledge of Bridge/Skat, and rewriting inference algorithm).
   - **Difficulty:** Hard (each game has different hand categories, rules, and information structure).
   - **Already addressed?** Skat is similar to Dou Dizhu (hand categories, trick-taking), but no known DeltaDou-style solution for Skat. Bridge is more complex (bidding, contract, tricks). Not addressed in the provided paper set.

### Implied future work [Interpretation]

1. **Computational efficiency:** Reducing the 2-month training time by 10–100×. This could enable:
   - More frequent retraining on new human game data (e.g., tournament recordings).
   - Easier academic reproduction.
   - Faster exploration of hyperparameters and architecture choices.
   - This is addressed implicitly in DouZero (trains in 10 days with 4 GPUs, vs 2 months with 68 CPUs in DeltaDou).

2. **Evaluation rigor:** Larger sample sizes, formal statistical tests, breakdown by phase/role.
   - Addressed in DouZero and DouZeroPlus via Botzone leaderboard (hundreds of games against diverse opponents).

3. **Cooperation learning:** Better mechanisms for Peasants to signal and coordinate without communication.
   - Addressed in DouZeroPlus via opponent modeling (learning which teammate has which cards).

4. **Bidding phase:** Integrate bidding (role assignment) into the learning loop.
   - Not addressed in the provided paper set. (DouZero and DouZeroPlus focus on card-play phase only.)

---

## 14. Connection to Other Papers

### Earlier papers DeltaDou builds on

**Within this set:**
- **NFSP (Heinrich & Silver, 2016, paper 05_NFSP_2016):** DeltaDou uses the core idea of Neural Fictitious Self-Play—separate networks for each player, each trained to best-respond to a slowly-updating aggregate of past policies. DeltaDou modifies this by (a) using MCTS instead of policy gradients, (b) adding Bayesian inference for hidden information, (c) applying it to multi-player games with asymmetric roles.
- **Deep CFR (Brown et al., ICML 2019, paper 06_Deep_CFR_ICML2019):** DeltaDou does not cite or build on Deep CFR (which is for two-player games and uses regret minimization). No direct connection.

**Outside this set (mentioned in paper):**
- **AlphaZero (Silver et al., 2018):** DeltaDou explicitly positions itself as "extending AlphaZero to multi-player imperfect-information games" (§4, title; p. 5). It borrows:
  - Self-play training loop.
  - Policy-value network trained on (MCTS policy, game outcome) pairs.
  - p-UCT exploration.
- **MCTS and ISMCTS (Cowling et al., 2012):** DeltaDou addresses the strategy-fusion and non-locality problems of ISMCTS by proposing FPMCTS.
- **Poker AI (Ponsen et al., 2010, on integrating opponent models with MCTS):** Similar idea of sampling opponent moves from a policy, but Ponsen et al. used traditional MCTS with rollouts, while DeltaDou uses value networks.

### Later papers that build on or critique DeltaDou

**DouZero (Zha et al., ICML 2021, paper 01_DouZero_ICML2021):**
- **What it changes:** 
  - Replaces MCTS with Monte Carlo sampling (no tree search, just sample trajectories from policy).
  - Removes Bayesian inference (no explicit hidden-hand inference; opponent cards are implicitly handled via sampling).
  - Removes heuristic kicker network (action encoding + mask instead).
  - Trains from scratch (no heuristic bootstrapping).
- **Why:** DouZero notes that DeltaDou's MCTS + Bayesian inference are "computationally expensive. It takes more than two months to train DeltaDou even when initializing the networks with supervised regression to heuristics" (p. 2). DouZero aims for a simpler, faster approach.
- **Result:** DouZero "outperforms DeltaDou in three days and ten days of training in terms of WP and ADP, respectively" (DouZero paper, p. 8) on 4 GPUs (vs 2 months on 68 CPUs for DeltaDou). DouZero achieves comparable or better performance with ~20× faster training.
- **New limitation:** DouZero removes explicit inference, potentially losing interpretability about opponent hand beliefs. But empirically, it performs better, suggesting that explicit inference is not necessary for strong play in Dou Dizhu.

**DouZeroPlus (Zhao et al., 2022, paper 02_DouZeroPlus_2022):**
- **What it adds to DouZero:**
  - Opponent modeling: Learn a separate model predicting opponent hand composition at each turn.
  - Coach network: A classifier that selects "valuable" games for training, skipping deterministic or low-value deals.
- **Why:** "Due to the large information space in this game, the training of the AI program for DouDizhu costs a lot of time. Considering the fact that the outcome of DouDizhu depends heavily on the initial cards of three players, quite a few games are of little value for learning" (p. 1–2). Also, "Opponent modeling aims to determine a likely probability distribution for the opponents' hidden cards" (p. 2), echoing DeltaDou's explicit inference but within the DouZero framework.
- **Result:** DouZeroPlus "ranks the first on the Botzone leaderboard among more than four hundred agents, including DouZero" (p. 2).
- **Relation to DeltaDou:** DouZeroPlus re-introduces opponent modeling (DeltaDou's explicit Bayesian inference concept) but in the context of a Monte Carlo sampler (DouZero's efficiency). This suggests that explicit opponent hand inference is valuable but can be combined with simpler algorithms than FPMCTS.

### Evolution chain

```
Poker research (Ponsen 2010)
    ↓ [limitation: only 2-player, uses rollouts]
AlphaZero (Silver 2018)
    ↓ [limitation: perfect information only]
NFSP (Heinrich & Silver 2016)
    ↓ [limitation: slow convergence, ISMCTS suffers from fusion/non-locality]
DeltaDou (2019)
    → [introduces FPMCTS + Bayesian inference; solves Dou Dizhu but expensive (2 months)]
    ↓ [limitation: computationally expensive, heuristic-dependent]
DouZero (2021)
    → [replaces MCTS with Monte Carlo sampling; no explicit inference; faster (10 days)]
    → [solves Dou Dizhu ~20× faster, simpler approach]
    ↓ [limitation: no explicit opponent modeling; unclear hand beliefs]
DouZeroPlus (2022)
    → [adds opponent modeling back; coach network for sampling]
    → [best performance; top of Botzone leaderboard]
```

**[Interpretation] Evolution:**
1. DeltaDou tackled multi-player imperfect-information games by combining ideas from AlphaZero (self-play), NFSP (separate networks), MCTS (search), and Bayesian inference (hand inference). It proved this works but was expensive.
2. DouZero simplified by removing MCTS and explicit inference, using plain Monte Carlo sampling instead. Counterintuitively, this was faster and better, suggesting that explicit tree search and inference are not necessary for strong play in Dou Dizhu.
3. DouZeroPlus refined DouZero by re-introducing opponent modeling and intelligent game selection, achieving state-of-the-art performance.

**[Interpretation] Why DouZero is better than DeltaDou:**
- **Simpler:** No FPMCTS to implement; just sample trajectories from policies.
- **Faster:** 10 days vs 2 months (20× speedup). The paper attributes this to (a) no MCTS tree overhead, (b) no Bayesian inference, (c) better parallelization on GPU.
- **Comparable performance:** Both achieve human-expert level or better, but DouZero is more sample-efficient.
- **Lesson:** For large action spaces and complex hidden information, Monte Carlo sampling can outperform explicit tree search + inference. This challenges the intuition that search is always better.

---

## 15. Project Ideas for Me

### Idea 1: Scaled-down FPMCTS with ablations

**Research question:** Can a simpler FPMCTS variant (fewer playouts, no Bayesian inference, smaller network) achieve acceptable performance on Dou Dizhu within a CS230 timeline?

**Baseline:** XDou (from the paper) or a simple heuristic AI.

**Modification:** 
- Implement FPMCTS with 50–100 playouts (vs paper's 400).
- Replace Bayesian inference with uniform random determinization or simple heuristic (e.g., "uniform over hands consistent with dealt cards").
- Train a small policy-value network (2–3 residual blocks instead of 10).
- Pre-train on ~2,000 heuristic games (instead of 200,000).

**Hypothesis:** Simplified FPMCTS will reach 40–50% WP vs a strong heuristic, validating that FPMCTS is the key technique (not Bayesian inference or large networks).

**Experiment:** 
- Self-play for 5–10 episodes (500–1,000 games total).
- Evaluate every episode on 20–50 games vs XDou.
- Ablate: (a) search depth (50 vs 100 vs 200 playouts), (b) network size (2 vs 5 vs 10 blocks), (c) pre-training data (0 vs 1,000 vs 5,000 heuristic games).

**Metrics:** WP, ADP, training time (should be <1 week on 1 GPU).

**Difficulty:** Moderate (FPMCTS implementation is non-trivial; Bayesian inference code can be skipped).

**Compute:** Low (1 GPU, ~1 week).

**Novelty:** This is not novel compared to the paper, but it is a credible **reproduction study** demonstrating the core algorithmic idea. For a CS230 student early in RL, this is good practice in implementation and evaluation. Novelty could come from the ablations or scaling analysis.

---

### Idea 2: Opponent modeling for Peasant cooperation

**Research question:** Can explicit opponent hand inference (predicting teammate's cards) improve Peasant win rate in Dou Dizhu?

**Baseline:** Standard DeltaDou (as described in paper) or DouZero (from later paper).

**Modification:**
- Train an additional network to predict teammate's hand composition at each turn (similar to DouZeroPlus).
- During MCTS search, use the predicted teammate hand to guide Peasant actions (e.g., "if teammate likely has high cards, play low cards to let them win").
- Compare two variants:
  - (a) Blind Peasant (no hand prediction).
  - (b) Informed Peasant (with hand prediction feedback).

**Hypothesis:** Informed Peasants will have higher win rate when paired together, due to better cooperation. Estimated +5–10% WP.

**Experiment:**
- Train hand-prediction network on self-play game logs (supervised learning: predict hand composition from state + action history).
- Integrate into MCTS: compute hand predictions for both opponents at each node, use to weight move selection.
- Play tournaments:
  - Blind vs Blind Peasants.
  - Informed vs Informed Peasants.
  - Blind vs Informed (control).
- Measure WP gains and ADP.

**Metrics:** WP (Peasant win rate when both are Informed), ADP, training time.

**Difficulty:** Moderate (requires implementing hand-prediction network + integrating into search).

**Compute:** Medium (2 GPUs, ~2 weeks).

**Novelty:** Moderate. DouZeroPlus already has opponent modeling, so this is incremental, but applying it specifically to Peasant cooperation is a reasonable research angle. Novelty lies in the integration and ablation: does hand prediction help if used only for Peasant-Peasant cooperation (not against Landlord)?

---

### Idea 3: Action abstraction comparison (rank+category vs full moves)

**Research question:** How much does action abstraction (separating rank+category from kicker) hurt performance in Dou Dizhu, and can a learned kicker network recover the loss?

**Baseline:** Full-action network (all ~10,000 moves encoded separately) trained with a strong action embedding (e.g., learned embedding of card combinations).

**Modifications:**
- (a) **Rank+category only** (paper's approach): Network outputs rank+category; kicker is selected by heuristic rule. Fast but potentially suboptimal.
- (b) **Rank+category+learned kicker**: Network outputs rank+category; a separate kicker network predicts the best kicker. More flexible than heuristic, but requires training two networks.
- (c) **Full-action network** (baseline): All ~10,000 moves encoded and output by a single network. Slow but potentially optimal.

**Hypothesis:** Abstraction (a) is fast but loses ~5–10% WP; learned kicker (b) recovers most of the loss; full-action (c) is slower but gains ~5% WP over (a).

**Experiment:**
- Train all three variants with the same architecture and hyperparameters, varying only the action encoding + network.
- Measure: (a) training time per 1,000 games, (b) WP vs a heuristic baseline, (c) inference time per move.
- Ablate: (a) no abstraction (full moves) vs (b) rank+category only (no kicker network) vs (c) rank+category+learned kicker.

**Metrics:** WP, inference time, training time, ADP.

**Difficulty:** Moderate (requires implementing and training three variants; careful experimental setup).

**Compute:** Medium (1–2 GPUs, ~2 weeks).

**Novelty:** Good. This directly addresses a design choice in DeltaDou (action abstraction) and proposes an improvement (learned kicker network). The evaluation is straightforward: if the learned kicker is faster than full actions and nearly as accurate, it's a win.

---

### Idea 4: Transfer learning from human games to DeltaDou

**Research question:** Can pretraining on human expert game logs (supervised learning) accelerate self-play in Dou Dizhu and improve final performance?

**Baseline:** DeltaDou trained from scratch (or pre-trained on heuristic games only, as in the paper).

**Modification:**
- Collect or use a dataset of human expert games (e.g., tournament records from Botzone or a Dou Dizhu platform).
- Pretraining phase: Supervised learning on human games for the policy (behavioral cloning) and value (game outcomes).
- Self-play phase: Fine-tune the pretrained network via self-play, starting from the human-initialized policy.
- Compare three variants:
  - (a) Scratch (random init).
  - (b) Heuristic pretraining (paper's approach).
  - (c) Human pretraining.

**Hypothesis:** Human pretraining will converge faster than heuristic pretraining, reaching expert level in ~1 week instead of ~2 months.

**Experiment:**
- Collect ~10,000 human games (if available via RLCard, Botzone, or a public dataset).
- Pretrain on human games; measure policy accuracy (% top-1 action match) and value error (MSE on outcomes).
- Self-play for 5–10 episodes; measure WP vs heuristic baseline per episode.
- Plot learning curves: (a) scratch, (b) heuristic, (c) human.

**Metrics:** Convergence speed (episodes to reach 50% WP), final WP, training time, policy/value losses during pretraining.

**Difficulty:** Moderate to Hard (requires access to human game data; behavioral cloning can be brittle if human strategies are suboptimal or diverse).

**Compute:** Medium (1–2 GPUs, ~2 weeks).

**Novelty:** Good. Transfer learning from human games is a practical direction for real-world AI systems (e.g., systems trained on expert play may be faster and more robust). The CS230 novelty is in the evaluation: does human pretraining actually help in this domain, and by how much?

---

## 16. Important Implementation Components

### MUST UNDERSTAND AND IMPLEMENT/MODIFY

1. **Fictitious Play MCTS (FPMCTS) algorithm (§4.1, Algorithm 1):**
   - What: Asymmetric tree search where only the acting player branches; opponents' moves are sampled from policies.
   - Why: Avoids strategy fusion and non-locality; faster than full MCTS.
   - How: (1) Maintain separate search trees per player. (2) At decision nodes, use p-UCT with policy priors. (3) At chance nodes, sample opponent moves. (4) Expand by two levels.
   - Where to modify: (a) Number of playouts (affects search depth and time per move). (b) p-UCT coefficient $c_{pUCT}$ (affects exploration). (c) Policy sampling temperature (affects how deterministic opponent play is).

2. **Policy-value network and loss function (§4.2, Eq. 2):**
   - What: A neural network that outputs both a policy distribution $p$ (over moves) and a value $v$ (expected outcome).
   - Why: (a) Policy distills MCTS decisions into a fast, differentiable model. (b) Value guides search and bootstraps leaf evaluation.
   - How: (1) Residual blocks with batch norm. (2) 1D-convolutions to exploit card rank structure. (3) Loss combines value MSE + policy cross-entropy + L2 regularization.
   - Where to modify: (a) Network depth (affects expressiveness and speed). (b) Input/output encoding (state representation, action space). (c) Loss weights $C_1, C_2$ (balance value vs policy).

3. **Self-play loop:**
   - What: Generate games using current policies, collect (state, policy, outcome) tuples, update policies, repeat.
   - Why: Provides non-stationary but improving opponents, driving self-improvement.
   - How: (1) Parallel game generation (on CPUs). (2) Accumulate trajectories in a buffer. (3) Mini-batch training on GPU. (4) Checkpoint and repeat.
   - Where to modify: (a) Episode size (8,000 games per episode; affects stability and iteration speed). (b) Temperature schedule (controls exploration over time). (c) Replay buffer size (affects diversity of training data).

4. **State and action representation:**
   - State: 15×7 matrix of [own cards | opponent card counts | recent actions | played cards].
   - Action: Network outputs rank+category buckets (not individual moves); separate kicker network applies heuristics.
   - Why: Reduces branching factor and speeds up search.
   - Where to modify: (a) State encoding (add/remove features, change matrix size). (b) Action encoding (change rank/category bucketing, try learned kickers instead of heuristics).

### MUST UNDERSTAND CONCEPTUALLY

1. **Bayesian inference for hidden hands (§4.3, Eq. 3):**
   - What: Compute posterior $p(s|H)$ over possible hidden hands, given observed action history $H$.
   - Why: Handles imperfect information; allows search to reason under belief.
   - How: Two-phase approximation: (1) Filter random hands using a neural network. (2) Reweight using Bayes' rule with policy likelihood.
   - Implementation note: Expensive (requires scoring thousands of hands); can be simplified or skipped for a CS230 project.

2. **Game-theoretic equilibrium:**
   - What: The self-play learning is implicitly finding a Nash equilibrium or locally stable policy.
   - Why: Ensures robustness (at equilibrium, agents cannot improve unilaterally).
   - How: Each self-play iteration updates toward best response to current opponent policies; at convergence, all players are best-responding.
   - Implementation note: Conceptually important, but implementation is automatic (just run self-play). No explicit equilibrium solver needed.

3. **Information sets and imperfect information (§3.2):**
   - What: An information set $u^i$ groups game states indistinguishable to player $i$.
   - Why: Formalize the gap between player's knowledge and true state; basis for belief-based reasoning.
   - How: Two states are in the same information set if player $i$ observes the same own cards and action history.
   - Implementation note: Conceptually central to imperfect-information games; no explicit data structure usually needed (just use observed cards + action history as state representation).

4. **Action abstraction and kicker networks:**
   - What: Reduce action space by decoupling rank+category from kicker selection.
   - Why: Manageable branching factor; faster search.
   - How: Main network outputs rank+category distributions; kicker network (or heuristic rules) selects kicker.
   - Implementation note: Trade-off between speed and optimality. Understanding this is crucial for scaling RL to complex games.

### CAN REUSE AS INFRASTRUCTURE

1. **RLCard framework:** (From paper 07_RLCard_IJCAI2020.) A PyTorch-based toolkit for card game RL. Provides:
   - Game engine for Dou Dizhu (state, legal moves, rewards).
   - Agent interfaces (for policy networks, MCTS).
   - Parallelization utilities for self-play.
   - Evaluation harness.
   - Can directly reuse for a CS230 project; reduces implementation burden to policy-value network + FPMCTS.

2. **DouZero codebase:** (From paper 01_DouZero_ICML2021, available on GitHub.) Provides:
   - Optimized game engine and state representation.
   - Parallel game generation on CPUs.
   - Network architecture (residual blocks, 1D-convolutions).
   - Self-play training loop.
   - Evaluation utilities and Botzone integration.
   - More advanced and faster than DeltaDou, but can serve as a template. Can reuse network architecture and training loop, then modify to add FPMCTS + Bayesian inference.

3. **Standard deep RL libraries:** PyTorch (for neural networks), NumPy (for matrix operations), Ray (for distributed self-play).

---

## 17. Questions I Should Ask My CS230 Staff

1. **Evaluation rigor:** How many games are needed to claim a statistically significant difference between two Dou Dizhu AIs? The paper uses 100 games per matchup; would you recommend more (e.g., 500–1000)? Should I use binomial tests or other formal statistical tests, and should I report confidence intervals?

2. **Reproduction scale:** I'm considering a CS230 project to reproduce a simplified version of DeltaDou (e.g., FPMCTS without Bayesian inference, 10 self-play episodes). Is this realistic within the quarter (10 weeks) and compute constraints (1–2 GPUs)? Should I simplify further, or would you recommend starting from the DouZero codebase instead?

3. **Novelty and contribution:** The main contribution of DeltaDou (FPMCTS + Bayesian inference) is now superceded by DouZero (simpler, faster). For a CS230 project, should I aim to reproduce DeltaDou as a learning exercise, or should I propose a novel modification (e.g., improved kicker selection, opponent modeling, transfer learning)? What counts as sufficient novelty for a CS230 final project?

4. **Baseline selection:** The paper evaluates against heuristic baselines (XDou, RHCP) and human experts. For my project, would it be acceptable to use a simple learned heuristic (e.g., supervised learning on a small dataset of human games) as a baseline, or should I replicate one of the paper's baselines for fairness?

5. **Multi-player RL challenges:** Dou Dizhu is a three-player game with both competition (Landlord vs Peasants) and cooperation (Peasants with each other). The paper mentions DeltaDou is weak as Peasants, possibly due to poor cooperation. How would you suggest tackling cooperation in multi-player RL? Should I focus on communication/signaling, reward shaping, or opponent modeling?

---

## 18. Beginner Concepts I Need to Learn

| Concept | Plain-language explanation | Where in paper | Timing |
|---------|---------------------------|----------------|--------|
| **Information set** | A set of game states that look identical to a player (same own cards, same action history, different hidden opponent cards). Formalizes imperfect information. | §3.2 (extensive-form games) | NOW (prerequisite for understanding imperfect info) |
| **Bayesian inference** | Using observed data (action history) to update beliefs about unknown quantities (opponent hands). Bayes' rule: posterior ∝ prior × likelihood. | §4.3 (card inference) | NOW (core algorithm) |
| **MCTS (Monte Carlo Tree Search)** | A search algorithm that builds a tree by repeatedly sampling trajectories from the current state. Uses value estimates to guide expansion. | §4.1 (FPMCTS) | NOW (core algorithm) |
| **p-UCT (policy-prior UCT)** | A variant of MCTS's exploration strategy (UCB) that uses neural network priors to focus search on promising actions. | §4.1, Eq. 1 | NOW (core algorithm) |
| **Policy distillation** | Training a network to imitate the distribution of actions from MCTS. Used to speed up inference (MCTS is slow, network is fast). | §6 (loss function) | NOW (training paradigm) |
| **Self-play** | Training an agent by playing against itself (or past versions). Non-stationary opponent improves gradually, driving agent improvement. | §4.4, §7 | NOW (training procedure) |
| **Fictitious play** | A game-theoretic concept: assume opponents play a fixed (but powerful) strategy, compute best response. DeltaDou uses this assumption in MCTS. | §4.1, related work | SOON (explains FPMCTS intuition) |
| **Value network** | A neural network that predicts the expected reward (value) for a player at a given state. Used as a leaf evaluator in tree search (instead of rollouts). | §3 (RL formulation) | NOW (network component) |
| **Action abstraction** | Reducing the action space by grouping actions or decoupling choices (e.g., "choose rank+category first, then kicker"). Trades optimality for speed. | §4.2 | NOW (important design choice) |
| **Determinization** | Sampling a specific assignment of hidden cards to opponents. One of many possible "worlds" consistent with observed history. | §4.3 | NOW (core to inference) |
| **Nash equilibrium** | A game state where no player can unilaterally improve by changing strategy. Self-play algorithms aim to find local Nash equilibria. | Implicit in §7 | SOON (theoretical justification for self-play) |
| **Imperfect information** | A game where players don't observe the full state (e.g., hidden cards in Poker). Harder than perfect-information games (AlphaGo). | §3.2 | NOW (problem definition) |
| **Parallel actors** | Multiple CPU processes running games in parallel, generating data for a GPU training process. Enables scalable self-play. | §11 | SOON (engineering) |
| **Non-stationarity in RL** | The environment/opponents change over time (as the agent improves), making the learning target non-stationary. Can slow convergence. | Implicit in §7, 12B | SOON (challenge in self-play) |
| **Monte Carlo return** | The cumulative discounted reward over a full episode (from current state to termination). Used as a training target instead of bootstrapping. | §6 | NOW (training target) |

---

## 19. Paper Scorecard

> **Cross-paper calibration:** this scorecard looks at this paper alone. Ranked against the whole pack for this Dou Dizhu project, the recommended verdict is **READ SELECTIVELY**. See `00_SYNTHESIS_AND_RECOMMENDATION.md` §20.9.

| Dimension | Score (1–5) | Justification |
|-----------|------------|---------------|
| **Importance for my project** | 5 | FPMCTS, Bayesian inference, self-play in imperfect-information games are directly relevant. First RL system to reach expert level in Dou Dizhu. |
| **Difficulty to understand** | 4 | Requires grasping MCTS, Bayesian inference, extensive-form games, self-play, and policy-value networks. Multiple interlocking ideas; reading time ~3 hours. |
| **Difficulty to reproduce** | 4 | Full reproduction (~2 months on 68 CPUs) is hard. Simplified reproduction (1–2 weeks on 1 GPU) is moderate. Core algorithms (FPMCTS, inference) are non-trivial to implement. |
| **Relevance to Dou Dizhu** | 5 | Dou Dizhu-specific. No generic applicability to other domains. But for a Dou Dizhu RL project, this is canonical. |
| **Relevance to self-play** | 5 | Self-play is the core training mechanism. The paper directly addresses challenges (role asymmetry, non-stationarity, opponent diversity) specific to multi-player self-play. |
| **Relevance to imperfect information** | 5 | Entire paper is about handling imperfect information (hidden cards). Bayesian inference is a key technical contribution. |
| **Potential for project ideas** | 4 | FPMCTS + Bayesian inference are the main techniques; most ideas are refinements or ablations. DouZero/DouZeroPlus have more obvious "next steps" (opponent modeling, action learning). But (a) scaling-down FPMCTS, (b) improving cooperation, (c) action abstraction comparison, (d) transfer learning all viable. |
| **Importance to read deeply** | 4 | Core paper for understanding Dou Dizhu RL via search + learning. But DouZero (which simplifies and improves upon DeltaDou) may be more practical as a baseline. Read both if time permits; prioritize DouZero if you must choose. |

**Verdict: READ DEEPLY, then cross-reference with DouZero (ICML 2021) for context.**

**Which sections to read first (if time is limited):**
1. §3 (Background on Dou Dizhu and extensive-form games) — to ground yourself in the problem.
2. §4.1 (FPMCTS algorithm) — the main algorithmic contribution.
3. §4.3 (Bayesian inference) — the main handling of imperfect information.
4. §5 (Experiments) — to see what works and what doesn't.
5. §9 (Evaluation critical assessment) — to spot-check the methodology.
6. §14 (comparison to DouZero) — to understand why DouZero is faster.

**Skip if time is very limited:**
- §4.2 (policy-value network details) — standard architecture; same as AlphaZero.
- §4.4 (self-play details) — standard AlphaZero-like procedure.
- §3.2 (extensive-form games formalism) — useful for theory, but skim if you know game theory.

---

## Appendix: Synthesis Facts (for cross-paper comparison table)

- **Paper:** DeltaDou: Expert-level Doudizhu AI through Self-play (Jiang et al., IJCAI 2019)
- **Year:** 2019
- **Environment(s):** Dou Dizhu (3-player imperfect-information card game; only card-play phase, not bidding).
- **Main algorithm:** Fictitious Play MCTS (FPMCTS) + Bayesian inference over hidden hands + policy-value networks + self-play.
- **State representation:** 15×7 matrix encoding own cards, opponent card counts, recent actions, played cards.
- **Action representation:** Rank+category (encoded by network), kicker (selected by separate kicker network + heuristics).
- **Self-play?:** Yes, with separate networks per role (Landlord, Peasant-D, Peasant-E).
- **Opponent modeling?:** Implicit via Bayesian posterior over opponent hands; no explicit opponent-type classifier.
- **Main loss:** Value MSE + policy cross-entropy (against MCTS visit distribution) + L2 regularization; targets are game outcomes and MCTS visit counts.
- **Main contribution:** (1) FPMCTS algorithm for multi-player imperfect-information games. (2) Policy-based Bayesian inference. (3) First RL system to reach expert level in Dou Dizhu.
- **Evaluation method:** Tournament against heuristic AIs (XDou, RHCP) and human experts; duplicate bridge format (same deals, swapped roles); metrics: Win Rate (WP), Average Difference in Points (ADP).
- **Compute:** 68 CPUs + unspecified GPU(s), ~2 months training time (~640,000 games over 80 episodes).
- **Main limitation:** Computationally expensive (2 months); relies on heuristic bootstrapping (200,000 games); slightly weaker than a human expert team as Peasants on 100 deals (61% vs 67%; not statistically established), although its edge over XDou comes mainly from the Peasant side; bidding phase not modeled.
- **Best idea to borrow:** (1) **Bayesian posterior over opponent hands** with two-phase filtering (neural network + likelihood reweighting) to handle intractability. (2) **Policy-based sampling in opponent moves** during MCTS (no best-response recursion). (3) **Action abstraction** (rank+category, kicker separate) to manage large action space. (4) **Separate networks per role** for asymmetric games.

