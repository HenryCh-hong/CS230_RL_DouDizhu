# Combining Deep Reinforcement Learning and Search for Imperfect-Information Games (ReBeL) (NeurIPS 2020)

> Brown, Bakhtin, Lerer, Gong (Facebook AI Research) · arXiv:2007.13544 · Tier C (Advanced) · RL+Search for imperfect-information games via belief state abstractions; superhuman performance in poker with minimal domain knowledge.

---

## 0. TL;DR for a CS230 student

- **What it is:** ReBeL extends AlphaZero's RL+Search paradigm to imperfect-information games (poker, Liar's Dice) by representing game states as *public belief states* (PBSs)—shared probability distributions over hidden card holdings.
- **The problem with perfect-information RL+Search in imperfect-information games:** In poker, a hand's value depends on *opponent beliefs* about your strategy, not just the objective game state. Naive one-ply lookahead fails because you can't decide whether to bluff without knowing opponent assumptions.
- **The core idea:** Convert any imperfect-information game into a perfect-information game over continuous belief states. Use iterative equilibrium-finding (CFR) at each belief state, with neural networks bootstrapping leaf values. At test time, sample a random search iteration to play an equilibrium policy *in expectation* (safe search).
- **Why it matters:** First general RL+Search algorithm with convergence guarantees for imperfect-information games. Achieves superhuman poker performance with ~100× less domain abstraction than prior agents (DeepStack, Libratus).
- **One idea to steal for Dou Dizhu:** Represent beliefs as explicit network input features (e.g., probability distributions over opponent card holdings). Train a value network via CFR subgame solving + self-play, then do small-scale search at test time to refine decisions.
- **The limitation you'll hit:** Input size scales linearly with the number of infostates per public state. In Dou Dizhu with ~10^300 card configurations, this is intractable. Dou Dizhu is also 3+ players, not 2-player zero-sum.
- **Learn this for the exam:** Theorem 1 (infostate values as supergradients) + Theorem 3 (safe search in expectation) are the theoretical core; Algorithms 1–2 show the recursive self-play loop.

---

## 1. Paper Overview

### 1a. Beginner-friendly explanation

Imagine you're playing poker but can't see your cards. Instead, you know:
- What cards you *might* be holding (a probability distribution).
- What cards your opponent *might* be holding (from your perspective).
- What betting actions have happened publicly.

You want to decide whether to bet or fold. The tricky part: **your hand's value depends on what your opponent *thinks* you're holding.** If they think you never bluff, they'll call your bets. If they think you always bluff, they'll fold. So the "value" of your actual cards changes based on your opponent's beliefs about your strategy.

AlphaZero (which solved Go and chess) doesn't work here because it assumes each board state has a unique, fixed value. Poker doesn't work that way.

**ReBeL's solution:** Make the belief distribution *part of the state*. Instead of "I'm holding 7♠5♥," the state is "I'm holding 7♠5♥ *with probability 0.8 and K♠Q♦ with probability 0.2, and my opponent thinks I play Rock 40%, Paper 40%, Scissors 20%*." Now the state has a fixed value again. We convert the imperfect-information game into a perfect-information game over these expanded "belief states."

During self-play training:
1. Start at a belief state (e.g., preflop with random cards).
2. Run an iterative equilibrium-finding algorithm (CFR: Counterfactual Regret Minimization) for T iterations.
3. On each iteration, evaluate leaf nodes using a trained neural network (like AlphaZero).
4. After T iterations, add the computed values to training data for the neural network.
5. Sample a leaf belief state and repeat.

At test time (playing a human who doesn't reveal their policy):
- Run the same algorithm, but sample a random iteration's policy.
- This guarantees you play a Nash equilibrium policy *in expectation*, even though you don't know the opponent's actual beliefs.

**Why this works:** The convex geometry of belief spaces means you can use iterative algorithms like CFR. The neural network learns which belief states are valuable through self-play, just like AlphaZero learns through self-play.

### 1b. Technical explanation

**Problem:** Prior RL+Search algorithms (AlphaZero, etc.) assume:
- Each state has a unique value (state value function $v(s)$).
- This is true in perfect-information games because all information is observable, so game progress deterministically leads to values.

In imperfect-information games:
- A state $s$ (sequence of actions + observations) is ambiguous.
- Player i's value depends on a probability distribution over what opponent policies are.
- Formally, value is not $v(s)$ but $v_i(s | \pi_j)$: agent i's value at state $s$ *given opponent plays policy $\pi_j$*.

Since agents don't know opponent policies, naive RL+Search breaks: you can't pin down a leaf value during search.

**Solution (ReBeL):** 
1. **Expand the state representation.** Use a *public belief state* (PBS): $\beta = (\Delta S_1(s^{pub}), \Delta S_2(s^{pub}), \ldots)$, a joint probability distribution over each player's possible infostates (information sets), conditioned on common-knowledge public observations and assuming all players' policies are common knowledge.
   - $\Delta S_i(s^{pub})$ = probability distribution over infostates player $i$ might be in, given public observations $s^{pub}$ and assuming all policies are known.
   - Example: 52 choose 2 possible hands for each of 2 players = 1326×1326 possible (hand1, hand2) pairs. A PBS might be "I hold K♠A♦ with prob 1, opponent holds any pair of the remaining 50 cards uniformly." Or a uniform distribution over all 1326×1326 pairs (initial state).

2. **PBS values are well-defined.** Each PBS $\beta$ has a unique value $V_i(\beta)$ when both players play a Nash equilibrium $\pi^*$ in the subgame rooted at $\beta$. (In 2-player zero-sum games, $V_1(\beta) = -V_2(\beta)$.)

3. **Search algorithm (CFR-D):** At each PBS, run T iterations of Counterfactual Regret Minimization (CFR):
   - On iteration $t$, CFR computes a policy $\pi^t$.
   - Set leaf values using a value network: $\hat{v}(\text{si}|\beta^t)$ for each possible infostate si.
   - CFR minimizes regret and converges to a Nash equilibrium.
   - Average policy $\bar{\pi}^T$ is a $O(1/\sqrt{T})$-Nash equilibrium.

4. **Self-play training:** 
   - Start at root PBS $\beta_r$ (e.g., both players unknown cards, uniform belief).
   - Solve subgame at $\beta_r$ using CFR + neural network.
   - Collect the computed infostate values $v^\pi_i(s_i | \beta_r)$ as training data.
   - Sample a leaf PBS $\beta_{r'}$ (using a random CFR iteration's policy) and repeat.
   - Network learns which belief states are valuable through this bootstrapped data.

5. **Safe search at test time:** At test time, opponent doesn't reveal policy. So we don't know our exact PBS. Solution: run CFR at test time, but pick a *random iteration* $t \sim \text{Uniform}\{1, T\}$ and assume opponent plays $\pi^t$. *Theorem 3*: this plays a $(O(\delta) + O(1/\sqrt{T}))$-Nash equilibrium *in expectation*, where $\delta$ is value function error.

**Key contributions:**
- First general RL+Search framework with convergence proofs for 2-player zero-sum imperfect-information games.
- Safe search without extra constraints (novel test-time procedure from Theorem 3).
- CFR-AVG: variant of CFR-D that improves empirical performance.
- Superhuman poker (HUNL) with minimal domain abstractions (1,326 infostates per card pair vs. hand clusters in prior work).

---

## 2. RL Formulation

| Component | Definition | Citation |
|-----------|-----------|----------|
| **Agent(s)** | Two players (2-player games); agent and opponent | (p. 1, Sec. 3) |
| **Environment** | Factored observation game (partially observable stochastic game) | (Sec. 3; extends [34]) |
| **State (world state)** | $w \in W$: complete game configuration (all cards, all history). | (Sec. 3, p. 3) |
| **Observation** | Private: $O^{priv}_i(w, a, w')$ (private cards, hidden actions). Public: $O^{pub}(w, a, w')$ (betting, public cards). | (Sec. 3, p. 3) |
| **Infostate (action-observation history)** | $s_i = (O^{priv}_i(\cdot), O^{pub}(\cdot), a^{priv}_i(\cdot))$: agent's own observations + actions. Also called AOH (action-observation history). | (Sec. 3, p. 3) |
| **Public state** | $s^{pub} = (O^{pub}_0, O^{pub}_1, \ldots)$: sequence of public observations (common knowledge). | (Sec. 3, p. 3) |
| **Public Belief State (PBS)** | $\beta = (\Delta S_1(s^{pub}), \ldots, \Delta S_N(s^{pub}))$: joint probability distribution over agents' possible infostates. | (Sec. 4, p. 4; [43, 45, 19]) |
| **Action** | $a_i \in A_i(w)$: discrete actions (fold, call, raise, etc.). | (Sec. 3, p. 3) |
| **Legal action handling** | $A_i(w)$: set of legal actions depends on world state $w$. Agent abstraction: reduce action space (e.g., ≤9 bet sizes in poker). Responds to off-tree actions at test time by adding them dynamically. | (Sec. 7, p. 8; Appendix D) |
| **Reward** | $R_i(w, a)$: immediate reward at state w after action a. In poker, final payoff (chips won/lost). | (Sec. 3, p. 3) |
| **Return** | Cumulative discounted reward (undiscounted in episodic games like poker). | (Implicit; Sec. 2, p. 2) |
| **Policy** | $\pi_i: s_i \to \Delta A_i$: map from infostate to probability distribution over actions. Described for both world-state representation and PBS representation. | (Sec. 3, p. 3; Sec. 5) |
| **Value function** | PBS value: $V_i(\beta)$ = expected payoff at PBS $\beta$ under Nash equilibrium $\pi^*$. Infostate value: $v^\pi_i(s_i\|\beta)$ = value of infostate $s_i$ in PBS $\beta$ under policy profile $\pi$. Network learns: $\hat{v}: B \to \mathbb{R}^{\|S_1\|+\|S_2\|}$. | (Sec. 4, p. 4; Sec. 5, p. 5) |
| **Fully observable?** | No; imperfect information (hidden cards, private observations). | (Sec. 1, p. 1) |
| **Single vs. multi-agent** | Two-agent (multi-agent: 2 players); extends to >2 empirically but theory limited to 2-player. | (Sec. 3, p. 3) |
| **Cooperative / Competitive / Mixed** | Competitive: zero-sum (one player's gain = other's loss). $R_1(w,a) = -R_2(w,a)$. | (Sec. 3, p. 4) |
| **Self-play?** | Yes; agents learn through self-play with previous checkpoints implicitly (one agent trained, plays itself). | (Sec. 5.2, p. 6) |
| **Assumption: Common knowledge of policies** | *[Paper]* Yes, assumed during training (Sec. 3, p. 3). Removed at test time via Section 6 (safe search). | (Sec. 3, p. 3; Sec. 6, p. 7) |
| **Assumption: Common knowledge of public information** | Yes; all agents observe same public state $s^{pub}$ and can mutually update beliefs. | (Sec. 3, p. 4) |

---

## 3. Model / Architecture

**Network Type:** Multilayer Perceptron (MLP) for both value and policy networks.

**Activation:** GeLU (Gaussian Error Linear Units, [27]).

**Normalization:** LayerNorm per hidden layer [3].

**Value Network $\hat{v}$:**
- **Input representation:**
  - Agent index: 1 dimension.
  - Public state representation: depends on game.
    - Poker: board cards (5 cards, 1 dim each = 5), pot size relative to stack (1), betting round flag (1).
    - Liar's Dice: last bid (one-hot over possible bids), acting player (1).
  - **Belief distribution over infostates:** 
    - Poker: probability distribution over 1,326 possible two-card hands for player 1, + 1,326 for player 2. Total: 2,652 dimensions (represented as card embeddings → MLP).
    - Liar's Dice: probability over dice outcomes.
  - **Total input size for HUNL:** $1 + 1 + 1 + 5 + 2 \times 1326 = 2660$ dimensions (Appendix E, p. 18: "1(agent index) + 1(acting agent) + 1(pot) + 5(board) + 2×1326(infostate beliefs)").
  
- **Hidden layers:** 6 layers, 1,536 units each.
- **Output:** Vector of size $|S_1| + |S_2|$, one value per possible infostate of each player.
  - For HUNL: $1326 + 1326 = 2652$ outputs (one value for each possible hand each player can hold).
- **Loss:** Pointwise Huber loss (robust to outliers).
- **Target:** Infostate values $v^\pi(s_i | \beta)$ computed during CFR iterations in subgames. Average over T iterations: $(\sum_t v^{\pi^t}(s_i | \beta_r)) / T$.

**Policy Network $\hat{\pi}$ (optional, used for poker):**
- **Input:** Similar to value network but includes separate pot size fractions for both players + betting round flag.
- **Hidden layers:** Same MLP architecture (6 layers, 1,536 units).
- **Output:** Probability distribution over legal actions for each infostate. Size: $|S_1| + |S_2|$ × $|A|$ (e.g., HUNL: $1326 \times 9$ for player 1 if 9 actions max).
- **Loss:** Mean squared error (MSE) over probabilities (empirically better than cross-entropy per Section 7).
- **Target:** Average policy $\bar{\pi}^T$ from CFR after T iterations on each subgame.
- **Purpose:** Warm-start CFR so it converges faster. Optional; value network converges correctly even without it.

**Card Representation:** Embeddings for board cards (similar to [7], Deep CFR). Then passed to MLP.

**Separate networks per player/role?** No; single value and policy network, conditioned on agent index. Outputs are indexed by agent.

**RNN/LSTM/Transformer?** No; feedforward MLP only. History is implicitly in the public state and belief distribution, not in recurrence.

**Why chosen?** (§7, p. 8)
- MLPs are standard for game-playing agents (AlphaZero-style).
- GeLU and LayerNorm empirically work better than ReLU + BatchNorm in preliminary experiments.
- Single network (not separate per player) simplifies training and is symmetric.

**Tensor shapes (example for HUNL):**
- Value net input: `[batch_size, 2660]` (1 agent index + 1 acting agent + 1 pot + 5 board + 2×1326 beliefs).
- Value net output: `[batch_size, 2652]` (1326 + 1326 values).
- Policy net input: `[batch_size, 2656]` (same + extra pot fractions/round flag).
- Policy net output: `[batch_size, 2652 × 9]` (for up to 9 actions per infostate; padded/masked for legal actions).

**Notable omission:** No information abstraction (hand clustering). Prior agents (DeepStack, Libratus) bucket similar hands; ReBeL does not. Each of 1,326 hands is distinct.

---

## 4. Training Algorithm

**High-level pipeline:**

```
Initialize value network θᵥ and policy network θπ
Initialize replay buffers Dᵥ (value data) and Dπ (policy data)

while training:
    PLAY ONE FULL GAME via self-play recursion:
    
    ✓ Start at root PBS βᵣ (e.g., initial game state with uniform beliefs over cards)
    
    ✓ Construct depth-limited subgame G rooted at βᵣ
      (e.g., solve one betting round only, not entire game)
    
    ✓ Initialize subgame policy:
      - If policy network available, use π̄, πᵗʷᵃʳᵐ = INITIALIZE_POLICY(θπ)
      - Else, uniform policy
    
    ✓ Set leaf values using value network:
      For each leaf node z in G, set v(z) = v̂(sᵢ(z) | βᶻ, θᵥ)
      (evaluated on the network)
    
    ✓ CFR iterations (run T iterations of CFR-D or CFR-AVG):
        for t = 1 to T:
            ✓ CFR updates policy: πᵗ = UPDATE_POLICY(G, πᵗ⁻¹)
            ✓ Update average policy: π̄ ← average(π̄, πᵗ)
            ✓ Re-evaluate leaf values: v(z) = v̂(..., θᵥ)  [NOW USES πᵗ]
            ✓ Store infostate values: vᵖⁱᵗ(βᵣ) = [v(s₁|βᵣ), v(s₂|βᵣ), ...]
            
            ✓ If t = tₛₐₘₚₗₑ (random iteration):
                Sample leaf PBS βᵣ' ← SAMPLE_LEAF(G, πᵗ)
                    (sample history from βᵣ, step through game via πᵗ, stop at leaf)
                Save βᵣ' for next iteration
    
    ✓ Collect training data:
    Add (βᵣ, average of values over all CFR iterations) to Dᵥ
    Add (β, π̄(β)) for all PBSs β in subgame G to Dπ
    
    ✓ Recurse: βᵣ ← βᵣ'  (move to sampled leaf PBS)
    
    ✓ Until terminal state, repeat from "Construct depth-limited subgame"

    TRAIN NETWORKS:
    
    ✓ Batch sample from Dᵥ, compute loss: L_v = Huber(v̂(s|β, θᵥ), target)
    ✓ Backprop, update θᵥ via Adam
    
    ✓ Batch sample from Dπ, compute loss: L_π = MSE(π̂(β, θπ), target_π)
    ✓ Backprop, update θπ via Adam
```

**Key algorithmic details (Algorithm 1, p. 5–6):**

1. **Subgame construction:** Fixed-depth subgame (e.g., "this betting round only"), not MCTS-style adaptive depth.

2. **Leaf node value setting:** 
   - On iteration $t$ of CFR-D, set $v(z) = \hat{v}(s_i(z) | \beta_z^{\pi^t}, \theta_v)$.
   - This means: look up the value network at PBS $\beta_z^{\pi^t}$ (the PBS at leaf z given policy $\pi^t$).
   - **Why?** Leaf values are *policy-dependent*. As policy improves toward equilibrium, leaf values change.
   - (CFR-AVG variant: set instead to $\hat{v}(s_i(z) | \beta_z^{\bar{\pi}^t}, \theta_v)$, average policy.)

3. **Infostate value collection:**
   - After T iterations of CFR, compute average value at each infostate: $v(s_i | \beta_r) = \sum_t v^{\pi^t}(s_i | \beta_r) / T$.
   - This is the training target for $\hat{v}(s_i | \beta_r)$.
   - **Intuition:** CFR's average policy converges to Nash. Average of values approximates Nash value at each infostate.

4. **Leaf PBS sampling:**
   - On a random iteration $t_{\text{sample}} \sim \text{Uniform}\{1, T\}$, sample a leaf PBS.
   - Sample history $h$ from root PBS $\beta_r$ uniformly, then step through game:
     - For each decision point: pick a random player $i^*$, explore with prob $\epsilon = 0.25$ (random action) else play $\pi^t(s_i(h))$.
     - Stop at leaf node, get PBS $\beta_r' = \text{PBS}(h)$.
   - **Why random iteration?** To train value network on PBSs encountered at *every* stage of search, not just final policies. (Sec. 5.2, p. 6.)
   - **Why random exploration $\epsilon = 0.25$?** To ensure sufficient coverage of off-policy PBSs. (Sec. 5.2, p. 6.)

5. **Bootstrapping vs. Monte Carlo:**
   - Described as bootstrapping: leaf values estimated by network, not by rollout.
   - Tradeoff: bias (network error) vs. variance (rollout stochasticity). (Sec. 5.2, p. 6.)

6. **Policy network warm-start (Sec. 5.3, Appendix J):**
   - If policy network available, initialize CFR with $\pi^0 = \hat{\pi}(\beta_r)$.
   - Use "soft best response" technique [10]: compute exact best response, scale regrets by factor of 15 to imitate 15 CFR iterations.
   - Result: CFR starts with good initial policy, converges faster.
   - Optional; value network works without it but converges slower.

7. **Training procedure for networks (Sec. 7, Appendix E, p. 8):**
   - **Value network:**
     - Loss: Huber loss (robust to outliers).
     - Optimizer: Adam with learning rate $3 \times 10^{-4}$ (halved every 800 epochs for HUNL).
     - Batch size: 1,024.
     - Circular replay buffer: 12M samples (old data discarded as new data generated).
     - Remove half of initial random-value-network data after 20 epochs (warm-up discarding).
   - **Policy network:**
     - Loss: MSE over probabilities (cross-entropy performed worse empirically).
     - Linear quantization applied to action probabilities to manage output size.
     - Same optimizer and buffer as value network.

---

## 5. Mathematics

### Equation 1: Infostate value in a PBS (p. 4)

**[MUST UNDERSTAND]**

$$v_i^{\pi^*}(s_i | \beta) = \max_{\pi_i} \sum_{h \in H(s_i)} p(h | s_i, \beta_{-i}) v_i^{(\pi_i, \pi_{-i}^*)}(h)$$

**Symbols:**
- $v_i^{\pi^*}(s_i | \beta)$: value of infostate $s_i$ for agent $i$, assuming all other players play Nash equilibrium $\pi^*$ in subgame rooted at PBS $\beta$.
- $\pi_i$: agent i's policy in the subgame (to be optimized).
- $\pi_{-i}^*$: Nash equilibrium policies of other agents.
- $H(s_i)$: set of all histories (world states) consistent with infostate $s_i$.
- $p(h | s_i, \beta_{-i})$: probability of being in history $h$ given that $s_i$ is reached and other players' belief distributions are $\beta_{-i}$.
- $v_i^{(\pi_i, \pi_{-i}^*)}(h)$: expected value starting from history $h$ if agent $i$ plays $\pi_i$ and others play $\pi^*$.

**Intuition:** 
- Agent $i$ doesn't know which history in $H(s_i)$ is true, only the belief distribution.
- Each history has a value (outcome from that point forward under the given policies).
- Agent $i$ can choose a policy to maximize their expected value, averaging over the hidden histories.
- This is the **best response** to the opponent's Nash equilibrium policy in the subgame.

**Why needed:** This is the value the neural network should learn for each infostate in each PBS. It's the "ground truth" from CFR-computed equilibria.

---

### Theorem 1: Infostate values are supergradients of PBS values (p. 4)

**[OPTIONAL / ADVANCED]**

**Statement:**
For any PBS $\beta = (\beta_1, \beta_2)$ and any Nash equilibrium policy $\pi^*$ of the subgame rooted at $\beta$:

$$v_1^{\pi^*}(s_1 | \beta) = V_1(\beta) + \bar{g} \cdot \hat{s}_1 \quad \text{...(2)}$$

where $\bar{g}$ is a supergradient of the extension $\tilde{V}_1(\beta)$ of PBS value to unnormalized belief distributions, and $\hat{s}_1$ is the unit vector in direction $s_1$.

**Symbols:**
- $V_1(\beta)$: PBS value (value of PBS $\beta$ when both players play Nash equilibrium).
- $\tilde{V}_1$: extension of $V_1$ from normalized beliefs (simplex) to unnormalized distributions (cone $\mathbb{R}_{\geq 0}$).
- Supergradient: generalization of gradient for concave functions. For concave $F$ at $x_0$, $g$ is a supergradient if $F(x) - F(x_0) \leq g \cdot (x - x_0)$ for all $x$.

**Intuition:** 
- The PBS value $V_1(\beta)$ is a concave function of belief $\beta_1$ (agent 1's belief distribution over infostates).
- Infostate values $v_1^{\pi^*}(s_1 | \beta)$ form a supergradient of this function.
- **Practical consequence:** you can compute PBS values via weighted sums of infostate values (linear combination of supergradients is itself a supergradient). 
- **Why advanced?** Requires convex analysis (concavity, supergradients). Doesn't directly affect the algorithm but justifies why infostate values are sufficient.

**Proof sketch (Appendix F, p. 19):**
1. Show $V_1(\beta)$ is concave in $\beta_1$ (since it's minimum of linear functions $V_1^{\pi_2}$ over $\pi_2$).
2. For Nash $\pi^*$, the gradient of the linear function $V_1^{\pi_2^*}(\beta)$ is a supergradient of $V_1(\beta)$ (since it's tangent).
3. Compute the gradient: $\nabla_{\beta_1} V_1^{\pi_2^*}(\beta) = \sum_{s_1 \in S_1} \hat{s}_1 v_1^{\pi_2^*}(s_1 | \beta)$.
4. Rearrange to get the formula.

---

### Theorem 2: Value network convergence (p. 6)

**[MUST UNDERSTAND (gist); OPTIONAL (proof)]**

**Statement:**
Consider an idealized value approximator that returns the most recent sample of the value for sampled PBSs and 0 otherwise. Running Algorithm 1 with T iterations of CFR in each subgame will produce a value approximator with error at most $\frac{C}{\sqrt{T}}$ for any PBS that could be encountered during play, where $C$ is a game-dependent constant.

**Symbols:**
- $T$: number of CFR iterations per subgame.
- $C$: game-specific constant (depends on payoff ranges, game depth).
- Error $\delta = C / \sqrt{T}$: upper bound on $|\hat{v}(s_i | \beta) - v^*_i(s_i | \beta)|$ for any sampled PBS.

**Intuition:**
- The error decreases as $O(1/\sqrt{T})$, typical for CFR-style algorithms.
- With more CFR iterations, the value function converges to true values.
- This assumes "perfect" function approximation (actually stores all sampled PBSs exactly), so real networks will have additional error.

**Why needed:** Justifies that the algorithm is theoretically sound—error bounds improve with more computation.

**Practical limit:** Real networks have function approximation error on top of this $O(1/\sqrt{T})$ error. Not specified how bad that is.

---

### Theorem 3: Safe search at test time (p. 8)

**[MUST UNDERSTAND]**

**Statement:**
If Algorithm 1 is run at test time with no off-policy exploration, a value network with error at most $\delta$ for any leaf PBS, and with $T$ iterations of CFR solving subgames, then the algorithm plays a $(\delta C_1 + \delta C_2 \sqrt{\frac{1}{T}})$-Nash equilibrium, where $C_1, C_2$ are game-specific constants.

**Symbols:**
- $\delta$: maximum value network error on leaf PBSs.
- $T$: CFR iterations used during test-time search.
- $(\delta C_1 + \delta C_2 / \sqrt{T})$-Nash equilibrium: policy whose exploitability is bounded by this expression.

**Intuition:**
- You can play a near-Nash equilibrium *in expectation* at test time, even if you don't know opponent's true policy.
- The procedure: run CFR at test time, pick a random iteration $t \sim \text{Uniform}\{1, T\}$, play $\pi^t$.
- Error comes from two sources:
  1. Value network error ($\delta C_1$ term): network might misestimate leaf PBSs.
  2. CFR convergence error ($\delta C_2 / \sqrt{T}$ term): CFR needs finite iterations.
- **Key novelty:** Previous safe search methods add constraints; this just randomly samples iterations—simpler and empirically better.

**Proof idea (Appendix G, p. 20):**
Inductive proof over game depth. Base case: terminal subgames where CFR plays a $O(1/\sqrt{T})$-Nash. Inductive case: assume leaf PBSs can be played $\delta$-approximately; then subgame rooted at parent PBS can also be played approximately via CFR.

---

### Equation (implicit in Algorithm 1): Value target for training

**[MUST UNDERSTAND]**

After running CFR for T iterations in subgame rooted at $\beta_r$:

$$\text{Target}_{\text{value}} = \frac{1}{T} \sum_{t=1}^{T} v^{\pi^t}(s_i | \beta_r)$$

**Symbols:**
- $v^{\pi^t}(s_i | \beta_r)$: value of infostate $s_i$ computed by CFR at iteration $t$ (with $\pi^t$ as the policy).
- Average: expected value under the converging average policy $\bar{\pi}^T$.

**Intuition:**
- CFR's average policy $\bar{\pi}^T$ converges to Nash.
- Average of values over iterations approximates the value at Nash equilibrium.
- This is the target for training $\hat{v}(s_i | \beta_r)$.

---

### Summary table of equations:

| Equation | Type | Purpose |
|----------|------|---------|
| Eq. 1: $v_i^{\pi^*}(s_i \| \beta)$ | MUST UNDERSTAND | Define infostate value (best response to opponent Nash in subgame) |
| Eq. 2 (Theorem 1): Supergradient | OPTIONAL/ADVANCED | Justify using infostate values in search; connects to convex analysis |
| Theorem 2: $O(1/\sqrt{T})$ convergence | OPTIONAL/ADVANCED | Prove value network error decreases with CFR iterations |
| Theorem 3: Safe search | MUST UNDERSTAND | Justify test-time algorithm (random iteration sampling) |
| Value target average | MUST UNDERSTAND | Training objective for value network |

---

## 6. Loss Function

**Value Network Loss:**
- **Loss function:** Pointwise Huber loss $L_{\text{smooth}}(\text{pred}, \text{target})$.
  - Defined as: $L(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } \|y - \hat{y}\| \leq \delta \\ \delta(\|y - \hat{y}\| - \frac{\delta}{2}) & \text{otherwise} \end{cases}$ (smooth approximation of L1 for large errors).
  - **Why Huber?** MSE was tried; Huber found empirically better (Sec. 7, p. 8). Robust to outliers in value estimates.
  
- **Target:** Infostate values $v^{\pi^t}(s_i | \beta_r)$ computed by CFR after $T$ iterations, averaged: $\frac{1}{T} \sum_t v^{\pi^t}(s_i | \beta_r)$.
- **CFR variant:** The paper uses CFR-AVG (Appendix I, p. 22–23), which is theoretically sound. However, an efficient implementation modification for poker is "not theoretically sound but empirically performs well in poker" (Appendix I, p. 23). Whether this modified form is theoretically sound remains an open question.

- **Where target comes from:** Bootstrapping. CFR solves subgames and computes values. Those values are supervised targets.

- **Target network?** No explicit target network. But there is a delay: value network trained on CFR outputs from previous data generation runs, not immediately on latest network.

- **Bootstrapping?** Yes; leaf node values in CFR come from value network. The network is trained on CFR outputs, which depend on the network itself. This creates circular dependency (self-play loop).

- **Why this works:** Self-play breaks the cycle if network improves over time. Early poor network produces poor subgame solutions; retraining on those solutions improves network; next self-play uses better network.

**Policy Network Loss:**
- **Loss function:** Mean squared error (MSE) over action probabilities. $L_{\text{policy}} = \frac{1}{|A|} \sum_{a \in A} (\hat{\pi}(a | s_i) - \bar{\pi}^T(a | s_i))^2$.
  - (Cross-entropy was tried; MSE performed better empirically per Sec. 7.)
  
- **Target:** Average policy $\bar{\pi}^T$ from CFR after T iterations. (Unlike value net, policy net trained on final averaged policy, not iteration-wise average.)

- **Purpose:** Warm-start CFR in next iteration. By predicting $\bar{\pi}^T$ from PBS, next CFR run initializes with a good policy and converges faster.

- **Where target comes from:** CFR directly outputs $\bar{\pi}^T$ at each subgame root.

- **Optional?** Yes; value network alone converges correctly (Sec. 5.3, p. 6). Policy network reduces CFR iterations needed for convergence.

**Multi-objective or single?** Separate training loops:
1. Generate data (CFR subgame solving + network evaluation), collect $(s_i, v)$ and $(s_i, \pi)$ pairs.
2. Batch-train value network on $(s_i, v)$ pairs.
3. Batch-train policy network on $(s_i, \pi)$ pairs (optional, only for poker).

**Auxiliary losses?** None mentioned. No regularization terms (e.g., L2 weight decay, entropy regularization).

**Why these choices?**
- Huber loss: robust to outliers in self-play (some subgames may have poorly estimated values early on).
- Separate networks: allows independent optimization; policy network optional.
- No auxiliary losses: simplicity. Regularization not needed due to large data volume and early stopping (300 epochs for TEH, 1,750 for HUNL).

---

## 7. Self-Play

**Who plays whom?** 
- Single agent trained via self-play against itself.
- Formally: both players execute the same network $\hat{v}$ and $\hat{\pi}$, playing subgame solving in parallel.
- No opponent pool; both players are trained simultaneously.

**Simultaneous or alternating updates?**
- Both players update simultaneously within a self-play game (no alternating roles per se; CFR alternates updates internally).
- Networks updated after each full game (batch of games) in training.

**Parameter sharing?**
- Yes; single value network $\hat{v}$ and policy network $\hat{\pi}$ shared by both players.
- Output indexed by agent ID: network outputs are $|S_1| + |S_2|$ dimensions, with first $|S_1|$ for player 1, next $|S_2|$ for player 2.

**Old checkpoints?**
- Paper does not explicitly mention maintaining old checkpoints (pool of previous policies).
- Implicit: all players always use latest checkpoint during data generation and training.
- Potential issue: non-stationarity (opponent's behavior changes as network improves). Not explicitly addressed.

**Opponent pool?**
- No; unlike some multi-agent methods, ReBeL doesn't maintain a pool of past policies.
- All games use current checkpoint.

**Exploration:**
- During leaf sampling, random agent $i^*$ and one random action with probability $\epsilon = 0.25$ (Appendix E, p. 17).
- This ensures sufficient PBS coverage even as policy converges.

**Non-stationarity and handling:**
- Not explicitly discussed in paper.
- [**Interpretation**] Self-play with a single evolving policy can suffer from instability: as network improves, opponent (also using network) changes, creating moving target. Mitigated by:
  1. Large replay buffer (12M samples, circular): recent high-quality data mixed with older data.
  2. Exploration $\epsilon$: prevents convergence to trivial best-response strategies too early.
  3. CFR's robustness: CFR solves subgames exactly (given value function), so even if value network is imperfect, search is still principled.

---

## 8. Imperfect Information

**How hidden information is handled:**

| Actually observed by agent | Hidden |
|-----|--------|
| Public observations: betting actions, public board cards (in poker), public bids (in Liar's Dice). | Private cards: opponent's hole cards, own private dice rolls (in Liar's Dice). |
| Own private observations: own cards, own dice. | Other agents' private observations. |
| Sequence of all public actions/events (public history). | Which specific private cards/outcomes opponent holds (private history). |

**Explicit representations:**

1. **Information sets (Infostates):**
   - Agent $i$'s infostate $s_i$ = sequence of own observations + actions (AOH: action-observation history).
   - Example: player has seen public cards [A♠, K♥] and their own cards [Q♠, J♦], and the action sequence was "check, bet, call, raise, call".
   - Multiple world states can map to same infostate (player doesn't distinguish them).

2. **Public states:**
   - Common knowledge: sequence of public observations only.
   - Example: public cards [A♠, K♥], bets (100 small blind, 200 big blind, 500 raise, 500 call, 1000 raise).
   - All agents agree on public state.

3. **Belief states (PBSs):**
   - Joint probability distribution over agents' infostates, conditioned on public state.
   - $\beta = (\Delta S_1(s^{pub}), \Delta S_2(s^{pub}))$: for each agent, probability distribution over possible infostates given public observations.
   - Example: I believe opponent's cards are uniformly distributed over all 1,326 possible hands (except mine is known), so $\beta_2$ is uniform; I know my cards exactly, so $\beta_1$ has probability 1 on my true infostate.
   - **Common knowledge:** all agents can compute $\beta$ from public state + knowledge of policies (assumption during training).
   - **At test time:** opponent's policy not known, so exact PBS unknown. Solved by sampling random CFR iteration (Section 6, Theorem 3).

4. **Card inference / Belief updates:**
   - Implicit in PBS representation; Bayesian updates are pre-computed when generating PBSs.
   - When moving from PBS $\beta$ via action $a$, the new PBS $\beta'$ reflects updated beliefs after observing $a$ (and assuming all players know all policies).
   - **Explicit features:** Network input includes belief distribution over opponent hands (e.g., for HUNL, 1,326-dim vector of probabilities summing to 1).

**Architecture for handling imperfect info:**

- **No RNN/memory:** Hidden history is not modeled sequentially. Instead, beliefs are collapsed into a single PBS, and network sees the PBS (belief + public state) as input.
- **Belief features:** Explicit probability distributions over cards are input features (not learned implicitly).
- **No opponent modeling in network:** Network doesn't predict opponent's hand; it only uses the belief distribution as input.

**Information sets in HUNL:**

- Number of infostates: $1,326$ (possible two-card hands for each player).
- Per public state (e.g., "flop with board [A♠K♥T♦]"), each player has up to 1,326 possible infostates (less if some hands were folded earlier).
- Network outputs a value for each infostate within PBS: $1,326 + 1,326 = 2,652$ outputs.

**Limitations:**

- [**Paper**] Input size scales with number of infostates per PBS (Conclusion, p. 9). Intractable for games with little common knowledge (e.g., Recon Chess, where each player has hidden pieces).
- [**For Dou Dizhu**] ~3 players with ~$10^{300}$ possible card configurations makes explicit belief representation infeasible.

---

## 9. Evaluation

### Baselines

**Poker (HUNL):**
- BabyTartanian8 (2016 Computer Poker Competition champion): strong baseline.
- Slumbot (recurring CPC champion): strong baseline.
- Libratus (2017, beat top humans): "gold standard" baseline (not directly compared; only aggregate human results from Libratus trials).
- DeepStack (2017, beat humans): value function approach (compared in related work, not head-to-head).
- Local Best Response (LBR): simple baseline that exploits fixed opponent [39].
- Dong Kim: top human expert (one of four humans who played Libratus; did best against Libratus).

**Liar's Dice:**
- Full-game tabular CFR / Fictitious Play: intractable as game grows, used as reference for exploitability.
- ReBeL's own baselines (same game, no search): compare with search vs. without.

**TEH (Turn Endgame Hold'em):**
- Perfect Value Net: oracle (exact leaf values).
- Self-Play Value Net trained via ReBeL.
- Self-Play Value/Policy Net (with warm-start).
- Random Beliefs Value Net: value network trained on random PBSs (without self-play).

### How policies compared

**Metric 1: Exploitability (Liar's Dice, TEH)**
- Definition: $\text{Expl}(\pi^*) = \sum_i \max_{\pi_i} v_i(\pi_i, \pi^*_{-i}) / |N|$.
  - Sum over agents of the maximum value each agent can achieve against the opponent.
  - Lower is better (closer to 0 = less exploitable = closer to equilibrium).
- Unit: thousandths of initial pot (TEH), unitless fraction (Liar's Dice).
- **Why useful:** Direct measure of Nash convergence; goal is low exploitability.

**Metric 2: Head-to-head win rate (HUNL)**
- Unit: millibig blinds (mbb) per game = expected winnings in thousandths of big blind per hand.
- Positive = winning, negative = losing.
- **Notation uncertainty:** Table 1 caption (p. 8) states "± shows one standard deviation", but Appendix E.1 (p. 18) states "± indicates one standard error" (SE = SD / √n, which is smaller). Different quantities; paper is inconsistent.
- Example: ReBeL vs. Dong Kim: 165 ± 69 mbb (ReBeL won on average; ±69 has high relative uncertainty).

**Metric 3: Variance reduction (HUNL human play)**
- Method: AIVAT (Action-Independent Variance Reduction Technique) [17].
- Why needed: Poker has high variance (luck of the draw). AIVAT reduces variance by normalizing outcomes against expected values at decision points.
- Effect: "Variance-reduced score" is ±69 vs. raw score of ±188 (Appendix E.1, p. 18).

### Test setup

**Sample size:**
- Dong Kim vs ReBeL: 7,500 hands (large sample).
- Computer vs computer (BabyTartanian8, Slumbot): unknown sample size (typical: 10,000–100,000 hands for statistical significance).
- Liar's Dice: policies averaged over 1,024 playthroughs per test.

**Fixed opponents?**
- Yes; BabyTartanian8 and Slumbot are fixed (don't adapt).
- Dong Kim: human expert, could adapt within session but likely limited within 7,500 hands.

**Roles evaluated separately?**
- Poker is symmetric (zero-sum, no role difference), but:
  - In poker, player 1 acts first preflop; player 2 acts first on later streets. Different advantage.
  - [Paper does not specify] Whether ReBeL played both roles against humans or only one.
  - (Libratus results show both roles matter; doesn't specify for ReBeL.)

**Cross-play?**
- Liar's Dice: ReBeL vs itself (self-play policies are evaluated by averaging over playthroughs, then exploitability computed).
- HUNL: ReBeL vs fixed opponents (not symmetric).
- [**Interpretation**] No cross-play with other methods (e.g., ReBeL vs DeepStack) due to computational cost and different abstractions.

**Multiple seeds?**
- [**Paper does not specify.**] 
- Typical practice: models trained once, tested once. If multiple seeds, not mentioned.
- Variance from single seed could inflate reported uncertainties.

### Results with numbers

**Table 1: Head-to-head results (HUNL, p. 8)**

| Opponent | ReBeL (mbb/game) | Notes |
|----------|------------------|-------|
| Slumbot | 45 ± 5 | ReBeL wins (positive). |
| BabyTartanian8 | 9 ± 4 | ReBeL wins (positive). |
| LBR (Local Best Response) | 881 ± 94 | ReBeL wins decisively (LBR is weak baseline). |
| Dong Kim (human) | 165 ± 69 | ReBeL wins; note high uncertainty (95% CI roughly ±138, so overlaps with near-zero). |
| DeepStack (from prior work) | - | -383 ± 112 vs. Kim (DeepStack lost to Kim). |
| Libratus (from prior work) | - | 147 ± 39 vs. Kim aggregate (Libratus beat Kim). |
| Modicum (from prior work) | - | 11 ± 5 vs. Slumbot, 6 ± 3 vs. BabyTartanian8. |

**Interpretation:**
- ReBeL beats BabyTartanian8 (9 ± 4) and Slumbot (45 ± 5).
- vs. Dong Kim: 165 ± 69 mbb means ReBeL won on average, but with large std error (±69 is about 42% of the mean). High uncertainty: 95% CI roughly ±138, so the confidence interval could be near-zero or even favor Kim.
- Libratus's 147 ± 39 vs. humans is stronger (but against 4 humans aggregate; Dong Kim alone might be different).
- Modicum (11 ± 5 vs. Slumbot) is weaker than ReBeL (45 ± 5).
- **Verdict**: ReBeL competitive or better than prior poker AIs; human performance measured with substantial variance.

**Table 2: Exploitability in Liar's Dice (p. 8)**

| Game | Full CFR | Full FP | ReBeL CFR-D | ReBeL FP |
|------|----------|---------|-------------|----------|
| 1x4f | 0.001 | 0.012 | 0.017 | 0.041 |
| 1x5f | 0.001 | 0.024 | 0.015 | 0.020 |
| 1x6f | 0.002 | 0.039 | 0.024 | 0.040 |
| 2x3f | 0.002 | 0.057 | 0.017 | 0.020 |

- Full-game tabular methods (CFR, FP) are more exploitable in larger games (as game size increases, more variance in sampling).
- ReBeL matches or beats tabular FP; underperforms full-game CFR (but full-game CFR intractable for larger games).
- **Note:** ReBeL runs depth-2 subgames with 1024 iterations, vs. full-game 1024 iterations for tabular methods. Different settings, so direct comparison limited.

**Figure 2: TEH convergence (p. 8)**
- X-axis: search iterations (0–250) and training epochs (0–300).
- Y-axis: exploitability (log scale, 10^-1 to 10^-3).
- **Key finding:** Self-Play Value Net reaches exploitability of ~0.001–0.01 (equivalent to ~125 full-game tabular CFR iterations).
- **Comparison:** 
  - Perfect Value Net (oracle): reaches ~0.001 fastest.
  - Self-Play Value Net: catches up around iteration 200.
  - Random Beliefs Value Net: flat, doesn't improve (self-play crucial).
  - Self-Play Value/Policy Net: slightly faster convergence than Value Net alone.

### Critical Evaluation

**Strengths:**

1. **Theoretical contributions:** First RL+Search algorithm with convergence proofs for imperfect-information games. Theorems 1–3 are novel and valuable.

2. **Empirical success:** Superhuman HUNL performance with far less domain knowledge than prior agents.
   - No information abstraction (1,326 hands vs. hand clusters).
   - No precomputed end-of-game values.
   - Single value network for all game depths (vs. separate networks in prior work).
   - Responds to arbitrary off-tree actions (vs. fixed action abstraction).

3. **Safe search without constraints:** Theorem 3 (random iteration sampling) is elegant and practical. No need for complex safe search algorithms from prior work.

4. **Open-source implementation:** Liar's Dice code released for reproducibility (poker code not released for safety reasons).

5. **Detailed experiments:** Convergence plots (Figure 2) show clear comparison of with/without self-play, value/policy networks.

**Weaknesses:**

1. **Limited human evaluation:**
   - Only 7,500 hands vs. Dong Kim (one player). Confidence intervals are wide (±69 mbb std error).
   - Libratus played 4 humans for longer; Modicum's results are older.
   - No detailed account of Dong Kim's play (did he adapt? take breaks? play at optimal focus?).
   - [**Analysis**] The 165 ± 69 result is positive but not statistically overwhelming; could be noise given variance.

2. **Liar's Dice results weaker than full-game CFR:**
   - ReBeL CFR-D is 17× worse than full CFR in 1x4f (0.017 vs. 0.001).
   - Advantage: handles larger games where full CFR intractable; not a "better algorithm," just more scalable.

3. **No comparison with Deep CFR:**
   - Deep CFR (2019) also uses neural networks + CFR [7].
   - Not head-to-head compared (different settings, implementations).
   - [**Interpretation**] Relationship unclear; Deep CFR may be orthogonal (single agent abstraction) or complementary (could combine).

4. **Bootstrapping circularity not analyzed:**
   - Value network trained on CFR outputs, which use the value network as leaf values.
   - Potential for bias (network error → poor subgame solutions → training targets remain bad).
   - Theorem 2 assumes "perfect approximator," but real networks don't achieve this.
   - No analysis of how imperfect networks affect convergence in practice.

5. **Scalability limitations for >2 players:**
   - Theory limited to 2-player zero-sum.
   - Empirical results mention "related techniques have been empirically successful in some settings with more players [14]" (i.e., Pluribus), but ReBeL not tested.
   - Dou Dizhu has 3+ players; ReBeL's approach not straightforward.

6. **Input dimensionality:** 
   - Network input grows linearly with infostates per PBS.
   - HUNL has manageable ~2,600 input dims, but games with larger info states (Recon Chess, Dou Dizhu) infeasible.
   - No dimensionality reduction techniques attempted (e.g., autoencoders, hand clustering).

7. **Variance reduction only for human play:**
   - AIVAT applied to human results but not to bot comparisons (why?).
   - Affects interpretation: bot results may have higher noise than reported.

### What I would improve for a CS230 project

1. **Ablation studies:**
   - Isolate contribution of each component: value network warm-start, policy network, exploration $\epsilon$, replay buffer size, loss function choice.
   - Only Table 2 (value vs. value+policy) and Figure 2 (value net comparison) provided; limited ablations.

2. **Failure case analysis:**
   - When does the algorithm play badly (e.g., against exploitative opponents)?
   - Test against agents trained to exploit ReBeL.

3. **Scaling experiments:**
   - Vary network size, data volume, computational budget.
   - Show compute/performance Pareto frontier.

4. **Interpretability:**
   - Visualize learned value networks: which PBSs are high/low value?
   - Probe network learned beliefs: are they sensible?

5. **Comparison with simpler baselines:**
   - Tabular CFR (if feasible on smaller games): direct comparison.
   - Pure policy gradients (PPO, A3C): how much does search help?

6. **Extend to 3+ players:**
   - Modify algorithm for 3-player Liar's Dice or toy poker variant.
   - Analyze how algorithm scales (e.g., does convergence degrade?).

---

## 10. Experiments and Ablations

### Major experiments

**Experiment 1: TEH convergence (Figure 2, p. 8)**
- **What changed:** Comparison of different value function approaches (perfect oracle, self-play, random PBSs) and with/without policy network.
- **Held constant:** Game (TEH), search depth, number of epochs, network architecture.
- **Metric:** Exploitability (lower = better).
- **Result:** 
  - Self-Play Value Net converges to ~0.001 (100 iterations) vs. full-game tabular CFR benchmark.
  - Random Beliefs Value Net flat (fails to learn).
  - Self-Play Value/Policy Net converges ~10% faster.
- **Conclusion:** Self-play is critical; random initialization fails; policy network marginally helpful.
- **Critical note:** Exploitability computed on one specific board (3♠7♥T♦K♠ with $20k stacks), so generalization to other boards not shown.

**Experiment 2: HUNL vs. bot agents (Table 1, p. 8)**
- **What changed:** Opponent agent (BabyTartanian8, Slumbot, LBR).
- **Held constant:** ReBeL configuration, number of hands per matchup (unspecified but likely 10k+).
- **Metric:** Win rate in mbb/game ± std error.
- **Results:**
  - Slumbot: ReBeL +45 ± 5 (wins).
  - BabyTartanian8: ReBeL +9 ± 4 (wins, smaller margin).
  - LBR: ReBeL +881 ± 94 (dominates weak baseline).
- **Conclusion:** ReBeL competitive or stronger than prior top bots.
- **Critical note:** Margins are modest vs. Slumbot/BabyTartanian8 (45±5 is ~9× the std error for Slumbot). Substantial overlap possible with more hands.

**Experiment 3: HUNL vs. human expert (Table 1, p. 8)**
- **What changed:** Opponent (Dong Kim, top human).
- **Held constant:** ReBeL configuration, 7,500 hands.
- **Metric:** Variance-reduced win rate ± 1 std error.
- **Result:** ReBeL won 165 ± 69 mbb/game (variance-reduced; high relative uncertainty ±69).
- **Conclusion:** ReBeL superhuman (beats expert).
- **Critical note:** 
  - Large std error (±69) relative to mean (165). 95% CI roughly 165 ± 138, so could range from near-zero to ~300. High uncertainty.
  - Only one human tested (small sample of human variation).
  - 7,500 hands is not huge for high-variance poker (Libratus played 120,000 hands against 4 humans).
  - Variance-reduced score (±69) masks true variance (raw std error was ±188).

**Experiment 4: Liar's Dice convergence (Table 2, p. 8)**
- **What changed:** Game variant (1x4f, 1x5f, 1x6f, 2x3f; f = faces, d = dice).
- **Held constant:** Search depth (2), number of search iterations (1024), network architecture.
- **Metric:** Exploitability of averaged policy over 1024 playthroughs (upper bound).
- **Results:** ReBeL CFR-D exploitability 0.015–0.024; ReBeL FP 0.020–0.041.
- **Comparison:** Full-game CFR exploitability 0.001–0.002; full-game FP 0.012–0.057.
- **Conclusion:** ReBeL converges to approximate Nash; worse than full-game methods but handles larger games.
- **Critical note:** 
  - Exploitability upper bound (averaged from 1024 samples), not true exploitability (requires full game tree).
  - Unfair comparison: full-game CFR uses 1024 iterations on entire game; ReBeL uses depth-2 subgames with 1024 iterations each (different computational effort).

**Experiment 5: Search iterations vs. convergence (Appendix I, Figure 4 and 5, p. 22–23)**
- **What changed:** Search iterations (64–16384) for different equilibrium-finding algorithms (FP, Linear FP, FLOP, Linear CFR).
- **Held constant:** Game, network, depth.
- **Metric:** Exploitability.
- **Results:** Linear CFR > FLOP > Linear FP > FP in convergence rate.
- **Conclusion:** Linear CFR (state-of-the-art) > novel FLOP; FLOP is good compromise.
- **Critical note:** FLOP introduced in paper; only compared within reachability not against baselines.

**Experiment 6: Warm-start CFR (Appendix J, p. 25)**
- **What changed:** Initial CFR policy (uniform vs. warm-start from policy network).
- **Held constant:** Network, search depth, game (TEH).
- **Metric:** Iterations needed to reach target exploitability.
- **Result:** Warm-start reduces iterations needed (from Figure 2, ~20 iterations faster with policy net).
- **Conclusion:** Policy network provides useful initialization.
- **Critical note:** Minimal quantitative data; mostly shown qualitatively in Figure 2.

### Reproducibility at small scale for CS230

**Feasible (weeks of 1–2 GPUs):**
1. **Liar's Dice (1x4f):** Reproduce Table 2. ~25M training examples, 1 GPU sufficient.
2. **Turn Endgame Hold'em (TEH):** Smaller version. Reduce network size (3 layers, 512 units) + fewer epochs. 1–2 GPUs.
3. **Toy poker game:** 2-card half-deck (6 cards), 2 players, simplified betting (fold/call/raise). Feasible on CPU.

**Unrealistic (require significant compute):**
1. **Full HUNL:** 90 DGX-1s (720 V100s) for data generation + weeks of training. Beyond student resources.
2. **Large Liar's Dice:** Require distributed training.

**Approach for CS230:**
- Implement ReBeL on Liar's Dice 1x4f as provided. Download/retrain only value and policy networks (data generation provided).
- Ablate: remove policy network, reduce search depth, reduce network size. Measure impact on exploitability.
- Test on 1-2 humans or bots (if available).

---

## 11. Compute and Engineering

**Hardware:**
- **Data generation for HUNL:** 90 DGX-1 machines, each with 8 × 32GB Nvidia V100 GPUs (Appendix E, p. 18). (General capability: up to 128 machines with 8 GPUs each, per Sec. 7, p. 8.)
  - Total for HUNL: 90 × 8 = 720 V100 GPUs.
  - For Liar's Dice: Single GPU for training + 60 CPU threads for data generation (Appendix E, p. 17).
- **Training:** Single machine with 8 GPUs; data generation was the bottleneck (Sec. 7, p. 8).

**Training duration:**
- **HUNL:** 1,750 epochs (Appendix E, p. 18). Epoch = 2.56M examples, batch size 1,024. ~2-3 weeks for full HUNL (estimated; exact wallclock time not specified).
- **TEH:** 300 epochs (faster, smaller game). ~3–5 days.
- **Liar's Dice:** 1,000 epochs (Appendix E, p. 17). ~1 week on 1 GPU.

**Data volume:**
- **HUNL:** 1,750 epochs × 2.56M examples/epoch = 4.48 billion training examples.
- **TEH:** 300 epochs × (unstated) examples/epoch.
- **Liar's Dice:** 1,000 epochs × 25,600 examples/epoch = 25.6M examples.

**Parallel actors:**
- **Data generation parallelized:** CFR + network evaluation in parallel across 128 machines (HUNL) / 1 machine (Liar's Dice).
- **Bottleneck:** Sequential CFR and leaf evaluation; parallelism limited.
- Sec. 7 (p. 8): "We found data generation to be the bottleneck due to the sequential nature of the FP and CFR algorithms and the evaluation of all leaf nodes on each iteration."

**Memory:**
- Replay buffer: 12M samples for HUNL (Appendix E, p. 18: "circular buffer of size 12M").
- Network parameters: 6 layers × 1,536 units = ~24M parameters (rough estimate; actual depends on layer connections).
- Typical: <1GB per network.

**Distributed training:**
- Not explicitly distributed (single machine for training).
- Data generation distributed but asynchronous (generate → store in buffer → train when ready).

**Implementation tricks:**
1. **Subgame caching (Appendix E.1, p. 18):** "Bot cached subgames it encountered on the preflop. When the same subgame was encountered again, it would simply reuse the solution it had already computed previously."
   - Trades memory for computation; speeds up test-time inference.
   
2. **Card embeddings (Appendix E, p. 17):** "We use card embedding for the board cards similar to [7]."
   - Reduces dimensionality of card representation before MLP.

3. **Linear quantization for policy network (Appendix E, p. 17):** "We apply linear quantization to the policy values."
   - Reduces size of target vector for policy network (which is up to 9× larger than value target).

4. **Warm-up data discarding (Appendix E, p. 18):** "Remove half of the data from the replay buffer after 20 epochs."
   - Data generated with random initial network is poor; discard it to improve training.

5. **Variance reduction during evaluation (Appendix E.1, p. 18):** "Variance was reduced by using AIVAT [17]."
   - Applied only to human evaluation; bot comparisons don't use AIVAT.

### Reproduction Difficulty: **Hard / Unrealistic for a quarter**

**Justification:**

- **Full HUNL:** Requires 720+ V100-equivalents and weeks of compute. Beyond typical student access (usually 1–4 GPUs).
- **Policy complexity:** CFR + neural network bootstrapping + self-play loops requires careful implementation. Standard RL libraries don't provide templates.
- **Convergence diagnostics:** Needs careful monitoring of exploitability, loss curves, value network error to catch bugs.
- **Computational bottleneck:** Data generation (CFR) not easily parallelizable on limited GPUs; single machine becomes slow.

**Realistic scaled-down version:**

1. **Liar's Dice 1x4f as proof-of-concept** (~1–2 weeks):
   - Network smaller (2 layers, 256 units).
   - Fewer epochs (100–300).
   - Single GPU.
   - Achieve exploitability of ~0.05–0.1 (vs. paper's 0.017).
   
2. **Modified poker:**
   - Toy 2-card poker (3-card deck) with simplified betting.
   - Network smaller.
   - Depth-1 subgames only.
   - ~1–2 weeks.

3. **Hybrid approach:**
   - Use paper's released Liar's Dice code as skeleton.
   - Modify to add policy network or explore different CFR variants.
   - Measure impact on convergence.

**Time estimate:** 2–4 weeks of part-time work (including debugging, baseline runs, analysis).

---

## 12. Limitations

### A. Explicitly mentioned by the authors

1. **Input size scales with infostates per PBS (Conclusion, p. 9):** 
   - "The input to its value and policy functions currently grows linearly with the number of infostates in a public state."
   - Becomes intractable in games with strategic depth but little common knowledge (Recon Chess: hidden opponent pieces).
   - **Impact:** Dou Dizhu has ~10^100–10^300 possible card configurations per public state (rough); infeasible.

2. **Theory limited to 2-player zero-sum (Conclusion, p. 9):**
   - "ReBeL's theoretical guarantees are also limited only to two-player zero-sum games."
   - Empirical success in multiplayer noted (Pluribus [14]) but not analyzed.
   - **Impact:** Dou Dizhu is 3+ players and not zero-sum (variable-sum with point system).

3. **Common knowledge assumption removed only at test time (Sec. 6, p. 7):**
   - Training assumes policies are common knowledge; test-time addresses this via random iteration sampling.
   - If opponent's true policy far from training distribution, value network may fail.
   - **Mitigation:** Test-time search still principled (Theorem 3), but approximation error compounds.

### B. Inferred from the methodology

1. **Bootstrapping circularity (self-play stability):**
   - Value network trained on CFR outputs; CFR uses network as leaf values.
   - If network is bad initially, CFR produces bad subgame solutions → poor training targets → network doesn't improve.
   - [**Paper does not analyze this**] Convergence proof (Theorem 2) assumes "perfect approximator"; real networks don't achieve this.
   - **Mitigation:** Exploration (ε=0.25) and replay buffer (12M samples) help, but rigorous analysis lacking.

2. **Non-stationarity in self-play:**
   - Both players use same network; as network improves, opponent's behavior changes.
   - Can cause instability or slow convergence (moving target).
   - [**Paper does not discuss**] No opponent pool or periodic checkpointing to stabilize training.
   - **Mitigation:** Large replay buffer mixing old and new data; CFR's robustness to value function errors.

3. **Test-time safe search still imperfect (Theorem 3):**
   - Error bounded by $\delta C_1 + \delta C_2 / \sqrt{T}$.
   - If $\delta$ (value network error) is large, guarantee is loose.
   - Random iteration sampling may select very early (poor) iterations (Sec. 6, p. 8).
   - **Mitigation:** Modern equilibrium-finding algorithms (Linear CFR) weight early iterations less.

4. **Limited diversity in training data:**
   - No opponent pool; always plays against same (self) policy.
   - Opponent's strategy may overfit to self-play behavior (e.g., always bluff in certain spots).
   - Real human opponents play differently; could exploit trained agent.
   - [**Paper does not analyze**] No experiments against diverse opponents or adversarially trained agents.

5. **Scalability of CFR:**
   - CFR is inherently sequential; each iteration updates all infostates.
   - Parallelization limited (Section 7, p. 8: "bottleneck is sequential nature of CFR").
   - For large games, even one CFR iteration takes long.
   - **Impact:** Depth-limited subgames necessary, but introduce approximation error.

6. **Information abstraction not used (vs. prior work):**
   - Paper avoids hand clustering; each hand distinct (1,326 per player).
   - Network input large (2,600+ dims) but manageable.
   - Approach doesn't scale to even larger info sets (e.g., Dou Dizhu with ~10^300 configurations).
   - [**Interpretation**] Advantage is no information loss; disadvantage is no dimensionality reduction.

7. **Bootstrapping bias:**
   - Leaf values are estimates (from network), not exact.
   - This introduces bias into CFR's value estimates.
   - Trade-off: bias vs. variance (Sec. 5.2, p. 6). Paper acknowledges but doesn't analyze quantitatively.

---

## 13. Future Work

### Explicitly mentioned by the authors

1. **Extend to multiplayer games (Conclusion, p. 9):**
   - Theory limited to 2-player zero-sum; empirical success in multiplayer (Pluribus) but not studied for ReBeL.
   - **Classify as:** Directly usable as CS230 project idea.
   - **Later work in set?** DouZero (paper 01) extends to 3-player; Suphx (paper 08, Mahjong) also multiplayer. Neither uses ReBeL approach.

2. **Handle games with strategic depth but little common knowledge:**
   - Recon Chess example (hidden pieces, few public observations).
   - Requires different state representation (not explicit belief distributions).
   - **Classify as:** Too ambitious / engineering-heavy for CS230.

3. **Scaling to billions of infostates per PBS (implied by Theorem 1, Appendix A, p. 14):**
   - "Theorem 1 proves that all of the algorithms described in this paper can in theory be conducted using only $V_1$, not $\hat{v}$."
   - Could learn single-value function instead of vector; reduces network output size.
   - **Classify as:** Mostly theoretical; implementation details unclear.

### Implied future work

[**Interpretation**]

1. **Opponent modeling / unknown opponent policies:**
   - Section 6 addresses test-time unknown policies via random iteration sampling.
   - Could improve by modeling opponent's likely policies (e.g., via Bayesian inference during play).
   - **Classify as:** Possible but difficult. Requires tracking beliefs over opponent policies (meta-beliefs).

2. **Hybrid RL+Search:**
   - Combine policy gradient (e.g., A3C) with search (ReBeL).
   - Could avoid bootstrapping circularity by using rollouts for some values.
   - **Classify as:** Directly usable; combines ReBeL with known RL techniques.

3. **Other equilibrium-finding algorithms:**
   - Paper tests CFR-D, CFR-AVG, Fictitious Play, Linear CFR.
   - Other algorithms: OSHW (optimistic smoothing), excessive gap technique, etc.
   - Could explore impact on convergence.
   - **Classify as:** Directly usable; mostly engineering.

4. **Dimensionality reduction for belief states:**
   - Instead of 1,326-dim hand distributions, learn compact embeddings (autoencoders, VAE).
   - Could enable scaling to larger info sets.
   - **Classify as:** Possible but difficult. Requires careful design to preserve strategic relevant info.

5. **Abstraction + ReBeL:**
   - Combine information abstraction (hand clustering) with ReBeL for mixed robustness.
   - Trade-off: smaller network vs. information loss.
   - **Classify as:** Directly usable; tests tradeoff.

6. **Safe search with stronger guarantees:**
   - Theorem 3 depends on value function error $\delta$; could be large in practice.
   - Develop tighter bounds or adaptive search strategies.
   - **Classify as:** Theoretical; unclear practical impact.

---

## 14. Connection to Other Papers

### Earlier papers in the set (build-up)

**NFSP (2016, paper 05):**
- Self-play neural network training for imperfect-information games.
- No search during training (only at test time with depth-1 rollout).
- Q-learning variant; no theoretical convergence guarantees for general games.
- **Relation to ReBeL:** ReBeL adds search during training + convergence proofs. Advancement over NFSP's test-only search.

**Deep CFR (2019, paper 06):**
- CFR algorithm using neural networks for value functions.
- Applied to single-agent game abstraction (simplified game, solves abstraction, transfers back).
- No self-play RL; value network trained on CFR outputs only (no policy network).
- **Relation to ReBeL:** 
  - [**Paper**] Related work (p. 2) mentions Deep CFR but doesn't cite it in detail.
  - [**Interpretation**] ReBeL's approach is complementary: Deep CFR + abstraction vs. ReBeL + explicit beliefs. Could be combined (ReBeL on abstracted game).
  - Difference: Deep CFR focuses on abstraction; ReBeL focuses on self-play.

**RLCard (2020, paper 07):**
- Toolkit for RL in card games.
- Provides environments (including Dou Dizhu) and baseline agents.
- No specific algorithm; infrastructure.
- **Relation to ReBeL:** If implementing ReBeL, could use RLCard environment instead of writing game engine.

**DeltaDou (2019, paper 04):**
- Self-play for Dou Dizhu using abstraction + imitation learning.
- No explicit belief-based search.
- 3-player, imperfect information, complex action space.
- **Relation to ReBeL:** 
  - Different approach (abstraction + imitation vs. belief states + CFR).
  - DeltaDou solves 3-player; ReBeL theory is 2-player zero-sum.
  - No mention of ReBeL in DeltaDou (published same time or before).

### Later papers in the set (extending ReBeL ideas)

**DouZero (2021, paper 01):**
- State-of-the-art for Dou Dizhu.
- Deep Monte Carlo (DMC) + self-play, no explicit belief search.
- 3-player, complex.
- **Relation to ReBeL:**
  - Extends self-play RL to 3-player but avoids belief-based search (scalability).
  - Uses "efficient rollout" (network predicts value, less compute than full rollout).
  - No mention of ReBeL or belief states.
  - [**Interpretation**] DouZero's scalability to 3-player suggests belief-based approach (ReBeL) may be overkill; simpler methods sufficient if training data large.

**DouZero+ (2022, paper 02):**
- Extends DouZero with opponent modeling and coach-guided learning.
- Doesn't use belief-based search.
- **Relation to ReBeL:** Similar spirit (opponent modeling) but different mechanism (no PBS).

**Suphx (2020, paper 08, Mahjong):**
- Self-play RL for multiplayer imperfect-information game.
- No explicit search during training; search at test time.
- **Relation to ReBeL:** 
  - Similar self-play paradigm but no belief-based search (Suphx uses different approach: policy network only, no value network).
  - Predates ReBeL (or concurrent); doesn't cite it.

### Analysis of lineage

[**Interpretation**]

```
NFSP (2016) ← Self-play RL for imperfect info
  ↓
Deep CFR (2019) ← CFR + networks for value functions
  ↓
ReBeL (2020) ← CFR + networks + self-play + beliefs + safe search
            (Combines ideas from NFSP + Deep CFR)
  ↓
DouZero (2021) ← Self-play RL for 3-player Dou Dizhu (avoids belief-based search for scalability)
  ↓
DouZero+ (2022) ← Opponent modeling on top of DouZero
```

**Evolution:**
- NFSP: self-play RL, no search during training.
- Deep CFR: search (CFR) + networks, no self-play RL.
- ReBeL: self-play RL + search + beliefs + convergence proofs.
- DouZero: abandons belief-based search (scales 3-player better without it).

**Key limitation ReBeL addresses:** NFSP and Deep CFR not theoretically sound for general imperfect-information games. ReBeL adds convergence guarantees.

**What DouZero abandons:** Explicit belief-based search. Dou Dizhu's complexity (3 players, large info sets) makes ReBeL's approach infeasible; simpler self-play RL sufficient.

---

## 15. Project Ideas for Me

### Idea 1: Belief-based Search on Simplified Dou Dizhu

**Research question:** Can explicit belief state representations improve convergence speed on a simplified 2-player 7-card Dou Dizhu variant?

**Baseline:** DMC (Deep Monte Carlo) agent trained via self-play on 7-card Dou Dizhu (simpler than full 54-card game).

**Modification:** 
- Add explicit belief features to network input (probability distribution over opponent's remaining cards).
- Implement depth-limited search (lookahead 1–2 cards) using simplified CFR on beliefs, then take max value as decision.
- Compare: DMC only vs. DMC + search vs. search only.

**Hypothesis:** Search will improve win rate and reduce convergence time by leveraging beliefs explicitly.

**Experiment:**
- Train both agents for same compute budget (~1 GPU-week).
- Evaluate vs. fixed opponents (random, rule-based) and vs. each other.
- Metric: win rate, convergence curve (loss over time).

**Metrics:** Win rate vs. baselines, number of games to convergence, value network accuracy on test set of belief states.

**Difficulty:** Moderate.
- Implementation: Use RLCard Dou Dizhu environment, add belief features to state representation, implement mini-CFR solver.
- Engineering: Integrating search into RL loop (scheduling, memory).

**Compute:** Medium (1–2 GPUs, 2–3 weeks).

**Novelty:** Applies ReBeL's belief-based approach to Dou Dizhu (prior work doesn't). Shows tradeoff between explicit belief modeling and computational cost on 3-player game.

---

### Idea 2: Safe Search Comparison in Imperfect-Information Games

**Research question:** How much does safe search (Theorem 3's random iteration sampling) improve over unsafe search in games where opponent's policy is unknown?

**Baseline:** Unsafe search (standard CFR at test time, assume equilibrium beliefs).

**Modification:** Implement both safe (random iteration) and unsafe search. Test on trained agents playing against humans or adversarial bot.

**Hypothesis:** Safe search will be less exploitable to adversarial opponents, even if both achieve similar exploitability vs. fixed opponents.

**Experiment:**
- Train agent on Liar's Dice using ReBeL.
- At test time, evaluate with safe vs. unsafe search.
- Test vs. fixed bot and vs. bot trained to exploit unsafe search.
- Metric: exploitability, win rate against adaptive opponents.

**Metrics:** Exploitability, win rate, robustness to novel opponents.

**Difficulty:** Easy.
- Implementation: Modify test-time search procedure (add flag for safe vs. unsafe).
- Theory: Already proven; just empirically validate.

**Compute:** Low (1 GPU, 1–2 weeks; mostly retraining on different configurations).

**Novelty:** Empirical validation of Theorem 3; shows practical importance of safe search in adversarial settings.

---

### Idea 3: Policy Network Warm-start Comparison

**Research question:** How much does a good policy network warm-start reduce CFR iterations needed for convergence?

**Baseline:** ReBeL without policy network (random uniform init).

**Modification:** 
- Train policy network via imitation learning on expert policies (e.g., Nash equilibrium policies from full-game solver or human expert).
- Use in ReBeL to warm-start CFR.
- Vary warm-start quality (random, imitation, expert) and measure convergence.

**Hypothesis:** Better warm-start will reduce CFR iterations and training time, but may limit exploration / create local optima.

**Experiment:**
- Three policy initializations: (1) uniform, (2) imitation from Nash, (3) imitation from human.
- Train agents for same number of epochs, measure exploitability over time.
- Metric: iterations to convergence, final exploitability, training time.

**Metrics:** Convergence speed, final performance, wall-clock training time.

**Difficulty:** Easy–Moderate.
- Requires expert policy data (can use full-game solver or prior checkpoints).
- Imitation learning straightforward (supervised learning on policy).

**Compute:** Medium (1–2 GPUs, 2–3 weeks).

**Novelty:** Ablates policy network importance; shows tradeoff between warm-start and exploration.

---

### Idea 4: Belief Representation Comparison

**Research question:** How do different belief representations (explicit distribution, hand rankings, abstractions) affect convergence and scalability?

**Baseline:** Full explicit belief distribution (1,326 dims per player in poker).

**Modification:** 
- Explicit rank distribution (e.g., "prob 0.3 opponent has high hand, 0.5 mid, 0.2 low").
- PCA-reduced belief embedding (compress to 50 dims).
- No explicit beliefs (DMC style; network learns implicitly).
- Hand clustering (information abstraction, reduce from 1,326 to ~100 clusters).

**Hypothesis:** Abstractions will speed up training but reduce value network accuracy. Sweet spot exists for Dou Dizhu's complexity.

**Experiment:**
- Train agents with each belief representation on Liar's Dice (varied sizes).
- Measure: network size, training time, final exploitability.
- Plot: Pareto frontier (accuracy vs. compute).

**Metrics:** Network size, training time, final exploitability.

**Difficulty:** Hard.
- Requires multiple representation implementations.
- Careful ablation to isolate belief representation effect.

**Compute:** High (2–4 GPUs, 4–6 weeks for multiple methods).

**Novelty:** Systematic comparison of belief representations. Informs choice of representation for Dou Dizhu.

---

## 16. Important Implementation Components

### MUST UNDERSTAND AND IMPLEMENT/MODIFY

1. **Public Belief State (PBS) construction and management:**
   - Compute joint distribution over infostates from public observations + player policies.
   - Bayesian update as actions are observed.
   - Compact representation (store only necessary fields, not full history).

2. **CFR (Counterfactual Regret Minimization) algorithm:**
   - Core subgame solver; essential to understanding and implementing ReBeL.
   - Iteratively compute policies to minimize regret.
   - Can use existing libraries (e.g., `pyker`, but limited) or implement from scratch.

3. **Neural network value and policy function approximators:**
   - MLP with GeLU, LayerNorm; standard Pytorch implementation.
   - Input: belief distributions + public state.
   - Output: value per infostate, policy per action per infostate.

4. **Self-play training loop:**
   - Recursive subgame solving starting at each state.
   - Data collection: store infostate values and policies from CFR.
   - Network training: batch gradient descent on collected data.

5. **Safe search at test time (Theorem 3):**
   - Run CFR at test time, pick random iteration's policy.
   - Implement this procedure correctly to achieve Theorem 3 guarantee.

### MUST UNDERSTAND CONCEPTUALLY

1. **Nash equilibrium and 2-player zero-sum game theory:**
   - Understand why Nash exists and is unique (in 2p0s).
   - Understand why value is well-defined per PBS.
   - Understand regret minimization → Nash convergence.

2. **Convex optimization and supergradients (Theorem 1):**
   - Why PBS values are concave in beliefs.
   - Why infostate values form supergradients.
   - (Not necessary to implement, but needed to understand soundness.)

3. **Proof sketches of Theorems 2 and 3:**
   - Understand error propagation and convergence rates.
   - Understand why random iteration sampling works at test time.
   - Needed to debug convergence issues.

4. **Game tree representation:**
   - How to represent extensive-form games (histories, infostates, public states).
   - How to traverse tree for CFR + value function evaluation.

5. **Variance reduction techniques (AIVAT):**
   - Conceptually understand why variance reduction matters in high-variance games.
   - (Implementation only needed for evaluation, not core algorithm.)

### CAN REUSE AS INFRASTRUCTURE

1. **RLCard game environments:**
   - Poker, Liar's Dice, Dou Dizhu implementations already provided.
   - Use to avoid writing game engine.
   - May need to adapt state representation to match ReBeL's belief-state format.

2. **CFR implementations from academic repos:**
   - Existing CFR solvers (e.g., in `pyker`, academic papers' code).
   - Can serve as starting point or reference implementation.
   - Caveat: most don't include neural network integration; need to modify.

3. **PyTorch standard components:**
   - MLP, optimization, data loading.
   - Standard practices (no custom layers needed for ReBeL).

4. **Existing Nash equilibrium solvers:**
   - For baseline comparison (e.g., `openspiel` library has full-game Nash solvers).
   - Useful for validation on small games.

5. **Replay buffer implementations:**
   - Circular buffer, experience replay.
   - Standard in RL libraries (e.g., `stable-baselines3`).

---

## 17. Questions I Should Ask My CS230 Staff

1. **Scope and compute:**
   - Can I access 2–4 GPUs for 4 weeks? (Determines feasibility: full HUNL unrealistic; Liar's Dice + ablations feasible.)
   - Are there compute grants available?

2. **Project focus:**
   - Should I focus on reproducing ReBeL exactly, or on an original variation (e.g., belief-based search for Dou Dizhu)?
   - How much novelty is expected for full credit?

3. **Game environment:**
   - Can I use RLCard? (Simplifies implementation.)
   - If I implement custom Liar's Dice, what's acceptable scope?

4. **Evaluation and baselines:**
   - If I can't compare vs. prior agents (due to compute), can I compare vs. rule-based baselines or simple RL agents?
   - How rigorous should statistical testing be (e.g., multiple seeds, confidence intervals)?

5. **Theory vs. empirics:**
   - Do I need to re-prove convergence theorems, or can I assume ReBeL's proofs are correct?
   - How much mathematical rigor is expected in write-up?

6. **Checkpoint and progress milestones:**
   - When should I have a working demo (Liar's Dice agent)?
   - How should I structure mid-quarter check-in?

---

## 18. Beginner Concepts I Need to Learn

| Concept | Plain-language explanation | Where it appears in paper | Learn? |
|---------|-----|--------|---|
| **Nash equilibrium** | Policy profile where no player can improve by unilaterally switching. Both players are simultaneously playing best response to each other. | Sec. 3 (p. 3), Theorems 1–3. | NOW |
| **Imperfect information** | Players don't observe all game state; hidden cards or hidden opponent actions. | Sec. 1 (p. 1), running example throughout. | NOW |
| **Information set / infostate** | Equivalence class of game histories indistinguishable to a player (grouped by what the player has observed). | Sec. 3 (p. 3). | NOW |
| **Public belief state (PBS)** | Probability distribution over infostates, conditioned on public observations that all agents agree on. Compact representation of "what I think opponent might be holding." | Sec. 4 (p. 4), core algorithm. | NOW |
| **Extensive-form game** | Sequential game tree representation (vs. normal form / payoff matrix). Specifies all possible histories and outcomes. | Sec. 3 (p. 3), implicit in tree search. | NOW |
| **Subgame solving** | Solve depth-limited portion of game tree (future only), using leaf values from value function. Repeat from new root each turn (recursive). | Sec. 5.1–5.2 (p. 5–6), core algorithm. | NOW |
| **Counterfactual regret minimization (CFR)** | Iterative algorithm that minimizes each player's regret (how much better they could have played). Converges to Nash. Computes policy and value. | Sec. 5.1 (p. 6), Theorems 2–3. | SOON |
| **Self-play** | Agent plays against itself (or copies of itself); learns by observing its own wins/losses. Generates training data without external opponents. | Sec. 5.2 (p. 6), Algorithm 1. | NOW |
| **Value function** | Maps state to expected sum of future rewards. Approximates how good a state is. | Sec. 2–5 passim. | NOW |
| **Policy network** | Neural network that outputs action probabilities for each state. Used to guide search (initialization). | Sec. 5.3 (p. 6), Algorithm 1. | SOON |
| **Bootstrapping** | Using network's own predictions (leaf values) as training targets. Creates self-referential loop; works if network improves over time. | Sec. 5.2 (p. 6), whole algorithm. | SOON |
| **Belief distribution** | Probability distribution over hidden outcomes (opponent cards, dice). Updates via Bayes' rule as new public info arrives. | Sec. 4 (p. 4), core representation. | NOW |
| **Common knowledge** | Information known by all agents and known by all to be known, etc. (infinite nesting). Enables exact belief computation during training. | Sec. 3 (p. 3). | SOON |
| **Exploitability** | How much value an agent can extract by best-responding to your policy. Lower = stronger policy. | Sec. 7 (p. 8), Tables 2, Figure 2. | NOW |
| **Regret** | For a past decision, regret = (value of best action) - (value of actual action). Minimizing regret → Nash. | Sec. 5.1 (p. 6), Appendix F. | OPTIONAL |
| **Supergradient** | Generalization of gradient for non-smooth (concave) functions. Approximates tangent plane. | Theorem 1 (p. 4), Appendix F. | OPTIONAL |
| **Convexity / concavity** | Convex function: line between any two points lies above function (U-shaped). Concave: upside-down U. Affects optimization properties. | Sec. 4 (p. 4), Theorem 1. | OPTIONAL |
| **Depth-limited search** | Search only T steps into future, not entire game. Use value function for leaf values. Tradeoff: less accurate but tractable. | Sec. 5.1 (p. 6). | NOW |
| **Zero-sum games** | One player's gain = other's loss. Rewards sum to zero. Simplifies analysis (Nash is unique value per state). | Sec. 3 (p. 4). | NOW |
| **Two-player zero-sum (2p0s)** | Exactly two players; rewards sum to zero. ReBeL theory proven for this class; empirical results mention >2 players as future work. | Sec. 3 (p. 4) passim. | NOW |

---

## 19. Paper Scorecard

> **Cross-paper calibration:** this scorecard looks at this paper alone. Ranked against the whole pack for this Dou Dizhu project, the recommended verdict is **REFERENCE ONLY**. See `00_SYNTHESIS_AND_RECOMMENDATION.md` §20.9.

| Dimension | Score | Comments |
|-----------|-------|----------|
| **Importance for my project** | 4/5 | Core paper on belief-based RL+search for imperfect-information games. Directly applicable to Dou Dizhu (though needs adaptation for 3+ players). Algorithms and theory form backbone of belief-based approach. |
| **Difficulty to understand** | 3.5/5 | Accessible intro; theory (Theorems 1–3) requires convex analysis background. Algorithm 1 is clear. Implementation details sometimes sparse (e.g., exact PBS dynamics, CFR details deferred to appendix). |
| **Difficulty to reproduce** | 4.5/5 | Full HUNL unrealistic (720+ V100s). Liar's Dice feasible (1 GPU). Code released for Liar's Dice but not HUNL. CFR integration non-trivial. 2–3 weeks part-time for simplified version. |
| **Relevance to Dou Dizhu** | 3/5 | Directly relevant for belief-based search; major limitation is 2-player zero-sum theory (Dou Dizhu is 3+ player, variable-sum). Belief representation works; scalability unclear. |
| **Relevance to self-play** | 5/5 | Core contribution: self-play RL + search. Theorems ensure convergence. Self-play loop integral to algorithm. |
| **Relevance to imperfect information** | 5/5 | Solves the hard problem: RL+search in imperfect-information games. Central to paper's contribution. |
| **Potential for project ideas** | 4/5 | Enables multiple directions: belief representations, safe search empirics, warm-start ablations, scaling to >2 players. Some ideas (multiplayer, scaling) ambitious for quarter. |
| **Importance to read deeply** | 5/5 | Foundational for understanding how to combine RL, search, and beliefs. Theorems (esp. 1, 3) essential for soundness. |

### **Verdict: READ DEEPLY**

**Rationale:**
- Foundational paper for belief-based imperfect-information RL.
- Theory (Theorems 1–3) is novel and non-obvious; worth understanding carefully.
- Algorithm generalizes to Dou Dizhu (with modifications); ideas are transferable.
- Open-source Liar's Dice code makes reproduction feasible.

**Which sections to read first (if short on time):**
1. **Section 1 (intro) + Figure 1a–b:** Intuition for why perfect-information RL+Search fails in imperfect info.
2. **Section 4 (PBS concept):** Core insight; understand PBS before algorithm.
3. **Sections 5 + Algorithm 1:** Main algorithm; follow pseudocode.
4. **Theorems 1, 2, 3 + proofs in Appendix F, G:** Theory; skim proofs if short, understand statements and intuitions.
5. **Section 7 + Experiments (Tables 1–2, Figure 2):** Results and implications.
6. **Optional/later:** Appendices on CFR-AVG, FLOP, warm-start (engineering details).

**Skip if time-constrained:**
- Appendix H (Fictitious Play variant): nice-to-have, not core.
- Appendix I (CFR-AVG proof): theoretical; understanding the algorithm is sufficient.
- Broader impact section (p. 9): important for ethics but not algorithmically necessary.

---

## Appendix: Synthesis Facts (for cross-paper comparison)

- **Paper:** Combining Deep RL and Search for Imperfect-Information Games (ReBeL)
- **Year:** 2020
- **Environment(s):** Heads-up no-limit Texas hold'em poker (HUNL, ~10^160 infostates), Liar's Dice (4 variants, 100s–1000s of infostates), Turn Endgame Hold'em (TEH, ~2×10^11 infostates).
- **Main algorithm:** Self-play RL + Counterfactual Regret Minimization (CFR-D/CFR-AVG) depth-limited subgame solving with neural network leaf value bootstrapping; safe search at test time via random iteration sampling.
- **State representation:** Public belief state (PBS): joint probability distribution over both players' infostates (possible card holdings), conditioned on public observations and known policies.
- **Action representation:** Discrete actions (fold, call, raise at fixed bet sizes) in poker; discrete in Liar's Dice. Abstraction: ≤9 actions per decision point in poker.
- **Self-play?** Yes; both players use shared networks, trained via recursive subgame solving.
- **Opponent modeling?** No explicit opponent modeling; assumes known policies during training. At test time, safe search achieves equilibrium in expectation without knowing opponent policy.
- **Main loss:** Value network: Huber loss on CFR-computed infostate values (bootstrap targets). Policy network: MSE on average CFR policy.
- **Main contribution:** First general RL+Search algorithm with convergence proofs for 2-player zero-sum imperfect-information games. Novel safe search (Theorem 3) without additional constraints. Superhuman HUNL performance with minimal domain knowledge (no information abstraction).
- **Evaluation method:** Exploitability (Liar's Dice, TEH); head-to-head match win rate (HUNL vs. bots and human expert Dong Kim); variance reduction via AIVAT.
- **Compute:** 90 DGX-1 machines (720 V100 GPUs) for data generation + 1 machine for training. ~1,750 epochs for HUNL (4.48B examples), ~300 for TEH, ~1,000 for Liar's Dice. Total: weeks of compute for full HUNL.
- **Main limitation:** Input size scales linearly with number of infostates per PBS (intractable for high-information games like Recon Chess or Dou Dizhu). Theory limited to 2-player zero-sum.
- **Best idea I could borrow:** Explicit belief state representation as network input features (probability distributions over opponent hands). Use iterative equilibrium-finding (CFR or simpler variant) at test time to refine decisions. Combine with self-play RL for training.

