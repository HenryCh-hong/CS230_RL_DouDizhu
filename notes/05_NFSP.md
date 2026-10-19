# Deep Reinforcement Learning from Self-Play in Imperfect-Information Games (NFSP, 2016)
> Johannes Heinrich & David Silver (University College London) · arXiv:1603.01121v2, June 2016 · Tier B (Foundations of self-play in imperfect-information games) · **The key insight: two separate neural networks (best response + average strategy) + two memories (RL + SL) enable self-play learning to converge to Nash equilibrium in imperfect-information games where vanilla DQN fails.**

## 0. TL;DR for a CS230 Student

- **What it is:** Neural Fictitious Self-Play (NFSP) combines game-theoretic fictitious play with deep RL to learn Nash equilibria in poker games without prior domain knowledge.
- **Why you care:** In the authors' words, it is "the first deep reinforcement learning method known to converge to approximate Nash equilibria in self-play" (§6, p. 9) — "known" here means shown experimentally (in Leduc Hold'em), not proven for neural networks—a fundamental problem for multi-agent RL and card games.
- **The core idea:** Maintain two networks in parallel: (1) a Q-network trained on past games to compute best responses, and (2) a policy network trained to imitate your own historical average behavior. Mix between them during play; this stabilizes learning better than greedy DQN self-play.
- **The big problem it solves:** Vanilla DQN in self-play creates a death spiral—both players play deterministic, highly correlated greedy policies, so the data distribution is narrow and hostile to learning. NFSP's slower-changing average policy provides more diverse experience.
- **Key innovation #1 (algorithmic):** Reservoir sampling for the supervised-learning memory ensures the policy network learns a true historical average, not a recent snapshot.
- **Key innovation #2 (game-theoretic):** The anticipatory parameter *η* bridges continuous-time fictitious play theory to discrete-time play; it lets the agent learn best responses to a predicted future of opponent strategies.
- **Results:** On Leduc poker (small), NFSP reaches exploitability ~0.06 (near Nash). On Limit Texas Hold'em (real-world scale), it is competitive with superhuman programs trained on expert abstractions, despite starting from scratch (achieves ~-20 mbb/h vs. expert humans' 40–60 mbb/h).
- **For Dou Dizhu:** RLCard (2020) shows NFSP is feasible but struggles with sparse rewards and huge state/action spaces—it beats DQN but plateaus early. DouZero (2021) uses a different approach (Deep Monte-Carlo + distillation) and leapfrogs NFSP, suggesting NFSP alone may not be the end goal.

## 1. Paper Overview

### 1a. Beginner-friendly Explanation

Imagine two poker players learning by playing thousands of games against each other. The naive approach: each player trains a neural network to win as much as possible, updating it after every game. This sounds reasonable, but it often *fails* in poker because both players start with random strategies, then drift into a narrow set of behaviors. Once both are playing "greedy" (always picking the single best move), their game becomes highly predictable. The data the other player sees is very biased—lots of the same situations repeated—so learning slows or diverges.

NFSP solves this by doing something clever: each player maintains **two** strategies:
1. A **best-response strategy** (trained like DQN: learn to beat whatever the opponent is doing right now).
2. An **average-strategy** (a slow-moving imitation of all the best responses the agent ever tried).

During actual play, the agent flips a coin: with probability *η*, play the best response; otherwise, play the average strategy. This keeps the game less predictable. Mathematically, this is inspired by **fictitious play**, a game-theory idea from 1951, but adapted for deep learning and self-play.

Why does mixing help? The average strategy evolves *slowly* (trained on a big historical memory). So the opponent sees a steadier, more diverse stream of behaviors. The best response is trained on *this* slowly-changing opponent, so the data distribution is more stable. The result: both players drift toward **Nash equilibrium**, a strategy neither could improve against if the other stuck to their current plan.

### 1b. Technical Explanation

**Problem solved:** Prior work on imperfect-information games relied on domain-specific abstractions (e.g., hand-strength heuristics for poker) or expensive full-game traversal (game-theoretic solvers). NFSP is the first *end-to-end* deep RL method that learns approximate Nash equilibria from raw play without hand-crafted features.

**Why prior methods failed:**
- **TD-Gammon** and other RL approaches worked in perfect-information games but *diverged* in imperfect-information games. The reason: they learned deterministic, greedy policies, which are beatable in games requiring mixed strategies (random play).
- **DQN** in self-play suffers from two ills: (a) the greedy policy is exploitable, (b) both agents' ε-greedy exploration produces highly correlated, non-stationary data, so the Q-network sees a shifting target that never stabilizes.

**Main contributions:**
1. **Combination of FSP + deep RL:** Extends Fictitious Self-Play (Heinrich et al., 2015) from exact game trees to function approximation.
2. **Two-network architecture with off-policy RL:** A Q-network computes best responses; a policy network learns the average strategy.
3. **Two memories:** M_RL (circular buffer, recent experience) for RL; M_SL (reservoir sample, historical archive) for supervised learning of the average strategy.
4. **Anticipatory dynamics:** A parameter *η* that lets agents predict and respond to opponent drift, inspired by continuous-time game theory.
5. **Empirical proof:** First deep RL method to converge to Nash in a small poker game (Leduc); competitive with handcrafted methods on real-world poker (Limit Texas Hold'em).

**What came before:**
- Fictitious Play (Brown, 1951): agents alternate between computing best responses and averaging historical play.
- Fictitious Self-Play / FSP (Heinrich et al., 2015): extended FP to extensive-form games; but required exact backward induction—impractical for large games.
- Deep Q-Network (Mnih et al., 2015): revolutionary for single-agent RL, but provably fails in multi-agent imperfect-information settings.

**What limitation motivated NFSP:** DQN's failure in imperfect-information games (shown in Figure 1c of the paper) + the scalability mismatch between domain-knowledge-heavy game-theoretic solvers and learning-based approaches.

---

## 2. RL Formulation

| Component | Value/Description | Citation |
|-----------|-------------------|----------|
| **Agent(s)** | Multiple agents (typically 2 in experiments), each running NFSP | §3, Algorithm 1 |
| **Environment** | Extensive-form (sequential) game with imperfect information. Specifically: Leduc Hold'em (small) and Limit Texas Hold'em (real-world scale). | §4.1–4.3 |
| **State (game-theoretic)** | Information state $s_i^t$: player $i$'s private cards + observed action history. Not full game state (opponents' hidden cards unknown). | §2.2 |
| **Observation** | Player $i$ observes: (a) own cards (e.g., 6 cards in Leduc, 52-card deck), (b) betting/action history (tensor encoding), (c) revealed community cards. | §4.1, "domain-independent encoding" |
| **Action** | Discrete: {fold, call/check, raise/bet}. Betting cap: fixed number of raises per round (varies by game). | §4.1 |
| **Legal-action handling** | The paper does not specify this clearly. [Interpretation: assumed to be enforced by the game engine; network output is over all actions, but only legal actions are sampled.] | — |
| **Reward** | Zero-sum: one player's gain = another's loss. Payoff at terminal node (hand outcome, chips won/lost). Intermediate reward is zero; only terminal reward counts. | §2.1, §4.3 |
| **Return** | $G_t = \sum_{i=t}^{T} R_{i+1}$ (cumulative future rewards from time $t$ to end of game). Used to compute target for Q-learning. | §2.1, §5 |
| **Policy (average strategy)** | $\pi(s, a)$ (Π network output): probability distribution over actions; learned via supervised learning (log-likelihood of past best-response plays). | §3, Algorithm 1 |
| **Policy (best response)** | $\beta = \epsilon\text{-greedy}(Q)$: ε-greedy policy derived from Q-network. With probability ε, random action; else max Q-value. | §3, Algorithm 1 |
| **Value function** | $Q(s, a \mid \theta_Q)$: action-value function (DQN-style). Expected payoff of taking action $a$ in state $s$ and then following best response. | §3 |
| **Fully vs. partially observable** | **Partially observable.** Player sees own cards and action history, not opponents' cards. Information sets group states indistinguishable to an agent. | §2.2 |
| **Single vs. multi-agent** | **Multi-agent.** NFSP trains multiple agents simultaneously via self-play. All players run NFSP. | §3, §4 |
| **Cooperative/competitive/both** | **Competitive.** Zero-sum games (one player's gain = loss to others). No cooperation modeled. | §4 |
| **Self-play used?** | **Yes, essential.** All agents learn from simultaneous self-play. Each agent's M_RL is populated by play against other agents' current/historical strategies (mixed policy σ). | §3, Algorithm 1 |

**Critical detail:** NFSP is formulated for **two-player zero-sum extensive-form games**. Convergence guarantees (if any) apply to this class. [Interpretation] Dou Dizhu is 3-player, asymmetric roles (1 landlord vs. 2 peasants, cooperative peasants). NFSP's theory does NOT directly cover this; whether it applies is an open question.

---

## 3. Model / Architecture

**Network architecture (Leduc Hold'em):** [Paper, §4.1, p. 5]
- **Type:** Fully connected (multi-layer perceptron / MLP).
- **Layers:** 1 hidden layer of 64 neurons with ReLU activation.
- **Outputs:** Action probabilities (Π network, policy); action values (Q-network, best response).
- **Input shape:** Vector of length 30 (6 cards k-of-n encoded + 2 rounds, up to 2 raises per round, 3 actions).
- **Rationale:** "We manually calibrated NFSP" (p. 5). Experiments tested multiple architectures (8–128 hidden units); larger networks performed better.

**Network architecture (Limit Texas Hold'em):** [Paper, §4.3, p. 6]
- **Type:** Fully connected MLP.
- **Layers:** Four hidden layers with 1024, 512, 1024, 512 neurons, all ReLU.
- **Outputs:** Π and Q networks (same outputs, different parameters).
- **Input shape:** Vector of length 288 (52 cards k-of-n encoded over 4 rounds + 80-element betting history tensor).
- **Rationale:** Larger game required deeper architecture.

**Card representation (domain-independent):** [Paper, §4.1, p. 4–5]
- **k-of-n encoding:** Each round, if $k$ cards are revealed from an $n$-card deck, use a binary vector of length $n$, with 1s at revealed card indices. For Leduc (6 cards, 3 duplicates), each round is a length-6 vector.
- **Why not hand-crafted features?** The paper explicitly avoids poker-domain heuristics (card strength, pot odds, etc.). Encoded directly from game state.

**Action representation:** [Paper, §4.1, p. 4–5]
- **Betting history tensor:** For Limit Hold'em, 4D tensor (player, round, raise_number, action_type). Flattened to length 80 (2 players × 4 rounds × 5 raise slots × 3 actions, minus fold).
- **Output:** Networks produce a probability (Π) or Q-value (Q) for each action (3 actions in Leduc, typically 3 in Hold'em).

**Separate networks per role:** No. [Paper, §3, p. 3–4] A *single* NFSP agent runs at each player position. Both players (heads-up game) have their own Π and Q networks. No role-specific networks. [Interpretation: Each player's Q-network is trained to compute best responses *against the opponent's current mixed strategy*, not a specific role.]

**Recurrent vs. feedforward:** **Feedforward (no RNN/LSTM).** Networks process flattened information states as vectors. [Interpretation] The betting history is explicitly encoded in the input, so sequential memory via LSTM is not used. This is a limitation: LSTMs could help compress long sequences, but the paper does not explore this.

**Key design choice:** The network does *not* condition on $\eta$ (anticipatory parameter) at inference time. Instead, $\eta$ determines the *data distribution* (how often each policy is played) and is incorporated into the target for Q-learning ($\mathbb{E}_{\beta^i, \sigma^{-i}}[\cdots]$, not $\mathbb{E}_{\pi^i, \pi^{-i}}[\cdots]$).

---

## 4. Training Algorithm

**Full step-by-step pipeline:**

1. **Initialization:** 
   - Create game instance Γ (Leduc or LHE).
   - For each player $i$, initialize:
     - Replay memories: $M_{RL}$ (circular buffer, size 200k for Leduc, 600k for LHE) and $M_{SL}$ (reservoir, size 2M for Leduc, 30M for LHE).
     - Networks: $\Theta_Q$ and $\Theta_\Phi$ (random weights).
     - Target network: $\Theta_Q^{\text{target}} \gets \Theta_Q$.
     - Anticipatory parameter $\eta$ (fixed, typically 0.1).

2. **For each episode:**
   - Choose which policy to follow: with probability $\eta$, use $\epsilon\text{-greedy}(Q)$ (best response); else use $\Theta_\Phi$ (average strategy).
   - Label this as the "mixed policy" $\sigma$.

3. **For each timestep $t$ in the episode (up to terminal state):**
   - Player $i$ observes current information state $s_t$.
   - Sample action $a_t$ from policy $\sigma(s_t)$.
   - Execute $a_t$ in the game; receive reward $r_{t+1}$ and next state $s_{t+1}$.
   - **Store in M_RL:** transition tuple $(s_t, a_t, r_{t+1}, s_{t+1})$ (always, regardless of which policy was used).
   - **Store in M_SL:** behavior tuple $(s_t, a_t)$ only if the agent followed the best-response policy ($\sigma = \epsilon\text{-greedy}(Q)$).

4. **After each timestep, training step (stochastic gradient updates):**
   - **Supervised Learning (average policy):** 
     - Sample mini-batch of $(s, a)$ pairs from $M_{SL}$ (size 128 for Leduc, 256 for LHE).
     - Compute loss: $L_\Phi = \mathbb{E}_{(s,a) \sim M_{SL}} [-\log \Theta_\Phi(s, a)]$ (cross-entropy / negative log-likelihood).
     - Update $\Theta_\Phi$ via SGD (learning rate 0.005 for Leduc, 0.01 for LHE).
   
   - **Reinforcement Learning (best response):**
     - Sample mini-batch of $(s, a, r, s')$ tuples from $M_{RL}$ (size 128 for Leduc, 256 for LHE).
     - Compute target: $y = r + \max_{a'} Q(s', a' \mid \Theta_Q^{\text{target}})$.
     - Compute loss: $L_Q = \mathbb{E}_{(s,a,r,s') \sim M_{RL}} [y - Q(s, a \mid \Theta_Q)]^2$ (mean-squared error).
     - Update $\Theta_Q$ via SGD (learning rate 0.1 for Leduc, 0.1 for LHE).

5. **Periodically (every 300 updates for Leduc, 1000 for LHE):**
   - Copy target network: $\Theta_Q^{\text{target}} \gets \Theta_Q$ (stabilize Q-learning targets).

6. **Repeat until convergence** (measured by exploitability or win-rate metrics).

**Pseudocode (Algorithm 1 from paper, annotated):**
```
Initialize game Γ and execute RUNAGENT for each player

function RUNAGENT(Γ)
    Initialize M_RL ← circular buffer (size 200k–600k)
    Initialize M_SL ← reservoir (size 2M–30M)
    Initialize Π(s, a | θ_Π) with random θ_Π          // Average-policy network
    Initialize Q(s, a | θ_Q) with random θ_Q          // Best-response Q-network
    θ_Q^target ← θ_Q                                  // Target network for stability
    Initialize η (anticipatory parameter, e.g., 0.1)
    
    for each episode do
        // Choose policy: mix between best response and average strategy
        σ ← ε-greedy(Q) with probability η
             Π        with probability (1 - η)
        
        Observe initial state s₁ and reward r₁
        
        for t = 1 to T do
            Sample action aₜ from policy σ
            Execute aₜ; observe rₜ₊₁, sₜ₊₁
            Store (sₜ, aₜ, rₜ₊₁, sₜ₊₁) in M_RL          // RL memory (all transitions)
            
            if agent played σ = ε-greedy(Q) then
                Store (sₜ, aₜ) in M_SL                  // SL memory (only on best-response plays)
            end if
            
            // Supervised learning: train Π to match past best responses
            Update θ_Π via SGD on L(θ_Π) = 𝔼_{(s,a)~M_SL} [-log Π(s, a | θ_Π)]
            
            // Reinforcement learning: train Q as DQN-style off-policy learner
            Update θ_Q via SGD on:
              L(θ_Q) = 𝔼_{(s,a,r,s')~M_RL} [r + max_{a'} Q(s', a' | θ_Q^target) - Q(s, a | θ_Q)]²
            
            Periodically: θ_Q^target ← θ_Q              // Refresh target network
        end for
    end for
end function
```

**Key design decisions:**
- **Two memories:** $M_{RL}$ is a circular buffer (fast write, recent data only); $M_{SL}$ is a reservoir sample (uniform historical distribution, but slower to converge). [Interpretation: RL needs recent data (opponent is changing), but SL needs a representative archive (to learn the true average).]
- **On-the-fly training:** Updates happen every step, not batched after the episode. This is online off-policy learning.
- **Exploration decay:** $\epsilon$ starts at 0.06 (Leduc) or 0.08 (LHE) and decays as $\epsilon(t) = \epsilon_0 / \sqrt{t}$ (inverse square-root decay).
- **No momentum:** "Vanilla SGD" (p. 5), no Adam or momentum. This is a 2016 choice; modern versions might benefit from Adam.

---

## 5. Mathematics

### MUST UNDERSTAND (for CS230)

**Equation 1: Realization-equivalent behavioral strategy**

$$\sigma(s, a) \propto \lambda_1 x_{\pi_1}(s) \pi_1(s, a) + \lambda_2 x_{\pi_2}(s) \pi_2(s, a) \quad \forall s, a$$

**Symbol definitions:**
- $\sigma(s, a)$: realization-equivalent behavioral strategy (how to play in information state $s$).
- $\lambda_1, \lambda_2$: weights in a convex combination (e.g., $\lambda_1 = 0.5, \lambda_2 = 0.5$ for 50-50 mix).
- $\pi_1, \pi_2$: two base normal-form strategies.
- $x_{\pi_i}(s)$: realization-probability of information state $s$ under strategy $\pi_i$. Mathematically, $x_{\pi_i}(s_i^t) = \prod_{k=1}^{t-1} \pi_i(s_i^k, a_i^k)$ (probability of reaching $s_i^t$).
- The denominator (not shown, implicit in $\propto$): $\lambda_1 x_{\pi_1}(s) + \lambda_2 x_{\pi_2}(s)$ (normalizing constant per information state).

**Intuition:** In an extensive-form game, you can't just take a probabilistic average of two strategies in normal form; you have to adjust by how likely each strategy was to reach this state. Otherwise, you'd double-count actions from strategies that rarely reached this point.

**Why needed:** Fictitious play, in principle, averages all past best responses. Equation (1) is the *realization-equivalent* way to compute a mixed strategy that represents this average in behavioral form (ready to play in the game).

**Relation to RL:** This is pure game theory. It doesn't directly appear in the algorithm, but it *motivates* why the SL memory must track historical best-response plays with the correct frequency distribution (via reservoir sampling).

[Paper, §2.3, p. 3; also cited from Heinrich et al. 2015 and Von Stengel 1996]

---

**Equation 2: Q-function target (DQN-style Bellman)**

$$y = r + \max_{a'} Q(s', a' \mid \theta_Q^{\text{target}})$$

**Symbol definitions:**
- $y$: bootstrapped target (what we want the Q-function to predict).
- $r$: immediate reward (in poker, usually 0 except at terminal state).
- $s'$: next state.
- $\max_{a'} Q(\cdot)$: greedy action value at next state (used in standard DQN; note: NFSP uses this for computing best response, not for behavioral policy selection).
- $\theta_Q^{\text{target}}$: parameters of the *target network*, updated periodically for stability.

**Intuition:** The Bellman backup tells us: if the environment is stationary, $Q(s, a)$ should equal the immediate reward plus the discounted value of the next state. We use a target network with frozen parameters to reduce the risk of chasing a moving target (which destabilizes learning).

**Why used:** Standard off-policy Q-learning technique. Reduces variance in the TD update.

**Relation to RL:** This is the core of DQN (Mnih et al., 2015), adapted to imperfect-information games.

[Paper, §3, Algorithm 1, p. 4; also referenced as "Fitted Q Iteration" and DQN literature]

---

**Equation 3: Q-learning loss**

$$L(\theta_Q) = \mathbb{E}_{(s,a,r,s') \sim M_{RL}} \left[ \left( r + \max_{a'} Q(s', a' \mid \theta_Q^{\text{target}}) - Q(s, a \mid \theta_Q) \right)^2 \right]$$

**Symbol definitions:**
- $L(\theta_Q)$: loss (MSE between predicted and target Q-values).
- $\mathbb{E}_{(s,a,r,s') \sim M_{RL}}$: expectation over transitions sampled from the RL memory.
- The squared term: mean-squared error (MSE).

**Intuition:** Minimize the difference between what the network predicts and the Bellman target. Smaller loss = better predictions.

**Implementation:** Each training step, sample a mini-batch of $(s, a, r, s')$ tuples, compute the MSE, backpropagate.

[Paper, §3, Algorithm 1, p. 4; also standard DQN loss]

---

**Equation 4: Supervised learning loss (average policy)**

$$L(\theta_\Phi) = \mathbb{E}_{(s, a) \sim M_{SL}} \left[ -\log \Theta_\Phi(s, a) \right]$$

**Symbol definitions:**
- $L(\theta_\Phi)$: loss (cross-entropy / negative log-likelihood).
- $\Theta_\Phi(s, a)$: probability of action $a$ in state $s$ according to the policy network.
- $\mathbb{E}_{(s, a) \sim M_{SL}}$: expectation over $(s, a)$ pairs sampled from the SL memory.
- $-\log \Theta_\Phi(s, a)$: cross-entropy. Minimizing this makes the network assign high probability to observed actions.

**Intuition:** The average-policy network learns to imitate the distribution of actions the agent took *when playing best-response*. It's supervised learning: "given this state, try to predict the action the agent historically took."

**Why this objective?** We want the average policy to converge to a mixture of all past best responses, weighted by frequency. A cross-entropy / log-likelihood loss naturally encourages the network to match the empirical action distribution.

[Paper, §3, Algorithm 1, p. 4; also called "behavior cloning" or "imitation learning"]

---

**Equation 5: Anticipatory dynamics (continuous-time)**

$$\pî_t^{-i} + \eta \frac{d\pî_t^{-i}}{dt}$$

**Symbol definitions:**
- $\pî_t^{-i}$: opponent's average strategy at time $t$ (normal-form representation).
- $\eta$: anticipatory parameter (tuned per game; typical value 0.1).
- $\frac{d\pî_t^{-i}}{dt}$: rate of change of opponent's strategy (derivative).

**Intuition:** Instead of computing a best response to the opponent's *current* average strategy, compute it to a prediction of where they'll be: current + (step size) × (change direction). This is inspired by continuous-time dynamic fictitious play (Shamma & Arslan, 2005).

**In discrete time (NFSP's approximation):**

$$\pî_t^{-i} + \eta (\betâ_{t+1}^{-i} - \pî_t^{-i})$$

or equivalently,

$$\sigma^{-i} = \pî^{-i} + \eta (\betâ^{-i} - \pî^{-i}) = (1 - \eta) \pî^{-i} + \eta \betâ^{-i}$$

**Interpretation:** The agent computes a best response to a weighted average of the opponent's average strategy and their latest best response. The weight $\eta$ controls how much we "look ahead." High $\eta$ = aggressive extrapolation; low $\eta$ = conservative.

**Why needed:** Without anticipation, an agent learning to best-respond to the opponent's average would not generate experience with its own best response (since it never plays it if $\eta = 0$). The anticipation ensures that the agent mixes in enough best-response play to collect data for the SL network.

**Theory:** Shamma & Arslan (2005) proved that for appropriate $\eta$ (game-dependent), continuous-time anticipatory fictitious play converges faster to Nash and is more stable. NFSP uses $\eta$ as a heuristic parameter.

[Paper, §3, p. 4, and §3 background on Shamma & Arslan 2005]

---

**Equation 6: NFSP's mixed policy**

$$\sigma \equiv (1 - \eta) \hat{\pi} + \eta \hat{\beta}$$

**Symbol definitions:**
- $\sigma$: policy actually played during the game (mixture).
- $\hat{\pi}$: the agent's average strategy (network $\Theta_\Phi$).
- $\hat{\beta} = \epsilon\text{-greedy}(Q)$: the agent's best-response policy (network $\Theta_Q$, ε-greedy).
- $\eta$: anticipatory parameter (e.g., 0.1).

**Intuition:** With probability $\eta$, act greedily (best response); with probability $1 - \eta$, sample from the learned policy (average strategy). This is a *deterministic mixture*: we commit to the mix before each episode.

**Why this form?** This ensures the opponent sees data generated from two sources: (a) a slowly-changing average strategy (gives stable targets for Q-learning), (b) an improving best response (allows progress toward Nash). The fixed $\eta$ means both learning signals are always present.

[Paper, §3, Algorithm 1, p. 3–4]

---

### OPTIONAL / ADVANCED

**Equation 7: Q-learning in off-policy multi-agent setting**

$$Q^i(s, a) \approx \mathbb{E}_{\beta^i, \sigma^{-i}} \left[ G_t \mid S_t = s, A_t = a \right]$$

**Interpretation:** The Q-function estimates the expected return under the agent's best-response policy $\beta^i$ and the opponents' mixed strategy $\sigma^{-i}$ (which is itself a mixture of their average and best response, updated every step). This is *off-policy* because the agent's behavior policy (the mixed $\sigma$) differs from the target policy ($\beta^i$ against $\sigma^{-i}$).

**Why advanced?** Off-policy learning is tricky in multi-agent settings because the environment (opponent) is non-stationary. The paper doesn't provide a detailed convergence proof, so this remains somewhat heuristic.

[Paper, §3, p. 3; discussed but not formalized with full mathematical rigor]

---

**Equation 8: Realization-probability (used in game theory)**

$$x_{\pi^i}(s_i^t) = \prod_{k=1}^{t-1} \pi^i(s_i^k, a_i^k)$$

**Symbol definitions:**
- $x_{\pi^i}(s_i^t)$: probability that player $i$'s strategy $\pi^i$ reaches their information state $s_i^t$.
- $\prod_{k=1}^{t-1}$: product over all timesteps up to $t$.
- $\pi^i(s_i^k, a_i^k)$: probability of taking action $a_i^k$ in state $s_i^k$ under strategy $\pi^i$.

**Intuition:** To reach information state $s_i^t$, player $i$ must have taken a specific sequence of actions, each with some probability under $\pi^i$. The realization-probability is the product of all these individual probabilities.

**Why needed?** In game theory, when computing mixed strategies, you must account for the probability of reaching each state. A strategy that wins often but is never played doesn't contribute to the average.

**Relation to RL:** This is pure game-theoretic notation. In NFSP, we approximate this by reservoir sampling: store best-response actions with equal weight, so the empirical distribution naturally reflects realization probabilities.

[Paper, §2.3, p. 3, citing Von Stengel 1996]

---

**Equation 9: Fictitious play update (normal form)**

$$\hat{\pi}_T^i = \frac{1}{T} \sum_{t=1}^{T} \betâ_t^i$$

**Symbol definitions:**
- $\hat{\pi}_T^i$: average strategy for player $i$ after $T$ rounds.
- $\betâ_t^i$: the best response player $i$ computed in round $t$.
- $\frac{1}{T}$: uniform average.

**Intuition:** In classic fictitious play, each round a player computes a best response to the opponent's current average strategy. Then, the player's own average strategy is updated to include this new best response. Over many rounds, both players' averages converge to Nash.

**NFSP's approximation:** Instead of exactly computing and averaging best responses, NFSP trains a neural network to learn $\hat{\pi}$ via supervised learning on the historical plays (stored in $M_{SL}$), and uses another neural network ($Q$) to approximate the best response. Reservoir sampling ensures the SL network sees a representative historical distribution.

[Paper, §2.3 and §3, p. 3–4; introduced by Brown (1951)]

---

## 6. Loss Function

**NFSP uses *two* loss functions, one for each network:**

### Q-Network Loss (Best-Response Training)

**Loss:** Mean-squared Bellman error
$$L_Q(\theta_Q) = \mathbb{E}_{(s,a,r,s') \sim M_{RL}} \left[ \left( r + \gamma \max_{a'} Q(s', a' \mid \theta_Q^{\text{target}}) - Q(s, a \mid \theta_Q) \right)^2 \right]$$

**Where:**
- **Target:** $y = r + \gamma \max_{a'} Q(s', a' \mid \theta_Q^{\text{target}})$ (Bellman bootstrap).
- $r$: immediate reward (in poker, often 0 except at terminal node, where $r$ is the payoff).
- $\gamma$: discount factor (the paper does **not specify** $\gamma$ for poker; [Interpretation: likely $\gamma = 1$ for episodic finite-horizon games like poker, where the episode always terminates in finite time]).
- **Target network:** $\theta_Q^{\text{target}}$ is frozen; refreshed every 300–1000 updates. This stabilizes learning and reduces variance.
- **Optimizer:** Vanilla SGD (no momentum).
- **Learning rate:** 0.1 (Leduc), 0.1 (LHE).

**Why this loss?**
- Standard DQN objective. Minimizing Bellman error encourages the Q-function to satisfy the Bellman optimality equation.
- MSE is simpler to backprop than absolute-value loss; scales naturally with reward magnitude.
- Target network reduces the risk of chasing a moving target (non-stationarity from simultaneous opponent adaptation).

**Data source:** $M_{RL}$ (circular buffer of 200k–600k transitions). All transitions from play (regardless of which policy was used) go into $M_{RL}$.

**Bootstrap strategy:** Yes, bootstrapping via target network. This is a form of temporal-difference (TD) learning: each step, we estimate the Q-value based on one-step lookahead + estimated future value.

**[Paper]** Does not explicitly state why *not* to use Monte-Carlo targets (full episode return). [Interpretation] TD bootstrapping allows online updates and may provide better variance-bias tradeoff in large games. DouZero later explores Deep Monte-Carlo and finds it better for Dou Dizhu, suggesting NFSP's TD choice may not be optimal for all games.

---

### Policy Network Loss (Average-Strategy Training)

**Loss:** Cross-entropy (negative log-likelihood)
$$L_\Phi(\theta_\Phi) = \mathbb{E}_{(s, a) \sim M_{SL}} \left[ -\log \Theta_\Phi(s, a) \right]$$

**Where:**
- **Target:** The empirical action distribution stored in $M_{SL}$. The network learns to assign high probability to actions observed during best-response play.
- **Optimizer:** Vanilla SGD.
- **Learning rate:** 0.005 (Leduc), 0.01 (LHE).

**Why this loss?**
- Cross-entropy is the canonical objective for classification/imitation learning. It encourages the predicted probability $\Theta_\Phi(s, a)$ to match the target (observed action frequency).
- This is *supervised* learning (behavior cloning): "copy the historical actions of the best-response agent."

**Data source:** $M_{SL}$ (reservoir sample of ~2M–30M behavior tuples $(s, a)$). Tuples are added *only* when the agent played the best-response policy. This ensures the network learns the distribution of best-response actions, not the mixed policy.

**Memory strategy:** Reservoir sampling (Vitter, 1985).
- Why not circular buffer? A circular buffer (recent window) would cause the policy network to drift toward the *most recent* best response, not the historical average. Reservoir sampling ensures uniform weight to all historical best responses, approximating the true average.
- Trade-off: Slower convergence (no recent-data bias) but more stable and theoretically sound (approximates Equation 9's averaging).

[Paper, §3, p. 3–4, and Appendix discussion on XFP robustness]

---

**Why two losses?**
- **Separation of concerns:** $Q$ is trained on all data to become a good best-response learner. $\Phi$ is trained on a curated subset (best-response plays only) to become a good average-strategy learner.
- **Stability:** If we trained a single network on both tasks, the best-response signal (RL) and averaging signal (SL) would conflict. Separate networks allow each to specialize.
- **Theory:** Fictitious play theory requires maintaining both a best response (moving target) and an average (stationary target). Having separate networks with separate objectives mirrors this duality.

---

**Comparison to alternatives:**
- **Pure DQN:** Trains only a Q-function to maximize reward. In self-play, DQN agents learn deterministic policies, which are exploitable and create non-stationary data. [Paper, §4.2, Figure 1c shows DQN diverges.]
- **Policy gradient (e.g., A3C):** Trains a policy directly to maximize return. Also unstable in self-play. [Paper does not test this, but references argue RL methods generally fail.]
- **Deep CFR:** Regret minimization (different paradigm, discussed in later papers). NFSP is simpler and more scalable for large games.

---

## 7. Self-Play

**Who plays whom:** All agents play all other agents. In heads-up poker (2 players), both players run NFSP and play each other repeatedly in episodes.

**Simultaneous updates?** [Paper, §3, p. 3] "All players of the game are controlled by separate NFSP agents that learn from simultaneous play against each other." Yes, simultaneous updates. Both agents update their Q- and Π-networks *every step*, so both are non-stationary.

**How do they handle non-stationarity?** Via the mixed policy $\sigma$ and the anticipatory parameter $\eta$. The opponent's average strategy changes slowly (reservoir-sampled SL memory), so the agent's Q-learning target is relatively stable. The agent's own best response improves, but is moderated by the fixed $\eta$ (not the greedy policy, but a mix).

**Parameter sharing?** No. Each agent has its own Q-network and Π-network. In a 2-player game, there are 4 networks total. [Interpretation] This makes sense: Player A's best response to Player B is different from Player B's best response to Player A.

**Old checkpoints / opponent pool?** The paper does **not specify** how often (if ever) to save and replay against old checkpoints. [Interpretation] In classic fictitious play, agents play the opponent's historical average, which is implicitly an averaged pool. In NFSP, the opponent's mixed policy $\sigma$ is (1 - η)$\pi$ + η$\beta$, where $\pi$ is the long-term average. So NFSP doesn't explicitly maintain an opponent pool; instead, the opponent's average policy acts as a proxy for a pool of past selves.

**Exploration:** Via $\epsilon$-greedy. Starting exploration $\epsilon = 0.06$ (Leduc) or $\epsilon = 0.08$ (LHE), decaying as $1/\sqrt{t}$. This decays to 0 as training progresses, so the best-response policy becomes (nearly) deterministic, while the average policy remains stochastic.

**Non-stationarity handling (key insight):**
- **DQN self-play problem:** Both agents' $\epsilon$-greedy policies are highly correlated (both follow similar patterns) and drift together. The data distribution is narrow and shifts non-stationarily.
- **NFSP solution:** Agents mix in the average strategy (which changes slowly) with the best response (which improves). The average strategy is effectively a "slow player" that provides stable, diverse experience. The data distribution is wider (more diverse plays from the average policy) and changes more slowly (due to reservation sampling).
- **Anticipatory dynamics:** The parameter $\eta$ ensures that each agent generates enough best-response data to train the Π network, while still providing diverse mixed play to the opponent.

[Paper, §3, p. 4: "Thus, their experience varies more smoothly, resulting in a more stable data distribution, and therefore more stable neural networks."]

---

**Convergence to Nash:** [Paper, §4.1, p. 5, Figure 1a] In Leduc Hold'em, NFSP reaches exploitability ~0.06, which the paper notes "full-width XFP typically achieves after around 1000 full-width iterations." This is strong empirical evidence of convergence, but [Paper does not provide theoretical convergence guarantee]. The appendix (§A) shows that adding noise to XFP (to approximate NFSP's stochastic gradient updates) does not prevent convergence, suggesting NFSP inherits FP's convergence properties under function approximation.

---

## 8. Imperfect Information

### How hidden information is handled:

**Explicit representation:** [Paper, §4.1, p. 4–5] Each agent's input is an **information state**: a vector encoding what that agent observes.
- Private cards: k-of-n encoding (binary vector, 1 if card is known, 0 otherwise).
- Public cards (community cards in Hold'em): similarly k-of-n encoded.
- Action history: betting history tensor (player, round, raise_number, action_type).
- **Crucially:** No explicit distribution over hidden cards; no belief state; no inference.

**Action history as proxy for beliefs:** [Interpretation] The action history implicitly carries information about opponents' hands. E.g., if an opponent aggressively raises, the network can (in principle) learn to infer they have a strong hand. The network learns this implicit inference from data, not via explicit inference algorithms.

**Why not explicit opponent modeling?** [Paper does not discuss] Presumably for simplicity and to avoid domain knowledge. The paper emphasizes learning "directly from raw inputs" without "engineering abstractions."

**Information sets:** [Paper, §2.2, p. 2] "Information sets group states indistinguishable to an agent." NFSP does **not** explicitly group information sets. Instead, the information state (cards + action history) acts as the unique identifier. Two states are different if they have different action histories or different private cards, even if they represent the same "strategic situation." [Interpretation] This is fine for neural networks: redundancy in the input space is not a problem if the network can learn to ignore it (e.g., via unsupervised feature learning).

**Imperfect recall:** [Paper, §2.2, p. 2] "We assume games with perfect recall." Agents never forget their information state. This is true for poker but not for all games.

### Observation vs. Hidden Information Table

| Aspect | Actually Observed by Agent | Hidden from Agent |
|--------|---------------------------|-------------------|
| **Own cards** | Yes. k-of-n encoded. | — |
| **Opponents' private cards** | No. | Yes. Inferred implicitly from action history. |
| **Community cards** | Yes (in Hold'em). Revealed progressively per round. | — |
| **Current round** | Yes. Implicit in action history. | — |
| **Previous actions (all players)** | Yes. Betting history tensor. | — |
| **Payoffs (before game end)** | No (zero until terminal node). | Yes. Determined by hidden cards. |
| **Opponents' strategies** | No. But mixed policy σ is known (fixed η). | Yes. But σ is stationary. |
| **Opponents' intentions** | No. | Yes. |
| **Chip stacks / hand strength** | Yes (observable from cards and history). | — |

---

**Why imperfect information is hard:** In imperfect-information games, a *pure strategy* (always take action X in state S) is often suboptimal. You need *mixed strategies* (randomize between actions). Vanilla DQN learns pure, greedy strategies, so it fails. NFSP learns a mixed average strategy, which is required.

**Handling via neural networks:** The network doesn't explicitly model opponents' hidden information. Instead, it implicitly learns conditional distributions. E.g., given the history, the network learns "with probability 0.7, the opponent has a strong hand." This emerges from experience, not from explicit probabilistic inference.

---

## 9. Evaluation

**Baselines:** [Paper, §4 and comparison experiments]
1. **Full-width XFP:** Exact fictitious play on the full game tree. Infeasible for large games but provides a theoretical benchmark.
2. **DQN (greedy policy):** Vanilla DQN, playing deterministically. Expected to fail.
3. **DQN (average strategy):** DQN + passive SL memory (Π network doesn't affect play). Intended to test whether averaging alone helps without the anticipatory dynamics.
4. **Variants of NFSP:** Removed components (circular buffer vs. reservoir for SL, high η = 0.5, exponentially-averaged reservoir) to isolate importance of each component.

**Evaluation metrics:**

### In Leduc Hold'em:
- **Primary:** Exploitability (units: milli-big-blinds per hand, mbb/h or just numeric).
  - Definition: "the expected average payoff that a best response profile achieves against it" (§4, p. 5).
  - An exploitability of 2δ yields at least a δ-Nash equilibrium.
  - Lower is better; 0 = Nash equilibrium.
  - Computed via best-response calculation (expensive, so not done frequently).
- **Secondary:** Win rate against random agents (not primary in Leduc, but shown in Figure 1a learning curves).

### In Limit Texas Hold'em:
- **Primary:** Win rate vs. top computer programs (mbb/h, including confidence intervals).
  - Evaluation: 25,000 hands of symmetric play per opponent.
  - Confidence intervals: "estimated standard error of each evaluation is less than 10 mbb/h" [Figure 2a].
  - Compared to: SmooCT (detailed in §4.3), plus ACPC 2014 top 3 (Hyperborean, SmooCT, escabeche).
  - Result: NFSP's greedy-average strategy achieves ~-20 mbb/h (slightly negative but competitive; expert humans achieve 40–60 mbb/h in online high-stakes games).
- **Additional:** Best response and greedy-average strategies tested separately.

### General notes:
- **No multiple seeds reported:** [Interpretation] Leduc results appear to be single runs. LHE results do not report multiple seeds, only confidence intervals from tournament play.
- **Role symmetry:** Heads-up poker is symmetric (dealer alternates), so single-seed is arguably OK, but multiple seeds would be better.
- **Evaluation scope:** Only heads-up (2-player) games. No 3-player or team games.

---

## Critical Evaluation

### Strengths:

1. **Convergence in imperfect-information games:** "First deep reinforcement learning method known to converge to approximate Nash equilibria in self-play" (§6, p. 9) — demonstrated experimentally in Leduc Hold'em, not proven. In the same experiments, DQN self-play (even with its behaviour averaged) failed to approach a Nash equilibrium (its exploitability stagnated), while NFSP's exploitability kept falling.

2. **Scalability without domain knowledge:** No hand-crafted features. Learns from raw cards + action history. Competitive with methods that use extensive domain expertise (e.g., poker bot abstractions).

3. **Clear algorithmic contribution:** The combination of two networks + two memories + anticipatory dynamics is elegant and well-motivated by game theory.

4. **Ablations:** Figure 1b isolates the importance of reservoir sampling, anticipatory parameter, and other components. Shows each matters.

5. **Theoretical grounding:** Built on well-understood game-theoretic foundations (fictitious play, Nash equilibrium, information sets). Not purely empirical.

---

### Weaknesses:

1. **Convergence guarantee scope:** [Paper, §5, p. 7] "In this work, we focused on imperfect-information two-player zero-sum games." The paper provides **no theoretical convergence proof**, only empirical evidence and an appeal to fictitious play theory. Convergence guarantees for FSP apply to two-player zero-sum extensive-form games, but NFSP's approximations (function approximation + stochastic gradient) may break this. The Appendix (§A) shows empirical robustness to noise, but this is not a proof.

2. **3-player games (e.g., Dou Dizhu) not addressed:** [Paper, §5, p. 7] Mentions cooperative potential games and suggests NFSP might work, but provides no evidence. Fictitious play convergence proofs for 3-player games are weaker or absent. **Critical limitation for Dou Dizhu.**

3. **Sparse rewards:** Both Leduc and LHE have rewards only at terminal nodes (end of hand), not intermediate steps. Poker games naturally have this structure, but it makes learning harder. The paper does not compare to reward shaping or dense-reward variants.

4. **Single-run experiments (Leduc):** Figure 1a appears to show one random seed per architecture. No error bars, no multiple seeds. Leduc is small enough that two runs might differ significantly.

5. **Limited LHE comparisons:** Only compared to ACPC 2014 programs (2 years old at time of publication). No comparison to contemporaneous deep RL methods (e.g., policy gradient approaches). No ablation on network depth (fixed to 4 hidden layers).

6. **Hyperparameter sensitivity:** The paper reports that "we manually calibrated NFSP" (p. 5). No systematic sensitivity analysis. How much do results depend on learning rate, η, memory size, target update frequency?

7. **No off-policy stability analysis:** NFSP is off-policy (Q-learning from mixed policy data, trained on best-response target). In multi-agent settings, off-policy stability is delicate. The paper provides no formal analysis, only empirical observations.

8. **Exploration decay:** ε decays to ~0, so the best-response policy becomes nearly deterministic. Does this hurt robustness? What if ε is kept constant?

9. **Information state design is manual:** While the paper avoids domain-specific features, the choice of what to encode (k-of-n for cards, tensor for betting history) is still hand-picked. [Interpretation] Automatic feature learning (e.g., via CNN or attention) might be better but would increase complexity.

10. **Scalability unknown:** Leduc is tiny (~1e6 states); LHE is medium (~1e17 game states, but maybe 1e8–1e9 information states). Dou Dizhu has ~1e20 information states. [Interpretation] It's unclear if NFSP scales beyond LHE. The RLCard paper (2020) tested NFSP on Dou Dizhu and found it "highly unstable" with "minor improvements during learning."

---

## What I Would Improve for a CS230 Project:

1. **Test on Dou Dizhu end-to-end:** RLCard showed NFSP struggles on Dou Dizhu. Why? Test hypotheses:
   - Is it the large action space (many ways to play a card set)?
   - Sparse rewards (rewards only at game end, no intermediate signal)?
   - Non-stationary multi-agent dynamics (3 players, role asymmetry)?
   - Network architecture (did they use enough capacity)?
   - Hyperparameters (learning rates, η tuned for poker, not Dou Dizhu)?

2. **Compare memory strategies:** Reservoir sampling (SL memory) vs. circular buffer—which is better for Dou Dizhu's longer episodes (poker hands are ~10-20 actions; Dou Dizhu hands can be 50+)? Does exponential averaging of the reservoir (as in LHE) help?

3. **Ablate on η:** The paper uses η = 0.1 (fixed per game). What if η decays? What if it's learned? The anticipatory parameter is crucial; more analysis would help.

4. **Add opponent modeling:** NFSP doesn't explicitly model opponents. What if you added opponent modeling (e.g., a separate network to predict opponent's next action)? Would this improve learning?

5. **Combine with policy gradient:** NFSP is Q-learning-only. Hybrid methods (critic + actor, A3C-style) might be more sample-efficient.

6. **Use modern deep RL tools:** Adam optimizer, layer normalization, experience replay prioritization—techniques developed post-2016. Would they help NFSP?

7. **Evaluate statistical significance:** Run multiple seeds, report confidence intervals or significance tests. Single runs hide variance.

---

## 10. Experiments and Ablations

### Experiment 1: Leduc Hold'em, varying network sizes

**What changed:** Network hidden-layer size: 8, 16, 32, 64, 128 neurons (1-layer MLP).

**Held constant:** Memory sizes (200k, 2M), learning rates (0.1, 0.005), target update frequency (300), η (0.1), exploration schedule (ε: 0.06 → 0).

**Metric:** Exploitability (lower = better; 0 = Nash).

**Result (Figure 1a, p. 6):** All sizes converge to Nash, but larger networks converge faster and to lower exploitability. 64 and 128 hidden neurons both reach ~0.06; 8 and 16 plateau higher (~0.15–0.2).

**Authors' conclusion:** "We observe a monotonic performance increase with size of networks. NFSP achieved an exploitability of 0.06, which full-width XFP typically achieves after around 1000 full-width iterations."

**Critical note:** No error bars; appears to be single run per architecture.

---

### Experiment 2: Leduc Hold'em, ablations (Figure 1b)

**What changed (four variants):**
1. "NFSP with sliding window SL memory": Replace reservoir with circular buffer for M_SL.
2. "NFSP with exponentially-averaging SL reservoir": Reservoir with exponential decay (recent samples weighted more).
3. "NFSP with anticipatory parameter of 0.5": Increase η from 0.1 to 0.5.
4. Baseline NFSP (128 hidden neurons, standard parameters).

**Held constant:** Same as Experiment 1.

**Metric:** Exploitability.

**Result (Figure 1b, p. 6):**
- Sliding-window SL memory: **diverges** (exploitability increases over time, goes off-chart).
- Exponentially-averaging reservoir: **noisy**, doesn't converge as well (plateaus higher).
- High η (0.5): **plateaus** at high exploitability (~0.1–0.15), does not improve past ~10k iterations.
- Baseline NFSP: converges smoothly to ~0.06.

**Authors' conclusion (p. 5–6):** "Figure 1b shows that these modifications led to decremental performance. In particular, using a fixed-size sliding window to store experience of the agents' own behaviour led to divergence... high anticipatory parameter of 0.5 that most likely violates the game-dependent stability conditions of dynamic fictitious play (Shamma and Arslan, 2005)."

**Critical note:**
- These ablations clearly show the importance of reservoir sampling and moderate η.
- But they don't isolate the contribution of each component quantitatively. E.g., how much of NFSP's success is due to two networks vs. anticipatory dynamics?
- No error bars; single runs.

---

### Experiment 3: Leduc Hold'em, comparison to DQN (Figure 1c)

**What changed:** Algorithm.
- **DQN (greedy):** Vanilla DQN, plays ε-greedy(Q), trained from self-play.
- **DQN (average strategy):** DQN + passive SL memory (Π network doesn't affect play, just observes). Implemented as NFSP with η = 1 (always play best-response, never average policy).
- **Baseline NFSP:** η = 0.1.

**Held constant:** Network size (128 hidden neurons), memory sizes (2M for SL memory / 2M reservoir in DQN average-strategy variant).

**Hyperparameter tuning for DQN:** "We trained DQN with all combinations of the following parameters: Learning rate {0.2, 0.1, 0.05}, decaying exploration starting at {0.06, 0.12} and reinforcement learning memory {2m reservoir, 2m sliding window}. We then chose the best-performing result of DQN."

**Result (Figure 1c, p. 6):**
- DQN (greedy): **diverges** sharply (exploitability increases to ~1).
- DQN (average strategy): doesn't converge; plateaus at high exploitability (~0.2–0.3) with noisy performance.
- NFSP: converges smoothly to ~0.06.

**Authors' conclusion (p. 5, §4.2):** "The problem is that DQN agents exclusively generate self-play experience according to their ε-greedy strategies. These experiences are both highly correlated over time, and highly focused on a narrow distribution of states. In contrast, NFSP agents use an ever more slowly changing (anticipated) average policy to generate self-play experience. Thus, their experience varies more smoothly, resulting in a more stable data distribution, and therefore more stable neural networks."

**Critical note:**
- This is the paper's strongest result: DQN provably fails in imperfect-information self-play; NFSP succeeds.
- But DQN was hyperparameter-tuned. Did they try other RL methods (policy gradient, actor-critic)? Probably not feasible in 2016.
- The DQN average-strategy variant (η = 1) confirms that the average strategy alone is not enough; you need the mixed play to generate good data for RL.

---

### Experiment 4: Limit Texas Hold'em, learning curves (Figure 2a)

**What changed:** Training time / number of iterations.

**Held constant:** Game (LHE, heads-up); evaluation (25,000-hand tournament vs. SmooCT), network size (4 layers, 1024–512–1024–512 neurons), all other hyperparameters as listed in §4.3 (p. 6).

**Metric:** Win rate vs. SmooCT (mbb/h). Lower (more negative) = worse.

**Evaluation schedule:** "While training, we periodically evaluated NFSP's performance against SmooCT from symmetric play for 25000 hands each."

**Result (Figure 2a, p. 7):**
- **Best-response strategy (deterministic):** Noisy performance, ranges from -50 to 0 mbb/h, no clear trend.
- **Greedy-average strategy (deterministic, choosing action with highest probability from Π):** Stable improvement, reaches ~-20 mbb/h (slightly negative but reasonable).
- **Average strategy (stochastic, sampling from Π):** Stable improvement, reaches ~-50 mbb/h (worse than greedy-average).

**Interpretation:** The greedy-average strategy (deterministic choices from the policy network) performs best. The average strategy (sampling) is more conservative (mixed). The best-response strategy is unstable and exploitable (greedy against a moving target).

**Statistical notes:** "The estimated standard error of each evaluation is less than 10 mbb/h." So error bars would be roughly ±10 mbb/h.

---

### Experiment 5: Limit Texas Hold'em, tournament vs. ACPC 2014 top 3 (Figure 2b and Table 3)

**What changed:** Opponent.

**Held constant:** NFSP training complete; final greedy-average strategy.

**Evaluation:** "10,000 games" (hands) per matchup, symmetric play. [Paper, Table 3, p. 6]

**Metric:** Win rate (mbb/h) against each opponent.

**Result (Table 3, p. 6):**
| Opponent | NFSP vs. Opponent |
|----------|-------------------|
| SmooCT | -52.1 ± 8.5 |
| Hyperborean | -17.4 ± 9.0 |
| escabeche | -13.6 ± 9.2 |

**Interpretation:** NFSP loses to all three, but:
- vs. Hyperborean: only -17.4 mbb/h (competitive; human experts achieve 40–60 mbb/h).
- vs. SmooCT: -52.1 mbb/h (large loss; SmooCT is strong).
- vs. escabeche: -13.6 mbb/h (smallest loss).

**Authors' conclusion (p. 6–7):** "NFSP achieves winrates similar to those of the top half of computer agents in the ACPC 2014 and thus is competitive with superhuman computer poker programs."

**Critical note:**
- The paper claims NFSP is "competitive with superhuman programs," but the numbers show NFSP loses to all three top programs.
- However, in the ACPC 2014, top programs also lose to each other by similar margins (mbb/h). So NFSP is in the ballpark of top programs, even if not the absolute best.
- The result is impressive because NFSP learned from scratch in a few days/weeks, whereas these programs incorporate years of domain knowledge.

---

### Experiment 6: Limit Texas Hold'em, three strategy variants

**What changed:** Strategy type.
1. Average strategy (sampling from Π): π(s, a).
2. Greedy-average strategy (deterministic): argmax_a Π(s, a).
3. Best-response strategy (deterministic): argmax_a Q(s, a).

**Held constant:** Training complete; same trained networks; evaluation vs. SmooCT.

**Metric:** Win rate (mbb/h).

**Result (Figure 2a, p. 7):**
- Best-response: -50 to 0 mbb/h (highly variable, exploitable).
- Greedy-average: ~-20 mbb/h (stable, best).
- Average: ~-50 mbb/h (more conservative, worse).

**Interpretation:** The greedy-average strategy is a good compromise: more deterministic than the average policy (less mixed) but more robust than the pure best-response (which overfits to the training data).

---

### Appendix Experiment A: Robustness of XFP (Figures 3a, 3b)

**What changed:** Variants of full-width XFP to mimic NFSP's approximations.

**Experiment A.1 (Figure 3a):** Constant stepsizes in average-policy updates.
- XFP normally uses stepsize 1/T (smaller over time). What if stepsize is constant (0.5, 0.1, 0.05, 0.01)?
- Result: Smaller stepsizes → slower initial progress but better asymptotic. With stepsize 1/T (adaptive), performance is best.
- Interpretation: Reservoir sampling in NFSP approximates stepsize 1/T.

**Experiment A.2 (Figure 3b):** Added noise to best-response computation.
- XFP computes best response via exact backward induction. NFSP uses noisy stochastic gradient. How robust is XFP to noise?
- At each step, backpropagate a random action's value with probability ε instead of the best.
- Result: Performance decreases monotonically with noise, but remains stable and improving for all noise levels.
- Interpretation: NFSP's stochasticity is not fatal; XFP remains convergent even with noise.

---

## Experiments Reproducible at Small Scale for CS230:

1. **Leduc Hold'em (Figure 1, all variants):** Smallest game. ~1M states. Reproducible on CPU in ~1–2 hours. Requires: game engine (e.g., OpenSpiel), PyTorch/TensorFlow.
   - Estimated compute: **1–4 hours on CPU per experiment.**
   - Recommendation: Reproduce Experiment 1 (varying network sizes) and Experiment 3 (NFSP vs. DQN).

2. **Simpler toy game (if Leduc seems too large):** Single-card poker or Rock-Paper-Scissors with imperfect information. ~100 states. Reproducible in seconds. Useful for debugging.

3. **Limit Texas Hold'em (Figure 2):** Larger; ~1e8–1e9 information states. Probably not reproducible from scratch in a quarter.
   - Would require GPU and weeks of training. Can evaluate pre-trained models or run for limited iterations.

4. **Dou Dizhu (not in paper, but cf. RLCard):** Very large (~1e20 states). Not reproducible from scratch. Could test NFSP on RLCard's Dou Dizhu with pre-implemented algorithms.

---

## 11. Compute and Engineering

**Hardware (for LHE):** [Paper, not explicitly stated; inferred from context]
- Single machine with unspecified GPU. Probably NVIDIA K40 or similar (common in 2016).
- [Interpretation] Based on the training time and algorithm complexity, likely 4–16 CPU cores for game engine, 1 GPU for neural networks.

**Training duration:**
- **Leduc:** [Not explicitly stated]. Figure 1a shows iterations up to 1M; each iteration is one game (both players take actions). Estimate: ~hours on CPU.
- **LHE:** [Not explicitly stated]. Figure 2a shows iterations up to 3.5M. Estimate: **weeks on GPU**, or **days** with parallel actors (if available).

**Number of samples:**
- **Leduc:** Figure 1a goes to ~1M iterations (episodes). Each episode is ~10 actions (two players, ~5 rounds × 2 players). Total transitions: ~10M. Both memories (MRL, MSL) are ~200k and 2M, so trained on ~2-5 passes through data.
- **LHE:** Figure 2a goes to ~3.5M iterations. Each episode is ~20–50 actions. Total transitions: ~70M–175M. Memories are 600k and 30M.

**Parallel actors:** [Paper does not mention parallelization]. Likely single-threaded game + network updates, so not scalable. [Interpretation] For large-scale reproduction, you'd want parallel game servers (like DouZero or RLCard's distributed setup).

**Memory requirements:**
- Networks: 1-2 hidden layers, thousands of parameters. Negligible (~MB).
- Memories: Leduc 2.2M transitions (16 bytes each ≈ 35MB). LHE 30.6M (≈500MB). Manageable on modern hardware.

**Implementation tricks:**
1. **Target network:** Refresh every 300–1000 updates. Reduces variance in TD targets.
2. **Reservoir sampling:** Key to stable SL training. Must implement carefully (maintain random index, replace with probability 1/count).
3. **Exploration decay:** ε(t) = ε_0 / √t. Smooth decay, not step-wise.
4. **Stochastic gradient updates:** Two gradient steps per timestep (one for RL, one for SL). High update frequency keeps networks fresh.

---

## Reproduction Difficulty: **Moderate to Hard** for a quarter.

**Justification:**

**Easy parts:**
- Algorithm is clear (pseudocode in Algorithm 1).
- Hyperparameters are published.
- Leduc Hold'em is small and reproducible.

**Hard parts:**
- Leduc requires a game engine (or API like OpenSpiel, RLCard).
- Off-policy learning in multi-agent settings is tricky (require careful implementation).
- Convergence is slow; need to run for many iterations to see results.
- LHE is huge; training from scratch takes weeks.
- No open-source official implementation provided (though RLCard has an implementation).

**Realistic timeline for a CS230 project:**
- **Week 1–2:** Set up game engine, implement NFSP from scratch or adapt RLCard.
- **Week 3–4:** Debug and run Leduc experiments (should converge to Nash within days).
- **Week 5–8:** Run ablations, compare to DQN. Report results.
- **Week 9–10:** Write up, explore modifications (e.g., different η, opponent modeling).
- **Total:** Feasible as a **final project**, not a mid-quarter checkpoint.

---

## What a Scaled-Down Reproduction Would Look Like:

1. **Use RLCard library:** They provide pre-implemented NFSP, Leduc, and Dou Dizhu. Saves weeks.
2. **Small Leduc subset:** Train only on a simplified variant (e.g., fewer rounds, lower card count).
3. **Shorter training:** Run for 100k–500k iterations instead of millions. Enough to see convergence trend, not full convergence.
4. **Batch evaluation:** Instead of 10,000-game tournaments, evaluate on 1,000–5,000 hands.
5. **Single-seed runs:** Acceptable for mid-quarter checkpoint; multiple seeds for final results.

---

## 12. Limitations

### A. Explicitly Mentioned by Authors:

1. **Two-player zero-sum games only** [§5, p. 7]: "In this work, we focused on imperfect-information two-player zero-sum games." No guarantee for 3-player, cooperative, or asymmetric games. Dou Dizhu is 3-player and asymmetric (landlord vs. peasants).

2. **Theoretical convergence proof absent** [§5, p. 7]: "NFSP is the first deep reinforcement learning method known to converge to approximate Nash equilibria in self-play." They claim convergence based on empirical evidence (Leduc) and appeal to FSP theory, but don't prove NFSP converges under function approximation.

3. **Possible extension to cooperative games** [§5, p. 8]: "It is therefore conceivable that NFSP can be successfully applied to cooperative, potential games (Monderer and Shapley, 1996). ... It remains to be seen whether their approach converges in practice or theory" [citing related work on other methods]. Speculative; no evidence provided.

4. **Continuous-action games not addressed** [§5, p. 8]: "Recent developments in continuous-action reinforcement learning (Lillicrap et al., 2015) could enable NFSP to be applied to continuous-action games, which current game-theoretic methods cannot deal with directly." Future work, not implemented.

---

### B. Inferred from Methodology:

1. **No theoretical convergence guarantee under function approximation:** FSP is proven to converge on exact game trees, but NFSP uses neural network approximation (function approximation error), stochastic gradient descent (gradient error), and off-policy learning (error from behavior policy ≠ target policy). Any of these can break convergence. The paper provides **no formal analysis**, only empirical evidence.

2. **Hyperparameter sensitivity unknown:** The paper states "we manually calibrated NFSP." No systematic sensitivity analysis. Results may depend heavily on η, learning rates, memory sizes, network architecture. Small perturbations could cause failure.

3. **Single-seed Leduc experiments:** Figure 1a, 1b, 1c show no error bars or multiple runs. Leduc is small enough that randomness could matter. Reproducibility unclear.

4. **Scalability to large games uncertain:** Leduc is tiny (~1e6 states); LHE is medium (~1e17 states, 1e8–1e9 info sets); Dou Dizhu is huge (~1e20 info sets). RLCard (2020) tested NFSP on Dou Dizhu and found it "highly unstable" with poor convergence. Scalability remains an open question.

5. **Sparse rewards only:** Rewards are zero except at terminal states. This is fine for poker but makes learning harder. No exploration with dense rewards or reward shaping.

6. **No opponent pool / checkpointing strategy:** The paper doesn't discuss saving checkpoints or maintaining an opponent pool (as in later works like AlphaGo). All agents play each other's current mixture, which is the simplest approach but may not be optimal.

7. **Exploration becomes zero:** ε → 0, so best-response policy becomes deterministic. This may reduce robustness to adversarial opponents or distribution shift.

8. **Off-policy learning stability in multi-agent setting:** Q-learning assumes a stationary MDP, but in multi-agent self-play, the environment (opponents) changes every step. The paper provides no formal stability analysis. Empirical convergence in Leduc is encouraging but not a proof.

9. **No explicit opponent modeling:** The agent doesn't track what opponents might have (belief state, hand strength inference). This is by design (to avoid domain knowledge), but it means implicit inference is learned, which may be inefficient.

10. **Information state design is manual:** While the paper avoids domain-specific features (like hand strength), the choice of input representation (k-of-n, betting tensor) is still hand-crafted. Automatic feature learning (e.g., CNN) might be better.

11. **No comparison to policy gradient / actor-critic methods:** Only compares to DQN and XFP. Other RL methods (e.g., A3C, PPO) might do better.

12. **Deterministic tie-breaking in argmax:** When multiple actions have the same Q-value, how is the action chosen? The paper doesn't specify. Neural network outputs have ties, so this matters. [Interpretation] Likely ties are broken arbitrarily (first index), which introduces bias.

---

## 13. Future Work

### Explicitly Mentioned by Authors (with citations and classification):

1. **"Cooperative, potential games"** [§5, p. 8; (Monderer and Shapley, 1996)]
   - **Idea:** Apply NFSP to games with cooperation (e.g., both players prefer high joint payoff).
   - **Classification:** **Possible but difficult.** FSP theory supports it, but no empirical evidence. Would require careful extension to handle asymmetric payoffs (e.g., peasants in Dou Dizhu cooperate but have different roles).
   - **Later work in this set:** None explicitly address FSP-style cooperation. DouZero (2021) tackles 3-player asymmetric games but uses Deep Monte-Carlo, not averaging strategies.

2. **"Continuous-action games"** [§5, p. 8; (Lillicrap et al., 2015)]
   - **Idea:** Extend NFSP to games with continuous action spaces (e.g., poker with arbitrary bet sizes).
   - **Classification:** **Possible but difficult.** Requires changes to the Q-network (output continuous actions or regression) and policy network (continuous probability distributions, e.g., Gaussian). Not inherently harder than discrete, but not explored in poker literature.
   - **Later work in this set:** None address continuous-action games in card games.

3. **"Scalability to larger games"** [Implied from discussion of LHE; not explicit]
   - **Idea:** [Interpretation] The paper doesn't state this, but implicitly, scaling from Leduc (~1e6 states) to LHE (~1e8–1e9 info states) is a 1000× jump. Dou Dizhu is another 10^11×. How to scale further?
   - **Classification:** **Too ambitious for a quarter.** Requires fundamental algorithmic or architectural improvements (e.g., state abstraction, hierarchical RL, model-based methods).
   - **Later work in this set:** DouZero (2021) addresses Dou Dizhu via Deep Monte-Carlo + action encoding + parallel actors. RLCard shows NFSP plateaus on Dou Dizhu.

4. **"Application to real-world games (security, finance, traffic)"** [Introduction, §1, p. 1]
   - **Idea:** Mentioned as motivation. Not a concrete future-work item, but implicit direction.
   - **Classification:** **Too ambitious / too applied for a quarter.** Requires domain expertise and significant engineering.
   - **Later work in this set:** None address real-world applications.

---

### Implied Future Work (not explicitly stated, but reasonably inferred):

[**Interpretation**]

5. **"Opponent pool / self-play with checkpointed policies"** [Not mentioned, but implied by discussion of data distribution]
   - **Idea:** Instead of playing the current opponent, maintain a pool of past checkpoints (like AlphaGo or DouZero). This ensures consistent (slowly-changing) opponent behavior, not simultaneous drift.
   - **Classification:** **Directly usable as a CS230 project.** Simple to implement: save a network every N iterations, randomly sample opponents from the pool. Hypothesis: better than simultaneous learning because opponent becomes stationary.
   - **Later work in this set:** DouZero uses opponent pool / baseline models for evaluation (not training). RLCard doesn't mention it.

6. **"Anticipatory parameter scheduling (decay η over time)"** [Not mentioned, but motivated by discussion of η]
   - **Idea:** Currently η is fixed (e.g., 0.1). What if η is annealed, starting high (explore, learn best responses) then decaying to low (play average policy, converge)?
   - **Classification:** **Directly usable as a CS230 project.** Easy to implement. Hypothesis: adaptive η could speed convergence or improve robustness.
   - **Later work in this set:** Not addressed.

7. **"Policy distillation or model compression"** [Not mentioned]
   - **Idea:** [Interpretation] After training, combine Π and Q networks into a single network for faster inference. This is useful for deployment.
   - **Classification:** **Engineering-heavy, not deep research.**
   - **Later work in this set:** DouZero mentions distillation briefly; Combinational Q-Learning paper uses distillation.

8. **"Multi-agent off-policy learning with importance weighting"** [Motivated by off-policy discussion in §3]
   - **Idea:** The behavior policy (mixed σ) differs from the target policy (β against σ^-i). Importance weighting could correct for this divergence.
   - **Classification:** **Too ambitious / theoretical.** Requires careful stability analysis.
   - **Later work in this set:** Not addressed in these papers.

9. **"Application to Dou Dizhu specifically"** [Not mentioned, but implied by relevance of game]
   - **Idea:** [Interpretation] Leduc and LHE are 2-player. Dou Dizhu is 3-player, asymmetric, huge state space. Does NFSP work? What modifications help?
   - **Classification:** **Directly usable as a CS230 project.** Test NFSP (via RLCard) on Dou Dizhu, identify failure modes, propose modifications (e.g., opponent modeling, action encoding, reward shaping).
   - **Later work in this set:** RLCard (2020) tests NFSP on Dou Dizhu and finds it struggles. DouZero (2021) uses a different algorithm entirely.

10. **"Comparison to Deep CFR or other regret-minimization methods"** [Not mentioned; implied by related work]
    - **Idea:** [Interpretation] The paper briefly discusses CFR as an alternative. Direct comparison on Leduc/LHE would be informative.
    - **Classification:** **Possible but difficult.** CFR is computationally expensive; only feasible on small games.
    - **Later work in this set:** Deep CFR (Brown et al., ICML 2019) combines CFR with neural networks. Addresses the comparison.

---

## 14. Connection to Other Papers

**Papers in this set NFSP relates to:**

### Direct predecessors:
1. **Fictitious Self-Play (FSP), Heinrich et al., 2015** [Cited extensively in NFSP]
   - **What it does:** Extends fictitious play to extensive-form games. Computes exact best responses (via backward induction) and updates average strategy exactly.
   - **What NFSP keeps:** The two-strategy framework (best response + average strategy). The core idea that averaging stabilizes self-play.
   - **What NFSP changes:** Replaces exact computation with neural networks + stochastic gradient descent. Enables scalability to large games.
   - **Why the change improves performance:** FSP is infeasible for large games (backward induction is exponential). NFSP's sampling-based approach scales to LHE.
   - **New limitation:** NFSP loses the convergence guarantee (FSP is proven; NFSP is empirical).

### Direct successors (in this set):
2. **Deep CFR, Brown, Lerer, Gross, Sandholm, ICML 2019**
   - **What it does:** Combines regret minimization (CFR) with neural networks. Learns a policy that minimizes regret against any opponent.
   - **How it relates:** Alternative approach to imperfect-information games. Doesn't use fictitious play or averaging strategies. Focuses on regret bounds, not Nash convergence.
   - **Comparison to NFSP:** Deep CFR is theoretically grounded (regret bounds); NFSP is empirically grounded (convergence observation). Deep CFR has been applied to heads-up poker and large games.
   - **Indication of progression:** CFR is orthogonal to NFSP; both are tools in the imperfect-information RL toolkit.

3. **DeltaDou, Jiang et al., IJCAI 2019**
   - **What it does:** Applies expert-domain knowledge + RL + Monte Carlo tree search to Dou Dizhu. Reaches expert level.
   - **How it relates:** First strong result on Dou Dizhu. Uses self-play RL (similar to NFSP) but combines with search. Asymmetric roles (landlord vs. peasants).
   - **Comparison to NFSP:** DeltaDou is heavier-weight (search-based); NFSP is model-free. DeltaDou uses domain knowledge (card abstraction); NFSP is domain-agnostic.
   - **Indication of progression:** Dou Dizhu is harder than poker; simple self-play (NFSP) struggles. Search + RL works better.

4. **RLCard, Zha et al., IJCAI 2020**
   - **What it does:** Toolkit for RL in card games. Implements multiple algorithms (DQN, NFSP, CFR) and evaluates on multiple games (Leduc, LHE, Dou Dizhu, Mahjong).
   - **How it relates:** Provides open-source NFSP implementation. Evaluates NFSP on Dou Dizhu, finding it "highly unstable" with sparse convergence.
   - **Comparison to NFSP paper:** Confirms NFSP works on Leduc/LHE (reproducing paper results). Shows NFSP fails on Dou Dizhu (large game with sparse rewards).
   - **Indication of progression:** NFSP is a baseline; not sufficient for complex games. Need better algorithms or modifications.

5. **DouZero, Zha et al., ICML 2021**
   - **What it does:** Deep Monte-Carlo method for Dou Dizhu. Reaches superhuman performance in days.
   - **How it relates:** Addresses the limitations of NFSP-style methods on Dou Dizhu. Uses a different algorithm (Deep MC instead of averaging) + architectural innovations (action encoding, parallel actors).
   - **Comparison to NFSP:** DouZero is purpose-built for Dou Dizhu; NFSP is domain-agnostic but struggles on complex games. DouZero's success shows that averaging strategies (NFSP) may not be optimal for large games with sparse rewards.
   - **Indication of progression:** Dou Dizhu requires more sophisticated methods than NFSP. Different games, different algorithms.

---

### Evolution chain:

```
Fictitious Play (Brown, 1951)
  → advantage: convergence to Nash
  → limitation: only normal-form games (single-step)

Fictitious Self-Play (Heinrich et al., 2015)
  → improvement: extends to extensive-form (multi-step) games
  → limitation: exact computation infeasible for large games

Neural FSP (NFSP, Heinrich & Silver, 2016)
  → improvement: adds neural networks, scales to real-world poker (LHE)
  → limitation: only 2-player zero-sum; no convergence proof under function approximation

Deep CFR (Brown et al., 2019)
  → alternative improvement: uses regret minimization + neural networks
  → scales to heads-up no-limit poker (larger than LHE)
  → limitation: computationally expensive, only 2-player

DeltaDou (Jiang et al., 2019)
  → novel approach: RL + MCTS + expert features, for 3-player asymmetric Dou Dizhu
  → limitation: domain knowledge needed, slow training

RLCard (Zha et al., 2020)
  → toolkit: implements NFSP, DQN, CFR; evaluates on 7 games
  → finding: NFSP works on Leduc/LHE, fails on Dou Dizhu (large, sparse rewards)

DouZero (Zha et al., 2021)
  → new algorithm: Deep Monte-Carlo + action encoding + distillation
  → solves Dou Dizhu end-to-end; reaches superhuman level
  → implication: averaging strategies (NFSP-style) not optimal for large games
```

---

## 15. Project Ideas for Me (CS230 student)

### Idea 1: Opponent Modeling in NFSP

**Research question:** Does explicit opponent modeling (predicting opponent's next action distribution) improve NFSP convergence speed or final performance?

**Baseline:** Standard NFSP (from RLCard) on Leduc Hold'em.

**Modification:** Add a third network $O_i(s_i, a^{-i} | \theta_O)$ that predicts the opponent's action probabilities, trained via supervised learning on observed plays. Use predictions to augment the Q-network input or to compute an opponent-aware target for the Q-learning loss.

**Hypothesis:** Opponent modeling helps the agent learn faster because it can predict opponent behavior and adapt. In imperfect-information games, inferring opponent's hand is crucial; explicit modeling might accelerate this.

**Experiment:**
1. Train standard NFSP on Leduc for N iterations.
2. Train NFSP + opponent modeling variant for N iterations.
3. Compare convergence speed (exploitability over time) and final exploitability.
4. Ablate: opponent modeling alone (network trained but not used in play).

**Metrics:**
- Exploitability (primary).
- Win rate vs. DQN baseline.
- Training time to reach a threshold exploitability.

**Difficulty:** Moderate. Requires modifying the Q-network architecture and loss; also implementing supervised learning for the opponent model.

**Compute:** Low. Can run on CPU; Leduc is small.

**Novelty:** NFSP (2016) predates modern opponent modeling; adding it is a natural extension. Not groundbreaking, but plausible improvement.

---

### Idea 2: Adaptive Anticipatory Parameter

**Research question:** Can learned or scheduled η outperform the fixed-η baseline?

**Baseline:** Standard NFSP with η = 0.1 (hand-tuned).

**Modification:**
- **Option A (scheduled):** Anneal η over training (e.g., η(t) = 0.1 / (1 + t / T)).
- **Option B (learned):** Add η as a learnable parameter, trained via meta-learning or policy gradient to maximize convergence speed.
- **Option C (adaptive):** Compute η dynamically based on opponent's strategy drift (e.g., if opponent changes a lot, increase η to explore more).

**Hypothesis:** Fixed η is suboptimal. Early training benefits from high η (explore many best responses). Late training benefits from low η (play stable average policy). Adaptive η adapts to the game dynamics.

**Experiment:**
1. Train standard NFSP (η = 0.1) on Leduc.
2. Train variants with scheduled/learned η.
3. Compare convergence and final exploitability.
4. Sensitivity analysis: try different schedules and learning rules.

**Metrics:**
- Exploitability over time.
- Robustness to initial η choice.

**Difficulty:** Easy to Moderate. Scheduling is trivial; learning is harder.

**Compute:** Low.

**Novelty:** Modest. Adaptive hyperparameters are a known technique, but haven't been applied to NFSP's η specifically.

---

### Idea 3: NFSP on Dou Dizhu with Action Encoding

**Research question:** Can simple action encoding (e.g., representing card sets as vectors) improve NFSP's performance on Dou Dizhu?

**Baseline:** NFSP on Dou Dizhu using raw action one-hot encoding (from RLCard).

**Modification:** Instead of one-hot encoding 27,472 possible actions in Dou Dizhu, encode actions as semantic features:
- **Card composition:** counts of high/mid/low cards, pairs, triples, etc.
- **Play type:** single, pair, triple, sequence, bomb, etc.
- **Relative strength:** comparison to table state (winning or losing).

**Hypothesis:** Dou Dizhu's huge action space (27K actions) is overwhelming for the Q-network. Semantic action encoding reduces dimensionality and provides useful inductive bias, allowing NFSP to learn better on Dou Dizhu.

**Experiment:**
1. Train standard NFSP (RLCard baseline) on Dou Dizhu for limited iterations (e.g., 100k hands).
2. Train NFSP with action encoding variant.
3. Compare learning curves (reward or win rate vs. random opponent).
4. Evaluate convergence and stability.

**Metrics:**
- Win rate vs. random baseline.
- Training stability (variance in performance over time).
- Sample efficiency (win rate per training step).

**Difficulty:** Moderate. Requires understanding Dou Dizhu's action space and designing encodings. Implementation in RLCard may be nontrivial.

**Compute:** Medium. Dou Dizhu is large; training takes longer than Leduc.

**Novelty:** Inspired by DouZero (which uses action encoding successfully). Applying it to NFSP is a reasonable ablation to understand why DouZero works better.

---

### Idea 4: Average-Policy vs. Latest-Policy Self-Play

**Research question:** In NFSP, how much does the average-policy network contribute to convergence vs. just using the latest best-response?

**Baseline:** Standard NFSP (average policy + best response mixture).

**Modification:**
- **Variant A:** Remove average policy; use only best-response self-play (like DQN).
- **Variant B:** Replace average policy with exponential moving average (EMA) of recent best responses instead of reservoir sampling.
- **Variant C:** Use no mixing (only average policy, no best-response exploration).

**Hypothesis:** The average policy is crucial for stability. Pure best-response self-play diverges (Figure 1c). But a fast-moving average (EMA) might hurt. The right compromise (slow historical average via reservoir) is optimal.

**Experiment:**
1. Train each variant on Leduc for fixed iterations.
2. Compare convergence (exploitability) and final performance.
3. Analyze data distribution (state visitation frequency) from each variant to understand why averaging helps.

**Metrics:**
- Exploitability.
- Win rate vs. DQN and NFSP.
- Distribution divergence (JS divergence of state visitations between variants).

**Difficulty:** Easy. Mostly disabled features and comparing existing implementations.

**Compute:** Low.

**Novelty:** Diagnostic paper. No new algorithm, but provides understanding of NFSP's components. Useful for designing better methods.

---

---

## 16. Important Implementation Components

### MUST UNDERSTAND AND IMPLEMENT/MODIFY:

1. **Two-network architecture**
   - Π network: MLP, outputs probability distribution over actions. Trained via cross-entropy loss.
   - Q network: MLP, outputs scalar Q-value per action. Trained via MSE loss with Bellman target.
   - Separate, independent parameters. Different learning rates.

2. **Reservoir sampling (for M_SL)**
   - Maintain a fixed-size buffer of (s, a) tuples with uniform historical probability.
   - Insert new tuples by replacing a random entry with probability size / count.
   - Ensures SL network learns true historical average, not recent snapshot.

3. **Mixed policy σ = (1 - η)π + ηβ**
   - Deterministic mixture: at each episode start, commit to either best-response or average policy.
   - Not a probabilistic mixture during play. The mixture probability is part of the sampling distribution, not the network output.

4. **Off-policy Q-learning with target network**
   - Maintain two copies of Q-network: θ_Q (training) and θ_Q^target (frozen).
   - Use θ_Q^target to compute bootstrapped Bellman targets.
   - Periodically update: θ_Q^target ← θ_Q.

5. **Ε-greedy exploration**
   - ε starts high (e.g., 0.06), decays to 0 over training.
   - Schedule: ε(t) = ε_0 / √t.
   - Applied to Q-network only (best-response policy), not average policy.

6. **Memory management**
   - M_RL: circular buffer (recent window). Fast write/read; old data is evicted.
   - M_SL: reservoir sample (historical archive). Slower insertion but uniform historical distribution.

7. **Training loop**
   - For each episode: play one game, generate transitions.
   - For each timestep: update both networks via SGD.
   - Frequency: 2 SGD steps per step in game (one per network, per mini-batch).

---

### MUST UNDERSTAND CONCEPTUALLY:

1. **Fictitious play and Nash equilibrium**
   - Why averaging strategies stabilizes learning.
   - Why pure strategies are exploitable in imperfect-information games.
   - Convergence theory (FSP converges; NFSP is empirically convergent).

2. **Off-policy learning in multi-agent settings**
   - Q-learning assumes a stationary MDP, but multi-agent self-play violates stationarity.
   - How does NFSP handle this? Via slowly-changing average strategy + anticipatory parameter.
   - Not rigorously proven; empirically stable.

3. **Information states and imperfect information**
   - Why implicit hand inference (via action history) works better than explicit belief updates in large games.
   - Trade-off: implicit inference is learned, not hand-engineered.

4. **Target network and temporal-difference learning**
   - Why bootstrapping reduces variance.
   - Why a frozen target network is key to DQN stability.
   - When to refresh the target network.

5. **Anticipatory dynamics**
   - Motivation from continuous-time game theory.
   - Discrete approximation: predict opponent's next strategy and respond.
   - Effect on learning: enables agent to generate own best-response data while playing against stable (averaged) opponent.

---

### CAN REUSE AS INFRASTRUCTURE:

1. **OpenSpiel or RLCard:** Game environments (Leduc, LHE, Dou Dizhu).
2. **PyTorch / TensorFlow:** Neural network libraries. NFSP implementation requires < 500 lines of code with a framework.
3. **Scalable RL platforms:** If scaling to large games, consider frameworks like OpenAI's baselines, Stable Baselines, or Ray RLlib.
4. **Exploitability computation:** For Leduc, compute best response via game-tree traversal or RL. For LHE, evaluate via tournament play.

---

## 17. Questions I Should Ask My CS230 Staff

1. **On algorithm scope:** "NFSP is proven only for 2-player zero-sum games. Dou Dizhu is 3-player with asymmetric roles and cooperation (peasants vs. landlord). Should I assume NFSP's convergence still applies, or design an experiment to test whether it converges on Dou Dizhu specifically?"

2. **On evaluation rigor:** "The paper reports single-run experiments on Leduc (Figure 1). How many random seeds should I use for my project's baseline experiments to demonstrate statistical significance? Is 3 seeds sufficient for a quarter project, or should I aim for 10+?"

3. **On reproducibility scope:** "The paper trains on LHE for weeks (Figure 2). For a quarter-long project, is it more important to reproduce one result end-to-end (e.g., Leduc convergence) or to run many experiments at smaller scale (fewer iterations) to test hypotheses faster?"

4. **On modifications:** "The paper uses fixed η = 0.1 (Leduc) or 0.1 (LHE). If I propose an adaptive η method, should I first verify that the baseline (fixed η) converges as in the paper, or can I skip reproducing if the code already exists (RLCard)?"

5. **On novelty vs. reproduction:** "The paper's core idea (two networks + averaging) is now 8 years old. For a project to feel novel, should I implement NFSP from scratch (to show I understand it), modify it (opponent modeling, action encoding), or use existing implementations and focus on the modification? What's the right balance between understanding and innovation?"

---

## 18. Beginner Concepts I Need to Learn

| Concept | Plain-language Explanation | Where It Appears in NFSP | Priority |
|---------|---------------------------|--------------------------|----------|
| **Nash Equilibrium** | A strategy profile where no player can improve by changing their own strategy unilaterally. In poker, both players play unexploitable strategies. | §2.2 (core motivation); §4 (goal of learning) | NOW |
| **Imperfect Information** | Players don't observe all state; e.g., in poker, you don't see opponent's cards. Requires mixing strategies (randomization) to be unpredictable. | §2.2 (definition); §8 (how NFSP handles it) | NOW |
| **Extensive-Form Game** | A sequential game with multiple rounds and hidden information. Represented as a game tree (but tree is huge for real games). | §2.2 (definition); §1 (motivation) | NOW |
| **Best Response** | The optimal strategy against a fixed opponent strategy. In Fictitious Play, each player computes and plays best responses. | §2.3 (definition); Algorithm 1 (Q-network learns best response) | NOW |
| **Mixed Strategy** | A probability distribution over actions, not a single action. Required in imperfect-information games to be unpredictable. | §2.2 (why needed); §3 (why NFSP uses Π network) | NOW |
| **Fictitious Play** | A learning algorithm: alternately compute best responses to opponent's historical average strategy, then update your own average. Converges to Nash in certain games. | §2.3 (formal definition); §3 (NFSP's foundation) | NOW |
| **Exploitability** | How much a best-response player can win against your strategy. Lower = closer to Nash. Used to measure NFSP's convergence. | §4 (definition and metric); Figure 1 (primary result metric) | NOW |
| **Off-Policy Learning** | Learning from data generated by a different (behavior) policy than the policy being learned (target policy). Q-learning is off-policy. | §2.1 (brief mention); Algorithm 1 (Q-network learns β off-policy) | SOON |
| **Information State** | What a player observes at a decision point: own cards + action history. Two players in same information state have identical observations. | §2.2 (definition); §4.1 (input to NFSP networks) | NOW |
| **Information Set** | A set of game nodes indistinguishable to a player (same information state). In poker, all nodes where you have the same cards and see the same history are in one info set. | §2.2 (definition); [Interpretation] NFSP doesn't explicitly group info sets | SOON |
| **Temporal-Difference (TD) Learning** | A bootstrapping RL method: estimate value as immediate reward + discounted next value. One-step lookahead. Faster than Monte Carlo but biased. | §2.1 (Q-learning is TD); §6 (NFSP's Q-loss is TD) | SOON |
| **Target Network** | A frozen copy of the Q-network used to compute Bellman targets. Reduces variance by preventing chasing a moving target. | §3 (mentioned); Algorithm 1 (refreshed periodically) | SOON |
| **Experience Replay** | Storing past transitions and training from mini-batches sampled from the buffer. Breaks correlation, improves stability. | §2.1 (definition); Algorithm 1 (M_RL and M_SL) | NOW |
| **Reservoir Sampling** | A method to maintain a random historical sample of data with uniform probability. Used by NFSP for M_SL to approximate true average strategy. | §3, p. 3–4; Algorithm 1 (M_SL); Appendix §A | SOON |
| **Stochastic Gradient Descent (SGD)** | Optimization method: update parameters by moving opposite the gradient of a mini-batch loss. Noisy but fast; used by NFSP for both networks. | §2.1 (standard RL); Algorithm 1 (every timestep) | SOON |
| **Epsilon-Greedy** | Simple exploration strategy: with probability ε, take random action; else take greedy action. ε typically decays over time. | §3 (β = ε-greedy(Q)); Algorithm 1 | NOW |
| **Bootstrapping** | In RL, using estimates of future values to compute targets for current values. E.g., Bellman backup uses Q(s', a') to estimate Q(s, a). | §2.1 (Q-learning); §6 (NFSP's Q-loss) | SOON |
| **Cross-Entropy Loss** | Loss function for classification: penalizes discrepancy between predicted and true probability distributions. Used by NFSP for policy network. | §5 (L_Π definition); standard supervised learning | SOON |
| **Realization-Probability** | In extensive-form games, the probability of reaching an information state under a given strategy. Used in game theory to weight strategies in mixtures. | §2.3 (Equation 1); [Interpretation] NFSP approximates this via reservoir sampling | OPTIONAL |
| **Behavioral Strategy** | A strategy represented as an action distribution per information state (vs. normal-form, which conditions on full history). Matches how games are actually played. | §2.3 (Equation 1); §4.1 (card + history encoding) | SOON |

---

## 19. Paper Scorecard

> **Cross-paper calibration:** this scorecard looks at this paper alone. Ranked against the whole pack for this Dou Dizhu project, the recommended verdict is **READ SELECTIVELY**. See `00_SYNTHESIS_AND_RECOMMENDATION.md` §20.9.

| Dimension | Score (1–5) | Reasoning |
|-----------|-------------|-----------|
| **Importance for my project** | 5 | Foundational paper on self-play and imperfect information. Whether or not you use NFSP directly, you need to understand its insights (averaging for stability, two networks, etc.). |
| **Difficulty to understand** | 3 | Algorithm is clear (Algorithm 1 is well-written). Game theory concepts (Nash, fictitious play) are new, but explained at accessible level. The connection between discrete and continuous-time dynamics (anticipatory parameter) is the hardest part. |
| **Difficulty to reproduce** | 3 | Leduc is reproducible in a few hours/days on CPU. LHE requires GPU and weeks. Hyperparameter tuning adds complexity. With RLCard's implementation, difficulty drops to 2. |
| **Relevance to Dou Dizhu** | 2 | NFSP was tested on Dou Dizhu (via RLCard) and found to be unstable. Not a direct fit. However, understanding why it fails is crucial for designing better methods. |
| **Relevance to self-play** | 5 | Self-play is the paper's core. It directly addresses how to stabilize self-play learning in imperfect-information games. Essential reading. |
| **Relevance to imperfect information** | 5 | The whole paper is about imperfect information. Information states, hidden cards, inferring opponent behavior—all central. Must read. |
| **Potential for project ideas** | 4 | Many extensions possible: opponent modeling, adaptive η, action encoding, multi-player variants, comparison to DQN/CFR. Solid foundation for projects. |
| **Importance to read deeply** | 4 | Not every proof or mathematical detail is essential, but the algorithmic ideas (two networks, two memories, anticipatory dynamics) and empirical results are critical. |

---

### Verdict: **READ DEEPLY**

**Which sections to prioritize:**
1. **Introduction (§1):** Context and motivation. 10 min.
2. **Background (§2):** Game theory basics. **Must read carefully**; new concepts. 30 min.
3. **NFSP (§3):** Algorithm. **Core; read slowly**. Algorithm 1 is the key. 30 min.
4. **Experiments (§4):** Results. Skim Leduc (Figure 1), focus on DQN comparison (Figure 1c, §4.2). Look at LHE results (Figure 2) but don't memorize numbers. 20 min.
5. **Math (§5):** Equations. Understand each equation's intuition; derivations less important. 15 min.
6. **Limitations (§12, implied):** Scope limitations. Crucial for understanding where NFSP fails (e.g., 3-player games, Dou Dizhu). 10 min.

**Sections to skim:**
- Appendix §A (XFP robustness): nice to see empirically, but not essential.
- Full related work (§5): gives context; not essential.
- Hyperparameter details (§4.1, 4.3): useful for reproduction, not for understanding ideas.

**Skip (unless needed for a specific project):**
- Detailed proofs or derivations (not provided in this paper anyway).
- Full list of references (skim for interesting citations).

**Estimated time to read deeply: 2–3 hours.**

---

## Appendix: Synthesis Facts

| Item | Value |
|------|-------|
| **Paper** | Deep Reinforcement Learning from Self-Play in Imperfect-Information Games (NFSP) |
| **Year** | 2016 |
| **Environment(s)** | Leduc Hold'em (2-player, 1e6 states), Limit Texas Hold'em (2-player, ~1e17 states / ~1e8–1e9 information states) |
| **Main algorithm** | Neural Fictitious Self-Play: two networks (Q for best response, Π for average policy), two memories (circular buffer for RL, reservoir for SL), mixed policy σ = (1-η)π + ηβ. |
| **State representation** | Vector: k-of-n encoding of cards + betting history tensor. Domain-independent, no hand-crafted features. |
| **Action representation** | Discrete: {fold, call, raise}. Output as vector of action values (Q-network) or action probabilities (Π-network). |
| **Self-play?** | Yes. Essential. Both agents learn simultaneously via self-play; mixed policy ensures stable data distribution. |
| **Opponent modeling?** | No. Action history serves as implicit proxy; no explicit belief updates or hand inference. |
| **Main loss (RL)** | Mean-squared Bellman error: MSE(r + max_a' Q(s', a'; θ^target) - Q(s, a; θ)). Standard DQN loss. |
| **Main loss (SL)** | Cross-entropy: E[-log Π(s, a)]. Imitation of best-response behavior. |
| **Main contribution** | "First end-to-end deep RL approach to learning approximate Nash equilibria of imperfect-information games from self-play" (§6) — convergence shown experimentally in Leduc, not proven with function approximation. Combines game-theoretic stability (fictitious play) with deep learning scalability. |
| **Evaluation method** | Leduc: exploitability via best-response calculation. LHE: win rate vs. ACPC 2014 top programs (confidence intervals from 25k-hand tournaments). |
| **Compute** | Leduc: ~hours–1 day on CPU. LHE: weeks on GPU (not reported; inferred). Parallel actors not mentioned. |
| **Main limitation** | Theory applies only to 2-player zero-sum games. No convergence proof under function approximation. RLCard (2020) shows NFSP struggles on 3-player Dou Dizhu (huge state space, sparse rewards). |
| **Best idea to borrow** | Two-network + two-memory architecture for stability in self-play. The insight that averaging slower-changing opponent policy stabilizes learning is powerful and applicable to other multi-agent RL problems. |

---

**End of Analysis**

---

