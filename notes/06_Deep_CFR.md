# Deep Counterfactual Regret Minimization (ICML 2019)

> Noam Brown, Adam Lerer, Sam Gross, Tuomas Sandholm · [PMLR](https://proceedings.mlr.press/v97/brown19a.html) · Tier B (Foundations) · **Neural networks approximate regrets in CFR without manual game abstraction; first non-tabular CFR to scale to large two-player zero-sum games.**

---

## 0. TL;DR for a CS230 student

- **What it is:** Deep Counterfactual Regret Minimization (Deep CFR) combines CFR (an equilibrium-finding algorithm for imperfect-information games) with deep neural networks to scale to huge games without hand-coded abstraction.

- **Why you care:** Dou Dizhu is too complex for pure CFR, but Deep CFR shows how to apply RL to imperfect-info games while maintaining convergence guarantees. Helps you understand regret minimization vs. reward maximization.

- **Problem it solves:** Tabular CFR requires abstraction (manual bucketing of game states), which is tedious and loses strategic nuances. Deep CFR learns to approximate regrets via neural networks instead.

- **Main idea:** During game tree traversals, collect samples of instantaneous regrets (advantages). Train a neural network from scratch each iteration to predict these advantages. Use regret matching on the network's predictions to determine strategy.

- **Why it is hard:** Function approximation breaks convergence guarantees. The paper proves Deep CFR still converges if approximation error is small—requires careful memory management (reservoir sampling) and retraining from scratch (not fine-tuning).

- **One idea to steal:** External sampling MCCFR produces unbiased advantage estimates (Lemma 1)—exactly what you want neural networks to learn. This is the key insight linking CFR theory to deep learning.

- **Limitation:** Theory holds only for two-player zero-sum games. Dou Dizhu is three-player with cooperation, so Deep CFR's guarantees don't apply (you'll need RL approaches like DouZero instead).

---

## 1. Paper Overview

### 1a. Beginner-friendly explanation

Imagine you're building AI for poker. The traditional approach, Counterfactual Regret Minimization (CFR), works by iteratively refining strategies to minimize **regret** (how much worse you do than an optimal opponent). But CFR needs to visit millions of game tree nodes—10^17 nodes for full Texas Hold'em. To make it tractable, experts manually simplify the game by grouping similar situations ("abstraction")—e.g., treating all flush hands the same. This abstraction is tedious, domain-specific, and can hide important nuances.

Deep CFR's insight: instead of visiting every node and computing regrets exactly, traverse a sampled subset of nodes, collect **advantage estimates** (a form of regret), and let a neural network learn to predict these advantages for any state. The network acts like a compressed regret memory. This eliminates manual abstraction.

**Why it's hard:** Neural networks approximate—they make errors. Does the error accumulate? Will the strategy still converge to equilibrium? The paper proves yes, with an error bound.

### 1b. Technical explanation

**Problem:** Tabular CFR converges to Nash equilibrium (§2.1, p. 2–3) with regret bound O(√T) but requires storing and updating regrets at every information set. Large games have 10^14+ information sets, making tabular CFR infeasible. Abstraction (manually bucketing states) is the standard workaround, but:
- It is domain-specific and labor-intensive.
- Coarser abstractions converge faster but are more exploitable.
- Determining a good abstraction requires knowing the equilibrium—a circular problem (§1, p. 1).

**Prior approaches:**
- Tabular CFR with domain-specific abstraction: solved 2-player poker (Bowling et al. 2015) but requires expert domain knowledge.
- Neural Fictitious Self Play (NFSP, Heinrich & Silver 2016): combines deep learning with Fictitious Play, but Fictitious Play converges slower than CFR in theory and practice (§3, p. 3).
- Regression CFR (RCFR, Waugh et al. 2015): uses hand-crafted regression trees instead of learning features; still requires full tree traversals and only evaluated on toy games (§3, p. 3).

**Deep CFR's solution:**
1. Use external-sampling Monte Carlo CFR (MCCFR) to traverse only a fraction of the game tree each iteration (§2.2, p. 3).
2. Collect samples of **instantaneous regrets** (advantages) from these traversals.
3. Train a neural network to predict advantages at any infoset, using MSE loss.
4. Use regret matching on the network's predictions to determine strategy.
5. Prove (Theorem 1, §4, p. 4–5) that average regret remains bounded by O(√T) plus a term proportional to the network's approximation error.

**Key innovation:** Lemma 1 (§4, p. 4) proves that external sampling MCCFR produces **unbiased estimates** of the advantage (difference in payoff for playing a specific action vs. the current mixed strategy). This is exactly what you want neural networks to learn—unbiased targets.

**Scope:** Theory applies to two-player zero-sum games. Deep CFR is experimentally competitive with domain-specific abstractions and outperforms NFSP in heads-up poker (§5–6, p. 4–7).

---

## 2. RL Formulation

[Paper] Deep CFR is formally a regret minimization algorithm for extensive-form games, not an MDP-based RL algorithm. However, it can be understood through an RL lens:

| **Component** | **Value** | **Citation / Note** |
|---|---|---|
| **Agent(s)** | Two players (P1, P2), each learning a separate regret network | §4, p. 4; uses alternating updates (§2.1, p. 3) |
| **Environment** | Imperfect-information extensive-form game (poker-like) | §2, p. 2; defined by nodes, actions, infosets, terminal payoffs |
| **State** | Game history h (sequence of actions from root to current node) | §2, p. 2 |
| **Observation** | Infoset I(h) = player's view of the game (own cards + public history); opponent's cards are hidden | §2, p. 2–3 |
| **Action** | Legal action a ∈ A(I) at the current infoset | §2, p. 2 |
| **Legal action handling** | All actions in A(I) are legal; external sampling explores all of them at traverser infosets | §2.2, p. 3; Algorithm 2 (TRAVERSE), p. 6 |
| **Reward** | Only at terminal nodes: u_p(z) = payoff to player p at terminal z | §2, p. 2 |
| **Return** | Cumulative payoff from current node to terminal (discounted if applicable; no discounting here) | [Interpretation: payoff is undiscounted, as is standard in games] |
| **Policy** | Strategy σ(I, a) ∈ [0,1] = probability of action a at infoset I; computed via regret matching on network outputs | §2.1, p. 3; Eq. 4, p. 3 |
| **Value function / Advantage** | V(I, a \| θ) = predicted advantage (regret proxy) for action a at I. Not a state value, but an action advantage: the difference in payoff between playing a vs. the current mixed strategy. Targets are r̃^t(I, a) = sampled instantaneous regret (Lemma 1, p. 4) | §4, p. 4; Eq. 2, p. 3 (instantaneous regret definition) |
| **Observability** | Partially observable: player observes own cards + public history but not opponents' cards | §2, p. 2–3 |
| **Single vs. multi-agent** | Two-agent game; agents learn simultaneously but CFR is defined for two-player (not n-player) games | §2.1, p. 3; alternating updates used in practice (§2.1, p. 3) |
| **Cooperative / Competitive / Both** | Competitive (zero-sum): u_1 = −u_2 | §2, p. 2 |
| **Self-play used?** | Yes: players alternate as traverser; both collect experience and train simultaneously | §4, p. 4; §5.2, p. 5 |

**[Interpretation]** The advantage network V(I, a \| θ) is analogous to a Q-function in tabular Q-learning, but it is trained on off-policy regret targets (not returns), and it is reset every iteration rather than bootstrapped.

---

## 3. Model / Architecture

### Network Topology [Paper]

(Figure 1, p. 5; full spec referenced in Appendix C per §5.1, p. 5—appendix not included in provided PMLR version)

**Shared architecture for both value network V and policy network Π:**

```
Input: infoset I = {cards, bet history}
  ↓
[Card Branch: 3 FC layers]  [Bet Branch: 2 FC layers]
  ↓                           ↓
  [Merge: Concatenate]
  ↓
[3 additional FC layers]
  ↓
Output: logits over |A(I)| actions

Activation: ReLU(Ax [+x])   (skip connection [+x] if dims match)
Normalization: Zero mean, unit variance on last-layer features
Depth: 7 layers total
Parameters: 98,948
```

### Input Representation [Paper, §5.1]

**Cards:**
- Rank embedding: 1–13 (13 ranks)
- Suit embedding: 1–4 (4 suits)  
- Card embedding: 1–52 (52 unique cards)
- These three embeddings are **summed** for each set of permutation-invariant cards (hole cards, flop, turn, river).
- The sums are concatenated.

**Bet history:** For each of N_rounds betting rounds, up to 6 sequential actions per round.
- Each betting position encoded as: (1) binary flag: has a bet occurred? (2) float value: bet size.

### Output Representation [Paper, §5.1]

- **Value network V:** outputs logits (raw advantage predictions), one per action.
- **Policy network Π:** outputs logits (log probabilities), one per action; softmax applied at inference.

### Why This Architecture [Paper, §5.1]

"The network architecture was not highly tuned, but **normalization and skip connections** were used because they were found to be **important to encourage fast convergence** when running preliminary experiments on pre-computed equilibrium strategies in FHP."

### Separate Networks Per Player [Paper]

Yes: each player p has:
- Own value network V_p to predict advantages from own perspective.
- One shared policy network Π trained on both players' strategy samples.

---

## 4. Training Algorithm

### Algorithm 1: Deep Counterfactual Regret Minimization [Paper, p. 6]

```
Initialize:
  - Each player p's advantage network V(I, a | θ_p) → 0 for all inputs
  - Reservoir-sampled advantage memories M_V,1, M_V,2
  - Strategy memory M_Π

for CFR iteration t = 1 to T do
  for each player p ∈ {1, 2} do
    for traversal k = 1 to K do
      TRAVERSE(∅, p, θ_1, θ_2, M_V,p, M_Π, t)

    Train θ_p from scratch on loss:
      L(θ_p) = E_{(I,t',r̃^t') ~ M_V,p} [t' · (Σ_a r̃^t'(a) - V(I, a | θ_p))^2]

  Train θ_Π on loss:
    L(θ_Π) = E_{(I,t',σ^t') ~ M_Π} [t' · (Σ_a σ^t'(a) - Π(I, a | θ_Π))^2]

return θ_Π
```

### Algorithm 2: CFR Traversal with External Sampling [Paper, p. 6]

```
function TRAVERSE(h, p, θ_1, θ_2, M_V, M_Π, t)
  if h is terminal:
    return payoff u_p(h)
  else if h is chance node:
    sample a ~ σ(h)
    return TRAVERSE(h·a, p, θ_1, θ_2, M_V, M_Π, t)
  else if P(h) = p:  // Traverser's infoset
    Compute σ^t(I) from V(I(h), a | θ_p) using regret matching
    for a ∈ A(h):
      v(a) ← TRAVERSE(h·a, p, θ_1, θ_2, M_V, M_Π, t)
    for a ∈ A(h):
      r̃(I, a) ← v(a) - Σ_{a'} σ(I, a') · v(a')
    Insert (I, t, r̃^t(I)) into M_V
  else:  // Opponent's infoset
    Compute σ^t(I) from V(I(h), a | θ_{3-p}) using regret matching
    Insert (I, t, σ^t(I)) into M_Π with weight t
    sample a ~ σ^t(I)
    return TRAVERSE(h·a, p, θ_1, θ_2, M_V, M_Π, t)
```

---

## 5. Mathematics

### Instantaneous Regret [Paper, Eq. 2, p. 2–3]

**MUST UNDERSTAND**

$$r_t(I, a) = v^{\sigma^t}(I, a) - v^{\sigma^t}(I)$$

**Symbols:** r_t(I, a) = instantaneous regret (advantage) of action a. v^{σ^t}(I, a) = counterfactual value if a is played. v^{σ^t}(I) = counterfactual value under current strategy.

**Intuition:** How much better is action a than your current mixed strategy?

**Relation to RL:** This is the **advantage function** A(s, a) = Q(s, a) - V(s) in policy gradient RL. Deep CFR learns to predict it.

### Counterfactual Value [Paper, Eq. 1, p. 2–3]

**MUST UNDERSTAND**

$$v^\sigma(I) = \sum_{z \in Z_I} \pi_{-p}^\sigma(z[I]) \pi^\sigma(z[I] \to z) u_p(z)$$

**Symbols:** π_{-p}^σ = reach probability (contribution of opponents + chance), π^σ = full reach probability, u_p = payoff.

**Intuition:** Expected payoff at infoset I, weighted by probability of reaching I.

### Regret Matching [Paper, Eq. 4, p. 3]

**MUST UNDERSTAND**

$$\sigma^{t+1}(I, a) = \frac{R_+^t(I, a)}{\sum_{a'} R_+^t(I, a')}$$

where $R_+^t = \max\{R^t, 0\}$ (positive cumulative regret).

**Intuition:** Play actions in proportion to positive cumulative regret. Provably minimizes regret.

### Cumulative Counterfactual Regret [Paper, Eq. 3, p. 3]

**MUST UNDERSTAND**

$$R^T(I, a) = \sum_{t=1}^T r_t(I, a)$$

**Intuition:** Sum of regrets over all iterations. If large, you should have played a more often.

### Lemma 1: Unbiased Advantages [Paper, Lemma 1, p. 4]

**MUST UNDERSTAND** (Core insight)

$$\mathbb{E}_{Q \in \mathcal{Q}^t} [r_p^\sigma(I, a) | Z_I \cap Q \ne \emptyset] = \frac{v^{\sigma^t}(I, a) - v^{\sigma^t}(I)}{\pi_{-p}^{\sigma^t}(I)}$$

**Intuition:** External sampling produces **unbiased** advantage estimates. Perfect targets for neural networks.

**Why:** Unlike some off-policy RL (importance-weighted Q-learning), these samples need no correction.

### Theorem 1: Deep CFR Convergence [Paper, Theorem 1, p. 4–5]

**MUST UNDERSTAND** (Core theoretical result)

Assume approximation error $L_V^t - L_V^{t*} \leq L$ (bounded MSE loss).

With probability $1 - \rho$:

$$R_p^T \leq \left(1 + \frac{\sqrt{2}}{\sqrt{\rho K}}\right) \Delta |I_p| \sqrt{|A|} \sqrt{T} + 4T |I_p| \sqrt{|A|} \Delta L$$

**Corollary 1:** As $T \to \infty$:

$$\frac{R_p^T}{T} \leq 4 |I_p| \sqrt{|A|} \Delta L$$

**Intuition:**
- First term: O(√T), same as vanilla MCCFR.
- Second term: linear in L (network loss). Small loss → small regret.
- Together: convergence to ε-Nash if L is small.

**Caveat:** No convergence bound proven for Linear CFR weighting in the Monte Carlo setting (§4, p. 5), though Figure 4 shows it accelerates convergence in practice.

### CFR Convergence Bound (Vanilla CFR) [Paper, after Eq. 4, p. 3]

**MUST UNDERSTAND**

If both players play regret matching every iteration:

$$R_p^T \leq \Delta \sqrt{|I_p| |A(I)| T}$$

Thus: $R_p^T / T \to 0$ as $T \to \infty$.

**Intuition:** Regret grows as O(√T), so time-averaged strategies converge to Nash equilibrium.

---

## 6. Loss Function

### Q&A

**Q: What loss is used?**  
[Paper] Mean squared error (MSE) on sampled instantaneous regrets (value network) and on sampled strategy probabilities (policy network), §5.2, p. 5 and Theorem 1, p. 4.

**Q: Training targets?**  
[Paper] Value network targets: $\tilde{r}^{t'}(I, a)$ = sampled instantaneous regret collected during prior traversals. Policy network targets: $\sigma^{t'}(I)$ = sampled strategy collected during prior traversals.

**Q: Where do targets come from?**  
Direct samples from game tree traversals (Algorithm 2, p. 6), stored in memory buffers (M_V, M_Π). Not bootstrapped.

**Q: Bootstrapping?**  
No. Targets come from past traversals, not from the current network.

**Q: Monte Carlo?**  
Yes—targets are empirical payoffs/strategies from sampled trajectories.

**Q: Target network?**  
No separate target network. Networks are reset each iteration (not carried forward).

**Q: Policy loss?**  
Yes, separate: $L(\theta_\Pi) = \mathbb{E} [t' (\Sigma_a \sigma^{t'}(a) - \Pi(I, a | \theta_\Pi))^2]$.

**Q: Value loss?**  
Yes: $L(\theta_p) = \mathbb{E} [t' (\Sigma_a \tilde{r}^{t'}(a) - V(I, a | \theta_p))^2]$.

**Q: Auxiliary loss?**  
No.

**Q: Multiple losses?**  
Two separate losses (V and Π), trained sequentially.

**Q: Why these choices?**  
MSE is a Bregman divergence, suitable for convergence analysis. Iteration weighting (t' factor) is Linear CFR, faster in practice. Two networks align with CFR theory: V for strategy learning, Π for the final average strategy.

---

## 7. Self-Play

### Q&A

**Q: Who plays whom?**  
[Paper] Two players alternate as traverser. When P1 traverses, P2's strategy comes from P2's value network (Algorithm 2, line "P(h) ≠ p", p. 6).

**Q: Simultaneous updates?**  
No. [Paper] "Alternating which player updates their regrets on each iteration rather than updating simultaneously" (§2.1, p. 3) is faster and proven to work.

**Q: Parameter sharing?**  
No. Each player p has separate network V_p (§4, p. 4). One shared policy network Π.

**Q: Separate networks per player?**  
Yes. Each p: separate value network V_p, shared policy network Π.

**Q: Old checkpoints?**  
Standard: train one final policy network Π. Alternative (Steinberger 2019, not used): store all value networks; sample one for play.

**Q: Opponent pool?**  
Not discussed. Only one opponent at a time.

**Q: Exploration?**  
External sampling: all actions explored at traverser infosets. No explicit exploration bonus.

**Q: Non-stationarity?**  
Opponent's network changes each iteration. CFR's regret minimization handles this; theory guarantees convergence despite non-stationarity.

---

## 8. Imperfect Information

### Actually Observed vs. Hidden [Interpretation]

| **Actually Observed** | **Hidden** |
|---|---|
| Own cards | Opponent's cards |
| Community cards | Opponent's payoffs (until terminal) |
| Betting history (own + opponent's actions) | Opponent's strategies (learned implicitly) |
| Legal actions available | Opponent's private beliefs |
| Reach probabilities (computed, not observed) | Opponent's information sets (inferred) |

### Information Sets (Infosets) [Paper, §2, p. 2–3]

Infosets group game states indistinguishable to a player. Player chooses one strategy for entire infoset. Example: "A♠K♦, opponent bet" in poker is one infoset.

### No Explicit Opponent Modeling [Paper]

The paper does **not** maintain belief distributions over opponent cards. Opponent's strategy determined by opponent's value network; payoffs computed by solving game tree.

### Recurrent Models [Paper]

No. Bet history is flat features. Assumption: network learns long-range dependencies from flat encoding.

### Card Inference [Paper]

No explicit inference. Hidden information remains hidden; symmetry of game handles it implicitly.

---

## 9. Evaluation

### Evaluation Q&A

**Q: Baselines?**  
[Paper] (1) Domain-specific abstractions (40K, 368K, 3.6M, 234M clusters), solved via Linear Monte Carlo CFR. (2) Neural Fictitious Self Play (NFSP). (3) Vanilla Linear CFR (gold-standard tabular reference).

**Q: Metrics?**  
Exploitability (mbb/g = milli-big-blinds per game). Sample efficiency (nodes touched vs. exploitability). Head-to-head expected value (Table 1).

**Q: Fixed opponents?**  
No. Agents play self-play. Final evaluation: head-to-head between converged strategies.

**Q: Cross-play?**  
Yes. Table 1: Deep CFR vs. NFSP and each abstraction.

**Q: Seeds?**  
Figure 4: 5 replicates (std ~2.25 mbb/g). Not reported for HULH.

**Q: Number of games?**  
Not explicit. Confidence intervals (±2 mbb/g) suggest sufficient samples.

**Q: Significance testing?**  
Confidence intervals used; no formal tests stated.

**Q: Roles separately?**  
No. Position advantage (dealer vs. non-dealer) not reported separately.

### **Strengths** [Interpretation]

1. Principled theory (Theorem 1) connects approximation error to convergence.
2. Outperforms NFSP (37 vs. 47 mbb/g in FHP).
3. Competitive with abstractions but 2–3 orders fewer nodes sampled (Figure 2).
4. Thorough ablations (Figure 4) validate design choices.
5. Same algorithm works for FHP and HULH (generality).

### **Weaknesses** [Interpretation]

1. Limited to poker; generality to other games unknown.
2. Wall-clock time improvements modest despite sample efficiency gains (overhead acknowledged).
3. Hyperparameters require tuning; architecture "not highly tuned."
4. NFSP re-implemented, not directly from prior work.
5. Small statistical sample (5 replicates), no formal significance tests.
6. Position-specific analysis absent.
7. "Sufficiently large" memories vague; 40M infosets chosen empirically.

### **What I Would Improve for a CS230 Project**

1. **Architecture search:** Systematic ablation of network depth, width, embeddings.
2. **Opponent pool:** Sample opponent from prior checkpoints to reduce non-stationarity.
3. **Belief-state augmentation:** Train auxiliary network to infer opponent cards.
4. **Small-scale reproduction:** Leduc Hold'em target (10x smaller than FHP).
5. **Cross-game transfer:** Train on FHP; evaluate on HULH.

---

## 10. Experiments and Ablations

### Major Experiments [Paper]

**Experiment 1: FHP Convergence (Figure 2)**

| **Changed** | **Constant** | **Metric** | **Result** | **Conclusion** | **Critical Note** |
|---|---|---|---|---|---|
| Deep CFR vs. abstractions (40K–234M clusters) vs. NFSP vs. Linear CFR tabular | Games, networks | Exploitability (mbb/g) vs. nodes | Deep CFR: 37 mbb/g; NFSP: 47 mbb/g; 3.6M abstraction: ~37 mbb/g (asymptotic); 2–3 order node reduction vs. lossless | "Deep CFR asymptotically reaches similar level as 3.6M abstraction, converges substantially faster." | Final exploitability not strictly better than 3.6M abstraction; they asymptote together. Wall-clock improvement modest. |

**Experiment 2: FHP Hyperparameters (Figure 3)**

| **Changed** | **Metric** | **Result** | **Conclusion** | **Critical Note** |
|---|---|---|---|---|
| Traversals/iter (3K–1M) | Exploitability vs. iteration | Fewer traversals → slower convergence, same asymptote | "Takes more iterations to collect enough data" | Tradeoff between sample efficiency and wall-clock time per iteration. |
| SGD steps (1K–32K) | Exploitability vs. iteration | Fewer steps → faster convergence but higher asymptote (worse final quality) | "Model loss decreases with more training steps" (Theorem 1 predicts) | Clear speed vs. quality tradeoff. 4K is sweet spot. |
| Model size (8–256 dims) | Exploitability vs. parameters | Larger models → lower exploitability up to 128 dims, then plateau | "Model size decreases final exploitability up to a point" | No overfitting observed; saturation may indicate architectural limits. |

**Experiment 3: FHP Design Choices (Figure 4)**

| **Changed** | **Result** | **Conclusion** | **Critical Note** |
|---|---|---|---|
| Retraining from scratch vs. fine-tuning | Fine-tuning produces ~50% higher final exploitability than retraining from scratch each iteration | "Model gets stuck in local minima as objective changes" | **Most important ablation.** Retraining is critical (Figure 4 left shows 5 replicates with std ≈ 2.25 mbb/g at t=450, so difference is statistically significant). |
| Regret matching variants | Standard RM (uniform on negative) performs worse; highest-regret action (probability 1) performs better | "Helps RM better cope with approximation error" (line 244-245) | Empirical observation noted by authors; no theoretical justification provided. |
| Reservoir sampling vs. sliding-window | Sliding-window: convergence breaks once buffer fills. Reservoir: continues improving | "Crucial" | Dramatic difference; reservoir sampling essential for memory management. |

**Experiment 4: HULH Head-to-Head (Table 1)**

| **Comparison** | **Result** | **Conclusion** | **Critical Note** |
|---|---|---|---|
| Deep CFR vs. NFSP | Deep CFR wins +43 ± 2 mbb/g | "Beats NFSP" | Significant margin. |
| Deep CFR vs. 3.3×10^8-bucket abstraction | Deep CFR loses −11 ± 2 mbb/g | "Competitively close" (Polaris 2007 was +52 mbb/g vs. humans with 3×10^8 buckets) | Gap non-negligible but acceptable without hand-crafted domain knowledge. |

### Reproducible at Small Scale

1. **Leduc Hold'em:** ~10^3 nodes, ~100 infosets. Hours on CPU. Target: ~20–30 mbb/g.
2. **FHP:** ~10^12 nodes. Days on 1 GPU. Target: 100–200 mbb/g (loose goal). Shrink: T=100 iterations, K=1000 traversals, 1M infosets, 1000 SGD steps.

---

## 11. Compute and Engineering

### Hardware [Paper, §5]

- **Type:** MLPs, no specialized hardware required but slow on CPU.
- **Assumed:** 1–4 GPUs (estimated). PyTorch code.

### Training Duration [Paper]

Not stated in the paper. [Interpretation] Estimated: FHP ~1 GPU-week, HULH ~1 GPU-month, based on algorithm complexity and typical modern GPU training times for similar-scale models.

### Memory and Samples [Paper, §5.2]

- **Memory:** 40M infosets per player (advantage + strategy buffers).
- **Samples:** K=10,000 traversals/player/iteration; 4,000 SGD iterations (FHP), batch size 10,000.

### Distributed Setup

Not discussed. Traversals are embarrassingly parallel but not detailed.

### Tricks [Paper + Figure 4 ablations]

1. **Retraining from scratch** (critical).
2. **Reservoir sampling** (critical).
3. **Iteration weighting** (Linear CFR, faster).
4. **All actions at traverser** (variance reduction).
5. **Skip connections + normalization** (convergence speed).

### **Reproduction Difficulty: MODERATE for Leduc, HARD for FHP/HULH**

**Scaled-down:** Use Leduc (RLCard), K=1000, T=100, 1M infosets, 1000 SGD steps. Target: 50–100 mbb/g exploitability. ~8 GPU-hours.

---

## 12. Limitations

### A. Explicitly Mentioned

1. **No bound for Linear CFR weighting** (§4, p. 4). "Convergence rate of Linear CFR has not been shown in the Monte Carlo case. However, Figure 4 shows moderately faster convergence in practice."

2. **External sampling impractical for huge branching factors** (§4, p. 4). Outcome sampling may be preferred in those cases.

3. **Scalability to larger games unclear** (§7, p. 8). "Will likely require more scalable sampling strategies... and strategies to reduce high variance in sampled payoffs."

4. **Limited to two-player zero-sum** (§2, p. 2). Theory assumes $u_1 = -u_2$. Multi-player and cooperative games unaddressed.

### B. Inferred

1. **Approximation error must be globally bounded** [Interpretation]. Theorem 1 assumes $L_V^t - L_V^{t*} \leq L$ uniformly. Out-of-distribution infosets may have higher error.

2. **"Sufficiently large" memory vague** [Interpretation]. No concrete requirement given.

3. **Wall-clock speed not optimized** [Interpretation]. Neural network overhead reduces total time improvements.

4. **Limited experimental scope** [Interpretation]. Only poker tested.

5. **NFSP re-implemented** [Interpretation]. Makes it hard to assess whether improvements come from CFR framework vs. architecture choices.

6. **Architecture admittedly not optimized** [Interpretation]. Student could improve.

7. **Position-specific effects not analyzed** [Interpretation]. Dealer advantage not reported.

8. **Theorem 1 assumes perfect regret matching** [Interpretation]. Noisy advantages may break the proof's assumptions.

9. **No ablation of memory size** [Interpretation]. Sensitivity to buffer size unknown.

---

## 13. Future Work

### From Authors

1. **More scalable sampling strategies** (§7, p. 8).
   - Classification: **Directly usable** (implement outcome sampling; compare to external sampling).
   - Later work: ReBeL (2020) uses depth-limited subgame solving. DouZero (2021) abandons tree traversal.

2. **Variance reduction in sampled payoffs** (§7, p. 8).
   - Classification: **Engineering-heavy** (baseline subtraction, importance weighting).
   - Later work: Schmid et al. (2019) on VR-MCCFR; could adapt.

### [Interpretation] Implied

3. **Opponent modeling** (not explicit).
   - Classification: **Possible but difficult** (belief-state augmentation).
   - Later work: DouZero+ (2022), DouZero (2021) use opponent modeling.

4. **Extension to multi-player games** (not addressed).
   - Classification: **Mostly theoretical** (CFR framework breaks down; needs new foundations).
   - Later work: DouZero (2021) solves three-player Dou Dizhu with deep RL instead.

5. **Parameter sharing and transfer learning** (not addressed).
   - Classification: **Directly usable** (test FHP→HULH transfer).
   - Later work: Not in the literature set.

---

## 14. Connection to Other Papers

### Prior Work

**NFSP (Heinrich & Silver 2016):**
- **Keeps:** Self-play, neural networks, poker evaluation.
- **Changes:** Algorithm (Fictitious Play → CFR). CFR converges faster and better.
- **Outcome:** Deep CFR 37 vs. NFSP 47 mbb/g in FHP (20% better).
- **New limitation:** Deep CFR limited to two-player zero-sum.

### Subsequent Work

**ReBeL (Brown et al., NeurIPS 2020):**
- **Keeps:** Value networks, self-play.
- **Changes:** Depth-limited subgame solving + search (not full-tree traversals).
- **Why:** Subgame solving is more tractable for very large games.
- **Scope:** Still two-player zero-sum.

**DouZero (Zha et al., ICML 2021):**
- **Keeps:** Self-play, value networks.
- **Changes:** Abandons tree-based methods entirely. Pure deep RL.
- **Why:** Dou Dizhu is three-player (zero-sum assumption breaks). CFR "is often not sound in three-player settings" (DouZero paper).
- **Scope:** Multi-player, huge action spaces.

### Evolution Chain

$$\text{CFR (2007)} \to \text{Abstraction (manual)} \to \text{Deep CFR (neural, 2019)} \to \text{ReBeL (subgame solving, 2020)} \to \text{DouZero (deep RL, multi-player, 2021)}$$

---

## 15. Project Ideas for Me

### Idea 1: Opponent Modeling via Auxiliary Network

**Research question:** Does an auxiliary network inferring opponent hand strength improve Deep CFR's convergence in FHP?

**Baseline:** Standard Deep CFR on Leduc Hold'em.

**Modification:** Add network $\text{Opp}(I, h_{opp} | \phi)$ predicting opponent strength. Include prediction as feature in value network.

**Hypothesis:** Explicit opponent modeling reduces approximation error L, improving convergence.

**Experiment:** Leduc, 100 iterations. Measure: exploitability, convergence speed (iterations to 50 mbb/g).

**Metrics:** Exploitability (mbb/g), convergence rate.

**Difficulty:** Moderate.

**Compute:** Medium (1 GPU, ~12 hours).

**Novelty:** Opponent modeling not combined with Deep CFR's value networks.

---

### Idea 2: Recurrent Value Networks for Longer Histories

**Research question:** Does an LSTM-based value network outperform MLPs on games with long bet histories?

**Baseline:** MLP value network (paper).

**Modification:** Replace MLP with LSTM over bet history sequence.

**Hypothesis:** RNNs better exploit long-range dependencies in bet history, reducing approximation error.

**Experiment:** Leduc + FHP. Measure: exploitability, MSE loss, convergence speed.

**Metrics:** Exploitability, network loss, convergence speed.

**Difficulty:** Moderate.

**Compute:** Medium (1 GPU, ~24 hours).

**Novelty:** Addresses acknowledged lack of architecture tuning.

---

### Idea 3: Memory Buffer Sensitivity

**Research question:** How sensitive is Deep CFR to buffer size? Is 40M infosets necessary?

**Baseline:** 40M infosets (paper).

**Modification:** Vary buffer size (1M, 5M, 10M, 20M, 40M).

**Hypothesis:** Smaller buffers increase reservoir sampling bias but save memory/time. Tradeoff exists.

**Experiment:** Leduc, 100 iterations, 5 buffer sizes. Measure: exploitability, memory, time.

**Metrics:** Exploitability, memory (GB), training time.

**Difficulty:** Easy.

**Compute:** Low (1 GPU, ~40 hours for 5 variants).

**Novelty:** Empirically characterizes vague "sufficiently large" requirement from Theorem 1.

---

### Idea 4: Cross-Game Transfer (FHP → HULH)

**Research question:** Does a network trained on FHP transfer to HULH?

**Baseline:** Cold-start Deep CFR on HULH.

**Modification:** Warm-start HULH with FHP's trained network.

**Hypothesis:** Learned FHP representation reduces initial approximation error in HULH, speeding convergence.

**Experiment:** FHP train to 35 mbb/g. HULH: compare cold-start vs. warm-start convergence.

**Metrics:** HULH convergence speed (iterations to 50 mbb/g), final exploitability.

**Difficulty:** Moderate (layer-wise matching).

**Compute:** High (~600 GPU-hours total).

**Novelty:** Transfer learning in imperfect-info games understudied.

---

## 16. Important Implementation Components

| **Component** | **Must Understand / Can Reuse** | **Why** | **Effort** |
|---|---|---|---|
| External sampling MCCFR (Algorithm 2) | MUST IMPLEMENT | Core sampling strategy; determines regret targets | Moderate |
| Regret matching (Eq. 4) | MUST IMPLEMENT | Converts advantages to strategies | Easy |
| Reservoir sampling | MUST IMPLEMENT | Memory management; Figure 4 critical | Easy |
| Network architecture | MUST UNDERSTAND CONCEPTUALLY | Input representation affects generalization | Moderate |
| Iteration-weighted MSE loss | MUST IMPLEMENT | Linear CFR weighting | Easy |
| Retraining from scratch | MUST IMPLEMENT | Figure 4: ~50% importance | Easy |
| Self-play loop | MUST UNDERSTAND CONCEPTUALLY | Non-standard RL paradigm | Moderate |
| Separate value + policy networks | MUST UNDERSTAND CONCEPTUALLY | Different roles in CFR theory | Moderate |
| Game tree representation | CAN REUSE (RLCard) | Simulation engine; high effort if building from scratch | High if custom |
| Best-response computation | CAN REUSE OR SKIP | Exploitability evaluation; only feasible on small games | Hard |
| Logging / visualization | CAN REUSE | Convergence plots (matplotlib) | Easy |

---

## 17. Questions I Should Ask My CS230 Staff

1. **Reproduction timeline:** Can I reproduce Deep CFR on Leduc Hold'em before the mid-quarter milestone (2 weeks), or should I use a pre-existing implementation and focus on modifications?

2. **Baseline selection:** Should I re-implement Deep CFR from scratch (learning, slow) or build on an existing codebase (e.g., PokerRL)? What codebases are recommended?

3. **Statistical rigor:** How many runs are sufficient to claim significance in a CS230 project? The paper uses 5 replicates (std ~2.25 mbb/g). Is that enough?

4. **Multi-player scope:** If my final project targets Dou Dizhu, should I focus on deep RL (DouZero-style) instead of Deep CFR, since CFR does not apply? How much does Deep CFR knowledge transfer?

5. **Compute needs:** Is 1 GPU sufficient, or do I need to request extra resources? FHP reportedly takes 1–10 GPU-days; acceptable for CS230, or should I stick to Leduc (smaller)?

---

## 18. Beginner Concepts I Need to Learn

| **Concept** | **Plain-Language Explanation** | **Where in Paper** | **NOW / SOON / OPTIONAL** |
|---|---|---|---|
| **Imperfect-information games** | Games where players have hidden info (e.g., opponent's cards). Different from MDPs (usually fully observed). | §2, p. 2 | NOW |
| **Information set (infoset)** | Group of game states indistinguishable to a player. Player chooses one strategy for entire infoset. | §2, p. 2–3 | NOW |
| **Regret** | Payoff difference between playing an action vs. current mixed strategy. High regret = should play that action more. | §2.1, p. 3 | NOW |
| **Counterfactual regret** | Regret weighted by probability of reaching an infoset. Decomposes game tree. | §2.1, p. 3 | SOON |
| **Regret matching** | Algorithm: play actions in proportion to accumulated regret. Converges to Nash equilibrium. | §2.1, p. 3 | NOW |
| **Nash equilibrium** | Strategy profile where no player benefits from unilateral deviation. Worst-case optimal in zero-sum. | §2.1, p. 3 | NOW |
| **Exploitability** | Payoff loss vs. optimal opponent (best response), compared to Nash equilibrium. Measured in mbb/g. | §2, p. 2–3 | SOON |
| **External sampling (MCCFR)** | CFR variant: traverse subset of game tree each iteration. Reduces computation at cost of variance. | §2.2, p. 3 | SOON |
| **Self-play** | Training: two agents play each other. Both improve simultaneously. | §4, p. 4 | NOW |
| **Function approximation** | Neural network approximates value/regret/policy instead of exact table. Enables scaling. | §4, p. 4 | NOW |
| **Convergence guarantee** | Mathematical proof algorithm reaches solution (Nash equilibrium) in bounded steps/samples. | Theorem 1, §4, p. 4–5 | SOON |
| **Abstraction** | Simplify game by bucketing similar states (e.g., hand strength groups). Reduces size but loses details. | §1, p. 1 | SOON |
| **MSE loss / Approximation error** | Mean squared error between predicted and true values. Theorem 1 bounds regret in terms of MSE loss L. | §6, p. 6 | SOON |
| **Reservoir sampling** | Randomized algorithm: maintain random sample from stream without storing entire stream. Used for memory management. | §4, p. 4 | OPTIONAL |

---

## 19. Paper Scorecard

> **Cross-paper calibration:** this scorecard looks at this paper alone. Ranked against the whole pack for this Dou Dizhu project, the recommended verdict is **REFERENCE ONLY**. See `00_SYNTHESIS_AND_RECOMMENDATION.md` §20.9.

| **Criterion** | **Score (1–5)** | **Rationale** |
|---|---|---|
| **Importance for my project** | 4 | Essential for two-player games; background for Dou Dizhu (three-player). |
| **Difficulty to understand** | 4 | CFR theory is non-standard. Infosets, regret, counterfactual value require time. §2–4 core; §5 engineering. |
| **Difficulty to reproduce** | 3 | Leduc: easy (1–2 weeks, 1 GPU). FHP: hard (1 GPU-week). HULH: very hard. |
| **Relevance to Dou Dizhu** | 1 | Explicitly not applicable (three-player, cooperation). Good background on why CFR fails. |
| **Relevance to self-play** | 4 | Core self-play algorithm. Both players train simultaneously, alternating. |
| **Relevance to imperfect information** | 5 | Entire paper on imperfect-info solving. Infosets, hidden cards, partial observability central. |
| **Potential for project ideas** | 4 | Many feasible ablations (opponent modeling, RNNs, memory size, transfer). |
| **Importance to read deeply** | 3 | Deep if two-player/poker project. Skim if Dou Dizhu or general RL. |

### Verdict

**READ DEEPLY** if two-player imperfect-info (poker, bridge) or interested in regret minimization.

**READ SELECTIVELY** if Dou Dizhu (read §0, §1a, §2, §8; understand why CFR fails).

**REFERENCE ONLY** if fully-observable or single-agent RL.

**First sections if selective (60–70 min total):** §0 (5 min), §1a (10 min), §2 & §8 (20 min), §5 MUST UNDERSTAND (15 min), §9 strengths/weaknesses (10 min).

---

## Appendix: Synthesis Facts

- **Paper:** Brown et al., Deep Counterfactual Regret Minimization (ICML 2019).
- **Year:** 2019.
- **Environment(s):** Heads-up Flop Hold'em (FHP ~10^12 nodes, 10^9 infosets); Heads-up Limit Texas Hold'em (HULH ~10^17 nodes, 10^14 infosets).
- **Main algorithm:** External-sampling MCCFR + MSE regression on sampled advantages + regret matching on network outputs.
- **State representation:** Game history reduced to infoset (observed cards + bet history). Card: rank/suit/ID embeddings summed. Bets: binary + float per position.
- **Action representation:** Discrete legal actions; network outputs |A(I)| logits (advantages or probabilities).
- **Self-play?** Yes; alternating traverser; both players' networks train simultaneously.
- **Opponent modeling?** Implicit via self-play; no explicit hand inference.
- **Main loss:** MSE on instantaneous regrets (value network V) and strategy probabilities (policy network Π). Samples weighted by iteration t.
- **Main contribution:** First non-tabular, theoretically justified CFR to scale to large two-player zero-sum games without manual abstraction. Theorem 1 proves convergence to ε-Nash despite function approximation error.
- **Evaluation method:** Exploitability (mbb/g, best-response play); head-to-head comparison (Table 1); sample efficiency (nodes touched); ablations (Figure 4).
- **Compute:** ~1 GPU-week FHP, ~1 GPU-month HULH (estimated). 40M infosets memory per player.
- **Main limitation:** Limited to two-player zero-sum. Linear CFR convergence not proven for Monte Carlo setting, though empirically helpful. Memory size requirement vague ("sufficiently large"). Not applicable to multi-player (e.g., Dou Dizhu) or games with huge action spaces (10^4+).
- **Best idea to borrow:** External sampling produces unbiased advantages (Lemma 1)—exactly what neural networks learn. Link game-tree sampling to function approximation targets.
