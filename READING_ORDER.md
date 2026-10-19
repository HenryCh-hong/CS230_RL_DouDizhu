# Recommended Reading Schedule

## Pass 1 — Project discovery

### 1. DouZero
Read first: Abstract, Introduction, Figure 1/2, Method overview, Experiments, Conclusion.
Do not get stuck on every tensor dimension yet.
Focus on: why DQN/A3C struggle, Deep Monte Carlo, action encoding, self-play, evaluation.

### 2. Combinatorial Q-Learning for Dou Di Zhu
Focus on: how an earlier method handles the huge/combinatorial action space and how it differs from DouZero.

### 3. DeltaDou
Focus on: MCTS, hidden-card inference, policy/value networks, self-play, compute cost, human heuristics.

### 4. DouZero+
Focus on: what limitation of DouZero is targeted, opponent modeling, coach-guided learning, and the ablations.

After these four, stop and draft 3 candidate CS230 research questions.

## Pass 2 — Imperfect-information foundations

### 5. NFSP
Focus on: fictitious self-play, approximate equilibrium, why ordinary self-play can be unstable.

### 6. Deep CFR
Focus on: information sets, regret, function approximation, evaluation via exploitability.

You do not need to master CFR mathematics immediately; understand the motivation and framework.

## Pass 3 — Inspiration / extensions

### 7. RLCard
Focus on: environments, state/action/reward interfaces, supported baselines. This is especially useful for implementation planning.

### 8. Suphx
Focus on: Mahjong-specific partial observability, oracle guiding, global reward prediction, run-time adaptation.

### 9. ReBeL
Advanced. Read last. Focus on the big idea: combining self-play, search, value learning, and game-theoretic reasoning.

## Suggested first-week workload

Day 1-2: DouZero Abstract + Introduction + Method overview
Day 3: DouZero experiments + notes
Day 4: CQL introduction/method/conclusion
Day 5: DeltaDou introduction/method/conclusion
Weekend: DouZero+ + comparison table across the four papers
