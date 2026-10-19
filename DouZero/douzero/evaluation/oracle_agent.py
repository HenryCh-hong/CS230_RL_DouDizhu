"""Greedy evaluation agent for one seat of the oracle arm, the positive control that sees the true hands."""
from douzero.dmc.oracle import oracle_labels
from douzero.evaluation.belief_agent import BeliefDeepAgent


class OracleAgent(BeliefDeepAgent):
    """Like BeliefDeepAgent, but for oracle checkpoints only: its belief input is the true hidden hands."""
    ARMS = ('oracle',)

    def _labels(self, infoset, x_no_action):
        return oracle_labels(infoset, self.position, x_no_action)
