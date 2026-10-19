"""Greedy evaluation agent for one seat of a trained hand-belief arm (M0c, M1, M2c or M2)."""
import torch

from douzero.dmc.models import load_arm_checkpoint
from douzero.dmc.utils import actor_belief
from douzero.env.env import get_obs


class BeliefDeepAgent:
    """Plays one seat greedily with a run's Q-network and frozen predictor, loaded strictly from its model.tar.

    The arm is read from the checkpoint's metadata, never chosen by the caller. The oracle, which
    reads the true hands, has its own OracleAgent; DouZero weights files use DeepAgent.
    """
    ARMS = ('none', 'next', 'derived_both', 'learned_both')

    def __init__(self, position, checkpoint_path, predictor_dir=None, allow_untrained=False):
        arm = load_arm_checkpoint(checkpoint_path, predictor_dir, allow_untrained=allow_untrained)
        if arm['hand_model'] not in self.ARMS:
            raise ValueError('{} holds a {!r} run, but {} plays only {}'.format(
                checkpoint_path, arm['hand_model'], type(self).__name__, ', '.join(self.ARMS)))
        self.position, self.hand_model = position, arm['hand_model']
        self.model, self.predictor = arm['q_networks'][position], arm['predictors'][position]

    def _labels(self, infoset, x_no_action):
        return None  # these arms never see the true hands

    def _belief(self, infoset, obs):
        if infoset.player_position != self.position:
            raise ValueError('a {} agent was asked to play for {}'.format(self.position, infoset.player_position))
        x_no_action = torch.from_numpy(obs['x_no_action'])
        return actor_belief(self.hand_model, self.predictor, self.position, x_no_action, torch.from_numpy(obs['z']),
                            self._labels(infoset, x_no_action))

    def belief(self, infoset):
        """(1, 150) belief input at infoset."""
        return self._belief(infoset, get_obs(infoset))

    def values(self, infoset):
        """Q of every legal action at infoset."""
        obs = get_obs(infoset)
        beliefs = self._belief(infoset, obs).expand(len(obs['legal_actions']), -1)
        with torch.no_grad():
            return self.model(torch.from_numpy(obs['z_batch']), torch.from_numpy(obs['x_batch']), return_value=True,
                              belief=beliefs)['values'][:, 0]

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]
        return infoset.legal_actions[int(torch.argmax(self.values(infoset)))]
