"""True-hand access for the oracle arm, the positive control.

The oracle is the only arm allowed to see the true hidden hands. Training and evaluation
code reads them here and nowhere else, so the access is easy to audit: tests/test_leakage.py
checks which files may call oracle_labels.
"""
import torch

from douzero.dmc.belief import union_counts_from_obs
from douzero.env.hand_belief import hand_labels, union_counts


def oracle_labels(infoset, position, x_no_action):
    """(1, 2, 15) true counts (next, other) for the decision whose observation is x_no_action.

    Fails loudly if the infoset belongs to another seat or another state than x_no_action,
    so the labels can never be paired with the wrong decision.
    """
    if infoset.player_position != position:
        raise ValueError('infoset of {} given for a {} decision'.format(infoset.player_position, position))
    if not torch.equal(union_counts_from_obs(x_no_action.cpu()), torch.from_numpy(union_counts(infoset)).long()):
        raise ValueError('infoset and observation describe different states')
    return torch.from_numpy(hand_labels(infoset)).long()[None]
