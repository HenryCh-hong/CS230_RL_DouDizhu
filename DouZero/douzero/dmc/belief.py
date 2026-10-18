"""Torch belief operations and the common belief input of the CS230 hand-belief study.

These are the torch counterparts of the numpy reference in douzero/env/hand_belief.py, for
use inside models without leaving torch; tests/test_belief_torch.py checks that they agree.

Every Q-network arm receives the same (B, 150) belief tensor [next-player slot | other-player
slot]. Each 75-wide slot is a (15, 5) belief flattened kind-major (index 5 * kind + count),
with kinds in hand_belief.KINDS order. build_belief() is the only place where the arms differ.
"""
import torch
import torch.nn.functional as F

from douzero.env.hand_belief import NUM_COUNTS, NUM_KINDS, UNION_SLICE

SLOT_DIM = NUM_KINDS * NUM_COUNTS
BELIEF_DIM = 2 * SLOT_DIM

# --hand_model choices, with the number of hands each arm's predictor outputs (0: no predictor).
PREDICTOR_TARGETS = {'none': 0,          # M0c
                     'next': 1,          # M1
                     'derived_both': 1,  # M2c: the M1 predictor plus its conservation reflection
                     'learned_both': 2,  # M2
                     'oracle': 0}        # true hands, positive control only
HAND_MODELS = tuple(PREDICTOR_TARGETS)


def union_counts_from_obs(x_no_action):
    """(..., D) x_no_action -> (..., 15) counts of the unseen cards, decoded from x[54:108]."""
    cards = x_no_action[..., UNION_SLICE].long()
    ranks = cards[..., :52].unflatten(-1, (13, 4)).sum(dim=-1)
    return torch.cat([ranks, cards[..., 52:]], dim=-1)


def legal_count_mask(union):
    """(..., 15) unseen counts u -> (..., 15, 5) mask of the feasible counts k <= u."""
    return torch.arange(NUM_COUNTS, device=union.device) <= union.unsqueeze(-1)


def mask_logits(logits, mask):
    """Logits with infeasible counts set to -inf, so that their probability is exactly zero."""
    return logits.masked_fill(~mask, float('-inf'))


def masked_softmax(logits, mask):
    return torch.softmax(mask_logits(logits, mask), dim=-1)


def reflect_belief(belief_next, union):
    """Belief over the other hidden hand derived by conservation.

    P(other holds k) = P(next holds u - k) for k <= u, and 0 for k > u.
    """
    source = union.unsqueeze(-1) - torch.arange(NUM_COUNTS, device=union.device)
    feasible = source >= 0
    return belief_next.gather(-1, source.clamp(min=0)).masked_fill(~feasible, 0.0)


def one_hot_counts(counts):
    """(..., 15) counts -> (..., 15, 5) one-hot beliefs."""
    return F.one_hot(counts.long(), NUM_COUNTS).float()


def flatten_belief(belief):
    """(..., 15, 5) belief -> (..., 75), kind-major: index 5 * kind + count."""
    return belief.flatten(start_dim=-2)


def hand_cross_entropy(logits, labels):
    """Masked logits (B, K, 15, 5) and true counts (B, K, 15) -> (K,) cross-entropy per hand.

    Each entry is the mean over states and card kinds of -log p(true count), in nats, as in
    DouZero+. For the two-hand predictor the entries are CE_next and CE_other.
    """
    nll = F.cross_entropy(logits.permute(0, 3, 1, 2), labels.long(), reduction='none')
    return nll.mean(dim=(0, 2))


def build_belief(hand_model, x_no_action, probs=None, labels=None):
    """The (B, 150) belief input of one arm: [next-player slot | other-player slot].

    probs: (B, K, 15, 5) predictor probabilities with K = PREDICTOR_TARGETS[hand_model]. They
        are detached here, so a Q-loss can never reach a predictor.
    labels: (B, 2, 15) true counts (next, other). Only the oracle arm takes them.
    """
    if hand_model not in PREDICTOR_TARGETS:
        raise ValueError('unknown hand_model {!r}'.format(hand_model))
    if (labels is not None) != (hand_model == 'oracle'):
        raise ValueError('only the oracle arm takes true hands')
    num_targets = PREDICTOR_TARGETS[hand_model]
    if (0 if probs is None else probs.shape[1]) != num_targets:
        raise ValueError('{} takes predictor probabilities for {} hands'.format(hand_model, num_targets))

    zeros = torch.zeros(len(x_no_action), SLOT_DIM, device=x_no_action.device)
    if hand_model == 'none':
        return torch.cat([zeros, zeros], dim=-1)
    if hand_model == 'oracle':
        return flatten_belief(one_hot_counts(labels)).flatten(start_dim=1)
    probs = probs.detach()
    p_next = flatten_belief(probs[:, 0])
    if hand_model == 'next':
        return torch.cat([p_next, zeros], dim=-1)
    if hand_model == 'derived_both':
        p_other = reflect_belief(probs[:, 0], union_counts_from_obs(x_no_action))
        return torch.cat([p_next, flatten_belief(p_other)], dim=-1)
    return flatten_belief(probs).flatten(start_dim=1)  # learned_both: both hands from one predictor
