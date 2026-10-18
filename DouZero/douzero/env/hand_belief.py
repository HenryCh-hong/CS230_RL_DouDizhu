"""Hidden-hand belief utilities for the CS230 hand-belief study.

From the acting player's seat the two other hands are hidden, but their union U
(InfoSet.other_hand_cards, also encoded in x_no_action[54:108]) and their sizes are
public. Card conservation, H_other = U - H_next for every card kind, makes a belief over
the second hidden hand recoverable from a belief over the next player's hand.

A belief has shape (..., 15, 5): for each card kind in KINDS, a distribution over how
many copies (0-4) the target hand holds. Labels hold the true counts, shape (..., 15).

hand_labels() reads InfoSet.all_handcards, the true hands. Only training-data generation
and the oracle control may call it; tests/test_leakage.py enforces this.
"""
import math
from collections import Counter

import numpy as np

# Card kinds in DouZero's order: 3..14 (A), 17 ('2'), 20 (small joker), 30 (big joker).
KINDS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 30)
NUM_KINDS = len(KINDS)
NUM_COUNTS = 5
KIND_INDEX = {card: i for i, card in enumerate(KINDS)}

# DouZero+ predicts the hand of the player who acts next (turn order L -> D -> U -> L).
NEXT_PLAYER = {'landlord': 'landlord_down',
               'landlord_down': 'landlord_up',
               'landlord_up': 'landlord'}
OTHER_PLAYER = {'landlord': 'landlord_up',
                'landlord_down': 'landlord',
                'landlord_up': 'landlord_down'}

UNION_SLICE = slice(54, 108)
# One-hot remaining-card counts in x_no_action, as (next player, other player).
HIDDEN_SIZE_SLICES = {'landlord': (slice(287, 304), slice(270, 287)),
                      'landlord_down': (slice(398, 415), slice(378, 398)),
                      'landlord_up': (slice(378, 398), slice(398, 415))}


def hidden_players(position):
    """The two hidden players seen from position, as (next player, other player)."""
    return NEXT_PLAYER[position], OTHER_PLAYER[position]


def card_counts(cards):
    """List of cards -> (15,) count of each kind."""
    counts = np.zeros(NUM_KINDS, dtype=np.int8)
    for card, num in Counter(cards).items():
        counts[KIND_INDEX[card]] = num
    return counts


def decode_card_array(cards_array):
    """Inverse of env._cards2array: (..., 54) card array -> (..., 15) counts."""
    cards_array = np.asarray(cards_array)
    ranks = cards_array[..., :52].reshape(cards_array.shape[:-1] + (13, 4)).sum(axis=-1)
    return np.concatenate([ranks, cards_array[..., 52:]], axis=-1).astype(np.int8)


def union_counts(infoset):
    """Counts of the cards the two hidden players hold together."""
    return card_counts(infoset.other_hand_cards)


def union_counts_from_obs(x_no_action):
    return decode_card_array(np.asarray(x_no_action)[..., UNION_SLICE])


def hidden_sizes_from_obs(position, x_no_action):
    """Remaining-card counts of (next player, other player) decoded from x_no_action."""
    x_no_action = np.asarray(x_no_action)
    return np.stack([x_no_action[..., s].argmax(axis=-1) + 1
                     for s in HIDDEN_SIZE_SLICES[position]], axis=-1)


def hand_labels(infoset):
    """True counts of the hidden hands, (2, 15) with rows (next player, other player).

    Reads InfoSet.all_handcards: training-data generation and the oracle control only.
    """
    return np.stack([card_counts(infoset.all_handcards[player])
                     for player in hidden_players(infoset.player_position)])


def legal_count_mask(union):
    """(..., 15) unseen counts u -> (..., 15, 5) mask of the feasible counts k <= u."""
    return np.arange(NUM_COUNTS) <= np.asarray(union)[..., None]


def masked_softmax(logits, mask):
    """Softmax over the count axis with exactly zero mass where mask is False."""
    logits = np.where(mask, logits, -np.inf)
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def reflect_belief(belief_next, union):
    """Belief over the other hidden hand derived by conservation.

    H_other = U - H_next per kind, so P(other holds k) = P(next holds u - k) for k <= u,
    and 0 for k > u.
    """
    source = np.asarray(union)[..., None] - np.arange(NUM_COUNTS)
    feasible = source >= 0
    reflected = np.take_along_axis(np.asarray(belief_next), np.where(feasible, source, 0), axis=-1)
    return np.where(feasible, reflected, 0.0)


def expected_counts(belief):
    """(..., 15, 5) belief -> (..., 15) expected count of each kind."""
    return np.asarray(belief) @ np.arange(NUM_COUNTS)


def one_hot_counts(counts):
    """(..., 15) counts -> (..., 15, 5) one-hot beliefs."""
    return np.eye(NUM_COUNTS)[np.asarray(counts)]


def belief_cross_entropy(belief, labels):
    """Mean over card kinds of -log belief[kind, true count], in nats."""
    labels = np.asarray(labels, dtype=np.intp)[..., None]
    p_true = np.take_along_axis(np.asarray(belief), labels, axis=-1)[..., 0]
    return -np.log(p_true).mean(axis=-1)


def hypergeometric_belief(union, hand_size):
    """Card-counting baseline: the hand is a uniformly random hand_size-subset of the unseen cards.

    P(holds k of kind r) = C(u_r, k) C(N - u_r, n - k) / C(N, n), with N = sum(u), n = hand_size.
    """
    union = [int(u) for u in union]
    hand_size = int(hand_size)
    total = sum(union)
    belief = np.zeros((NUM_KINDS, NUM_COUNTS))
    for r, u in enumerate(union):
        for k in range(min(u, hand_size) + 1):
            belief[r, k] = math.comb(u, k) * math.comb(total - u, hand_size - k)
    return belief / math.comb(total, hand_size)
