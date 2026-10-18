import itertools
from collections import Counter

import numpy as np
import pytest

from douzero.env import hand_belief as hb
from douzero.env.env import _cards2array

POSITIONS = ('landlord', 'landlord_down', 'landlord_up')
DECK = [k for k in range(3, 15) for _ in range(4)] + [17] * 4 + [20, 30]


def _random_belief(rng, union):
    return hb.masked_softmax(rng.randn(hb.NUM_KINDS, hb.NUM_COUNTS) * 3, hb.legal_count_mask(union))


# ---------------------------------------------------------------- encoding

def test_card_counts_follow_douzero_kind_order():
    assert hb.KINDS == (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 30)
    expected = np.zeros(hb.NUM_KINDS, dtype=int)
    expected[[0, 11, 12, 13, 14]] = [2, 1, 3, 1, 1]
    np.testing.assert_array_equal(hb.card_counts([3, 3, 14, 17, 17, 17, 20, 30]), expected)


def test_decode_card_array_inverts_douzero_encoding():
    rng = np.random.RandomState(3)
    for _ in range(200):
        hand = [DECK[i] for i in rng.choice(len(DECK), rng.randint(0, 21), replace=False)]
        np.testing.assert_array_equal(hb.decode_card_array(_cards2array(hand)), hb.card_counts(hand))


# ---------------------------------------------------------------- T1: conservation

def test_labels_of_both_hidden_hands_sum_to_the_union(random_play_states):
    for s in random_play_states:
        labels = hb.hand_labels(s['infoset'])
        assert labels.shape == (2, hb.NUM_KINDS)
        np.testing.assert_array_equal(labels[0] + labels[1], hb.union_counts(s['infoset']))


def test_union_decoded_from_the_observation_matches_the_engine(random_play_states):
    for s in random_play_states:
        np.testing.assert_array_equal(hb.union_counts_from_obs(s['obs']['x_no_action']),
                                      hb.union_counts(s['infoset']))


# ---------------------------------------------------------------- T2: hand sizes

def test_each_label_sums_to_that_players_remaining_card_count(random_play_states):
    for s in random_play_states:
        infoset = s['infoset']
        next_player, other_player = hb.hidden_players(infoset.player_position)
        labels = hb.hand_labels(infoset)
        assert labels[0].sum() == infoset.num_cards_left_dict[next_player]
        assert labels[1].sum() == infoset.num_cards_left_dict[other_player]


def test_hidden_hand_sizes_decoded_from_the_observation_match_the_labels(random_play_states):
    for s in random_play_states:
        infoset = s['infoset']
        sizes = hb.hidden_sizes_from_obs(infoset.player_position, s['obs']['x_no_action'])
        assert tuple(sizes) == tuple(hb.hand_labels(infoset).sum(axis=1))


# ---------------------------------------------------------------- T3: seat mapping

def test_seat_mapping_matches_douzero_plus():
    # DouZero+ predicts the player who acts next (oppo_modeling/env/env.py:251, 337, 460).
    assert hb.NEXT_PLAYER == {'landlord': 'landlord_down',
                              'landlord_down': 'landlord_up',
                              'landlord_up': 'landlord'}
    for position in POSITIONS:
        next_player, other_player = hb.hidden_players(position)
        assert {position, next_player, other_player} == set(POSITIONS)
    # For landlord_down, the next player is its teammate.
    assert hb.hidden_players('landlord_down') == ('landlord_up', 'landlord')


def test_next_player_is_the_seat_that_actually_acts_next(random_play_states):
    pairs = 0
    for prev, cur in zip(random_play_states, random_play_states[1:]):
        if prev['game'] == cur['game']:
            assert hb.NEXT_PLAYER[prev['infoset'].player_position] == cur['infoset'].player_position
            pairs += 1
    assert pairs > 1000


# ---------------------------------------------------------------- T4: legal mask

def test_mask_allows_exactly_counts_up_to_the_unseen_count():
    union = np.array([0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 2, 2, 4, 1, 0])
    mask = hb.legal_count_mask(union)
    assert mask.shape == (hb.NUM_KINDS, hb.NUM_COUNTS) and mask.dtype == bool
    np.testing.assert_array_equal(mask, np.arange(hb.NUM_COUNTS)[None, :] <= union[:, None])


def test_true_labels_are_always_legal(random_play_states):
    for s in random_play_states:
        mask = hb.legal_count_mask(hb.union_counts(s['infoset']))
        labels = hb.hand_labels(s['infoset'])
        for slot in range(2):
            assert mask[np.arange(hb.NUM_KINDS), labels[slot]].all()


def test_joker_counts_above_one_are_never_legal(random_play_states):
    for s in random_play_states:
        mask = hb.legal_count_mask(hb.union_counts(s['infoset']))
        assert not mask[13:, 2:].any()


def test_masked_softmax_puts_exactly_zero_mass_on_illegal_counts(random_play_states):
    rng = np.random.RandomState(0)
    for s in random_play_states[::10]:
        mask = hb.legal_count_mask(hb.union_counts(s['infoset']))
        p = hb.masked_softmax(rng.randn(hb.NUM_KINDS, hb.NUM_COUNTS) * 3, mask)
        assert (p[~mask] == 0).all()
        assert (p[mask] > 0).all()
        np.testing.assert_allclose(p.sum(axis=-1), 1.0, rtol=0, atol=1e-12)


# ---------------------------------------------------------------- T5: reflection (M2c)

def test_reflection_follows_the_conservation_formula(random_play_states):
    rng = np.random.RandomState(1)
    for s in random_play_states[::25]:
        union = hb.union_counts(s['infoset'])
        p = _random_belief(rng, union)
        q = hb.reflect_belief(p, union)
        for r in range(hb.NUM_KINDS):
            for k in range(hb.NUM_COUNTS):
                expected = p[r, union[r] - k] if k <= union[r] else 0.0
                assert q[r, k] == expected


def test_reflection_is_normalized_and_supported_on_legal_counts(random_play_states):
    rng = np.random.RandomState(2)
    for s in random_play_states[::5]:
        union = hb.union_counts(s['infoset'])
        q = hb.reflect_belief(_random_belief(rng, union), union)
        assert (q[~hb.legal_count_mask(union)] == 0).all()
        np.testing.assert_allclose(q.sum(axis=-1), 1.0, rtol=0, atol=1e-12)


def test_expected_counts_of_next_and_reflected_other_sum_to_the_union(random_play_states):
    rng = np.random.RandomState(3)
    for s in random_play_states[::5]:
        union = hb.union_counts(s['infoset'])
        p = _random_belief(rng, union)
        total = hb.expected_counts(p) + hb.expected_counts(hb.reflect_belief(p, union))
        np.testing.assert_allclose(total, union, rtol=0, atol=1e-12)


def test_reflection_is_an_involution(random_play_states):
    rng = np.random.RandomState(4)
    for s in random_play_states[::5]:
        union = hb.union_counts(s['infoset'])
        p = _random_belief(rng, union)
        np.testing.assert_allclose(hb.reflect_belief(hb.reflect_belief(p, union), union), p, rtol=0, atol=0)


def test_reflected_belief_scores_the_other_hand_exactly_as_the_original_scores_the_next_hand(random_play_states):
    rng = np.random.RandomState(5)
    for s in random_play_states[::5]:
        union = hb.union_counts(s['infoset'])
        labels = hb.hand_labels(s['infoset'])
        p = _random_belief(rng, union)
        ce_next = hb.belief_cross_entropy(p, labels[0])
        ce_other = hb.belief_cross_entropy(hb.reflect_belief(p, union), labels[1])
        assert ce_other == pytest.approx(ce_next, rel=0, abs=1e-12)


def test_reflecting_the_true_next_hand_gives_the_true_other_hand(random_play_states):
    for s in random_play_states:
        union = hb.union_counts(s['infoset'])
        labels = hb.hand_labels(s['infoset'])
        np.testing.assert_array_equal(hb.reflect_belief(hb.one_hot_counts(labels[0]), union),
                                      hb.one_hot_counts(labels[1]))


def test_one_hot_counts_marks_each_true_count():
    counts = np.array([0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 4, 1, 0])
    one_hot = hb.one_hot_counts(counts)
    assert one_hot.shape == (hb.NUM_KINDS, hb.NUM_COUNTS)
    np.testing.assert_array_equal(one_hot.argmax(axis=-1), counts)
    np.testing.assert_array_equal(one_hot.sum(axis=-1), 1)


# ---------------------------------------------------------------- T5b: hypergeometric card counting

def test_hypergeometric_belief_matches_brute_force_enumeration():
    union = np.zeros(hb.NUM_KINDS, dtype=int)
    union[[0, 1, 13]] = [2, 3, 1]  # two 3s, three 4s, the small joker
    cards = [kind for kind in range(hb.NUM_KINDS) for _ in range(union[kind])]
    for target_size in range(len(cards) + 1):
        expected = np.zeros((hb.NUM_KINDS, hb.NUM_COUNTS))
        subsets = list(itertools.combinations(range(len(cards)), target_size))
        for subset in subsets:
            counts = Counter(cards[i] for i in subset)
            for r in range(hb.NUM_KINDS):
                expected[r, counts[r]] += 1
        expected /= len(subsets)
        np.testing.assert_allclose(hb.hypergeometric_belief(union, target_size), expected, rtol=0, atol=1e-12)


def test_hypergeometric_belief_is_normalized_legal_and_has_the_proportional_mean(random_play_states):
    for s in random_play_states[::5]:
        union = hb.union_counts(s['infoset'])
        next_size = hb.hand_labels(s['infoset'])[0].sum()
        h = hb.hypergeometric_belief(union, next_size)
        np.testing.assert_allclose(h.sum(axis=-1), 1.0, rtol=0, atol=1e-12)
        assert (h[~hb.legal_count_mask(union)] == 0).all()
        np.testing.assert_allclose(hb.expected_counts(h), next_size * union / union.sum(), rtol=0, atol=1e-12)


def test_hypergeometric_beliefs_of_the_two_hidden_hands_are_reflections(random_play_states):
    for s in random_play_states[::5]:
        union = hb.union_counts(s['infoset'])
        next_size, other_size = hb.hand_labels(s['infoset']).sum(axis=1)
        np.testing.assert_allclose(hb.reflect_belief(hb.hypergeometric_belief(union, next_size), union),
                                   hb.hypergeometric_belief(union, other_size), rtol=0, atol=1e-12)


def test_hypergeometric_belief_gives_the_true_hands_positive_probability(random_play_states):
    for s in random_play_states[::5]:
        union = hb.union_counts(s['infoset'])
        labels = hb.hand_labels(s['infoset'])
        for slot in range(2):
            h = hb.hypergeometric_belief(union, labels[slot].sum())
            assert (h[np.arange(hb.NUM_KINDS), labels[slot]] > 0).all()
