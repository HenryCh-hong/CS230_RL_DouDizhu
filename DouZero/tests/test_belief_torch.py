"""T12: the torch belief operations agree with the numpy reference in douzero/env/hand_belief.py."""
import numpy as np
import torch

from douzero.dmc import belief
from douzero.env import hand_belief as hb

POSITIONS = ('landlord', 'landlord_down', 'landlord_up')


def _random_unions(rng, size):
    union = rng.randint(0, 5, size=(size, hb.NUM_KINDS))
    union[:, 13:] = rng.randint(0, 2, size=(size, 2))  # one copy of each joker
    return union


def _random_beliefs(rng, union):
    logits = rng.randn(*union.shape, hb.NUM_COUNTS) * 3
    return hb.masked_softmax(logits, hb.legal_count_mask(union))


def test_union_decoding_matches_numpy(random_play_states):
    for position in POSITIONS:
        x = np.stack([s['obs']['x_no_action'] for s in random_play_states
                      if s['infoset'].player_position == position])
        expected = hb.union_counts_from_obs(x)
        for x_no_action in (torch.from_numpy(x), torch.from_numpy(x).float()):
            np.testing.assert_array_equal(belief.union_counts_from_obs(x_no_action).numpy(), expected)


def test_legal_count_mask_matches_numpy():
    union = _random_unions(np.random.RandomState(0), 500)
    np.testing.assert_array_equal(belief.legal_count_mask(torch.from_numpy(union)).numpy(),
                                  hb.legal_count_mask(union))


def test_masked_softmax_matches_numpy_with_exact_zeros_on_illegal_counts():
    rng = np.random.RandomState(1)
    union = _random_unions(rng, 500)
    mask = hb.legal_count_mask(union)
    logits = rng.randn(500, hb.NUM_KINDS, hb.NUM_COUNTS) * 3
    probs = belief.masked_softmax(torch.from_numpy(logits), torch.from_numpy(mask))
    np.testing.assert_allclose(probs.numpy(), hb.masked_softmax(logits, mask), rtol=0, atol=1e-12)
    assert (probs[~torch.from_numpy(mask)] == 0).all()


def test_reflection_matches_numpy():
    rng = np.random.RandomState(2)
    union = _random_unions(rng, 500)
    p_next = _random_beliefs(rng, union)
    np.testing.assert_array_equal(
        belief.reflect_belief(torch.from_numpy(p_next), torch.from_numpy(union)).numpy(),
        hb.reflect_belief(p_next, union))


def test_torch_reflection_satisfies_the_conservation_identities():
    rng = np.random.RandomState(3)
    union = torch.from_numpy(_random_unions(rng, 500))
    p_next = torch.from_numpy(_random_beliefs(rng, union.numpy()))
    p_other = belief.reflect_belief(p_next, union)
    counts = torch.arange(hb.NUM_COUNTS, dtype=p_next.dtype)
    assert (p_other[~belief.legal_count_mask(union)] == 0).all()
    torch.testing.assert_close(p_other.sum(dim=-1), torch.ones(500, hb.NUM_KINDS, dtype=p_next.dtype),
                               rtol=0, atol=1e-12)
    torch.testing.assert_close((p_next + p_other) @ counts, union.to(p_next.dtype), rtol=0, atol=1e-12)
    assert torch.equal(belief.reflect_belief(p_other, union), p_next)


def test_reflecting_true_next_hands_gives_true_other_hands(random_play_states):
    labels = torch.from_numpy(np.stack([hb.hand_labels(s['infoset']) for s in random_play_states]))
    union = torch.from_numpy(np.stack([hb.union_counts(s['infoset']) for s in random_play_states]))
    reflected = belief.reflect_belief(belief.one_hot_counts(labels[:, 0]), union)
    assert torch.equal(reflected, belief.one_hot_counts(labels[:, 1]))


def test_one_hot_counts_matches_numpy():
    counts = _random_unions(np.random.RandomState(4), 500)
    np.testing.assert_array_equal(belief.one_hot_counts(torch.from_numpy(counts)).numpy(),
                                  hb.one_hot_counts(counts))


def test_flatten_is_kind_major():
    beliefs = torch.arange(3 * hb.NUM_KINDS * hb.NUM_COUNTS, dtype=torch.float64).reshape(3, 15, 5)
    flat = belief.flatten_belief(beliefs)
    assert flat.shape == (3, belief.SLOT_DIM) and belief.SLOT_DIM == 75 and belief.BELIEF_DIM == 150
    for r in range(hb.NUM_KINDS):
        for k in range(hb.NUM_COUNTS):
            assert torch.equal(flat[:, 5 * r + k], beliefs[:, r, k])
    np.testing.assert_array_equal(flat.numpy(), beliefs.numpy().reshape(3, 75))


def test_cross_entropy_per_hand_matches_numpy(random_play_states):
    rng = np.random.RandomState(5)
    states = random_play_states[::7]
    union = np.stack([hb.union_counts(s['infoset']) for s in states])
    labels = np.stack([hb.hand_labels(s['infoset']) for s in states])  # (N, 2, 15)
    mask = np.broadcast_to(hb.legal_count_mask(union)[:, None], (len(states), 2, 15, 5))
    logits = np.where(mask, rng.randn(len(states), 2, 15, 5) * 3, -np.inf)
    cross_entropy = belief.hand_cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels))
    probs = hb.masked_softmax(logits, mask)
    expected = [np.mean([hb.belief_cross_entropy(probs[i, t], labels[i, t]) for i in range(len(states))])
                for t in range(2)]
    assert cross_entropy.shape == (2,)
    np.testing.assert_allclose(cross_entropy.numpy(), expected, rtol=0, atol=1e-10)
