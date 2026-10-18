"""Leakage guards: an agent's decisions may depend only on what the acting player knows.

T6a  Exchanging cards between the two hidden hands, keeping their sizes and union, gives a
     state the acting player cannot tell apart; observations and decisions must not change.
T6b  The true hands (InfoSet.all_handcards) are read only through hand_belief.hand_labels,
     and only allow-listed modules may call it.
"""
import copy
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from douzero.env import hand_belief as hb
from douzero.env.env import _cards2array, get_obs
from douzero.evaluation.deep_agent import DeepAgent

REPO_ROOT = Path(__file__).resolve().parents[1]
SEAT_ORDER = ('landlord', 'landlord_up', 'landlord_down')  # GameEnv.get_infoset builds the union in this order
ADP_CHECKPOINTS = {position: REPO_ROOT / 'baselines' / 'douzero_ADP' / (position + '.ckpt')
                   for position in SEAT_ORDER}


def swap_hidden_cards(infoset, rng):
    """Copy of infoset in which the two hidden players exchange one card of different kinds.

    Only all_handcards and the seat-ordered other_hand_cards depend on how the unseen cards
    are split, so those are the fields that change. Sizes, the union and the public history
    stay the same, and the landlord keeps its still-unplayed bottom cards, so the new state
    is in the acting player's information set. Returns None if no exchange is possible.
    """
    next_player, other_player = hb.hidden_players(infoset.player_position)

    def givable(player):
        counts = Counter(infoset.all_handcards[player])
        if player == 'landlord':
            counts.subtract(infoset.three_landlord_cards)
        return sorted(card for card, num in counts.items() if num > 0)

    exchanges = [(a, b) for a in givable(next_player) for b in givable(other_player) if a != b]
    if not exchanges:
        return None
    from_next, from_other = exchanges[rng.randint(len(exchanges))]
    swapped = copy.deepcopy(infoset)
    for player, give, take in ((next_player, from_next, from_other), (other_player, from_other, from_next)):
        hand = swapped.all_handcards[player]
        hand.remove(give)
        hand.append(take)
        hand.sort()
    swapped.other_hand_cards = [card for position in SEAT_ORDER if position != swapped.player_position
                                for card in swapped.all_handcards[position]]
    return swapped


@pytest.fixture(scope='module')
def swaps(random_play_states):
    """(state, swapped infoset) for every third random-play state that admits an exchange."""
    rng = np.random.RandomState(7)
    pairs = []
    for state in random_play_states[::3]:
        swapped = swap_hidden_cards(state['infoset'], rng)
        if swapped is not None:
            pairs.append((state, swapped))
    return pairs


def differing_keys(obs_a, obs_b):
    assert obs_a.keys() == obs_b.keys()
    return [key for key in obs_a
            if not (np.array_equal(obs_a[key], obs_b[key]) if isinstance(obs_a[key], np.ndarray)
                    else obs_a[key] == obs_b[key])]


# ---------------------------------------------------------------- T6a: hidden-hand swap invariance

def test_swap_changes_the_hidden_hands_but_nothing_the_acting_player_knows(swaps):
    assert len(swaps) > 1000
    for state, swapped in swaps:
        original = state['infoset']
        assert not np.array_equal(hb.hand_labels(swapped), hb.hand_labels(original))
        np.testing.assert_array_equal(hb.union_counts(swapped), hb.union_counts(original))
        assert {p: len(hand) for p, hand in swapped.all_handcards.items()} == original.num_cards_left_dict
        assert Counter(swapped.three_landlord_cards) <= Counter(swapped.all_handcards['landlord'])
        assert swapped.player_hand_cards == original.player_hand_cards


def test_observation_is_identical_after_a_hidden_hand_swap(swaps):
    for state, swapped in swaps:
        assert differing_keys(get_obs(swapped), state['obs']) == [], (state['game'], state['step'])


def _reads_a_true_hand(infoset):
    next_player, _ = hb.hidden_players(infoset.player_position)
    return _cards2array(infoset.all_handcards[next_player])


def _reads_the_union_order(infoset):
    # other_hand_cards concatenates the hidden hands in seat order, so its order reveals them.
    first_seat = next(p for p in SEAT_ORDER if p != infoset.player_position)
    return _cards2array(infoset.other_hand_cards[:infoset.num_cards_left_dict[first_seat]])


@pytest.mark.parametrize('leaky_feature', [_reads_a_true_hand, _reads_the_union_order],
                         ids=['true_hand', 'union_order'])
def test_swap_exposes_features_that_leak_a_hidden_hand(swaps, leaky_feature):
    for state, swapped in swaps[::10]:
        assert not np.array_equal(leaky_feature(swapped), leaky_feature(state['infoset']))


@pytest.fixture(scope='module')
def adp_agents():
    missing = [path.name for path in ADP_CHECKPOINTS.values() if not path.exists()]
    if missing:
        pytest.skip('DouZero-ADP checkpoints not in baselines/douzero_ADP: {}'.format(missing))
    torch.set_num_threads(1)
    return {position: DeepAgent(position, str(path)) for position, path in ADP_CHECKPOINTS.items()}


def _adp_values(agent, infoset):
    obs = get_obs(infoset)
    with torch.no_grad():
        return agent.model.forward(torch.from_numpy(obs['z_batch']), torch.from_numpy(obs['x_batch']),
                                   return_value=True)['values']


def test_douzero_adp_values_and_actions_are_identical_after_a_hidden_hand_swap(adp_agents, swaps):
    checked = Counter()
    for state, swapped in swaps[::10]:
        original = state['infoset']
        if len(original.legal_actions) < 2:
            continue
        agent = adp_agents[original.player_position]
        assert torch.equal(_adp_values(agent, swapped), _adp_values(agent, original))
        assert agent.act(swapped) == agent.act(original)
        checked[original.player_position] += 1
    assert min(checked[position] for position in SEAT_ORDER) >= 20, checked


# ---------------------------------------------------------------- T6b: static audit of true-hand reads

SOURCE_FILES = sorted(path.relative_to(REPO_ROOT).as_posix() for path in REPO_ROOT.rglob('*.py')
                      if 'tests' not in path.relative_to(REPO_ROOT).parts)

# Adding a file to these sets is a reviewed decision: that file can see the true hidden hands.
MENTIONS_ALL_HANDCARDS = {'douzero/env/game.py',         # the engine fills InfoSet.all_handcards
                          'douzero/env/hand_belief.py'}  # hand_labels(), the only reader
MENTIONS_HAND_LABELS = {'douzero/env/hand_belief.py'}    # later: predictor-data generation, oracle agent


def _files_mentioning(name):
    pattern = re.compile(r'\b{}\b'.format(name))
    return {path for path in SOURCE_FILES
            if pattern.search((REPO_ROOT / path).read_text(encoding='utf-8', errors='replace'))}


def test_only_the_engine_and_hand_belief_mention_the_true_hands():
    assert _files_mentioning('all_handcards') == MENTIONS_ALL_HANDCARDS


def test_only_allow_listed_modules_mention_hand_labels():
    assert _files_mentioning('hand_labels') == MENTIONS_HAND_LABELS
