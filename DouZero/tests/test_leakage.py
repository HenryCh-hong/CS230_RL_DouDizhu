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

from douzero.dmc import belief, inprocess, models, oracle, utils
from douzero.dmc.arguments import parser
from douzero.env import hand_belief as hb
from douzero.env.env import Env, _cards2array, get_obs
from douzero.evaluation.belief_agent import BeliefDeepAgent
from douzero.evaluation.deep_agent import DeepAgent
from douzero.evaluation.oracle_agent import OracleAgent
from helpers import saved_run

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


@pytest.fixture(scope='module')
def belief_arms():
    """Per seat, randomly initialized arms with frozen predictors; M1 and M2c share one."""
    torch.manual_seed(0)
    arms = {}
    for position in SEAT_ORDER:
        next_predictor = models.HandPredictor(position, 1).eval().requires_grad_(False)
        predictors = {'none': None, 'next': next_predictor, 'derived_both': next_predictor,
                      'learned_both': models.HandPredictor(position, 2).eval().requires_grad_(False), 'oracle': None}
        arms[position] = {hand_model: models.make_arm(position, hand_model, predictor)
                          for hand_model, predictor in predictors.items()}
    return arms


def _arm_decision(arm, hand_model, infoset):
    """Belief and Q-values of one arm at infoset, through the actor's own code (actor_belief, oracle_labels)."""
    q_network, predictor = arm
    obs = get_obs(infoset)
    x_no_action = torch.from_numpy(obs['x_no_action'])
    labels = oracle.oracle_labels(infoset, infoset.player_position, x_no_action) if hand_model == 'oracle' else None
    beliefs = utils.actor_belief(hand_model, predictor, infoset.player_position, x_no_action,
                                 torch.from_numpy(obs['z']), labels)
    with torch.no_grad():
        values = q_network(torch.from_numpy(obs['z_batch']), torch.from_numpy(obs['x_batch']), return_value=True,
                           belief=beliefs.expand(len(obs['legal_actions']), -1))['values']
    return beliefs, values


def test_belief_arm_decisions_are_identical_after_a_hidden_hand_swap(belief_arms, swaps):
    for state, swapped in swaps[::10]:
        arms = belief_arms[state['infoset'].player_position]
        for hand_model in ('none', 'next', 'derived_both', 'learned_both'):
            beliefs, values = _arm_decision(arms[hand_model], hand_model, state['infoset'])
            swapped_beliefs, swapped_values = _arm_decision(arms[hand_model], hand_model, swapped)
            assert torch.equal(swapped_beliefs, beliefs) and torch.equal(swapped_values, values), hand_model


def test_the_oracle_arm_sees_the_hidden_hand_swap(belief_arms, swaps):
    for state, swapped in swaps[::10]:
        arm = belief_arms[state['infoset'].player_position]['oracle']
        beliefs, _ = _arm_decision(arm, 'oracle', state['infoset'])
        swapped_beliefs, _ = _arm_decision(arm, 'oracle', swapped)
        assert not torch.equal(swapped_beliefs, beliefs)


# ---------------------------------------------------------------- T21: swap invariance of the wired actor and agents

class SwapEnv(Env):
    """Env that, at every decision where a swap exists, swaps the hidden hands and hands the actor the
    swapped infoset and the observation built from it."""
    swap = True

    def __init__(self, objective):
        super().__init__(objective)
        self.rng, self.swapped = np.random.RandomState(0), 0

    def _observe(self):
        swapped = swap_hidden_cards(self.infoset, self.rng) if self.swap else None
        if swapped is not None:
            self.infoset, self.swapped = swapped, self.swapped + 1
        return self._leak(get_obs(self.infoset))

    def _leak(self, obs):
        return obs

    def reset(self):
        super().reset()
        return self._observe()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return (obs if done else self._observe()), reward, done, info


class LeakyEnv(SwapEnv):
    """A deliberately broken observation: the own-hand features carry the next player's true hand."""
    swap = False

    def _leak(self, obs):
        next_player, _ = hb.hidden_players(self.infoset.player_position)
        leaked = _cards2array(self.infoset.all_handcards[next_player])
        obs['x_no_action'] = np.concatenate([leaked, obs['x_no_action'][54:]])
        obs['x_batch'] = obs['x_batch'].copy()
        obs['x_batch'][:, :54] = leaked
        return obs


class LeakySwapEnv(LeakyEnv):
    swap = True


def _stored_games(hand_model, env_class, monkeypatch):
    """Two games of the real act() for one arm, as flushed rows per seat, and the env it played in."""
    flags = parser.parse_args(['--hand_model', hand_model, '--unroll_length', '4', '--num_buffers', '100',
                               '--seed', '7', '--actor_device_cpu', '--training_device', 'cpu'])
    torch.manual_seed(1)
    model = models.Model('cpu', belief_dim=belief.BELIEF_DIM)  # random belief weights, so beliefs move Q
    num_targets = belief.PREDICTOR_TARGETS[hand_model]
    predictors = {p: models.HandPredictor(p, num_targets).eval().requires_grad_(False) if num_targets else None
                  for p in SEAT_ORDER}
    made = {}
    monkeypatch.setattr(utils, 'create_env', lambda flags: made.setdefault('env', env_class(flags.objective)))
    buffers, _, full_queue = inprocess.run_actor(flags, model, predictors, max_episodes=2)
    return {p: inprocess.flushed_rows(buffers, full_queue, p) for p in SEAT_ORDER}, made['env']


@pytest.mark.parametrize('hand_model', ['none', 'next', 'derived_both', 'learned_both', 'oracle'])
def test_act_stores_the_same_games_when_every_infoset_is_swapped(hand_model, monkeypatch):
    plain, _ = _stored_games(hand_model, Env, monkeypatch)
    swapped, env = _stored_games(hand_model, SwapEnv, monkeypatch)
    assert env.swapped > 50
    identical = all(torch.equal(plain[p][key], swapped[p][key]) for p in SEAT_ORDER for key in plain[p])
    assert identical == (hand_model != 'oracle')  # only the oracle arm sees the true hands


def test_the_act_swap_test_catches_an_observation_that_leaks(monkeypatch):
    plain, _ = _stored_games('next', LeakyEnv, monkeypatch)
    swapped, env = _stored_games('next', LeakySwapEnv, monkeypatch)
    assert env.swapped > 50
    assert not all(torch.equal(plain[p][key], swapped[p][key]) for p in SEAT_ORDER for key in plain[p])


def test_evaluation_agents_are_blind_to_the_swap_except_the_oracle(tmp_path, swaps):
    agents = {}
    for hand_model, agent_class in (('derived_both', BeliefDeepAgent), ('oracle', OracleAgent)):
        path = saved_run(hand_model, tmp_path / hand_model)[0]
        agents[hand_model] = {p: agent_class(p, path) for p in SEAT_ORDER}
    for state, swapped in swaps[::20]:
        position = state['infoset'].player_position
        m2c, oracle_agent = agents['derived_both'][position], agents['oracle'][position]
        assert torch.equal(m2c.values(swapped), m2c.values(state['infoset']))
        assert m2c.act(swapped) == m2c.act(state['infoset'])
        assert not torch.equal(oracle_agent.belief(swapped), oracle_agent.belief(state['infoset']))


# ---------------------------------------------------------------- T6b: static audit of true-hand reads

SOURCE_FILES = sorted(path.relative_to(REPO_ROOT).as_posix() for path in REPO_ROOT.rglob('*.py')
                      if path.relative_to(REPO_ROOT).parts[0] not in ('tests', 'artifacts', 'baselines'))

# Adding a file to these sets is a reviewed decision: that file can see the true hidden hands.
MENTIONS_ALL_HANDCARDS = {'douzero/env/game.py',         # the engine fills InfoSet.all_handcards
                          'douzero/env/hand_belief.py'}  # hand_labels(), the only reader
MENTIONS_HAND_LABELS = {'douzero/env/hand_belief.py',    # defines it
                        'douzero/dmc/oracle.py',         # oracle_labels(), the oracle arm's one access point
                        'douzero/predictor/data.py',     # Stage 4 dataset: the training labels, stored as counts
                        'trace_decision.py'}             # debug trace: shows the labels, runs the oracle arm
# Each caller of oracle_labels can hand true hands to the oracle arm, so each is a reviewed oracle path.
MENTIONS_ORACLE_LABELS = {'douzero/dmc/oracle.py',               # defines it
                          'douzero/dmc/utils.py',                # act(), for the oracle arm only
                          'douzero/evaluation/oracle_agent.py',  # OracleAgent
                          'trace_episode.py',                    # debug trace of one game
                          'douzero/viz/record.py'}               # replay/comparison: oracle arm and teaching reveal;
                                                                 # never imported by training (tests/test_viz.py)


def _files_mentioning(name):
    pattern = re.compile(r'\b{}\b'.format(name))
    return {path for path in SOURCE_FILES
            if pattern.search((REPO_ROOT / path).read_text(encoding='utf-8', errors='replace'))}


def test_only_the_engine_and_hand_belief_mention_the_true_hands():
    assert _files_mentioning('all_handcards') == MENTIONS_ALL_HANDCARDS


def test_only_allow_listed_modules_mention_hand_labels():
    assert _files_mentioning('hand_labels') == MENTIONS_HAND_LABELS


def test_only_reviewed_oracle_paths_call_oracle_labels():
    assert _files_mentioning('oracle_labels') == MENTIONS_ORACLE_LABELS


def test_only_the_engine_and_count_decoders_touch_the_seat_ordered_union():
    # other_hand_cards concatenates the hidden hands in seat order; only order-free counts of it may be used.
    assert _files_mentioning('other_hand_cards') == {'douzero/env/env.py', 'douzero/env/game.py',
                                                    'douzero/env/hand_belief.py'}


def test_only_the_engine_touches_its_own_state():
    # Reaching past the InfoSet into the engine's per-seat state would also expose the hidden hands.
    assert _files_mentioning('info_sets') == {'douzero/env/game.py'}
    assert _files_mentioning('_env') == {'douzero/env/env.py'}
