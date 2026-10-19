"""The visualization tools: the episode replay and the one-state comparison of the five arms.

The replay must show the game the actors really play, so it is checked against trace_episode
(itself checked against act()). A NORMAL replay must not depend on hidden information at all,
and the comparison must apply each arm's belief rule to exactly the same state.
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

import render_episode_replay
import render_state_comparison
from douzero.dmc import belief, models
from douzero.env import hand_belief as hb
from douzero.viz import record as rec, render
from trace_episode import trace_episode

POSITIONS = ('landlord', 'landlord_up', 'landlord_down')
REPO_ROOT = Path(__file__).resolve().parents[1]
DECK = np.array([4] * 13 + [1, 1])  # copies of each of the 15 kinds


def frozen(num_targets, seed):
    torch.manual_seed(seed)
    return {p: models.HandPredictor(p, num_targets).eval().requires_grad_(False) for p in POSITIONS}


@pytest.fixture(scope='module')
def game():
    torch.manual_seed(0)
    model = models.Model('cpu', belief_dim=belief.BELIEF_DIM)  # random weights: fast, no checkpoint needed
    model.eval()
    predictors = frozen(1, 1)
    kwargs = dict(hand_model='derived_both', seed=4, model=model, predictors=predictors, exp_epsilon=0.2)
    return dict(model=model, predictors=predictors, kwargs=kwargs,
                revealed=rec.record_episode(reveal=True, **kwargs), normal=rec.record_episode(reveal=False, **kwargs))


# ---------------------------------------------------------------- the replay shows the real game

def test_the_replay_plays_the_same_game_as_trace_episode(game):
    _, traced = trace_episode('derived_both', seed=4, checkpoint_dir='', exp_epsilon=0.2, model=game['model'],
                              predictors=game['predictors'])
    steps = game['revealed']['steps']
    assert [s['action'] for s in steps] == [sorted(d['move']) for d in traced['decisions']]
    for step, decision in zip(steps, traced['decisions']):
        assert step['seat'] == decision['seat']
        assert step['selected']['q'] == pytest.approx(decision['q'], abs=1e-6)
        slots = decision['beliefs'][0].reshape(2, hb.NUM_KINDS, hb.NUM_COUNTS).numpy()
        assert np.allclose(step['belief']['next']['probs'], slots[0], atol=1e-4)
        assert np.allclose(step['belief']['other']['probs'], slots[1], atol=1e-4)
    assert game['revealed']['result']['returns'] == traced['returns']


def test_every_recorded_decision_is_consistent(game):
    record = game['revealed']
    explored = 0
    for step, hidden in zip(record['steps'], record['hidden']['steps']):
        top_q = [a['q'] for a in step['top_actions']]
        assert top_q == sorted(top_q, reverse=True) and len(step['top_actions']) == min(5, step['legal_count'])
        exploration = step['exploration']
        assert exploration['explored'] == (exploration['draw'] < exploration['epsilon'])
        if not exploration['explored']:
            assert step['selected']['rank'] == 1  # greedy: the best-scored legal action
        explored += exploration['explored']
        public = step['public']
        assert public['cards_left'][step['seat']] == len(step['own_hand'])
        assert sum(public['unseen']) == sum(public['hidden_sizes'])
        # Conservation: the three hands and every card played so far are exactly the deck.
        held = sum(hb.card_counts(hidden['hands'][p]) for p in POSITIONS)
        played = sum(hb.card_counts(public['played'][p]) for p in POSITIONS)
        assert np.array_equal(held + played, DECK)
        assert hidden['hands'][step['seat']] == step['own_hand']
    assert explored > 0  # epsilon 0.2 explores somewhere in a game, so both branches are exercised
    result = record['result']
    assert result['returns']['landlord_up'] == result['returns']['landlord_down'] == -result['returns']['landlord']


def test_records_are_plain_json_and_reproducible(game):
    assert json.loads(json.dumps(game['revealed'])) == game['revealed']
    assert rec.record_episode(reveal=True, **game['kwargs']) == game['revealed']


# ---------------------------------------------------------------- public versus hidden

def test_a_normal_record_holds_no_hidden_information(game):
    assert 'hidden' not in game['normal'] and not game['normal']['revealed']
    assert game['normal'] == rec.strip_hidden(game['revealed'])


def test_the_normal_replay_cannot_show_what_it_was_not_given(game):
    page = render.render_replay(game['revealed'], reveal=False)
    assert page == render.render_replay(rec.strip_hidden(game['revealed']), reveal=False)
    with pytest.raises(ValueError):
        render.render_replay(game['normal'], reveal=True)  # nothing hidden was recorded, so nothing to reveal


def test_the_teaching_replay_shows_every_hand_marked_as_hidden_truth(game):
    page = render.render_replay(game['revealed'], reveal=True)
    first = page[page.index('id="t0"'):page.index('id="t1"')]
    for position in POSITIONS:
        hand = game['revealed']['hidden']['steps'][0]['hands'][position]
        assert render.cards_html(hand, 'hidden' if position != 'landlord' else 'own') in first
    assert 'hidden from the actor' in first


def test_a_normal_oracle_replay_hides_everything_computed_from_the_true_hands():
    torch.manual_seed(3)
    model = models.Model('cpu', belief_dim=belief.BELIEF_DIM).eval()  # random belief weights: the belief moves Q
    revealed = rec.record_episode('oracle', seed=4, model=model, reveal=True)
    normal = rec.record_episode('oracle', seed=4, model=model, reveal=False)
    assert normal == rec.strip_hidden(revealed)
    for step, full in zip(normal['steps'], revealed['steps']):
        assert step['belief'] is None and step['top_actions'] is None  # Q of every move used the true hands
        assert step['selected'] == dict(cards=full['selected']['cards'], type=full['selected']['type'])  # public move
    page = render.render_replay(normal)
    assert page == render.render_replay(revealed, reveal=False)  # a NORMAL page strips whatever it is given
    assert '{:+.3f}'.format(revealed['steps'][0]['selected']['q']) not in page


def test_uniform_and_card_counting_baselines_are_scored_against_the_truth(game):
    for step, hidden in zip(game['revealed']['steps'], game['revealed']['hidden']['steps']):
        unseen = step['public']['unseen']
        expected = np.mean([np.log(u + 1) for u in unseen])  # P(true count) = 1 / (u + 1) for every kind
        assert hidden['cross_entropy']['uniform']['next'] == pytest.approx(expected, abs=1e-4)
        assert hidden['cross_entropy']['uniform']['other'] == pytest.approx(expected, abs=1e-4)


def test_every_arm_replays_and_renders_in_both_modes():
    for hand_model in ('baseline', 'none', 'next', 'learned_both'):
        record = rec.record_episode(hand_model, seed=6, checkpoint_dir='', reveal=True)
        assert (record['steps'][0]['belief'] is None) == (hand_model == 'baseline')
        for reveal in (True, False):
            assert render.render_replay(record, reveal=reveal).startswith('<!doctype html>')


# ---------------------------------------------------------------- one state, five arms

def test_decision_indexes_are_checked(game):
    for t in (-1, len(game['revealed']['steps'])):
        with pytest.raises(ValueError):
            rec.rebuild_state(game['revealed'], t)


def test_the_normal_comparison_hides_the_oracle_and_cannot_be_revealed_later(game):
    t = len(game['revealed']['steps']) // 2
    normal = rec.compare_state(game['revealed'], t, k1_predictors=frozen(1, 1), k2_predictors=frozen(2, 2),
                               checkpoint_dir='', reveal=False)
    oracle = next(a for a in normal['arms'] if a['hand_model'] == 'oracle')
    assert oracle['belief'] is None and oracle['top_actions'] is None and oracle['greedy_action'] is None
    assert 'hidden' not in normal
    revealed = rec.compare_state(game['revealed'], t, k1_predictors=frozen(1, 1), k2_predictors=frozen(2, 2),
                                 checkpoint_dir='', reveal=True)
    assert rec.strip_comparison(revealed) == normal
    assert render.render_comparison(revealed, reveal=False) == render.render_comparison(normal)
    assert '>Oracle<' in render.render_comparison(normal)
    with pytest.raises(ValueError):
        rec.compare_state(game['normal'], t, checkpoint_dir='', reveal=True)  # nothing hidden was recorded


def test_a_state_is_rebuilt_exactly_from_the_record(game):
    for t in (0, 5, len(game['revealed']['steps']) - 1):
        state = rec.rebuild_state(game['revealed'], t)
        step = game['revealed']['steps'][t]
        assert state.position == step['seat']
        assert sorted(state.infoset.player_hand_cards) == step['own_hand']
        assert rec.describe_public(state.infoset) == step['public']


def test_the_comparison_applies_each_arms_belief_rule_to_the_same_state(game):
    t = len(game['revealed']['steps']) // 2
    comparison = rec.compare_state(game['revealed'], t, k1_predictors=frozen(1, 1), k2_predictors=frozen(2, 2),
                                   checkpoint_dir='')
    arms = comparison['arms']
    assert [a['hand_model'] for a in arms] == ['none', 'next', 'derived_both', 'learned_both', 'oracle']
    probs = {a['hand_model']: {s: np.array(a['belief'][s]['probs']) for s in ('next', 'other')} for a in arms}
    unseen = comparison['state']['public']['unseen']
    true = comparison['hidden']['true_counts']
    assert not probs['none']['next'].any() and not probs['none']['other'].any()
    assert np.array_equal(probs['next']['next'], probs['derived_both']['next'])  # one shared K=1 predictor
    assert not probs['next']['other'].any()
    assert np.allclose(probs['derived_both']['other'], hb.reflect_belief(probs['derived_both']['next'], unseen),
                       atol=1e-4)
    assert np.array_equal(probs['oracle']['next'], hb.one_hot_counts(true['next']))
    assert np.array_equal(probs['oracle']['other'], hb.one_hot_counts(true['other']))
    assert probs['learned_both']['other'].any() and not np.allclose(probs['learned_both']['next'], probs['next']['next'])
    # Warm-started Q-networks have zero belief weights, so every arm still ranks the actions identically.
    assert comparison['identical_rankings']
    assert len({json.dumps(a['top_actions']) for a in arms}) == 1


def test_the_comparison_page_shows_all_five_arms(game):
    t = len(game['revealed']['steps']) // 2
    comparison = rec.compare_state(game['revealed'], t, k1_predictors=frozen(1, 1), k2_predictors=frozen(2, 2),
                                   checkpoint_dir='')
    page = render.render_comparison(comparison, reveal=True)
    for name in ('M0c', 'M1', 'M2c', 'M2', 'Oracle'):
        assert '>{}<'.format(name) in page
    assert page == render.render_comparison(comparison)  # reveal defaults to what was recorded


# ---------------------------------------------------------------- pages, command lines, isolation

def test_pages_are_self_contained_and_escape_text(game):
    tampered = json.loads(json.dumps(game['revealed']))
    tampered['predictors'] = '<script>alert(1)</script>'
    page = render.render_replay(tampered, reveal=True)
    assert '<script>alert(1)</script>' not in page and '&lt;script&gt;' in page
    assert not re.search(r'(src|href)="(https?:)?//', page)  # no network access: works offline
    assert page.count('<section class="step"') == len(game['revealed']['steps']) + 1  # every decision + the end


def test_the_command_lines_write_record_and_page(tmp_path):
    replay = tmp_path / 'replay'
    render_episode_replay.main(['--seed', '2', '--checkpoint_dir', '', '--reveal', '--out', str(replay)])
    record = json.loads((tmp_path / 'replay.json').read_text())
    assert record['schema'] == rec.SCHEMA_EPISODE and (tmp_path / 'replay.html').read_text().startswith('<!doctype html>')
    render_state_comparison.main(['--record', str(tmp_path / 'replay.json'), '--step', '3', '--checkpoint_dir', '',
                                  '--out', str(tmp_path / 'state')])
    assert json.loads((tmp_path / 'state.json').read_text())['schema'] == rec.SCHEMA_COMPARISON
    assert (tmp_path / 'state.html').exists()


def test_training_and_evaluation_never_import_the_visualization():
    for path in [*(REPO_ROOT / 'douzero').rglob('*.py'), REPO_ROOT / 'train.py', REPO_ROOT / 'evaluate.py']:
        if 'viz' not in path.relative_to(REPO_ROOT).parts:
            assert not re.search(r'\bviz\b', path.read_text()), path
