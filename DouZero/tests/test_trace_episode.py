"""T20: the one-game trace and the one-learner-update trace run end to end and show every stage."""
import os

import pytest
import torch

import trace_episode
import trace_learner_update
from douzero.dmc import belief, inprocess, models
from douzero.dmc.arguments import parser

POSITIONS = ('landlord', 'landlord_up', 'landlord_down')
# The DouZero-ADP warm start when its checkpoints are present, otherwise random weights.
CHECKPOINTS = trace_episode.DEFAULT_CHECKPOINT_DIR if os.path.isdir(trace_episode.DEFAULT_CHECKPOINT_DIR) else ''


def test_one_game_trace_shows_every_decision_and_the_outcome():
    text, record = trace_episode.trace_episode(hand_model='derived_both', seed=0, checkpoint_dir=CHECKPOINTS)
    for section in ('ONE GAME', 'DECISIONS', 'GAME OVER', 'WHAT THE BUFFER STORES'):
        assert section in text, section
    decisions = record['decisions']
    assert len(decisions) >= 10 and sum(record['decisions_per_seat'].values()) == len(decisions)
    assert record['winner'] in ('landlord', 'farmer')
    returns = record['returns']
    assert returns['landlord'] == -returns['landlord_up'] == -returns['landlord_down'] != 0
    assert all(abs(returns['landlord']) == 2 ** record['bombs'] for _ in [0])  # ADP: +-2^bombs
    assert len(text.splitlines()) < 30 + len(decisions)  # one line per decision plus a short header and footer


@pytest.mark.parametrize('hand_model', ['none', 'next', 'learned_both', 'oracle'])
def test_the_game_trace_stores_exactly_what_act_stores(hand_model):
    flags = parser.parse_args(['--hand_model', hand_model, '--unroll_length', '4', '--num_buffers', '64',
                               '--seed', '9', '--actor_device_cpu', '--training_device', 'cpu'])
    torch.manual_seed(2)
    model = models.Model('cpu', belief_dim=belief.BELIEF_DIM)
    num_targets = belief.PREDICTOR_TARGETS[hand_model]
    predictors = {p: models.HandPredictor(p, num_targets).eval().requires_grad_(False) if num_targets else None
                  for p in POSITIONS}
    _, record = trace_episode.trace_episode(hand_model=hand_model, seed=9, model=model, predictors=predictors)
    buffers, _, full_queue = inprocess.run_actor(flags, model, predictors, max_episodes=1)
    for position in POSITIONS:
        flushed = inprocess.flushed_rows(buffers, full_queue, position)
        traced = record['stored'][position]
        assert len(flushed['target']) >= 4 and len(traced) >= len(flushed['target'])
        for k in range(len(flushed['target'])):
            for key in flushed:
                assert torch.equal(flushed[key][k], traced[k][key]), (position, k, key)


@pytest.mark.parametrize('hand_model', ['derived_both', 'none'])
def test_one_learner_update_trace(hand_model):
    text, record = trace_learner_update.trace_learner_update(hand_model=hand_model, seed=0, checkpoint_dir=CHECKPOINTS)
    for section in ('BATCH', 'BEFORE THE UPDATE', 'AFTER THE UPDATE', 'PARAMETER CHANGE'):
        assert section in text, section
    assert record['actor_synced'] and record['q_network_change'] > 0
    assert record['predictor_change'] == (0.0 if hand_model != 'none' else None)
    if hand_model == 'none':  # M0c: an always-zero input gets no update
        assert record['belief_column_change'] == [0.0, 0.0]
    else:
        assert record['belief_column_change'][0] > 0 and record['belief_column_change'][1] > 0
