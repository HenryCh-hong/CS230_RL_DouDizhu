"""Model-layer tests for the hand-belief arms.

T8   shapes, legal masks, belief semantics and finite outputs for every seat and arm
T9   gradient flow: beliefs reach Q detached, and unused belief slots get exactly zero gradient
T10  function-preserving warm start from DouZero-ADP, and checkpoint round trips
T11  matched architecture: every arm has the same Q-network; predictors are separate modules
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from douzero.dmc import belief, models
from douzero.env import hand_belief as hb

POSITIONS = ('landlord', 'landlord_down', 'landlord_up')
X_NO_ACTION_DIM = {'landlord': 319, 'landlord_down': 430, 'landlord_up': 430}
ARMS = {'none': 0, 'next': 1, 'derived_both': 1, 'learned_both': 2, 'oracle': 0}  # --hand_model -> hands predicted
ADP_DIR = Path(__file__).resolve().parents[1] / 'baselines' / 'douzero_ADP'
WARM_START_ATOL = 1e-6


def _linear(fan_in, fan_out):
    return fan_in * fan_out + fan_out


LSTM_PARAMS = 4 * 128 * (162 + 128) + 2 * 4 * 128


def _num_params(module):
    return sum(p.numel() for p in module.parameters())


@pytest.fixture(scope='module')
def batches(random_play_states):
    """Per seat, a learner-style batch of 12 real decisions: one (state, first legal action) per row."""
    result = {}
    for position in POSITIONS:
        states = [s for s in random_play_states if s['infoset'].player_position == position][::40][:12]
        result[position] = dict(
            z=torch.from_numpy(np.stack([s['obs']['z'] for s in states])).float(),
            x_no_action=torch.from_numpy(np.stack([s['obs']['x_no_action'] for s in states])).float(),
            x=torch.from_numpy(np.stack([s['obs']['x_batch'][0] for s in states])),
            labels=torch.from_numpy(np.stack([hb.hand_labels(s['infoset']) for s in states])).long())
    return result


def _fresh_predictor(position, hand_model):
    """An untrained predictor of the size the arm needs (None for M0c and the oracle)."""
    return models.HandPredictor(position, ARMS[hand_model]) if ARMS[hand_model] else None


def _run_arm(position, hand_model, batch):
    q_network, predictor = models.make_arm(position, hand_model, _fresh_predictor(position, hand_model))
    out = predictor(batch['z'], batch['x_no_action']) if predictor is not None else None
    beliefs = belief.build_belief(hand_model, batch['x_no_action'],
                                  probs=None if out is None else out['probs'],
                                  labels=batch['labels'] if hand_model == 'oracle' else None)
    values = q_network(batch['z'], batch['x'], return_value=True, belief=beliefs)['values']
    return q_network, predictor, out, beliefs, values


# ---------------------------------------------------------------- T8: shapes, masks, semantics

def test_hand_model_names_and_predictor_sizes():
    assert belief.HAND_MODELS == tuple(ARMS)
    assert belief.PREDICTOR_TARGETS == ARMS


@pytest.mark.parametrize('hand_model', tuple(ARMS))
@pytest.mark.parametrize('position', POSITIONS)
def test_every_arm_produces_finite_outputs_of_the_right_shapes(position, hand_model, batches):
    torch.manual_seed(0)
    batch = batches[position]
    size = len(batch['z'])
    _, predictor, out, beliefs, values = _run_arm(position, hand_model, batch)
    num_targets = ARMS[hand_model]
    if num_targets:
        assert out['logits'].shape == out['probs'].shape == (size, num_targets, 15, 5)
        union = hb.union_counts_from_obs(batch['x_no_action'].numpy())
        legal = torch.from_numpy(hb.legal_count_mask(union))[:, None].expand(size, num_targets, 15, 5)
        assert (out['probs'][~legal] == 0).all() and (out['probs'][legal] > 0).all()
        assert torch.isneginf(out['logits'][~legal]).all() and torch.isfinite(out['logits'][legal]).all()
        torch.testing.assert_close(out['probs'].sum(dim=-1), torch.ones(size, num_targets, 15))
    else:
        assert predictor is None
    assert beliefs.shape == (size, belief.BELIEF_DIM) and torch.isfinite(beliefs).all()
    assert values.shape == (size, 1) and torch.isfinite(values).all()


@pytest.mark.parametrize('position', POSITIONS)
def test_belief_slots_follow_the_arm_definitions(position, batches):
    torch.manual_seed(0)
    batch = batches[position]
    size, x_no_action, labels = len(batch['z']), batch['x_no_action'], batch['labels'].numpy()
    next_predictor = models.HandPredictor(position, 1)  # M1 and M2c share it
    p_next = next_predictor(batch['z'], x_no_action)['probs']
    p_both = models.HandPredictor(position, 2)(batch['z'], x_no_action)['probs']
    p_next_np, p_both_np = p_next.detach().numpy(), p_both.detach().numpy()
    union = hb.union_counts_from_obs(x_no_action.numpy())
    zeros = np.zeros((size, 75), dtype=np.float32)
    expected = {
        'none': (zeros, zeros),
        'next': (p_next_np[:, 0].reshape(size, 75), zeros),
        'derived_both': (p_next_np[:, 0].reshape(size, 75),
                         hb.reflect_belief(p_next_np[:, 0], union).reshape(size, 75)),
        'learned_both': (p_both_np[:, 0].reshape(size, 75), p_both_np[:, 1].reshape(size, 75)),
        'oracle': (hb.one_hot_counts(labels[:, 0]).reshape(size, 75),
                   hb.one_hot_counts(labels[:, 1]).reshape(size, 75)),
    }
    inputs = {'none': {}, 'next': dict(probs=p_next), 'derived_both': dict(probs=p_next),
              'learned_both': dict(probs=p_both), 'oracle': dict(labels=batch['labels'])}
    for hand_model, (first_slot, second_slot) in expected.items():
        beliefs = belief.build_belief(hand_model, x_no_action, **inputs[hand_model]).numpy()
        np.testing.assert_array_equal(beliefs[:, :75], first_slot, err_msg=hand_model)
        np.testing.assert_array_equal(beliefs[:, 75:], second_slot, err_msg=hand_model)


def test_only_the_oracle_arm_accepts_true_hands_and_predictor_sizes_must_match(batches):
    batch = batches['landlord']
    x_no_action, labels = batch['x_no_action'], batch['labels']
    p_next = models.HandPredictor('landlord', 1)(batch['z'], x_no_action)['probs']
    p_both = models.HandPredictor('landlord', 2)(batch['z'], x_no_action)['probs']
    invalid = [('none', dict(labels=labels)), ('next', dict(probs=p_next, labels=labels)),
               ('derived_both', dict(probs=p_next, labels=labels)),
               ('learned_both', dict(probs=p_both, labels=labels)),
               ('oracle', {}), ('oracle', dict(probs=p_both, labels=labels)), ('none', dict(probs=p_next)),
               ('next', {}), ('next', dict(probs=p_both)), ('learned_both', dict(probs=p_next)),
               ('both', dict(probs=p_both))]
    for hand_model, kwargs in invalid:
        with pytest.raises(ValueError):
            belief.build_belief(hand_model, x_no_action, **kwargs)


# ---------------------------------------------------------------- T9: gradient flow

@pytest.mark.parametrize('hand_model', ['next', 'derived_both', 'learned_both'])
def test_q_loss_never_reaches_the_predictor_but_cross_entropy_does(hand_model, batches):
    torch.manual_seed(0)
    batch = batches['landlord_up']
    q_network, predictor = models.make_arm('landlord_up', hand_model, _fresh_predictor('landlord_up', hand_model))
    out = predictor(batch['z'], batch['x_no_action'])
    assert out['probs'].requires_grad
    beliefs = belief.build_belief(hand_model, batch['x_no_action'], probs=out['probs'])
    assert not beliefs.requires_grad
    values = q_network(batch['z'], batch['x'], return_value=True, belief=beliefs)['values']
    ((values.squeeze(-1) - 1.0) ** 2).mean().backward()
    assert all(p.grad is None for p in predictor.parameters())
    assert all(p.grad is not None for p in q_network.parameters())

    belief.hand_cross_entropy(out['logits'], batch['labels'][:, :ARMS[hand_model]]).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in predictor.parameters())
    assert predictor.lstm.weight_ih_l0.grad.abs().sum() > 0 and predictor.dense1.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('hand_model, filled_slots', [('none', ()), ('next', (0,)), ('derived_both', (0, 1)),
                                                      ('learned_both', (0, 1)), ('oracle', (0, 1))])
def test_only_filled_belief_slots_receive_gradient(hand_model, filled_slots, batches):
    torch.manual_seed(0)
    batch = batches['landlord']
    q_network, _, _, _, values = _run_arm('landlord', hand_model, batch)
    ((values.squeeze(-1) - 1.0) ** 2).mean().backward()
    belief_columns = q_network.dense1.weight.grad[:, -belief.BELIEF_DIM:]
    for slot in (0, 1):
        has_gradient = bool(belief_columns[:, 75 * slot:75 * (slot + 1)].abs().sum() > 0)
        assert has_gradient == (slot in filled_slots), slot


def test_unused_warm_started_slots_stay_exactly_zero_under_rmsprop(batches):
    torch.manual_seed(0)
    batch = batches['landlord']
    douzero = models.model_dict['landlord']()
    next_predictor = models.HandPredictor('landlord', 1)
    for hand_model, filled_slots in (('none', ()), ('next', (0,))):
        q_network = models.warm_start_from_douzero('landlord', douzero.state_dict())
        optimizer = torch.optim.RMSprop(q_network.parameters(), lr=1e-4, momentum=0, eps=1e-5, alpha=0.99)
        for _ in range(3):
            probs = next_predictor(batch['z'], batch['x_no_action'])['probs'] if hand_model == 'next' else None
            beliefs = belief.build_belief(hand_model, batch['x_no_action'], probs=probs)
            values = q_network(batch['z'], batch['x'], return_value=True, belief=beliefs)['values']
            optimizer.zero_grad()
            ((values.squeeze(-1) - 1.0) ** 2).mean().backward()
            optimizer.step()
        belief_columns = q_network.dense1.weight[:, -belief.BELIEF_DIM:]
        for slot in (0, 1):
            still_zero = bool((belief_columns[:, 75 * slot:75 * (slot + 1)] == 0).all())
            assert still_zero == (slot not in filled_slots), (hand_model, slot)


def test_q_network_rejects_a_missing_unexpected_or_undetached_belief(batches):
    batch = batches['landlord']
    size = len(batch['z'])
    douzero = models.model_dict['landlord']()
    belief_q_network = models.model_dict['landlord'](belief_dim=belief.BELIEF_DIM)
    with pytest.raises(ValueError):  # a belief the network would silently ignore
        douzero(batch['z'], batch['x'], return_value=True, belief=torch.zeros(size, belief.BELIEF_DIM))
    with pytest.raises(ValueError):
        belief_q_network(batch['z'], batch['x'], return_value=True)
    probs = models.HandPredictor('landlord', 1)(batch['z'], batch['x_no_action'])['probs']
    undetached = torch.cat([belief.flatten_belief(probs[:, 0]), torch.zeros(size, 75)], dim=-1)
    with pytest.raises(ValueError):  # would let the Q-loss reach the predictor
        belief_q_network(batch['z'], batch['x'], return_value=True, belief=undetached)


def test_frozen_predictor_gives_beliefs_without_gradients(tmp_path, batches):
    path = tmp_path / 'next.ckpt'
    models.save_hand_predictor(models.HandPredictor('landlord', 1), path)
    frozen = models.load_hand_predictor(path)
    assert not frozen.training and not any(p.requires_grad for p in frozen.parameters())
    batch = batches['landlord']
    assert not frozen(batch['z'], batch['x_no_action'])['probs'].requires_grad


# ---------------------------------------------------------------- T10: warm start and checkpoints

@pytest.fixture(scope='module')
def adp_state_dicts():
    paths = {position: ADP_DIR / (position + '.ckpt') for position in POSITIONS}
    missing = [path.name for path in paths.values() if not path.exists()]
    if missing:
        pytest.skip('DouZero-ADP checkpoints not in baselines/douzero_ADP: {}'.format(missing))
    return {position: torch.load(path, map_location='cpu') for position, path in paths.items()}


@pytest.fixture(scope='module')
def decision_batches(random_play_states):
    """Per seat, every legal action of a few real decisions: what an actor scores."""
    result = {}
    for position in POSITIONS:
        states = [s for s in random_play_states if s['infoset'].player_position == position][::60]
        result[position] = dict(z=torch.from_numpy(np.concatenate([s['obs']['z_batch'] for s in states])),
                                x=torch.from_numpy(np.concatenate([s['obs']['x_batch'] for s in states])))
    return result


def test_warm_start_copies_every_douzero_weight_and_zeroes_the_belief_columns(adp_state_dicts):
    for position, state_dict in adp_state_dicts.items():
        warm = models.warm_start_from_douzero(position, state_dict).state_dict()
        assert warm.keys() == state_dict.keys()
        for key, value in state_dict.items():
            if key == 'dense1.weight':
                assert torch.equal(warm[key][:, :value.shape[1]], value)
                assert torch.equal(warm[key][:, value.shape[1]:], torch.zeros(512, belief.BELIEF_DIM))
            else:
                assert torch.equal(warm[key], value), key


@pytest.mark.parametrize('position', POSITIONS)
def test_warm_start_reproduces_douzero_adp_for_any_belief(position, adp_state_dicts, decision_batches):
    original = models.model_dict[position]()
    original.load_state_dict(adp_state_dicts[position])
    warm = models.warm_start_from_douzero(position, adp_state_dicts[position])
    batch = decision_batches[position]
    size = len(batch['z'])
    torch.manual_seed(0)
    beliefs = [torch.zeros(size, 150), torch.rand(size, 150), torch.randn(size, 150) * 10, torch.ones(size, 150)]
    with torch.no_grad():
        q_original = original(batch['z'], batch['x'], return_value=True)['values']
        q_warm = [warm(batch['z'], batch['x'], return_value=True, belief=b)['values'] for b in beliefs]
    for q in q_warm:
        assert torch.equal(q, q_warm[0])  # zero belief columns: Q cannot depend on the belief
    assert (q_warm[0] - q_original).abs().max().item() <= WARM_START_ATOL


def test_belief_q_network_checkpoint_round_trip(tmp_path, adp_state_dicts, decision_batches):
    for position in POSITIONS:
        warm = models.warm_start_from_douzero(position, adp_state_dicts[position])
        torch.save(warm.state_dict(), tmp_path / (position + '.ckpt'))
        loaded = models.model_dict[position](belief_dim=belief.BELIEF_DIM)
        loaded.load_state_dict(torch.load(tmp_path / (position + '.ckpt')))
        for key, value in warm.state_dict().items():
            assert torch.equal(loaded.state_dict()[key], value), key
        batch = decision_batches[position]
        beliefs = torch.rand(len(batch['z']), belief.BELIEF_DIM)
        with torch.no_grad():
            assert torch.equal(loaded(batch['z'], batch['x'], return_value=True, belief=beliefs)['values'],
                               warm(batch['z'], batch['x'], return_value=True, belief=beliefs)['values'])


@pytest.mark.parametrize('num_targets', [1, 2])
def test_hand_predictor_checkpoint_round_trip(tmp_path, num_targets, batches):
    torch.manual_seed(0)
    predictor = models.HandPredictor('landlord_down', num_targets)
    path = tmp_path / 'predictor.ckpt'
    models.save_hand_predictor(predictor, path)
    batch = batches['landlord_down']
    for freeze in (True, False):
        loaded = models.load_hand_predictor(path, freeze=freeze)
        assert (loaded.position, loaded.num_targets) == ('landlord_down', num_targets)
        assert loaded.training != freeze and all(p.requires_grad != freeze for p in loaded.parameters())
        with torch.no_grad():
            got, want = loaded(batch['z'], batch['x_no_action']), predictor(batch['z'], batch['x_no_action'])
        assert torch.equal(got['logits'], want['logits']) and torch.equal(got['probs'], want['probs'])


# ---------------------------------------------------------------- T11: matched architecture

@pytest.mark.parametrize('position', POSITIONS)
def test_every_arm_has_the_same_q_network(position):
    shapes = {}
    for hand_model, num_targets in ARMS.items():
        q_network, predictor = models.make_arm(position, hand_model, _fresh_predictor(position, hand_model))
        shapes[hand_model] = [(name, tuple(p.shape)) for name, p in q_network.named_parameters()]
        assert not any(isinstance(module, models.HandPredictor) for module in q_network.modules())
        if num_targets:
            assert predictor.num_targets == num_targets and predictor.position == position
            assert not {id(p) for p in predictor.parameters()} & {id(p) for p in q_network.parameters()}
        else:
            assert predictor is None
    assert all(arm_shapes == shapes['none'] for arm_shapes in shapes.values())
    expected = (LSTM_PARAMS + _linear(128 + X_NO_ACTION_DIM[position] + 54 + 150, 512)
                + 4 * _linear(512, 512) + _linear(512, 1))
    assert _num_params(q_network) == expected
    assert _num_params(q_network) == _num_params(models.model_dict[position]()) + 150 * 512


@pytest.mark.parametrize('num_targets', [1, 2])
@pytest.mark.parametrize('position', POSITIONS)
def test_hand_predictor_has_the_douzero_plus_architecture(position, num_targets):
    heads = 13 * (_linear(512, 256) + _linear(256, 5)) + 2 * _linear(512, 5)
    expected = (LSTM_PARAMS + _linear(128 + X_NO_ACTION_DIM[position], 512) + 3 * _linear(512, 512)
                + num_targets * heads)
    predictor = models.HandPredictor(position, num_targets)
    assert _num_params(predictor) == expected
    assert (predictor.lstm.input_size, predictor.lstm.hidden_size) == (162, 128)
    trunk = (predictor.dense1, predictor.dense2, predictor.dense3, predictor.dense4)
    assert [(layer.in_features, layer.out_features) for layer in trunk] == [
        (128 + X_NO_ACTION_DIM[position], 512), (512, 512), (512, 512), (512, 512)]
    assert len(predictor.heads) == num_targets
    for kind_heads in predictor.heads:
        assert len(kind_heads) == 15
        for kind, head in enumerate(kind_heads):
            if kind < 13:  # ranks 3 .. 2: Linear(512, 256) - ReLU - Linear(256, 5)
                assert [type(module) for module in head] == [nn.Linear, nn.ReLU, nn.Linear]
                assert [(head[i].in_features, head[i].out_features) for i in (0, 2)] == [(512, 256), (256, 5)]
            else:  # jokers: Linear(512, 5)
                assert isinstance(head, nn.Linear) and (head.in_features, head.out_features) == (512, 5)


def test_make_arm_requires_an_explicit_predictor_for_predictor_arms():
    for hand_model in ('next', 'derived_both', 'learned_both'):
        with pytest.raises(ValueError):  # never a silently created, untrained predictor
            models.make_arm('landlord', hand_model)


def test_make_arm_rejects_a_predictor_of_the_wrong_kind():
    mismatches = [('none', models.HandPredictor('landlord', 1)), ('oracle', models.HandPredictor('landlord', 2)),
                  ('next', models.HandPredictor('landlord', 2)), ('derived_both', models.HandPredictor('landlord', 2)),
                  ('learned_both', models.HandPredictor('landlord', 1)),
                  ('next', models.HandPredictor('landlord_up', 1))]
    for hand_model, predictor in mismatches:
        with pytest.raises(ValueError):
            models.make_arm('landlord', hand_model, predictor)
