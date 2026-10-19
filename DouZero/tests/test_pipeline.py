"""Stage 3: the real actor -> buffer -> learner path, run in one process.

T7   buffer alignment: observation, action, belief and target rows of every arm and seat
T13  actor beliefs for every arm and seat
T14  the learner's Q inputs are exactly the actor's inputs for the chosen actions
T15  frozen predictors never change in a learner update
T16  the Q-network does change, and the actor copy receives the new weights
T17  M0c's belief weights stay zero, and its update equals a belief-free DouZero update
"""
import copy
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from douzero.dmc import belief, dmc, inprocess, models, utils
from douzero.dmc.arguments import parser
from douzero.env import hand_belief as hb
from douzero.env.env import Env, get_obs

ARMS = ('none', 'next', 'derived_both', 'learned_both', 'oracle')
POSITIONS = ('landlord', 'landlord_up', 'landlord_down')
T = 4       # unroll length: small, so every seat flushes many buffers
GAMES = 6


def tiny_flags(hand_model, seed=11, **overrides):
    flags = parser.parse_args(['--hand_model', hand_model, '--unroll_length', str(T), '--num_buffers', '100',
                               '--batch_size', '2', '--actor_device_cpu', '--training_device', 'cpu',
                               '--seed', str(seed)])
    for key, value in overrides.items():
        setattr(flags, key, value)
    return flags


def frozen_predictors(hand_model, seed=0):
    num_targets = belief.PREDICTOR_TARGETS.get(hand_model, 0)
    torch.manual_seed(seed)
    return {p: models.HandPredictor(p, num_targets).eval().requires_grad_(False) if num_targets else None
            for p in POSITIONS}


class RecordingEnv(Env):
    """Env that records the infoset of every decision (with its game) and every game's outcome."""

    def __init__(self, objective):
        super().__init__(objective)
        self.decisions, self.outcomes = [], []

    def reset(self):
        obs = super().reset()
        self.decisions.append((len(self.outcomes), self.infoset))
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if done:
            self.outcomes.append(dict(winner=self._game_winner, bombs=self._game_bomb_num, reward=reward))
        else:
            self.decisions.append((len(self.outcomes), self.infoset))
        return obs, reward, done, info


class RecordingModel:
    """Wraps a Model and records, at every actor decision, the Q inputs of the chosen action."""

    def __init__(self, model):
        self.model, self.chosen = model, []

    def forward(self, position, z, x, training=False, flags=None, belief=None):
        out = self.model.forward(position, z, x, training=training, flags=flags, belief=belief)
        index = int(out['action'])
        self.chosen.append(dict(position=position, z=z[index].clone(), x=x[index].clone(),
                                belief=None if belief is None else belief[index].clone()))
        return out


def play(hand_model, games=GAMES, seed=11, predictors=None, oracle_labels=None):
    """Runs the real act() in-process; returns everything it stored plus what really happened."""
    flags = tiny_flags(hand_model, seed=seed)
    torch.manual_seed(1)
    model = RecordingModel(models.Model('cpu', belief_dim=belief.BELIEF_DIM))
    predictors = predictors or frozen_predictors(hand_model)
    made = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(utils, 'create_env', lambda flags: made.setdefault('env', RecordingEnv(flags.objective)))
        if oracle_labels is not None:
            patch.setattr(utils, 'oracle_labels', oracle_labels)
        buffers, free_queue, full_queue = inprocess.run_actor(flags, model, predictors, max_episodes=games)
    return dict(flags=flags, model=model, predictors=predictors, env=made['env'], buffers=buffers,
                free_queue=free_queue, full_queue=full_queue)


def _forbidden(*args, **kwargs):
    raise AssertionError('a non-oracle arm asked for the true hands')


def _counting(calls):
    from douzero.dmc import oracle

    def counted(*args, **kwargs):
        calls.append(1)
        return oracle.oracle_labels(*args, **kwargs)
    return counted


@pytest.fixture(scope='module')
def collected():
    """Per arm, GAMES games of the real act(); non-oracle arms run with oracle_labels made to raise."""
    result = {}
    for hand_model in ARMS:
        calls = []
        result[hand_model] = play(hand_model, oracle_labels=_counting(calls) if hand_model == 'oracle' else _forbidden)
        result[hand_model]['oracle_calls'] = len(calls)
    return result


def seat_view(run, position):
    """The seat's decisions in actor order: (game, infoset, recorded Q inputs of the chosen action)."""
    env = run['env']
    decisions = [d for d in env.decisions if d[0] < len(env.outcomes)]  # drop the next game's unplayed opening
    chosen = run['model'].chosen
    assert len(decisions) == len(chosen)  # one Q-network call per decision
    return [(game, infoset, inputs) for (game, infoset), inputs in zip(decisions, chosen)
            if infoset.player_position == position]


def expected_belief(hand_model, predictor, x_no_action, z, infoset):
    """Recomputes one row's belief from its stored observation at batch size 1, as the actor does."""
    x = x_no_action.float()[None]
    if hand_model == 'oracle':
        return belief.build_belief('oracle', x, labels=torch.from_numpy(hb.hand_labels(infoset))[None])[0]
    probs = None
    if predictor is not None:
        with torch.no_grad():
            probs = predictor(z.float()[None], x)['probs']
    return belief.build_belief(hand_model, x, probs=probs)[0]


def alignment_errors(rows, view, outcomes, hand_model, predictor, position):
    """Every way the flushed rows of one seat disagree with what happened, as (check, row) pairs."""
    errors = []
    full_hand = 20 if position == 'landlord' else 17
    sign = 1 if position == 'landlord' else -1
    x_dim = rows['obs_x_no_action'].shape[1]
    for k in range(len(rows['target'])):
        game, infoset, inputs = view[k]
        obs = get_obs(infoset)
        if not np.array_equal(rows['obs_x_no_action'][k].numpy(), obs['x_no_action']):
            errors.append(('observation', k))
        if not np.array_equal(rows['obs_z'][k].numpy(), obs['z']):
            errors.append(('history', k))
        if not torch.equal(rows['obs_action'][k].float(), inputs['x'][x_dim:]):
            errors.append(('action', k))
        first = k == 0 or view[k - 1][0] != game
        last = k + 1 == len(view) or view[k + 1][0] != game
        own = hb.decode_card_array(rows['obs_x_no_action'][k, :54].numpy())
        if first and own.sum() != full_hand:
            errors.append(('first hand', k))
        if not first:
            before = hb.decode_card_array(rows['obs_x_no_action'][k - 1, :54].numpy())
            played = hb.decode_card_array(rows['obs_action'][k - 1].numpy())
            if not np.array_equal(before - played, own):
                errors.append(('hand evolution', k))
        target = sign * outcomes[game]['reward']
        if rows['target'][k].item() != target:
            errors.append(('target', k))
        if bool(rows['done'][k]) != last:
            errors.append(('done', k))
        if rows['episode_return'][k].item() != (target if last else 0.0):
            errors.append(('episode_return', k))
        if 'belief' in rows:
            if not torch.equal(rows['belief'][k], inputs['belief']):
                errors.append(('belief vs actor', k))
            if not torch.equal(rows['belief'][k], expected_belief(hand_model, predictor, rows['obs_x_no_action'][k],
                                                                  rows['obs_z'][k], infoset)):
                errors.append(('belief', k))
    return errors


# ---------------------------------------------------------------- T7: buffer alignment

@pytest.mark.parametrize('hand_model', ARMS)
def test_every_flushed_row_matches_its_decision(hand_model, collected):
    run = collected[hand_model]
    outcomes = run['env'].outcomes
    assert len(outcomes) == GAMES
    assert {o['winner'] for o in outcomes} == {'landlord', 'farmer'}  # both signs of the return occur
    for position in POSITIONS:
        rows = inprocess.flushed_rows(run['buffers'], run['full_queue'], position)
        view = seat_view(run, position)
        tail = len(view) - len(rows['target'])
        assert len(rows['target']) >= 3 * T and 1 <= tail <= T  # several flushes; only the last partial T is held back
        assert ('belief' in rows) == (hand_model != 'baseline')
        assert rows['belief'].shape == (len(rows['target']), belief.BELIEF_DIM)
        assert alignment_errors(rows, view, outcomes, hand_model, run['predictors'][position], position) == []


@pytest.mark.parametrize('hand_model', ['next', 'oracle'])
def test_alignment_check_catches_shifted_or_dropped_rows(hand_model, collected):
    run = collected[hand_model]
    position = 'landlord_down'
    rows = inprocess.flushed_rows(run['buffers'], run['full_queue'], position)
    view = seat_view(run, position)
    args = (view, run['env'].outcomes, hand_model, run['predictors'][position], position)

    def shifted(key):
        bad = dict(rows)
        bad[key] = torch.roll(rows[key], 1, dims=0)
        return bad
    found = {key: {check for check, _ in alignment_errors(shifted(key), *args)}
             for key in ('belief', 'obs_action', 'target')}
    assert 'belief' in found['belief'] and 'action' in found['obs_action'] and 'target' in found['target']
    dropped = {key: value[1:] for key, value in rows.items()}
    assert alignment_errors(dropped, *args)


def test_non_oracle_arms_never_ask_for_true_hands_and_the_oracle_asks_every_decision(collected):
    assert collected['oracle']['oracle_calls'] == len(collected['oracle']['model'].chosen)  # one per decision played
    for hand_model in ('none', 'next', 'derived_both', 'learned_both'):
        assert collected[hand_model]['oracle_calls'] == 0  # their act() ran with oracle_labels made to raise


def test_actor_seeds_never_collide_across_nearby_runs_devices_actors_or_attempts():
    pairs = [utils.actor_seeds(seed, device, i, attempt)
             for seed in range(4) for device in ('cpu', 0, 1) for i in range(5) for attempt in range(3)]
    assert len(set(pairs)) == len(pairs) and len({numpy_seed for numpy_seed, _ in pairs}) == len(pairs)
    assert utils.actor_seeds(3, 'cpu', 2) == utils.actor_seeds(3, 'cpu', 2)


def test_a_resumed_attempt_gives_its_actors_fresh_random_streams():
    numpy_state, torch_state = np.random.get_state(), torch.get_rng_state()  # seed_actor sets the global RNGs
    try:
        draws = {}
        for attempt in (0, 1, 0):
            utils.seed_actor(SimpleNamespace(seed=3, attempt=attempt), 'cpu', 0)
            draws.setdefault(attempt, []).append((np.random.rand(), torch.rand(1).item()))
    finally:
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
    assert draws[0][0] == draws[0][1]  # the same attempt replays its stream
    assert draws[1][0][0] != draws[0][0][0] and draws[1][0][1] != draws[0][0][1]  # a resume does not


def test_the_same_seed_gives_the_same_stored_games():
    first, second = play('derived_both', games=1, seed=5), play('derived_both', games=1, seed=5)
    for position in POSITIONS:
        a = inprocess.flushed_rows(first['buffers'], first['full_queue'], position)
        b = inprocess.flushed_rows(second['buffers'], second['full_queue'], position)
        assert all(torch.equal(a[key], b[key]) for key in a)


# ---------------------------------------------------------------- T13: actor beliefs

def _state_inputs(state):
    return torch.from_numpy(state['obs']['x_no_action']), torch.from_numpy(state['obs']['z'])


@pytest.mark.parametrize('hand_model', ARMS)
@pytest.mark.parametrize('position', POSITIONS)
def test_actor_belief_for_every_arm_and_seat(position, hand_model, random_play_states):
    predictors = frozen_predictors(hand_model)
    states = [s for s in random_play_states if s['infoset'].player_position == position][::50][:6]
    for state in states:
        x_no_action, z = _state_inputs(state)
        labels = torch.from_numpy(hb.hand_labels(state['infoset']))[None] if hand_model == 'oracle' else None
        beliefs = utils.actor_belief(hand_model, predictors[position], position, x_no_action, z, labels)
        assert beliefs.shape == (1, belief.BELIEF_DIM) and beliefs.dtype == torch.float32
        assert not beliefs.requires_grad and torch.isfinite(beliefs).all()
        assert torch.equal(beliefs[0], expected_belief(hand_model, predictors[position], x_no_action, z,
                                                       state['infoset']))


def test_actor_belief_rejects_a_predictor_for_another_seat_or_one_left_trainable(random_play_states):
    state = next(s for s in random_play_states if s['infoset'].player_position == 'landlord')
    x_no_action, z = _state_inputs(state)
    with pytest.raises(ValueError):
        utils.actor_belief('next', models.HandPredictor('landlord_up', 1).eval().requires_grad_(False),
                           'landlord', x_no_action, z)
    with pytest.raises(ValueError):  # still trainable
        utils.actor_belief('next', models.HandPredictor('landlord', 1), 'landlord', x_no_action, z)
    with pytest.raises(ValueError):  # eval mode alone is not frozen
        utils.actor_belief('next', models.HandPredictor('landlord', 1).eval(), 'landlord', x_no_action, z)


# ---------------------------------------------------------------- T14: actor inputs == learner inputs

def full_batch(run, position):
    """Every flushed buffer of one seat as one learner batch, via the real get_batch."""
    flags = tiny_flags(run['flags'].hand_model)
    flags.batch_size = len(run['full_queue'][position])
    full = inprocess.NonBlockingQueue(list(run['full_queue'][position]))
    batch = utils.get_batch(inprocess.NonBlockingQueue(), full, run['buffers'][position], flags, threading.Lock())
    return batch, flags.batch_size


@pytest.mark.parametrize('hand_model', ARMS)
def test_learner_inputs_are_exactly_the_actor_inputs(hand_model, collected):
    run = collected[hand_model]
    for position in POSITIONS:
        batch, num_buffers = full_batch(run, position)
        obs_z, obs_x, beliefs, target = dmc.learner_inputs(batch, torch.device('cpu'))
        view = seat_view(run, position)
        rows = inprocess.flushed_rows(run['buffers'], run['full_queue'], position)
        assert obs_x.shape == (T * num_buffers, 373 if position == 'landlord' else 484)
        assert obs_z.shape == (T * num_buffers, 5, 162) and beliefs.shape == (T * num_buffers, belief.BELIEF_DIM)
        for t in range(T):
            for b in range(num_buffers):
                row, decision = t * num_buffers + b, b * T + t  # (T, B) flattened row-major; buffers in flush order
                inputs = view[decision][2]
                assert torch.equal(obs_x[row], inputs['x']) and torch.equal(obs_z[row], inputs['z'])
                assert torch.equal(beliefs[row], inputs['belief'])
                assert target[row] == rows['target'][decision]


def test_learner_inputs_reject_a_misshapen_batch(collected):
    batch, _ = full_batch(collected['next'], 'landlord')
    bad = dict(batch, belief=batch['belief'][..., :75])
    with pytest.raises(ValueError):
        dmc.learner_inputs(bad, torch.device('cpu'))


def test_learner_inputs_reject_a_non_finite_belief(collected):
    batch, _ = full_batch(collected['next'], 'landlord')
    bad = dict(batch, belief=batch['belief'].clone())
    bad['belief'][0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        dmc.learner_inputs(bad, torch.device('cpu'))


# ---------------------------------------------------------------- T15-T17: one learner update

def rmsprop(model, flags):
    return torch.optim.RMSprop(model.parameters(), lr=flags.learning_rate, momentum=flags.momentum,
                               eps=flags.epsilon, alpha=flags.alpha)


def first_batch(run, position):
    """The seat's first batch_size flushed buffers, through the real get_batch."""
    flags = run['flags']
    full = inprocess.NonBlockingQueue(list(run['full_queue'][position])[:flags.batch_size])
    return utils.get_batch(inprocess.NonBlockingQueue(), full, run['buffers'][position], flags, threading.Lock())


def test_repeated_updates_on_one_batch_move_q_toward_the_targets(collected):
    run = collected['derived_both']
    torch.manual_seed(4)
    learner = models.model_dict['landlord'](belief_dim=belief.BELIEF_DIM)
    batch = first_batch(run, 'landlord')
    obs_z, obs_x, beliefs, target = dmc.learner_inputs(batch, torch.device('cpu'))

    def fit():
        with torch.no_grad():
            q = learner(obs_z, obs_x, return_value=True, belief=beliefs)['values'][:, 0]
        return ((q - target) ** 2).mean().item(), (q - target).mean().abs().item()
    loss_before, bias_before = fit()
    optimizer = rmsprop(learner, run['flags'])
    for _ in range(20):
        dmc.learn('landlord', {}, learner, batch, optimizer, run['flags'], threading.Lock())
    loss_after, bias_after = fit()
    assert loss_after < 0.5 * loss_before and bias_after < bias_before


def test_logged_grad_norm_is_the_norm_before_clipping(collected):
    run = collected['next']
    torch.manual_seed(5)
    learner = models.model_dict['landlord'](belief_dim=belief.BELIEF_DIM)
    twin = copy.deepcopy(learner)
    batch = first_batch(run, 'landlord')
    obs_z, obs_x, beliefs, target = dmc.learner_inputs(batch, torch.device('cpu'))
    ((twin(obs_z, obs_x, return_value=True, belief=beliefs)['values'][:, 0] - target) ** 2).mean().backward()
    expected = torch.sqrt(sum(p.grad.pow(2).sum() for p in twin.parameters())).item()
    flags = copy.copy(run['flags'])
    flags.max_grad_norm = expected / 2  # so that this update is clipped
    stats = dmc.learn('landlord', {}, learner, batch, rmsprop(learner, flags), flags, threading.Lock())
    assert stats['grad_norm_landlord'] == pytest.approx(expected, rel=1e-5) and stats['clipped_landlord'] == 1.0


def test_a_batch_without_a_finished_game_leaves_the_mean_return_unchanged(collected):
    run = collected['none']
    learner = models.model_dict['landlord'](belief_dim=belief.BELIEF_DIM)
    optimizer = rmsprop(learner, run['flags'])
    batch, _ = full_batch(run, 'landlord')  # every flushed buffer: it holds finished games
    first = dmc.learn('landlord', {}, learner, batch, optimizer, run['flags'], threading.Lock())
    quiet = dict(batch, done=torch.zeros_like(batch['done']))
    second = dmc.learn('landlord', {}, learner, quiet, optimizer, run['flags'], threading.Lock())
    assert np.isfinite(second['mean_episode_return_landlord'])
    assert second['mean_episode_return_landlord'] == first['mean_episode_return_landlord']


def test_learn_stops_before_the_weights_change_on_a_non_finite_loss(collected):
    run = collected['next']
    learner = models.model_dict['landlord'](belief_dim=belief.BELIEF_DIM)
    before = {k: v.clone() for k, v in learner.state_dict().items()}
    batch = first_batch(run, 'landlord')
    bad = dict(batch, target=torch.full_like(batch['target'], float('nan')))
    with pytest.raises(RuntimeError):
        dmc.learn('landlord', {}, learner, bad, rmsprop(learner, run['flags']), run['flags'], threading.Lock())
    assert all(torch.equal(before[k], v) for k, v in learner.state_dict().items())


def one_update(run, position, learner):
    """One real learn() step on two flushed buffers; returns (stats, actor copy)."""
    flags = run['flags']
    full = inprocess.NonBlockingQueue(list(run['full_queue'][position])[:flags.batch_size])
    batch = utils.get_batch(inprocess.NonBlockingQueue(), full, run['buffers'][position], flags, threading.Lock())
    actor = models.Model('cpu', belief_dim=learner.belief_dim)
    optimizer = torch.optim.RMSprop(learner.parameters(), lr=flags.learning_rate, momentum=flags.momentum,
                                    eps=flags.epsilon, alpha=flags.alpha)
    stats = dmc.learn(position, {'cpu': actor}, learner, batch, optimizer, flags, threading.Lock())
    return stats, actor, optimizer, batch


@pytest.mark.parametrize('hand_model', ['next', 'derived_both', 'learned_both'])
def test_frozen_predictors_never_change_in_a_learner_update(hand_model, collected):
    run = collected[hand_model]
    for position in POSITIONS:
        predictor = run['predictors'][position]
        before = {k: v.clone() for k, v in predictor.state_dict().items()}
        learner = models.model_dict[position](belief_dim=belief.BELIEF_DIM)
        _, _, optimizer, _ = one_update(run, position, learner)
        assert not predictor.training and not any(p.requires_grad for p in predictor.parameters())
        assert all(p.grad is None for p in predictor.parameters())
        in_optimizer = {id(p) for group in optimizer.param_groups for p in group['params']}
        assert not in_optimizer & {id(p) for p in predictor.parameters()}
        assert all(torch.equal(before[k], v) for k, v in predictor.state_dict().items())


@pytest.mark.parametrize('hand_model', ARMS)
def test_a_learner_update_changes_the_q_network_and_syncs_the_actor(hand_model, collected):
    run = collected[hand_model]
    torch.manual_seed(3)
    learner = models.model_dict['landlord'](belief_dim=belief.BELIEF_DIM)
    before = {k: v.clone() for k, v in learner.state_dict().items()}
    stats, actor, _, _ = one_update(run, 'landlord', learner)
    changed = [k for k, v in learner.state_dict().items() if not torch.equal(before[k], v)]
    assert changed and 'dense6.weight' in changed and 'lstm.weight_ih_l0' in changed
    synced = actor.get_model('landlord').state_dict()
    assert all(torch.equal(synced[k], v) for k, v in learner.state_dict().items())
    for key in ('loss', 'q_mean', 'q_std', 'grad_norm', 'clipped'):
        assert np.isfinite(stats[key + '_landlord'])


def test_m0c_keeps_zero_belief_weights_and_updates_exactly_like_douzero(collected):
    run = collected['none']
    for position in POSITIONS:
        douzero_state = models.model_dict[position]().state_dict()
        m0c = models.warm_start_from_douzero(position, douzero_state)
        douzero = models.model_dict[position]()
        douzero.load_state_dict(douzero_state)
        stats_m0c, _, _, batch = one_update(run, position, m0c)
        flags = run['flags']
        no_belief = {k: v for k, v in batch.items() if k != 'belief'}
        optimizer = torch.optim.RMSprop(douzero.parameters(), lr=flags.learning_rate, momentum=flags.momentum,
                                        eps=flags.epsilon, alpha=flags.alpha)
        stats_douzero = dmc.learn(position, {}, douzero, no_belief, optimizer, flags, threading.Lock())
        assert torch.equal(m0c.dense1.weight[:, -belief.BELIEF_DIM:], torch.zeros(512, belief.BELIEF_DIM))
        assert stats_m0c['loss_' + position] == pytest.approx(stats_douzero['loss_' + position], rel=1e-6)
        m0c_state, douzero_state = m0c.state_dict(), douzero.state_dict()
        for key, value in douzero_state.items():
            ours = m0c_state[key][:, :value.shape[1]] if key == 'dense1.weight' else m0c_state[key]
            torch.testing.assert_close(ours, value, rtol=0, atol=1e-6)
