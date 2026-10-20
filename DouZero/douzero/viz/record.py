"""Records one game, or one decision point, for the visualization tools (a teaching and debugging aid).

A record is a plain JSON-able dict. Each step holds only what the acting player could know (the
public state, its own hand and its count of the unseen cards) plus the model's side of the
decision: its belief input, the Q-values of the legal actions and the epsilon-greedy choice. The
true hidden hands go into a separate 'hidden' section, written only with reveal=True and read
through oracle.oracle_labels, the reviewed access point. A NORMAL record (reveal=False) also drops
everything the Oracle arm computes from the true hands: its belief, Q-values and ranking. The
record keeps the seed, so the code can always rebuild the deal; NORMAL is a teaching view, not a
secret. Training and evaluation never import this package.
"""
import contextlib
import copy
import os
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from douzero.dmc import belief, models, utils
from douzero.dmc.env_utils import _format_observation
from douzero.dmc.oracle import oracle_labels
from douzero.env import hand_belief as hb
from douzero.env.env import Env
from douzero.env.move_detector import get_move_type

SCHEMA_EPISODE = 'cs230.episode_replay/1'
SCHEMA_COMPARISON = 'cs230.state_comparison/1'
POSITIONS = ('landlord', 'landlord_down', 'landlord_up')  # the order of play
ARMS = ('none', 'next', 'derived_both', 'learned_both', 'oracle')
TOP_K = 5
DEFAULT_CHECKPOINT_DIR = str(Path(__file__).resolve().parents[2] / 'baselines' / 'douzero_ADP')
UNTRAINED = 'UNTRAINED (random weights): beliefs are noise until Stage 4 trains the predictors'

# Where each slot of an arm's belief input comes from: (next-player slot, other-player slot).
_PREDICTED = dict(kind='predicted', text='predicted by the K=1 next-hand predictor')
_LEARNED = dict(kind='learned', text='learned directly by the K=2 predictor')
_ZERO = dict(kind='zero', text='zeros: no information')
_ORACLE = dict(kind='oracle', text='the true hand (oracle control)')
SLOT_SOURCES = {'none': [_ZERO, _ZERO],
                'next': [_PREDICTED, _ZERO],
                'derived_both': [_PREDICTED, dict(kind='derived', text='derived by conservation: other = unseen - next')],
                'learned_both': [_LEARNED, _LEARNED],
                'oracle': [_ORACLE, _ORACLE]}
MOVE_TYPES = ('pass', 'single', 'pair', 'trio', 'bomb', 'rocket', 'trio + single', 'trio + pair', 'straight',
              'pair straight', 'airplane', 'airplane + singles', 'airplane + pairs', 'four + two singles',
              'four + two pairs', 'invalid')

State = namedtuple('State', 'env infoset position obs x_no_action z')


def cards(values):
    return sorted(int(card) for card in values)


def counts_to_cards(counts):
    return [kind for kind, num in zip(hb.KINDS, counts) for _ in range(int(num))]


def move_type(move):
    return MOVE_TYPES[get_move_type(list(move))['type']]


def action_entry(move, q):
    return dict(cards=cards(move), type=move_type(move), q=round(float(q), 6))


def slot_summary(probs):
    """A (15, 5) belief over the counts of one hand: its probabilities and the expected count of each kind."""
    probs = np.asarray(probs, dtype=np.float64)
    return dict(probs=[[round(float(p), 4) for p in row] for row in probs],
                expected=[round(float(e), 3) for e in hb.expected_counts(probs)])


def slots_of(beliefs):
    """The (1, 150) belief input as {'next': (15, 5), 'other': (15, 5)} arrays, or None (no belief input)."""
    if beliefs is None:
        return None
    slots = beliefs[0].reshape(2, hb.NUM_KINDS, hb.NUM_COUNTS).numpy().astype(np.float64)
    return dict(next=slots[0], other=slots[1])


def belief_summary(slots):
    return None if slots is None else {name: slot_summary(probs) for name, probs in slots.items()}


def counting_slots(public):
    """The hypergeometric baseline: each hidden hand is a uniformly random subset of the unseen cards."""
    return {name: hb.hypergeometric_belief(public['unseen'], size)
            for name, size in zip(('next', 'other'), public['hidden_sizes'])}


def cross_entropy(probs, true_counts):
    """Mean over kinds of -log P(true count) in nats; None when the belief gives the truth no mass (a zero slot)."""
    p_true = np.take_along_axis(np.asarray(probs), np.asarray(true_counts)[:, None], axis=-1)[:, 0]
    return round(float(-np.log(p_true).mean()), 4) + 0.0 if (p_true > 0).all() else None  # + 0.0: no '-0.0'


def describe_public(infoset):
    """What every player can see, plus the acting player's own count of the unseen cards."""
    position = infoset.player_position
    next_player, other_player = hb.hidden_players(position)
    to_beat = None
    if infoset.last_move:
        to_beat = dict(cards=cards(infoset.last_move), by=infoset.last_pid, type=move_type(infoset.last_move))
    return dict(cards_left={p: int(infoset.num_cards_left_dict[p]) for p in POSITIONS},
                played={p: cards(infoset.played_cards[p]) for p in POSITIONS},
                to_beat=to_beat,
                bombs=int(infoset.bomb_num),
                three_landlord_cards=cards(infoset.three_landlord_cards),
                unseen=[int(n) for n in hb.union_counts(infoset)],
                hidden_seats=[next_player, other_player],
                hidden_sizes=[int(infoset.num_cards_left_dict[p]) for p in (next_player, other_player)])


def describe_hidden(infoset, position, x_no_action, slots):
    """The ground truth of one decision (reveal mode only): all three hands and how good the belief was.

    cross_entropy compares the belief slots (raw, not rounded) with the true hands, the predictor
    metric of Stage 4 for this one decision, next to two baselines: card counting (a uniformly random
    deal of the unseen cards) and uniform (every feasible count equally likely).
    """
    labels = oracle_labels(infoset, position, x_no_action)[0].numpy()
    next_player, other_player = hb.hidden_players(position)
    hands = {position: cards(infoset.player_hand_cards), next_player: counts_to_cards(labels[0]),
             other_player: counts_to_cards(labels[1])}
    true_counts = dict(next=[int(n) for n in labels[0]], other=[int(n) for n in labels[1]])
    public = describe_public(infoset)
    counting = counting_slots(public)
    mask = hb.legal_count_mask(public['unseen'])
    uniform = mask / mask.sum(axis=-1, keepdims=True)
    ce = dict(card_counting={s: cross_entropy(counting[s], true_counts[s]) for s in ('next', 'other')},
              uniform={s: cross_entropy(uniform, true_counts[s]) for s in ('next', 'other')},
              belief=None if slots is None else {s: cross_entropy(slots[s], true_counts[s]) for s in ('next', 'other')})
    return dict(hands={p: hands[p] for p in POSITIONS}, true_counts=true_counts, cross_entropy=ce)


@contextlib.contextmanager
def seeded(seed, exp_epsilon):
    """Seeds numpy and torch exactly as actor 0 of a run with this --seed (first start), then restores them."""
    numpy_state, torch_state = np.random.get_state(), torch.get_rng_state()
    try:
        flags = SimpleNamespace(seed=seed, num_actors=1, exp_epsilon=exp_epsilon)
        utils.seed_actor(flags, 'cpu', 0)
        yield flags
    finally:
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def warm_started(position, checkpoint_dir, seed=0):
    """One seat's belief Q-network warm-started from DouZero, so its belief weights are zero, and its source."""
    path = os.path.join(checkpoint_dir, position + '.ckpt') if checkpoint_dir else ''
    if path and os.path.exists(path):
        state, source = torch.load(path, map_location='cpu', weights_only=True), 'DouZero-ADP warm start, belief weights zero'
    else:
        with torch.random.fork_rng(devices=[]):  # seeded, without changing the caller's random state
            torch.manual_seed(seed)
            state = models.model_dict[position]().state_dict()
        source = 'RANDOM DouZero weights (no checkpoint found), belief weights zero'
    return models.warm_start_from_douzero(position, state).eval(), source


def warm_started_model(hand_model, checkpoint_dir, seed=0):
    """The arm's Q-network Model for every seat, warm-started as RL training starts it, and its source."""
    model = models.Model('cpu', belief_dim=0 if hand_model == 'baseline' else belief.BELIEF_DIM)
    source = None
    for position in POSITIONS:
        if hand_model == 'baseline':
            path = os.path.join(checkpoint_dir, position + '.ckpt') if checkpoint_dir else ''
            if path and os.path.exists(path):
                model.get_model(position).load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
                source = 'DouZero-ADP'
            continue
        network, source = warm_started(position, checkpoint_dir, seed)
        model.get_model(position).load_state_dict(network.state_dict())
    model.eval()
    return model, source or 'RANDOM DouZero weights (no checkpoint found)'


def predictor_label(predictor_dir, predictors):
    """Where frozen predictors came from, for the page header; tiny or untrained ones are flagged in capitals."""
    provenances = {seat: p.provenance or {} for seat, p in predictors.items() if p is not None}
    provenance = next(iter(provenances.values()))
    if not all(p.get('trained') for p in provenances.values()):
        return 'UNTRAINED predictors from ' + predictor_dir
    counts = {seat: p.get('train_samples') for seat, p in provenances.items()}
    samples = None if None in counts.values() else min(counts.values())
    detail = ', '.join(part for part in (
        'Stage 4 run {}'.format(provenance['run_id']) if provenance.get('run_id') else '',
        'selected at step {}'.format(provenance['selected_step']) if 'selected_step' in provenance else '',
        'training samples per seat ' + ' / '.join('{} {}'.format(seat, n) for seat, n in counts.items())
        if samples is not None else '') if part)
    tiny = ('; TINY pipeline-check data: not a real predictor' if samples is not None and samples < 10000 else '')
    return 'frozen, from {} ({}{})'.format(predictor_dir, detail or 'trained', tiny)


def untrained_predictors(num_targets, seed):
    """Seeded untrained predictors for every seat; record_episode and compare_state use the same seed."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return {p: models.HandPredictor(p, num_targets).eval().requires_grad_(False) for p in POSITIONS}


def record_episode(hand_model='derived_both', seed=0, predictor_dir=None, checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                   exp_epsilon=0.01, reveal=True, model=None, predictors=None):
    """Plays one game as actor 0 of a CPU-actor run (--actor_device_cpu) with --seed seed would at its
    first start, and records it. GPU actors derive their seeds from the device index, so they play
    other games.

    The game is the one trace_episode.py shows: the same seeding, belief and epsilon-greedy forward
    as act(). model and predictors default to the warm start and to the frozen predictors in
    predictor_dir (untrained ones when it is None); tests pass their own. With reveal=False the
    record holds nothing hidden from the acting players.
    """
    model_source = 'given by the caller'
    if model is None:
        model, model_source = warm_started_model(hand_model, checkpoint_dir, seed)
    num_targets = belief.PREDICTOR_TARGETS.get(hand_model, 0)
    predictor_source = 'given by the caller' if predictors is not None else 'none for this arm'
    if num_targets and predictors is None:
        if predictor_dir:
            predictors = models.load_predictors(hand_model, predictor_dir, require_trained=False)
            predictor_source = predictor_label(predictor_dir, predictors)
        else:
            predictors, predictor_source = untrained_predictors(num_targets, seed), UNTRAINED
    predictors = predictors or {p: None for p in POSITIONS}

    steps, hidden_steps = [], []
    with seeded(seed, exp_epsilon) as flags:
        env = Env('adp')
        position, obs, x_no_action, z = _format_observation(env.reset(), 'cpu')
        while True:
            infoset = env.infoset
            beliefs = None
            if hand_model != 'baseline':
                labels = oracle_labels(infoset, position, x_no_action) if hand_model == 'oracle' else None
                beliefs = utils.actor_belief(hand_model, predictors[position], position, x_no_action, z, labels)
            legal_actions = obs['legal_actions']
            expanded = None if beliefs is None else beliefs.expand(len(legal_actions), -1)
            with torch.no_grad():
                values = model.forward(position, obs['z_batch'], obs['x_batch'], training=True,
                                       belief=expanded)['values'][:, 0].numpy()
                # Peek at the number the epsilon-greedy forward below is about to draw, without consuming it,
                # to record which branch it takes.
                draw = None
                if exp_epsilon > 0:
                    state = np.random.get_state()
                    draw = float(np.random.rand())
                    np.random.set_state(state)
                index = int(model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags,
                                          belief=expanded)['action'])
            order = np.argsort(-values, kind='stable')
            public = describe_public(infoset)
            slots = slots_of(beliefs)
            steps.append(dict(
                t=len(steps), seat=position, own_hand=cards(infoset.player_hand_cards), public=public,
                legal_count=len(legal_actions), belief=belief_summary(slots),
                card_counting=belief_summary(counting_slots(public)),
                top_actions=[action_entry(legal_actions[i], values[i]) for i in order[:TOP_K]],
                selected=dict(action_entry(legal_actions[index], values[index]),
                              rank=int(np.flatnonzero(order == index)[0]) + 1),
                exploration=dict(epsilon=exp_epsilon, draw=draw, explored=draw is not None and draw < exp_epsilon),
                action=cards(legal_actions[index])))
            if reveal:
                hidden_steps.append(describe_hidden(infoset, position, x_no_action, slots))
            raw_obs, reward, done, _ = env.step(legal_actions[index])
            if done:
                break
            position, obs, x_no_action, z = _format_observation(raw_obs, 'cpu')
        winner, bombs = env._game_winner, env._game_bomb_num

    returns = {p: float(reward if p == 'landlord' else -reward) for p in POSITIONS}  # the sign rule in act()
    record = dict(schema=SCHEMA_EPISODE, game_id='seed{}-actor0-game0'.format(seed), seed=seed,
                  hand_model=hand_model, arm=belief.ARM_NAMES[hand_model],
                  slot_sources=copy.deepcopy(SLOT_SOURCES.get(hand_model)), exp_epsilon=exp_epsilon,
                  q_network=model_source, predictors=predictor_source, revealed=reveal, steps=steps,
                  result=dict(winner='landlord' if winner == 'landlord' else 'peasants', bombs=int(bombs),
                              returns=returns,
                              decisions_per_seat={p: sum(s['seat'] == p for s in steps) for p in POSITIONS}))
    if reveal:
        last = steps[-1]
        final_hands = dict(hidden_steps[-1]['hands'])  # the hands after the last move: the losers' leftover cards
        final_hands[last['seat']] = counts_to_cards(hb.card_counts(final_hands[last['seat']]) -
                                                    hb.card_counts(last['action']))
        record['hidden'] = dict(steps=hidden_steps, final_hands=final_hands)
        return record
    return strip_hidden(record)


def strip_hidden(record):
    """The record without any ground truth: what a NORMAL replay is allowed to read.

    The 'hidden' section goes, and so does everything the Oracle arm computed from the true hands:
    its belief input, the Q-values and ranking of the moves. The move it played stays: it is public.
    """
    stripped = {k: copy.deepcopy(v) for k, v in record.items() if k != 'hidden'}
    stripped['revealed'] = False
    if record['hand_model'] == 'oracle':
        for step in stripped['steps']:
            step.update(belief=None, top_actions=None, oracle_masked=True,
                        selected=dict(cards=step['selected']['cards'], type=step['selected']['type']))
    return stripped


def rebuild_state(record, t):
    """The exact state of decision t of a recorded game: the same deal, then the recorded moves."""
    if not 0 <= t < len(record['steps']):
        raise ValueError('decision t={} does not exist: the game has decisions 0 to {}'.format(
            t, len(record['steps']) - 1))
    with seeded(record['seed'], record['exp_epsilon']):
        env = Env('adp')
        raw_obs = env.reset()
    for step in record['steps'][:t + 1]:
        infoset = env.infoset
        if infoset.player_position != step['seat'] or cards(infoset.player_hand_cards) != step['own_hand']:
            raise ValueError('decision {} does not match the record; was it made with other code?'.format(step['t']))
        if step['t'] == t:
            break
        move = next(a for a in raw_obs['legal_actions'] if cards(a) == step['action'])
        raw_obs, _, _, _ = env.step(move)
    position, obs, x_no_action, z = _format_observation(raw_obs, 'cpu')
    return State(env, env.infoset, position, obs, x_no_action, z)


def compare_state(record, t, k1_predictors=None, k2_predictors=None, checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                  reveal=None, arm_models=None, k1_source=None, k2_source=None):
    """Every arm's belief input and action ranking at decision t of a recorded game.

    M1 and M2c share the one K=1 predictor (k1_predictors), as in the experiment; M2 has its own
    K=2 predictor. By default every arm's Q-network is the DouZero warm start, whose belief weights
    are zero, so all arms still rank the actions identically. arm_models maps an arm to
    dict(q_network=..., predictor=...) for one seat, for example from a trained checkpoint.
    """
    reveal = record['revealed'] if reveal is None else reveal
    if reveal and not record['revealed']:
        raise ValueError('reveal=True needs a record made with reveal=True')
    state = rebuild_state(record, t)
    infoset, position, obs = state.infoset, state.position, state.obs
    k1_source = k1_source or 'given by the caller'
    k2_source = k2_source or 'given by the caller'
    if k1_predictors is None:
        k1_predictors, k1_source = untrained_predictors(1, record['seed']), UNTRAINED
    if k2_predictors is None:
        k2_predictors, k2_source = untrained_predictors(2, record['seed']), UNTRAINED  # as record_episode makes them
    shared, q_source = warm_started(position, checkpoint_dir, record['seed'])
    predictor_for = dict(none=None, next=k1_predictors[position], derived_both=k1_predictors[position],
                         learned_both=k2_predictors[position], oracle=None)
    labels = oracle_labels(infoset, position, state.x_no_action)  # the oracle arm's input
    legal_actions = obs['legal_actions']
    arms, orders, raw_slots = [], [], {}
    for hand_model in ARMS:
        custom = (arm_models or {}).get(hand_model, {})
        network = custom.get('q_network', shared)
        predictor = custom.get('predictor', predictor_for[hand_model])
        beliefs = utils.actor_belief(hand_model, predictor, position, state.x_no_action, state.z,
                                     labels if hand_model == 'oracle' else None)
        with torch.no_grad():
            values = network(obs['z_batch'], obs['x_batch'], return_value=True,
                             belief=beliefs.expand(len(legal_actions), -1))['values'][:, 0].numpy()
        order = np.argsort(-values, kind='stable')
        orders.append(order)
        raw_slots[hand_model] = slots_of(beliefs)
        arms.append(dict(hand_model=hand_model, arm=belief.ARM_NAMES[hand_model],
                         slot_sources=copy.deepcopy(SLOT_SOURCES[hand_model]),
                         belief=belief_summary(raw_slots[hand_model]),
                         q_network='custom' if 'q_network' in custom else q_source,
                         top_actions=[action_entry(legal_actions[i], values[i]) for i in order[:TOP_K]],
                         greedy_action=action_entry(legal_actions[order[0]], values[order[0]])))
        if hand_model == 'oracle' and not reveal:  # its input, and so its Q-values, come from the true hands
            arms[-1].update(belief=None, top_actions=None, greedy_action=None, oracle_masked=True)
    public = describe_public(infoset)
    comparison = dict(
        schema=SCHEMA_COMPARISON, game_id=record['game_id'], seed=record['seed'], t=t, played_by=record['arm'],
        state=dict(seat=position, own_hand=cards(infoset.player_hand_cards), public=public,
                   legal_count=len(legal_actions), card_counting=belief_summary(counting_slots(public)),
                   history=[dict(seat=s['seat'], action=s['action'], type=s['selected']['type'])
                            for s in record['steps'][:t]]),
        q_network=q_source, predictors=dict(k1=k1_source, k2=k2_source), arms=arms,
        # Over the four public arms: whether the Oracle agrees with them depends on the true hands.
        identical_rankings=all(np.array_equal(order, orders[0]) for order in orders[:4]), revealed=reveal)
    if reveal:
        hidden = describe_hidden(infoset, position, state.x_no_action, None)
        hidden['cross_entropy']['by_arm'] = {
            hand_model: {s: cross_entropy(slots[s], hidden['true_counts'][s]) for s in ('next', 'other')}
            for hand_model, slots in raw_slots.items()}
        hidden['oracle_ranks_identically'] = bool(np.array_equal(orders[4], orders[0]))
        comparison['hidden'] = hidden
    return comparison


def strip_comparison(comparison):
    """A comparison without any ground truth: what a NORMAL comparison page is allowed to read."""
    stripped = {k: copy.deepcopy(v) for k, v in comparison.items() if k != 'hidden'}
    stripped['revealed'] = False
    for arm in stripped['arms']:
        if arm['hand_model'] == 'oracle':
            arm.update(belief=None, top_actions=None, greedy_action=None, oracle_masked=True)
    return stripped
