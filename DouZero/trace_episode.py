"""Prints one complete game of one hand-belief arm, decision by decision (a learning and debugging tool).

The game is played the way an actor plays it: the same seeding (utils.seed_actor), the same
belief (utils.actor_belief), and the Q-network's epsilon-greedy forward. The terminal reward
then becomes the Monte Carlo targets, and a few of the rows the actor would store are shown.
tests/test_trace_episode.py checks that these rows equal what act() itself stores.
Run from DouZero/; one process, one CPU thread, a few seconds:

    python trace_episode.py --hand_model derived_both --seed 0

The Q-network starts from DouZero-ADP when its checkpoints are present. Without
--predictor_dir the predictors are untrained, so their beliefs are noise.
"""
import argparse
import os
from types import SimpleNamespace

import torch

from douzero.dmc import belief, models, utils
from douzero.dmc.env_utils import _format_observation
from douzero.dmc.oracle import oracle_labels
from douzero.env import hand_belief as hb
from douzero.env.env import Env

POSITIONS = ('landlord', 'landlord_up', 'landlord_down')
FACES = dict(zip(hb.KINDS, '3456789TJQKA2BR'))  # B and R: small and big joker
DEFAULT_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baselines', 'douzero_ADP')


def cards_text(cards):
    return ''.join(FACES[card] for card in sorted(cards)) or '-'


def action_text(action):
    return 'play ' + cards_text(action) if action else 'pass'


def decoded(cards_array):
    return [kind for kind, num in zip(hb.KINDS, hb.decode_card_array(cards_array.numpy())) for _ in range(num)]


def build_model(hand_model, checkpoint_dir):
    """The arm's Q-network Model, warm-started from checkpoint_dir, or random weights when it is empty."""
    belief_dim = 0 if hand_model == 'baseline' else belief.BELIEF_DIM
    model = models.Model('cpu', belief_dim=belief_dim)
    if checkpoint_dir:
        for position in POSITIONS:
            state = torch.load(os.path.join(checkpoint_dir, position + '.ckpt'), map_location='cpu', weights_only=True)
            if belief_dim:
                state = models.warm_start_from_douzero(position, state).state_dict()
            model.get_model(position).load_state_dict(state)
    model.eval()
    return model


def untrained_predictors(hand_model, seed):
    num_targets = belief.PREDICTOR_TARGETS.get(hand_model, 0)
    torch.manual_seed(seed)
    return {p: models.HandPredictor(p, num_targets).eval().requires_grad_(False) if num_targets else None
            for p in POSITIONS}


def expected_sizes(beliefs):
    """The belief's expected card count of each hidden hand ('-' for an empty slot)."""
    if beliefs is None:
        return '-', '-'
    slots = beliefs[0].reshape(2, hb.NUM_KINDS, hb.NUM_COUNTS)
    return tuple('{:.1f}'.format((slot * torch.arange(hb.NUM_COUNTS)).sum().item()) if slot.sum() > 0 else '-'
                 for slot in slots)


def trace_episode(hand_model='derived_both', seed=0, predictor_dir=None, checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                  exp_epsilon=0.01, model=None, predictors=None):
    """Plays one game as actor 0 of a run with --seed seed would at its first start, and returns (text, record).

    A resumed attempt reseeds its actors (utils.actor_seeds), so its games differ from this one.

    model and predictors default to the DouZero-ADP warm start and to the frozen predictors in
    predictor_dir (untrained ones when it is None); tests pass their own.
    """
    model_source = 'given by the caller'
    if model is None:
        model = build_model(hand_model, checkpoint_dir)
        model_source = 'DouZero-ADP warm start, belief weights zero' if checkpoint_dir else 'random weights'
    predictor_source = 'given by the caller'
    if not belief.PREDICTOR_TARGETS.get(hand_model, 0):
        predictor_source = 'none for this arm'
    elif predictors is None and predictor_dir:
        predictors, predictor_source = models.load_predictors(hand_model, predictor_dir, require_trained=False), \
            'frozen, from ' + predictor_dir
    elif predictors is None:
        predictors, predictor_source = untrained_predictors(hand_model, seed), 'UNTRAINED, so beliefs are noise'
    predictors = predictors or {p: None for p in POSITIONS}
    flags = SimpleNamespace(seed=seed, num_actors=1, exp_epsilon=exp_epsilon)
    utils.seed_actor(flags, 'cpu', 0)  # the random state act() starts from
    env = Env('adp')
    position, obs, x_no_action, z = _format_observation(env.reset(), 'cpu')
    decisions, stored = [], {p: [] for p in POSITIONS}
    while True:
        infoset = env.infoset
        beliefs = None
        if hand_model != 'baseline':
            labels = oracle_labels(infoset, position, x_no_action) if hand_model == 'oracle' else None
            beliefs = utils.actor_belief(hand_model, predictors[position], position, x_no_action, z, labels)
        expanded = None if beliefs is None else beliefs.expand(len(obs['legal_actions']), -1)
        with torch.no_grad():
            values = model.forward(position, obs['z_batch'], obs['x_batch'], training=True, belief=expanded)['values'][:, 0]
            index = int(model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags, belief=expanded)['action'])
        action = obs['legal_actions'][index]
        next_player, other_player = hb.hidden_players(position)
        decisions.append(dict(t=len(decisions), seat=position, own=len(infoset.player_hand_cards),
                              hidden=(infoset.num_cards_left_dict[next_player], infoset.num_cards_left_dict[other_player]),
                              legal=len(obs['legal_actions']), beliefs=beliefs, q=values[index].item(), move=action,
                              greedy=index == int(torch.argmax(values))))
        row = dict(obs_x_no_action=x_no_action, obs_action=utils._cards2tensor(action), obs_z=z, t=len(decisions) - 1)
        if beliefs is not None:
            row['belief'] = beliefs[0]
        stored[position].append(row)
        raw_obs, reward, done, _ = env.step(action)
        if done:
            break
        position, obs, x_no_action, z = _format_observation(raw_obs, 'cpu')

    returns = {p: reward if p == 'landlord' else -reward for p in POSITIONS}  # the sign rule in act()
    for p in POSITIONS:
        for k, row in enumerate(stored[p]):
            last = k == len(stored[p]) - 1
            row.update(target=torch.tensor(returns[p], dtype=torch.float32), done=torch.tensor(last),
                       episode_return=torch.tensor(returns[p] if last else 0.0, dtype=torch.float32))
    winner, bombs = env._game_winner, env._game_bomb_num

    lines = ['ONE GAME, ARM {} (--hand_model {}), seed {}'.format(belief.ARM_NAMES[hand_model], hand_model, seed),
             'Q-network: {}. Predictors: {}. Exploration epsilon {}.'.format(model_source, predictor_source,
                                                                            exp_epsilon),
             'own = cards in hand; hidden = public sizes of the next / other hidden hand; E|next| and E|other| are',
             'the sizes the belief expects. Q(chosen) is the predicted final score of the chosen move.',
             '', 'DECISIONS',
             '  {:>3}  {:13}  {:>3}  {:>6}  {:>5}  {:>7}  {:>8}  {:>9}  {:18}  {}'.format(
                 't', 'seat', 'own', 'hidden', 'legal', 'E|next|', 'E|other|', 'Q(chosen)', 'move', 'choice')]
    for d in decisions:
        e_next, e_other = expected_sizes(d['beliefs'])
        lines.append('  {:>3}  {:13}  {:>3}  {:>6}  {:>5}  {:>7}  {:>8}  {:>+9.3f}  {:18}  {}'.format(
            d['t'], d['seat'], d['own'], '{}/{}'.format(*d['hidden']), d['legal'], e_next, e_other, d['q'],
            action_text(d['move'])[:18], 'greedy' if d['greedy'] else 'random (exploration)'))
    lines += ['', 'GAME OVER',
              '  winner: {}  |  bombs and rockets played: {}  |  ADP return for the landlord = {}2^{} = {:+g}'.format(
                  winner, bombs, '+' if returns['landlord'] > 0 else '-', bombs, returns['landlord']),
              '  training return G per seat: ' + ', '.join('{} {:+g}'.format(p, returns[p]) for p in POSITIONS),
              '  decisions per seat: ' + ', '.join('{} {}'.format(p, len(stored[p])) for p in POSITIONS),
              "  every decision a seat made gets that seat's G as its Monte Carlo target, undiscounted",
              '', 'WHAT THE BUFFER STORES (3 of the {} rows; the learner never sees the other legal moves)'.format(
                  len(decisions))]
    middle = stored['landlord_up'][len(stored['landlord_up']) // 2]
    last_seat = decisions[-1]['seat']
    for label, row in (('landlord, first row', stored['landlord'][0]), ('landlord_up, a middle row', middle),
                       ('{}, last row'.format(last_seat), stored[last_seat][-1])):
        history = int((row['obs_z'].reshape(15, 54) != 0).any(dim=1).sum())
        belief_text = ('none (baseline network)' if 'belief' not in row else 'slot sums {:.1f} / {:.1f}'.format(
            row['belief'][:75].sum().item(), row['belief'][75:].sum().item()))
        lines += ['  {} (t={}): target {:+g}, done {}, episode_return {:+g}'.format(
                      label, row['t'], row['target'].item(), bool(row['done']), row['episode_return'].item()),
                  '    obs_x_no_action {} int8: own hand {};  obs_action (54,) int8: {}'.format(
                      tuple(row['obs_x_no_action'].shape), cards_text(decoded(row['obs_x_no_action'][:54])),
                      action_text(decoded(row['obs_action']))),
                  '    obs_z (5, 162) int8: {} non-pass moves among the last 15;  belief (150,) float32: {}'.format(
                      history, belief_text)]
    record = dict(decisions=decisions, winner=winner, bombs=bombs, returns=returns,
                  decisions_per_seat={p: len(stored[p]) for p in POSITIONS},
                  stored={p: [{k: v for k, v in row.items() if k != 't'} for row in stored[p]] for p in POSITIONS})
    return '\n'.join(lines), record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trace one complete game of one hand-belief arm.')
    parser.add_argument('--hand_model', default='derived_both', choices=('baseline',) + belief.HAND_MODELS)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--predictor_dir', default=None, help='frozen predictors; untrained ones when omitted')
    parser.add_argument('--checkpoint_dir', default=DEFAULT_CHECKPOINT_DIR,
                        help="DouZero checkpoints for the Q-network warm start ('' for random weights)")
    parser.add_argument('--exp_epsilon', type=float, default=0.01)
    parser.add_argument('--out', default=os.path.join('debug', 'episode_trace.txt'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    text, _ = trace_episode(args.hand_model, args.seed, args.predictor_dir, args.checkpoint_dir, args.exp_epsilon)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            f.write(text + '\n')
        print('\n(also written to {})'.format(args.out))
