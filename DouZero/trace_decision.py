"""Prints a readable trace of one decision for every hand-model arm (a learning and debugging tool).

A seeded deal is played with random legal moves for --num_random_moves moves. For the player
about to act, the trace shows what that player observes, what each arm's predictor outputs and
feeds to the Q-network, the Q-values of the legal actions, and the epsilon-greedy choice.

The Q-network is warm-started from DouZero-ADP with zero belief weights, so every arm scores
actions exactly as DouZero-ADP does until RL training moves those weights. The predictors are
untrained until offline pretraining exists: their beliefs show shapes and plumbing, not skill.
One process, one CPU thread, about a second:

    python trace_decision.py --seed 7 --num_random_moves 12
"""
import argparse
import os

import numpy as np
import torch

from douzero.dmc import belief, models
from douzero.env import hand_belief as hb
from douzero.env.env import Env

ARM_NAMES = {'none': 'M0c', 'next': 'M1', 'derived_both': 'M2c', 'learned_both': 'M2', 'oracle': 'Oracle'}
ARM_SLOTS = {'none': ('zeros', 'zeros'), 'next': ('p_next', 'zeros'),
             'derived_both': ('p_next', 'reflect(p_next)'), 'learned_both': ('p_next (M2)', 'p_other (M2)'),
             'oracle': ('one-hot true next', 'one-hot true other')}
FACES = dict(zip(hb.KINDS, '3456789TJQKA2BR'))  # B and R: small and big joker
DEFAULT_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baselines', 'douzero_ADP')


def cards_text(cards):
    return ''.join(FACES[card] for card in sorted(cards)) or 'pass'


def action_text(action):
    return 'play ' + cards_text(action) if action else 'pass'


def counts_to_cards(counts):
    return [kind for kind, num in zip(hb.KINDS, counts) for _ in range(num)]


def probs_text(probs):
    return '[' + ' '.join('{:.2f}'.format(p) for p in probs) + ']'


def describe(seat, position):
    if seat == 'landlord':
        return 'landlord'
    return seat + (' (teammate)' if position != 'landlord' else ' (peasant)')


def play_to_decision(seed, num_random_moves):
    """InfoSet and observation of the player to act after num_random_moves random moves of a seeded deal."""
    np.random.seed(seed)  # Env.reset deals with the global numpy RNG
    rng = np.random.RandomState(seed)
    env = Env('adp')
    obs = env.reset()
    for _ in range(num_random_moves):
        legal_actions = obs['legal_actions']
        obs, _, done, _ = env.step(legal_actions[rng.randint(len(legal_actions))])
        if done:
            raise ValueError('the game ended within {} moves; use fewer'.format(num_random_moves))
    return env.infoset, obs


def trace_decision(seed=7, num_random_moves=12, checkpoint_dir=DEFAULT_CHECKPOINT_DIR, exp_epsilon=0.01):
    """The trace of one decision, as text."""
    infoset, obs = play_to_decision(seed, num_random_moves)
    position = infoset.player_position
    next_player, other_player = hb.hidden_players(position)
    sizes = [infoset.num_cards_left_dict[next_player], infoset.num_cards_left_dict[other_player]]
    union = hb.union_counts(infoset)
    labels = hb.hand_labels(infoset)  # training-only; shown for comparison and used by the oracle arm
    legal_actions = obs['legal_actions']
    num_legal = len(legal_actions)
    to_beat = ('{} by {}'.format(cards_text(infoset.last_move), infoset.last_pid) if infoset.last_move
               else 'nothing (free lead)')
    lines = ['TRACE OF ONE DECISION  (seed {}, after {} random moves)'.format(seed, num_random_moves), '',
             'ACTING PLAYER',
             '  seat           {}. Acts next: {}. Other hidden seat: {}'.format(
                 position if position == 'landlord' else position + ' (peasant)',
                 describe(next_player, position), describe(other_player, position)),
             '  own hand       {} cards  {}'.format(len(infoset.player_hand_cards), cards_text(infoset.player_hand_cards)),
             '  to beat        ' + to_beat,
             '  hidden union   {} cards = {} {} + {} {} (sizes public, split hidden)'.format(
                 union.sum(), next_player, sizes[0], other_player, sizes[1]),
             '  unseen counts  ' + ' '.join('{}:{}'.format(FACES[kind], n) for kind, n in zip(hb.KINDS, union)),
             '  legal actions  {}'.format(num_legal), '',
             'OBSERVATION (public; douzero/env/env.py get_obs)',
             '  x_no_action  {:13s} int8     own hand | union | played cards | recent moves | hand sizes | bombs'.format(
                 str(obs['x_no_action'].shape)),
             '  z            {:13s} int8     last 15 moves, 3 per row: the LSTM history'.format(str(obs['z'].shape)),
             '  x_batch      {:13s} float32  one row per legal action: x_no_action | action (54)'.format(
                 str(obs['x_batch'].shape)),
             '  z_batch      {:13s} float32  z repeated for every legal action'.format(str(obs['z_batch'].shape)), '',
             'TRUE HIDDEN HANDS (training-only labels; of the five arms only Oracle sees them)',
             '  next   {:13s} {:2d} cards  {}'.format(next_player, sizes[0], cards_text(counts_to_cards(labels[0]))),
             '  other  {:13s} {:2d} cards  {}'.format(other_player, sizes[1], cards_text(counts_to_cards(labels[1]))),
             '']

    torch.manual_seed(seed)
    predictors = {1: models.HandPredictor(position, 1), 2: models.HandPredictor(position, 2)}
    z = torch.from_numpy(obs['z']).float()[None]
    x_no_action = torch.from_numpy(obs['x_no_action']).float()[None]
    with torch.no_grad():
        outputs = {k: predictor(z, x_no_action) for k, predictor in predictors.items()}
    p_next = outputs[1]['probs'][0, 0].numpy()
    p_reflected = hb.reflect_belief(p_next, union)
    p_two = outputs[2]['probs'][0].numpy()
    lines += ['PREDICTORS (untrained, random weights; input z (1, 5, 162) + x_no_action {})'.format(
                  tuple(x_no_action.shape)),
              '  M1 and M2c share one next-hand predictor: K=1, {:,} params -> logits, probs {}'.format(
                  sum(p.numel() for p in predictors[1].parameters()), tuple(outputs[1]['probs'].shape)),
              '  M2 has its own two-hand predictor:       K=2, {:,} params -> logits, probs {}'.format(
                  sum(p.numel() for p in predictors[2].parameters()), tuple(outputs[2]['probs'].shape)),
              '  P(count = 0..u) for the kinds with the most unseen copies:']
    row = '    {:4s} {:1s}  {:5s}  {:>4}  {:28s} {:35s} {}'
    lines.append(row.format('kind', 'u', 'hand', 'true', 'card counting', 'M1 / M2c', 'M2'))
    for kind in [r for r in sorted(range(hb.NUM_KINDS), key=lambda r: -union[r]) if union[r] > 0][:3]:
        u = union[kind]
        for slot, (hand, size, ours) in enumerate((('next', sizes[0], p_next), ('other', sizes[1], p_reflected))):
            counting = hb.hypergeometric_belief(union, size)[kind, :u + 1]
            lines.append(row.format(
                FACES[hb.KINDS[kind]] if slot == 0 else '', str(u) if slot == 0 else '', hand, labels[slot][kind],
                probs_text(counting), ('' if slot == 0 else 'reflect ') + probs_text(ours[kind, :u + 1]),
                probs_text(p_two[slot, kind, :u + 1])))
    lines.append('')

    beliefs = {hand_model: belief.build_belief(
        hand_model, x_no_action,
        probs=outputs[belief.PREDICTOR_TARGETS[hand_model]]['probs'] if belief.PREDICTOR_TARGETS[hand_model] else None,
        labels=torch.from_numpy(labels)[None] if hand_model == 'oracle' else None)
        for hand_model in belief.HAND_MODELS}
    lines.append('BELIEF INPUT (1, 150) = [next-player slot 75 | other-player slot 75]  (douzero/dmc/belief.py)')
    for hand_model, beliefs_one in beliefs.items():
        first, second = ARM_SLOTS[hand_model]
        lines.append('  {:6s} {:13s} [{:19s}| {:19s}]  slot sums {:4.1f} / {:4.1f}'.format(
            ARM_NAMES[hand_model], hand_model, first, second, beliefs_one[0, :75].sum().item(),
            beliefs_one[0, 75:].sum().item()))
    lines += ['  A filled slot holds 15 per-kind distributions, so it sums to 15.', '']

    checkpoint = os.path.join(checkpoint_dir, position + '.ckpt') if checkpoint_dir else None
    if checkpoint and os.path.exists(checkpoint):
        douzero_state_dict, source = torch.load(checkpoint, map_location='cpu'), 'DouZero-ADP warm start'
    else:
        douzero_state_dict = models.model_dict[position]().state_dict()
        source = 'no checkpoint: random DouZero weights, zero belief columns'
    q_network = models.warm_start_from_douzero(position, douzero_state_dict)
    z_batch, x_batch = torch.from_numpy(obs['z_batch']), torch.from_numpy(obs['x_batch'])
    with torch.no_grad():
        q = {hand_model: q_network(z_batch, x_batch, return_value=True,
                                   belief=beliefs_one.expand(num_legal, -1))['values'][:, 0]
             for hand_model, beliefs_one in beliefs.items()}
    values = q['none'].numpy()
    lines += ['Q-NETWORK (one network for every arm; {})'.format(source),
              '  input   LSTM(z_batch {}) -> 128 | x_batch {} | belief ({}, 150)  ->  dense1 input {}'.format(
                  tuple(z_batch.shape), tuple(x_batch.shape), num_legal, q_network.dense1.in_features),
              '  layers  {} -> 512 -> 512 -> 512 -> 512 -> 512 -> 1, {:,} parameters'.format(
                  q_network.dense1.in_features, sum(p.numel() for p in q_network.parameters())),
              '  output  Q ({}, 1): min {:+.3f}  max {:+.3f}  mean {:+.3f}'.format(
                  num_legal, values.min(), values.max(), values.mean()),
              '  identical for all five arms: {} (the new belief weights are still zero)'.format(
                  all(torch.equal(v, q['none']) for v in q.values())),
              '  Top 5 legal actions by Q:']
    for rank, i in enumerate(np.argsort(-values, kind='stable')[:5], 1):
        lines.append('    {}. Q = {:+.3f}  {}'.format(rank, values[i], action_text(legal_actions[i])))

    rng = np.random.RandomState(seed + 1)
    draw = rng.rand()
    if draw < exp_epsilon:
        choice, how = rng.randint(num_legal), 'random legal action'
    else:
        choice, how = int(np.argmax(values)), 'greedy (highest Q)'
    lines += ['', 'Epsilon-greedy (epsilon {}, the rule in models.py forward): draw {:.3f} {} {} -> {}: {}'.format(
        exp_epsilon, draw, '<' if draw < exp_epsilon else '>=', exp_epsilon, how, action_text(legal_actions[choice]))]
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trace one decision of every hand-model arm.')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--num_random_moves', type=int, default=12)
    parser.add_argument('--checkpoint_dir', default=DEFAULT_CHECKPOINT_DIR,
                        help='DouZero checkpoints used to warm-start the Q-network')
    parser.add_argument('--exp_epsilon', type=float, default=0.01)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(trace_decision(args.seed, args.num_random_moves, args.checkpoint_dir, args.exp_epsilon))
