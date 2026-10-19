"""Prints one learner update of one hand-belief arm, before and after (a learning and debugging tool).

The real actor (act(), run in this process) plays a few games, the real get_batch() takes a
batch for one seat, and the real learn() makes one update. The trace shows the batch's tensors,
the Q-values and loss before and after, the gradient norm, how far each part of the Q-network
moved, that the frozen predictor did not move at all, and that the actor's copy received the
new weights. Run from DouZero/; one process, one CPU thread, a few seconds:

    python trace_learner_update.py --hand_model derived_both --seat landlord --seed 0
"""
import argparse
import math
import os
import threading
from collections import Counter

import torch

from douzero.dmc import belief, dmc, inprocess, models, utils
from douzero.dmc.arguments import parser as training_parser

POSITIONS = ('landlord', 'landlord_up', 'landlord_down')
DEFAULT_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baselines', 'douzero_ADP')


def _norm(tensors):
    return math.sqrt(sum(t.pow(2).sum().item() for t in tensors))


def trace_learner_update(hand_model='derived_both', seat='landlord', seed=0, predictor_dir=None,
                         checkpoint_dir=DEFAULT_CHECKPOINT_DIR, unroll_length=8, batch_size=2, games=4):
    """Runs one real learner update and returns (text, record)."""
    num_targets = belief.PREDICTOR_TARGETS.get(hand_model, 0)
    args = ['--hand_model', hand_model, '--unroll_length', str(unroll_length), '--batch_size', str(batch_size),
            '--num_buffers', '64', '--seed', str(seed), '--actor_device_cpu', '--training_device', 'cpu']
    if checkpoint_dir:
        args += ['--warm_start_dir', checkpoint_dir]
    flags = training_parser.parse_args(args)
    torch.manual_seed(seed)
    actors, learner, optimizers = dmc.initial_models(flags, ['cpu'])
    if num_targets and predictor_dir:
        predictors, predictor_source = models.load_predictors(hand_model, predictor_dir, require_trained=False), \
            'frozen, from ' + predictor_dir
    elif num_targets:
        torch.manual_seed(seed)
        predictors = {p: models.HandPredictor(p, num_targets).eval().requires_grad_(False) for p in POSITIONS}
        predictor_source = 'UNTRAINED and frozen'
    else:
        predictors, predictor_source = {p: None for p in POSITIONS}, 'none for this arm'
    buffers, free_queue, full_queue = inprocess.run_actor(flags, actors['cpu'], predictors, max_episodes=games)
    if len(full_queue[seat]) < batch_size:
        raise RuntimeError('only {} buffers filled for {}; play more games'.format(len(full_queue[seat]), seat))
    lock = threading.Lock()
    batch = utils.get_batch(free_queue[seat], full_queue[seat], buffers[seat], flags, lock)
    network = learner.get_model(seat)
    obs_z, obs_x, beliefs, target = dmc.learner_inputs(batch, torch.device('cpu'))

    def q_and_loss():
        with torch.no_grad():
            q = network(obs_z, obs_x, return_value=True, belief=beliefs)['values'][:, 0]
        return q, ((q - target) ** 2).mean().item()

    q_before, loss_before = q_and_loss()
    before = {k: v.clone() for k, v in network.state_dict().items()}
    predictor = predictors[seat]
    predictor_before = None if predictor is None else {k: v.clone() for k, v in predictor.state_dict().items()}
    stats = dmc.learn(seat, actors, network, batch, optimizers[seat], flags, lock)
    q_after, loss_after = q_and_loss()

    after = network.state_dict()
    delta = {k: after[k] - before[k] for k in before}
    original = network.dense1.in_features - network.belief_dim
    belief_change = None
    if network.belief_dim:
        belief_change = [delta['dense1.weight'][:, original + 75 * s:original + 75 * (s + 1)].norm().item()
                         for s in (0, 1)]
    predictor_change = None if predictor is None else max(
        (predictor.state_dict()[k] - v).abs().max().item() for k, v in predictor_before.items())
    synced_state = actors['cpu'].get_model(seat).state_dict()
    actor_synced = all(torch.equal(synced_state[k], v) for k, v in after.items())
    record = dict(stats=stats, q_network_change=_norm(delta.values()), belief_column_change=belief_change,
                  predictor_change=predictor_change, actor_synced=actor_synced,
                  loss_before=loss_before, loss_after=loss_after)

    T, B = batch['target'].shape
    targets = Counter('{:+g}'.format(g) for g in target.tolist())
    grad_norm = stats['grad_norm_' + seat]
    lines = ['ONE LEARNER UPDATE, ARM {} (--hand_model {}), seat {}, seed {}'.format(
                 belief.ARM_NAMES[hand_model], hand_model, seat, seed),
             'Q-network: {}. Predictor: {}.'.format(
                 'DouZero-ADP warm start' if checkpoint_dir else 'random weights', predictor_source),
             '', 'BATCH (the real get_batch() after {} games of the real act(): T={} steps x B={} buffers = {} '
                 'decisions)'.format(games, T, B, T * B),
             '  obs_x_no_action {} int8 + obs_action {} int8  ->  obs_x {} float32 = [x_no_action | action]'.format(
                 tuple(batch['obs_x_no_action'].shape), tuple(batch['obs_action'].shape), tuple(obs_x.shape)),
             '  obs_z           {} int8  ->  {} float32'.format(tuple(batch['obs_z'].shape), tuple(obs_z.shape)),
             '  belief          {}'.format('{} float32  ->  {}, exactly as the actor computed it'.format(
                 tuple(batch['belief'].shape), tuple(beliefs.shape)) if beliefs is not None else 'none (baseline)'),
             '  target G        {} float32  ->  {}: {}'.format(tuple(batch['target'].shape), tuple(target.shape),
                                                               dict(sorted(targets.items()))),
             '', 'BEFORE THE UPDATE',
             '  Q mean {:+.4f}  std {:.4f}  |  loss = mean (Q - G)^2 = {:.4f}'.format(
                 q_before.mean().item(), q_before.std().item(), loss_before),
             '', 'THE UPDATE (dmc.learn: loss.backward(), clip_grad_norm_ to {}, one RMSprop step, sync to actors)'.format(
                 flags.max_grad_norm),
             '  gradient norm {:.4f} ({})'.format(grad_norm, 'clipped to {}'.format(flags.max_grad_norm)
                                                if grad_norm > flags.max_grad_norm else 'not clipped'),
             '', 'AFTER THE UPDATE (same batch)',
             '  Q mean {:+.4f}  std {:.4f}  |  loss = {:.4f} ({:+.4f})'.format(
                 q_after.mean().item(), q_after.std().item(), loss_after, loss_after - loss_before),
             '', 'PARAMETER CHANGE (L2 norm of new minus old weights)',
             '  Q-network, all parameters             {:.6f}'.format(record['q_network_change']),
             '    LSTM over z                          {:.6f}'.format(
                 _norm(v for k, v in delta.items() if k.startswith('lstm'))),
             '    dense1, its {} original inputs      {:.6f}'.format(
                 original, _norm([delta['dense1.weight'][:, :original], delta['dense1.bias']]))]
    if belief_change is not None:
        lines += ['    dense1, belief slot 1 (75 inputs)    {:.6f}'.format(belief_change[0]),
                  '    dense1, belief slot 2 (75 inputs)    {:.6f}'.format(belief_change[1])]
    lines += ['    dense2 .. dense6                     {:.6f}'.format(
                  _norm(v for k, v in delta.items() if k.startswith('dense') and not k.startswith('dense1'))),
              '  predictor                              {}'.format(
                  'none' if predictor_change is None else '{:g} (frozen: no gradient, no optimizer)'.format(
                      predictor_change)),
              "  actor copy equals the learner         {} (learn() syncs after every update)".format(actor_synced)]
    return '\n'.join(lines), record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trace one learner update of one hand-belief arm.')
    parser.add_argument('--hand_model', default='derived_both', choices=('baseline',) + belief.HAND_MODELS)
    parser.add_argument('--seat', default='landlord', choices=POSITIONS)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--predictor_dir', default=None, help='frozen predictors; untrained ones when omitted')
    parser.add_argument('--checkpoint_dir', default=DEFAULT_CHECKPOINT_DIR,
                        help="DouZero checkpoints for the Q-network warm start ('' for random weights)")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(trace_learner_update(args.hand_model, args.seat, args.seed, args.predictor_dir, args.checkpoint_dir)[0])
