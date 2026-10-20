"""Prints one real predictor training sample against both baselines, and the overfit-one-batch check.

A learning tool for Stage 4: what one sample holds, what the legal-uniform and hypergeometric
baselines and a trained predictor say about one card kind, and that the pipeline can memorize a
small batch. Reads a dataset from generate_predictor_data.py and, optionally, a predictor run.
One process, a few seconds:

    python trace_predictor_example.py --data artifacts/stage4_tiny_data --predictors artifacts/runs/stage4_tiny_k1/checkpoints
"""
import argparse

import numpy as np
import torch

from douzero.dmc import models
from douzero.env import hand_belief as hb
from douzero.predictor import data, metrics, train

FACES = dict(zip(hb.KINDS, '3 4 5 6 7 8 9 10 J Q K A 2 BJ RJ'.split()))


def cards(counts):
    return ' '.join(FACES[kind] for kind, n in zip(hb.KINDS, counts) for _ in range(int(n))) or '-'


def example(data_dir, predictor_dir=None, seat='landlord_down', phase=1, pick=3):
    arrays = data.load_split(data_dir, 'val', seat)
    i = int(np.flatnonzero(arrays['phase'] == phase)[pick])
    union, sizes, labels = arrays['union'][i].astype(int), arrays['sizes'][i], arrays['labels'][i].astype(int)
    next_player, other_player = hb.hidden_players(seat)
    lines = ['ONE SAMPLE: {}, validation split, game {}, decision t = {} of {} ({} phase)'.format(
                 seat, int(arrays['game'][i]), int(arrays['t'][i]), int(arrays['game_length'][i]),
                 data.PHASES[int(arrays['phase'][i])]),
             '  input   x_no_action {} int8 with {} ones; z {} int8 with {} ones'.format(
                 arrays['x_no_action'][i].shape, int(arrays['x_no_action'][i].sum()), arrays['z'][i].shape,
                 int(arrays['z'][i].sum())),
             '  public  {} unseen cards: {}; {} (next) holds {}, {} (other) holds {}'.format(
                 union.sum(), cards(union), next_player, sizes[0], other_player, sizes[1]),
             '  label   next hand {}  |  other hand {}'.format(cards(labels[0]), cards(labels[1]))]
    kind = int(np.argmax(union))
    lines.append('  kind {}: {} unseen, the next hand really holds {}'.format(FACES[hb.KINDS[kind]], union[kind],
                                                                             labels[0][kind]))
    beliefs = [('legal-uniform', metrics.uniform_probs(union[None])[0]),
               ('hypergeometric', metrics.hypergeometric_probs(union[None], sizes[0:1])[0])]
    if predictor_dir:
        predictor = models.load_hand_predictor('{}/{}.ckpt'.format(predictor_dir, seat), position=seat)
        with torch.no_grad():
            probs = predictor(torch.from_numpy(arrays['z'][i:i + 1]).float(),
                              torch.from_numpy(arrays['x_no_action'][i:i + 1]).float())['probs'][0, 0]
        samples = predictor.provenance.get('train_samples')
        beliefs.append(('predictor ({} samples)'.format(samples), probs.double().numpy()))
    for name, probs in beliefs:
        lines.append('    {:26s} P(0..{}) = {}   -log P(true) = {:.3f}   CE over 15 kinds = {:.3f}'.format(
            name, union[kind], ' '.join('{:.2f}'.format(p) for p in probs[kind][:union[kind] + 1]),
            -np.log(probs[kind][labels[0][kind]]), metrics.nll(probs[None], labels[0][None]).mean()))
    batch = data.load_split(data_dir, 'train', seat)
    batch = {key: value[:24] for key, value in batch.items()}
    lines.append('OVERFIT ONE BATCH (24 real training samples of {}, 150 Adam steps from fresh weights)'.format(seat))
    for num_targets in (1, 2):
        first, last = train.overfit_one_batch(batch, seat, num_targets, steps=150, seed=0)
        lines.append('    K={}: CE {:.3f} -> {:.4f}'.format(num_targets, first, last))
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print one predictor training sample and the overfit check.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--predictors', default=None, help='a K=1 predictor run checkpoints/ folder')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    text = example(args.data, args.predictors)
    print(text)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(text + '\n')
