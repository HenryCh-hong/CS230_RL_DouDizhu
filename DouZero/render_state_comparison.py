"""Compares the five arms at one decision point of a recorded game (a learning and debugging tool).

The state is rebuilt exactly from a replay record (render_episode_replay.py writes one): the same
deal, then the recorded moves. Each arm then builds its belief input for that state and scores
the legal moves. It writes <out>.json and <out>.html. With --reveal the page also shows the true
hidden hands and how far each belief is from them. One process, about a second:

    python render_state_comparison.py --record debug/replays/replay.json --step 20 --reveal
"""
import argparse
import json
import os

import torch

from douzero.dmc import models
from douzero.viz import record, render


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compare the five arms at one decision of a recorded game.')
    parser.add_argument('--record', required=True, help='a replay record (.json) from render_episode_replay.py')
    parser.add_argument('--step', type=int, default=None, help='decision index t; the middle of the game by default')
    parser.add_argument('--k1_predictor_dir', default=None, help='the shared M1/M2c predictor; untrained if omitted')
    parser.add_argument('--k2_predictor_dir', default=None, help='the M2 predictor; untrained if omitted')
    parser.add_argument('--checkpoint_dir', default=record.DEFAULT_CHECKPOINT_DIR,
                        help="DouZero checkpoints for the shared Q-network warm start ('' for random weights)")
    parser.add_argument('--reveal', action='store_true', help='teaching mode: show the true hidden hands')
    parser.add_argument('--out', default=os.path.join('debug', 'replays', 'state'),
                        help='output path without extension; writes .json and .html')
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    with open(args.record) as f:
        game = json.load(f)
    if args.reveal and not game['revealed']:
        parser.error('--reveal needs a record made with --reveal')
    t = len(game['steps']) // 2 if args.step is None else args.step
    if not 0 <= t < len(game['steps']):
        parser.error('--step must be between 0 and {} for this game'.format(len(game['steps']) - 1))
    k1 = models.load_predictors('next', args.k1_predictor_dir, require_trained=False) if args.k1_predictor_dir else None
    k2 = models.load_predictors('learned_both', args.k2_predictor_dir, require_trained=False) \
        if args.k2_predictor_dir else None
    comparison = record.compare_state(game, t, k1, k2, args.checkpoint_dir, reveal=args.reveal)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out + '.json', 'w') as f:
        json.dump(comparison, f, indent=1)
    with open(args.out + '.html', 'w') as f:
        f.write(render.render_comparison(comparison))
    print('Decision t={} ({} to act). Identical rankings across arms: {}. Wrote {}.json and {}.html'.format(
        t, comparison['state']['seat'], comparison['identical_rankings'], args.out, args.out))


if __name__ == '__main__':
    main()
