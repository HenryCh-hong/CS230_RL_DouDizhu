"""Records one game of one arm and renders it as a step-by-step HTML replay (a learning and debugging tool).

The game is played as actor 0 of a CPU-actor run (--actor_device_cpu) with --seed plays it at its
first start; trace_episode.py prints the same game as text, and GPU actors, seeded from their
device, play other games. It writes <out>.json, the record, and <out>.html, the page; open the
page in a browser and step with the arrow keys. --reveal (teaching mode) also records and shows
the true hidden hands; without it the page holds only what the acting player knows. One process,
a few seconds:

    python render_episode_replay.py --hand_model derived_both --seed 0 --reveal
"""
import argparse
import json
import os

import torch

from douzero.dmc import belief
from douzero.viz import record, render


def main(argv=None):
    parser = argparse.ArgumentParser(description='Render one game of one arm as an HTML replay.')
    parser.add_argument('--hand_model', default='derived_both', choices=('baseline',) + belief.HAND_MODELS)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--predictor_dir', default=None, help='frozen predictors; untrained ones when omitted')
    parser.add_argument('--checkpoint_dir', default=record.DEFAULT_CHECKPOINT_DIR,
                        help="DouZero checkpoints for the Q-network warm start ('' for random weights)")
    parser.add_argument('--exp_epsilon', type=float, default=0.01)
    parser.add_argument('--reveal', action='store_true', help='teaching mode: record and show the hidden hands')
    parser.add_argument('--out', default=os.path.join('debug', 'replays', 'replay'),
                        help='output path without extension; writes .json and .html')
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    game = record.record_episode(args.hand_model, args.seed, args.predictor_dir, args.checkpoint_dir,
                                 args.exp_epsilon, reveal=args.reveal)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out + '.json', 'w') as f:
        json.dump(game, f, indent=1)
    with open(args.out + '.html', 'w') as f:
        f.write(render.render_replay(game))
    print('{} decisions; {} wins. Wrote {}.json and {}.html'.format(
        len(game['steps']), game['result']['winner'], args.out, args.out))


if __name__ == '__main__':
    main()
