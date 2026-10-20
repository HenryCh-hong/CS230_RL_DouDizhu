"""Generates the Stage 4 predictor dataset from frozen DouZero-ADP self-play (see douzero/predictor/data.py).

Every game is seeded on its own, so the output does not depend on --workers. Before the manifest
is written, no deal may appear twice, within or across splits, and none may be one of the
evaluation deals of tools/dup_eval.py for the listed seeds; otherwise no manifest is written.
The full dataset runs remotely; on a laptop use a handful of games:

    python generate_predictor_data.py --out ../data/stage4_tiny --train_games 40 --val_games 10 --test_games 10
"""
import argparse
import importlib.util
import time
from pathlib import Path

import torch

from douzero.predictor import data

ROOT = Path(__file__).resolve().parents[1]


def evaluation_deal_hashes(seeds, count):
    """Deal hashes of the duplicate-deal evaluation sets tools/dup_eval.py would draw for these seeds."""
    spec = importlib.util.spec_from_file_location('dup_eval', ROOT / 'tools' / 'dup_eval.py')
    dup_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dup_eval)
    return {data.deal_hash(deal) for seed in seeds for deal in dup_eval.make_deals(count, seed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate the offline predictor dataset.')
    parser.add_argument('--out', required=True)
    parser.add_argument('--train_games', type=int, required=True)
    parser.add_argument('--val_games', type=int, required=True)
    parser.add_argument('--test_games', type=int, required=True)
    parser.add_argument('--checkpoint_dir', default='baselines/douzero_ADP', help='the frozen DouZero-ADP networks')
    parser.add_argument('--exp_epsilon', type=float, default=0.01, help="the RL actors' exploration rate")
    parser.add_argument('--shard_games', type=int, default=500)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--eval_seeds', type=int, nargs='*', default=[0, 1, 2],
                        help='dup_eval.py seeds whose deals must not be predictor deals')
    parser.add_argument('--eval_deals', type=int, default=10000, help='deals per evaluation seed to check against')
    parser.add_argument('--skip_eval_deal_check', action='store_true', help='tests only: no evaluation seeds')
    args = parser.parse_args(argv)
    if not args.eval_seeds and not args.skip_eval_deal_check:
        parser.error('give --eval_seeds (the evaluation deals to keep out), or --skip_eval_deal_check for tests')
    torch.set_num_threads(1)
    start = time.time()
    forbidden = evaluation_deal_hashes(args.eval_seeds, args.eval_deals) if args.eval_seeds else set()
    source = ('tools/dup_eval.py make_deals({}, seed) for seeds {}'.format(args.eval_deals, args.eval_seeds)
              if args.eval_seeds else None)
    manifest = data.generate(args.out, dict(train=args.train_games, val=args.val_games, test=args.test_games),
                             checkpoint_dir=args.checkpoint_dir, exp_epsilon=args.exp_epsilon,
                             shard_games=args.shard_games, workers=args.workers, forbidden_deals=forbidden,
                             forbidden_source=source)
    samples = {s: sum(v['samples'].values()) for s, v in manifest['splits'].items()}
    print('{} decisions ({}) in {:.0f} s; {} deals, all unique; evaluation deals checked: {}. Wrote {}'.format(
        sum(samples.values()), ', '.join('{} {}'.format(s, n) for s, n in samples.items()), time.time() - start,
        manifest['deal_check']['games'], manifest['deal_check']['evaluation'], args.out))


if __name__ == '__main__':
    main()
