"""Trains and selects the Stage 4 predictors, one per seat (see douzero/predictor/train.py).

K=1 is the next-hand predictor shared by M1 and M2c; K=2 is M2's two-hand predictor. The run
folder holds the selected checkpoints (checkpoints/<seat>.ckpt, ready for --predictor_dir),
metrics.csv, the baselines, the once-only test scores, summary.md and plots/. Real training runs
remotely; on a laptop only tiny pipeline checks:

    python train_predictor.py --data ../data/stage4_tiny --num_targets 1 --run_id k1_tiny --max_steps 60 --eval_every 20
"""
import argparse
import os

import torch

from douzero.predictor import train


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train the offline hidden-hand predictors.')
    parser.add_argument('--data', required=True, help='a dataset from generate_predictor_data.py')
    parser.add_argument('--num_targets', type=int, choices=(1, 2), required=True)
    parser.add_argument('--run_id', required=True)
    parser.add_argument('--savedir', default=os.path.join('artifacts', 'runs'))
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_steps', type=int, default=20000)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--examples', type=int, default=12)
    parser.add_argument('--threads', type=int, default=1, help='torch CPU threads')
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    run = train.train_predictor(args.data, args.num_targets, os.path.join(args.savedir, args.run_id),
                                batch_size=args.batch_size, lr=args.lr, max_steps=args.max_steps,
                                eval_every=args.eval_every, patience=args.patience, seed=args.seed,
                                examples=args.examples)
    print('Done. Selected predictors in {}/checkpoints; see {}/summary.md and {}/plots/index.html'.format(run, run, run))


if __name__ == '__main__':
    main()
