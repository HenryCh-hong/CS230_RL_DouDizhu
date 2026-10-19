"""Shared helpers for the Stage-3 tests: predictor files, flags and small saved runs."""
import os
import threading

import torch

from douzero.dmc import belief, dmc, models
from douzero.dmc.arguments import parser

POSITIONS = ('landlord', 'landlord_up', 'landlord_down')


def save_predictors(directory, num_targets, trained=True, seed=0):
    """Writes one seeded, untrained-weight predictor file per seat and returns the directory."""
    torch.manual_seed(seed)
    os.makedirs(directory, exist_ok=True)
    for position in POSITIONS:
        models.save_hand_predictor(models.HandPredictor(position, num_targets),
                                   os.path.join(directory, position + '.ckpt'),
                                   provenance=dict(trained=trained, data='test fixture'))
    return str(directory)


def arm_flags(hand_model, tmp_path, *extra):
    args = ['--hand_model', hand_model, '--actor_device_cpu', '--training_device', 'cpu', '--seed', '3',
            '--savedir', str(tmp_path / 'runs'), '--xpid', 'run_' + hand_model]
    num_targets = belief.PREDICTOR_TARGETS.get(hand_model, 0)
    if num_targets:
        args += ['--predictor_dir', save_predictors(tmp_path / 'predictors_k{}'.format(num_targets), num_targets)]
    return parser.parse_args(args + list(extra))


def one_optimizer_step(learner, optimizers):
    """A tiny made-up update per seat, so the optimizer state is not empty."""
    torch.manual_seed(0)
    for position in POSITIONS:
        network = learner.get_model(position)
        x = torch.rand(4, 373 if position == 'landlord' else 484)
        beliefs = torch.rand(4, belief.BELIEF_DIM) if network.belief_dim else None
        loss = network(torch.rand(4, 5, 162), x, return_value=True, belief=beliefs)['values'].pow(2).mean()
        optimizers[position].zero_grad()
        loss.backward()
        optimizers[position].step()


def saved_run(hand_model, tmp_path):
    """A randomly initialized run of one arm, saved with save_checkpoint as training would."""
    os.makedirs(tmp_path, exist_ok=True)
    flags = arm_flags(hand_model, tmp_path)
    predictors = models.load_predictors(flags.hand_model, flags.predictor_dir)
    metadata = dmc.run_metadata(flags, predictors)
    _, learner, optimizers = dmc.initial_models(flags, ['cpu'])
    one_optimizer_step(learner, optimizers)
    path = str(tmp_path / (hand_model + '.tar'))
    dmc.save_checkpoint(path, learner, optimizers, flags, metadata, [threading.Lock() for _ in POSITIONS],
                        stats={'loss_landlord': 0.5}, frames=3200, position_frames={p: 0 for p in POSITIONS},
                        updates={p: 1 for p in POSITIONS})
    return path, flags, predictors, metadata, learner, optimizers
