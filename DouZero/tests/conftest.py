import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from douzero.env.env import Env

torch.set_num_threads(1)  # local tests stay single-threaded


@pytest.fixture(autouse=True)
def no_training_processes(monkeypatch):
    """Unit tests may call train() only to see it refuse to start: one that got past its checks would spawn
    --num_actors actor processes (5 by default) and train for ever. This makes that fail at once instead.
    The opt-in tests/test_integration.py starts train.py as subprocesses, which this does not affect."""
    from douzero.dmc import dmc

    def refuse(*args, **kwargs):
        raise RuntimeError('a unit test tried to start training processes; only tests/test_integration.py may')
    monkeypatch.setattr(dmc.mp, 'get_context', refuse)


def play_random_games(num_games, seed):
    """Seeded random self-play. Returns one record per decision state, in play order.

    Each record holds the acting player's InfoSet (a deep copy made by GameEnv.get_infoset,
    so later steps never mutate it) and the observation built from it.
    """
    np.random.seed(seed)  # Env.reset deals with the global numpy RNG
    action_rng = np.random.RandomState(seed + 1)
    env = Env('adp')
    states = []
    for game in range(num_games):
        obs = env.reset()
        done = False
        step = 0
        while not done:
            states.append({'game': game, 'step': step, 'infoset': env.infoset, 'obs': obs})
            legal_actions = obs['legal_actions']
            action = legal_actions[action_rng.randint(len(legal_actions))]
            obs, _, done, _ = env.step(action)
            step += 1
    return states


@pytest.fixture(scope='session')
def random_play_states():
    return play_random_games(num_games=60, seed=2026)
