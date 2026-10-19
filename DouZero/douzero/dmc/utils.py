import os 
import typing
import logging
import traceback
import numpy as np
from collections import Counter
import time

import torch 
from torch import multiprocessing as mp

from .belief import BELIEF_DIM, build_belief
from .env_utils import Environment
from .oracle import oracle_labels
from douzero.env import Env
from douzero.env.env import _cards2array

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('doudzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
Buffers = typing.Dict[str, typing.List[torch.Tensor]]

def create_env(flags):
    return Env(flags.objective)

class StopLearning(Exception):
    """Raised in a learner thread that is waiting for a batch when training shuts down."""

def get_batch(free_queue,
              full_queue,
              buffers,
              flags,
              lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    A None index is train()'s shutdown signal and raises StopLearning.
    """
    with lock:
        indices = []
        while len(indices) < flags.batch_size:
            index = full_queue.get()
            if index is None:
                raise StopLearning
            indices.append(index)
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch

def create_optimizers(flags, learner_model):
    """
    Create three optimizers for the three positions
    """
    positions = ['landlord', 'landlord_up', 'landlord_down']
    optimizers = {}
    for position in positions:
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha)
        optimizers[position] = optimizer
    return optimizers

def create_buffers(flags, device_iterator, share_memory=True):
    """
    We create buffers for different positions as well as
    for different devices (i.e., GPU). That is, each device
    will have three buffers for the three positions.
    Hand-belief arms also store each decision's 150-d belief.
    share_memory=False makes plain tensors, for single-process use.
    """
    T = flags.unroll_length
    positions = ['landlord', 'landlord_up', 'landlord_down']
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in positions:
            x_dim = 319 if position == 'landlord' else 430
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
                obs_action=dict(size=(T, 54), dtype=torch.int8),
                obs_z=dict(size=(T, 5, 162), dtype=torch.int8),
            )
            if getattr(flags, 'hand_model', 'baseline') != 'baseline':
                specs['belief'] = dict(size=(T, BELIEF_DIM), dtype=torch.float32)
            _buffers: Buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):
                for key in _buffers:
                    if not device == "cpu":
                        _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device)))
                    else:
                        _buffer = torch.empty(**specs[key]).to(torch.device('cpu'))
                    _buffers[key].append(_buffer.share_memory_() if share_memory else _buffer)
            buffers[device][position] = _buffers
    return buffers

def actor_seeds(seed, device, i, attempt=0):
    """(numpy seed, torch seed) of actor i on device in a run with --seed seed (attempt counts resumes).

    Derived with numpy's SeedSequence from all four numbers, so consecutive run seeds no longer
    share streams the way --seed + i did. np.random.seed takes 32 bits, so two combinations collide
    with probability about 2^-32 per pair: negligible for the few hundred actor streams of this study.
    """
    device_index = 0 if device == 'cpu' else int(device) + 1
    numpy_seed, torch_seed = np.random.SeedSequence([seed, device_index, i, attempt]).generate_state(2)
    return int(numpy_seed), int(torch_seed)

def seed_actor(flags, device, i):
    """Seeds actor i's stream of deals and exploration from --seed (see actor_seeds)."""
    if getattr(flags, 'seed', None) is not None:
        numpy_seed, torch_seed = actor_seeds(flags.seed, device, i, getattr(flags, 'attempt', 0))
        np.random.seed(numpy_seed)
        torch.manual_seed(torch_seed)

def actor_belief(hand_model, predictor, position, x_no_action, z, labels=None):
    """(1, 150) belief input of one decision, built from the public observation only.

    x_no_action and z are this decision's observation tensors. The predictor (None for M0c
    and the oracle) must be frozen and for this seat, and runs without gradients. labels,
    the true hands, are passed only for the oracle arm.
    """
    x = x_no_action.float()[None]
    probs = None
    if predictor is not None:
        if predictor.position != position or predictor.training or any(p.requires_grad for p in predictor.parameters()):
            raise ValueError('a {} decision needs that seat\'s frozen predictor'.format(position))
        with torch.no_grad():
            probs = predictor(z.float()[None], x)['probs']
    return build_belief(hand_model, x, probs=probs, labels=labels)

def act(i, device, free_queue, full_queue, model, buffers, flags, predictors=None, max_episodes=None):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.

    For a hand-belief arm (flags.hand_model) every decision also gets its 150-d belief from
    actor_belief. The belief goes into the Q-network and is stored with the decision, so the
    learner sees exactly what the actor saw. predictors maps each seat to its frozen
    HandPredictor or None; max_episodes stops after that many games (tests and tools only).
    """
    positions = ['landlord', 'landlord_up', 'landlord_down']
    hand_model = getattr(flags, 'hand_model', 'baseline')
    predictors = predictors or {p: None for p in positions}
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)
        seed_actor(flags, device, i)

        env = create_env(flags)
        env = Environment(env, device)
        torch_device = torch.device('cpu' if device == 'cpu' else 'cuda:' + str(device))

        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_no_action_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        belief_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}
        episodes = 0

        position, obs, env_output = env.initial()

        while max_episodes is None or episodes < max_episodes:
            while True:
                obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
                obs_z_buf[position].append(env_output['obs_z'])
                belief = None
                if hand_model != 'baseline':
                    labels = None
                    if hand_model == 'oracle':  # the only arm allowed to see the true hands
                        labels = oracle_labels(env.env.infoset, position, env_output['obs_x_no_action']).to(torch_device)
                    belief = actor_belief(hand_model, predictors[position], position,
                                          env_output['obs_x_no_action'].to(torch_device),
                                          env_output['obs_z'].to(torch_device), labels)
                    belief_buf[position].append(belief[0].cpu())  # a host copy, like the other stored rows
                    belief = belief.expand(len(obs['legal_actions']), -1)
                with torch.no_grad():
                    agent_output = model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags, belief=belief)
                _action_idx = int(agent_output['action'].cpu().detach().numpy())
                action = obs['legal_actions'][_action_idx]
                obs_action_buf[position].append(_cards2tensor(action))
                size[position] += 1
                position, obs, env_output = env.step(action)
                if env_output['done']:
                    for p in positions:
                        diff = size[p] - len(target_buf[p])
                        if diff > 0:
                            done_buf[p].extend([False for _ in range(diff-1)])
                            done_buf[p].append(True)

                            episode_return = env_output['episode_return'] if p == 'landlord' else -env_output['episode_return']
                            episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
                            episode_return_buf[p].append(episode_return)
                            target_buf[p].extend([episode_return for _ in range(diff)])
                    break
            episodes += 1

            for p in positions:
                if hand_model != 'baseline' and len(belief_buf[p]) != len(obs_x_no_action_buf[p]):
                    raise RuntimeError('belief rows are out of step with observation rows for ' + p)
                while size[p] > T:
                    index = free_queue[p].get()
                    if index is None:
                        break
                    for t in range(T):
                        buffers[p]['done'][index][t, ...] = done_buf[p][t]
                        buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        buffers[p]['obs_x_no_action'][index][t, ...] = obs_x_no_action_buf[p][t]
                        buffers[p]['obs_action'][index][t, ...] = obs_action_buf[p][t]
                        buffers[p]['obs_z'][index][t, ...] = obs_z_buf[p][t]
                        if hand_model != 'baseline':
                            buffers[p]['belief'][index][t, ...] = belief_buf[p][t]
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                    obs_action_buf[p] = obs_action_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    belief_buf[p] = belief_buf[p][T:]
                    size[p] -= T

    except KeyboardInterrupt:
        pass  
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e

def _cards2tensor(list_cards):
    """
    Convert a list of integers to the tensor
    representation
    See Figure 2 in https://arxiv.org/pdf/2106.06135.pdf
    """
    matrix = _cards2array(list_cards)
    matrix = torch.from_numpy(matrix)
    return matrix
