"""Runs the real actor in the current process for a few games (tests and teaching tools).

In training, act() runs forever in its own process and learner threads drain the buffers it
fills. Here it plays max_episodes games into unshared buffers, with queues that start with
every buffer free and raise instead of blocking. The filled buffers can then be read back
in the order act() wrote them.
"""
from collections import deque

import torch

from douzero.dmc.utils import act, create_buffers

POSITIONS = ('landlord', 'landlord_up', 'landlord_down')


class NonBlockingQueue:
    """FIFO with the get/put interface that act() and get_batch() use; get() raises when empty."""

    def __init__(self, items=()):
        self._items = deque(items)

    def get(self):
        if not self._items:
            raise RuntimeError('queue is empty: in training this call would block forever')
        return self._items.popleft()

    def put(self, item):
        self._items.append(item)

    def __iter__(self):
        return iter(list(self._items))

    def __len__(self):
        return len(self._items)


def run_actor(flags, model, predictors=None, max_episodes=1, device='cpu'):
    """Plays max_episodes games with the real act(); returns (buffers, free_queue, full_queue), per seat."""
    buffers = create_buffers(flags, [device], share_memory=False)[device]
    free_queue = {p: NonBlockingQueue(range(flags.num_buffers)) for p in POSITIONS}
    full_queue = {p: NonBlockingQueue() for p in POSITIONS}
    act(0, device, free_queue, full_queue, model, buffers, flags, predictors, max_episodes=max_episodes)
    return buffers, free_queue, full_queue


def flushed_rows(buffers, full_queue, position):
    """Every row act() flushed for one seat, in decision order: {key: (num_flushed * T, ...)}."""
    indices = list(full_queue[position])
    return {key: torch.cat([buffers[position][key][i] for i in indices]) for key in buffers[position]}
