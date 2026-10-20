import contextlib
import copy
import hashlib
import io
import math
import os
import signal
import threading
import time
import timeit
import pprint
import traceback
from collections import deque
import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn

from . import run_dir
from .belief import ARM_NAMES, BELIEF_DIM, PREDICTOR_TARGETS
from .file_writer import FileWriter
from .models import Model, file_sha256, load_predictors, predictor_record, warm_start_from_douzero
from .utils import StopLearning, get_batch, log, create_env, create_buffers, create_optimizers, act

POSITIONS = ['landlord', 'landlord_up', 'landlord_down']
# Learner statistics, logged per seat as '<name>_<seat>'.
STAT_NAMES = ['mean_episode_return', 'loss', 'q_mean', 'q_std', 'grad_norm', 'clipped']
# Flags that change what is learned; a resumed run must keep them (--total_frames and the like may change).
RESUME_FIXED_FLAGS = ('objective', 'unroll_length', 'batch_size', 'learning_rate', 'alpha', 'momentum', 'epsilon',
                      'max_grad_norm', 'exp_epsilon', 'seed')
# How long the shutdown waits for the learners, and for each lock of the final checkpoint.
STOP_TIMEOUT_S = 60

mean_episode_return_buf = {p:deque(maxlen=100) for p in ['landlord', 'landlord_up', 'landlord_down']}

def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets)**2).mean()
    return loss

def stat_keys():
    """Every learner statistic train() logs."""
    return [name + '_' + position for position in POSITIONS for name in STAT_NAMES]

def learner_inputs(batch, device):
    """The Q-network inputs of a (T, B) batch, flattened row-major: row t * B + b is step t of buffer b.

    Returns obs_z (T*B, 5, 162), obs_x = [x_no_action | action] (T*B, 373 or 484), the stored
    belief (T*B, 150) or None for the original DouZero network, and the Monte Carlo target
    (T*B,). Raises ValueError if any part of the batch has the wrong shape.
    """
    T, B = batch['target'].shape
    x_dim = batch['obs_x_no_action'].shape[-1]
    expected = {'obs_x_no_action': (T, B, x_dim), 'obs_action': (T, B, 54), 'obs_z': (T, B, 5, 162),
                'done': (T, B), 'episode_return': (T, B)}
    if 'belief' in batch:
        expected['belief'] = (T, B, BELIEF_DIM)
    for key, shape in expected.items():
        if tuple(batch[key].shape) != shape:
            raise ValueError('batch[{!r}] has shape {}, expected {}'.format(key, tuple(batch[key].shape), shape))
    if x_dim not in (319, 430):
        raise ValueError('x_no_action has {} features, expected 319 or 430'.format(x_dim))
    obs_x = torch.cat((batch['obs_x_no_action'].to(device), batch['obs_action'].to(device)), dim=2).float()
    obs_x = torch.flatten(obs_x, 0, 1)
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    target = torch.flatten(batch['target'].to(device), 0, 1)
    belief = None
    if 'belief' in batch:
        belief = torch.flatten(batch['belief'].to(device), 0, 1)
        if not torch.isfinite(belief).all():
            raise ValueError('the batch holds a non-finite belief')
    return obs_z, obs_x, belief, target

def learn(position,
          actor_models,
          model,
          batch,
          optimizer,
          flags,
          lock):
    """Performs a learning (optimization) step."""
    if flags.training_device != "cpu":
        device = torch.device('cuda:'+str(flags.training_device))
    else:
        device = torch.device('cpu')
    obs_z, obs_x, belief, target = learner_inputs(batch, device)
    episode_returns = batch['episode_return'][batch['done']]
    if episode_returns.numel() > 0:  # a batch can hold no finished game
        mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))

    with lock:
        learner_outputs = model(obs_z, obs_x, return_value=True, belief=belief)
        values = learner_outputs['values']
        loss = compute_loss(values, target)
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite Q-loss for {}; stopping before any weight changes'.format(position))
        returns = mean_episode_return_buf[position]
        stats = {
            'mean_episode_return_'+position: torch.mean(torch.stack(list(returns))).item() if returns else float('nan'),
            'loss_'+position: loss.item(),
            'q_mean_'+position: values.mean().item(),
            'q_std_'+position: values.std().item(),
        }

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm).item()  # norm before clipping
        if not math.isfinite(grad_norm):
            raise RuntimeError('non-finite gradient for {}; stopping before any weight changes'.format(position))
        stats['grad_norm_'+position] = grad_norm
        stats['clipped_'+position] = float(grad_norm > flags.max_grad_norm)
        optimizer.step()

        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(model.state_dict())
        return stats

def check_flags(flags):
    """Rejects contradictory hand-belief settings before anything starts."""
    if not flags.freeze_predictor:
        raise NotImplementedError('joint predictor fine-tuning is not implemented; the main experiment freezes predictors')
    if (PREDICTOR_TARGETS.get(flags.hand_model, 0) > 0) != bool(flags.predictor_dir):
        raise ValueError('--predictor_dir is required by next, derived_both and learned_both, and only by them')
    if flags.load_model and flags.warm_start_dir:
        raise ValueError('--load_model resumes a run and --warm_start_dir starts a new one; give only one')
    if flags.fork_from_checkpoint:
        if not (flags.fork_reason or '').strip():
            raise ValueError('--fork_from_checkpoint needs a --fork_reason saying why the fork is made')
        if flags.load_model or flags.warm_start_dir:
            raise ValueError('--fork_from_checkpoint starts a new run; give it without --load_model and --warm_start_dir')
    elif flags.fork_reason is not None:
        raise ValueError('--fork_reason is only for --fork_from_checkpoint')
    if flags.unsafe_resume_across_code_change and not flags.load_model:
        raise ValueError('--unsafe_resume_across_code_change only applies to --load_model')
    if flags.num_buffers < flags.batch_size:
        raise ValueError('--num_buffers must be at least --batch_size, or a learner can never fill a batch')

def warm_start_record(warm_start_dir):
    """The DouZero files a run starts from, with their sha256 (None when it starts from scratch)."""
    if not warm_start_dir:
        return None
    files = {}
    for position in POSITIONS:
        path = os.path.abspath(os.path.join(warm_start_dir, position + '.ckpt'))
        files[position] = dict(path=path, sha256=file_sha256(path))
    return dict(dir=os.path.abspath(warm_start_dir), files=files)

def run_metadata(flags, predictors):
    """What every checkpoint carries to rebuild and audit its arm; plain types only (weights_only loading)."""
    return dict(arm=ARM_NAMES[flags.hand_model], hand_model=flags.hand_model,
                belief_dim=0 if flags.hand_model == 'baseline' else BELIEF_DIM, seed=flags.seed,
                predictor=predictor_record(flags.predictor_dir, predictors),
                warm_start=warm_start_record(flags.warm_start_dir), git=run_dir.git_info(),
                run=dict(xpid=flags.xpid, run_dir=os.path.abspath(os.path.join(flags.savedir, flags.xpid))),
                fork=None, invalid_for_final_results=False)

def warm_start_states(flags, belief_dim):
    """{seat: state_dict} the learner starts from with --warm_start_dir, or None."""
    if not flags.warm_start_dir:
        return None
    states = {}
    for position in POSITIONS:
        douzero_state = torch.load(os.path.join(flags.warm_start_dir, position + '.ckpt'), map_location='cpu',
                                   weights_only=True)
        states[position] = warm_start_from_douzero(position, douzero_state).state_dict() if belief_dim else douzero_state
    return states

def initial_models(flags, device_iterator, checkpoint_states=None):
    """Actor models (one Model per device), the learner Model and its optimizers, all with the same weights.

    The learner starts from the checkpoint being resumed or forked, else from --warm_start_dir, else from
    its random initialization. Every actor copy then gets exactly the learner's weights. DouZero copied
    them only when resuming, so warm-started actors would have played with random networks until
    the first learner update. A checkpoint without optimizer state (a weights file) starts fresh optimizers.
    """
    belief_dim = 0 if flags.hand_model == 'baseline' else BELIEF_DIM
    learner_model = Model(device=flags.training_device, belief_dim=belief_dim)
    optimizers = create_optimizers(flags, learner_model)
    if checkpoint_states is not None:
        for k in POSITIONS:
            learner_model.get_model(k).load_state_dict(checkpoint_states['model_state_dict'][k])
            if 'optimizer_state_dict' in checkpoint_states:
                optimizers[k].load_state_dict(checkpoint_states['optimizer_state_dict'][k])
                for group in optimizers[k].param_groups:  # this run's settings, not the parent's (equal on a resume)
                    group.update(lr=flags.learning_rate, momentum=flags.momentum, eps=flags.epsilon, alpha=flags.alpha)
    else:
        start = warm_start_states(flags, belief_dim)
        if start is not None:
            for k in POSITIONS:
                learner_model.get_model(k).load_state_dict(start[k])
    models = {}
    for device in device_iterator:
        model = Model(device=device, belief_dim=belief_dim)
        for k in POSITIONS:
            model.get_model(k).load_state_dict(learner_model.get_model(k).state_dict())
        model.share_memory()
        model.eval()
        models[device] = model
    return models, learner_model, optimizers

def frozen_copies(predictors, device):
    """The arm's frozen predictors on one actor device, in shared memory (None for seats without one)."""
    copies = {}
    for position, predictor in predictors.items():
        if predictor is not None:
            predictor = copy.deepcopy(predictor).to(torch.device('cpu' if device == 'cpu' else 'cuda:' + str(device)))
            predictor.share_memory()
        copies[position] = predictor
    return copies

def _atomic_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)  # if the process dies mid-write, the previous file stays intact

def save_checkpoint(path, learner_model, optimizers, flags, metadata, locks, progress=None, lock_timeout=None,
                    **counters):
    """Writes model.tar atomically, from a snapshot taken while no learner update is half-applied.

    locks are held while copying. progress, if given, returns the counters (stats, frames,
    position_frames, updates, elapsed_s) and is called under the same locks, so the counters match
    the weights; otherwise the counters come as keywords. lock_timeout, if given, bounds the wait for
    each lock and raises TimeoutError after it: the final checkpoint uses it, so a learner stuck while
    holding its lock cannot hang the shutdown. Returns the snapshot.
    """
    with contextlib.ExitStack() as held:
        for lock in locks:
            if not lock.acquire(timeout=-1 if lock_timeout is None else lock_timeout):
                raise TimeoutError('a checkpoint lock was still held after {} s'.format(lock_timeout))
            held.callback(lock.release)
        snapshot = dict(
            model_state_dict={k: {name: t.detach().clone() for name, t in learner_model.get_model(k).state_dict().items()}
                              for k in POSITIONS},
            optimizer_state_dict={k: copy.deepcopy(optimizers[k].state_dict()) for k in POSITIONS},
            **copy.deepcopy(progress() if progress is not None else counters))
    snapshot['flags'] = vars(flags)
    snapshot['cs230'] = metadata
    _atomic_save(snapshot, path)
    return snapshot

def save_weights(path, snapshot, metadata):
    """Writes every seat's weights at one point with the run metadata, so load_arm_checkpoint can rebuild that arm."""
    _atomic_save(dict(model_state_dict=snapshot['model_state_dict'], frames=snapshot['frames'], cs230=metadata), path)

def read_checkpoint(path, flag='--load_model'):
    if not os.path.isfile(path):
        raise FileNotFoundError('{}: no checkpoint at {}'.format(flag, path))
    return torch.load(path, map_location='cpu', weights_only=True)

def _predictor_hashes(record):
    return None if record is None else (record['num_targets'], {p: f['sha256'] for p, f in record['files'].items()})

def check_resume(states, metadata, flags, path):
    """Refuses to resume a run as a different experiment; returns True when a code change was overridden.

    The arm, the predictor files (compared by sha256, so a copied run still resumes), every flag in
    RESUME_FIXED_FLAGS and the code must match the saved run. The code matches when its git commit and
    code_sha256 (the tracked diff and the untracked files) are equal and both git states are known. A
    different code version is a different experiment, so it is refused unless --unsafe_resume_across_code_change
    (debugging only) lets it through; the caller then marks the run invalid for final results.
    """
    saved = states.get('cs230')
    if saved is None:
        raise ValueError('{} has no run metadata, so it cannot be resumed as this run'.format(path))
    hint = 'a different code version is a different experiment: fork it instead, with ' + _fork_hint(path)
    differences = [key for key in ('hand_model', 'belief_dim') if saved[key] != metadata[key]]
    if _predictor_hashes(saved['predictor']) != _predictor_hashes(metadata['predictor']):
        differences.append('predictor files')
    saved_flags = states.get('flags', {})
    differences += [key for key in RESUME_FIXED_FLAGS if saved_flags.get(key) != getattr(flags, key, None)]
    if differences:
        raise ValueError('cannot resume {}: {} differ from the saved run. {}'.format(
            path, ', '.join(differences), hint))
    for which, git in (('saved', saved['git']), ('current', metadata['git'])):
        if git['commit'] is None or git['dirty'] is None or git['code_sha256'] is None:
            raise ValueError('cannot resume {}: the git state of the {} code is unknown (commit {}, dirty {}), so it '
                             'cannot be shown to match the saved run. {}'.format(path, which, git['commit'],
                                                                                 git['dirty'], hint))
    code = [key for key in ('commit', 'code_sha256') if saved['git'][key] != metadata['git'][key]]
    if not code:
        return False
    if not flags.unsafe_resume_across_code_change:
        raise ValueError('cannot resume {}: the code differs from the saved run in {} (saved commit {}, sha256 {}; '
                         'now commit {}, sha256 {}). {}. For debugging only, '
                         '--unsafe_resume_across_code_change resumes anyway and marks the run invalid for final '
                         'results'.format(path, ', '.join(code), saved['git']['commit'], saved['git']['code_sha256'],
                                          metadata['git']['commit'], metadata['git']['code_sha256'], hint))
    return True

def _fork_hint(path):
    return '--fork_from_checkpoint {} --fork_reason TEXT and a new --xpid'.format(path)

def read_fork_parent(path, metadata, flags):
    """The parent checkpoint of a fork, and the fork record that config.json and every later checkpoint carry.

    path is a run's checkpoints/model.tar or weights_<frames>.tar. The parent must be a checkpoint of this study
    with the same network shape (belief_dim); its arm may differ, and the record names it. Its optimizer state
    is restored only when the file holds one. Returns (parent states, fork record).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError('--fork_from_checkpoint: no checkpoint at {}'.format(path))
    with open(path, 'rb') as f:  # hash and load the same bytes, even if the parent saves again meanwhile
        raw = f.read()
    sha256 = hashlib.sha256(raw).hexdigest()
    states = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    saved = states.get('cs230')
    if saved is None:
        raise ValueError('{} has no run metadata; it is not a checkpoint of this study, so it cannot be forked'.format(path))
    if saved['belief_dim'] != metadata['belief_dim']:
        raise ValueError('cannot fork {}: its network has belief_dim {} but this run has {}'.format(
            path, saved['belief_dim'], metadata['belief_dim']))
    checkpoint = os.path.realpath(path)
    if saved.get('run'):  # the run the file itself says it belongs to, wherever the file now is
        parent_run_id = dict(xpid=saved['run']['xpid'], run_dir=saved['run']['run_dir'])
    else:  # a checkpoint saved before runs recorded their identity: inferred from <run>/checkpoints/<file>
        run_path = os.path.dirname(os.path.dirname(checkpoint))
        parent_run_id = dict(xpid=os.path.basename(run_path), run_dir=run_path, inferred_from_path=True)
    record = dict(parent_run_id=parent_run_id, parent_checkpoint=dict(path=checkpoint, sha256=sha256),
                  parent_frames=states['frames'], parent_hand_model=saved['hand_model'],
                  parent_code=dict(commit=saved['git']['commit'], code_sha256=saved['git']['code_sha256']),
                  current_code=dict(commit=metadata['git']['commit'], code_sha256=metadata['git']['code_sha256']),
                  reason=flags.fork_reason.strip(), optimizer_state_restored='optimizer_state_dict' in states,
                  parent_fork=saved.get('fork'), parent_warm_start=saved.get('warm_start'),
                  actor_lineage=int(sha256[:8], 16) | 1)  # nonzero: the fork's actors get streams of their own
    return states, record

def load_checkpoint_states(path, metadata, flags):
    """The saved state of a run to resume, after check_resume has confirmed it is the same experiment."""
    states = read_checkpoint(path)
    check_resume(states, metadata, flags, path)
    return states

class Terminated(Exception):
    """Raised in the main thread on SIGTERM or SIGHUP (a timeout, a preemption, a closed terminal), so the
    run still saves a final checkpoint and summary."""

def _raise_terminated(signum, frame):
    raise Terminated(signal.Signals(signum).name)

def install_stop_handlers():
    """Makes SIGTERM and SIGHUP raise Terminated; returns {signal: previous handler} to restore on exit.

    A signal the run was started with ignored stays ignored: nohup ignores SIGHUP so that a run
    survives a closed terminal, and that must keep working.
    """
    previous = {}
    for signum in (signal.SIGTERM, signal.SIGHUP):
        if signal.getsignal(signum) is not signal.SIG_IGN:
            old = signal.signal(signum, _raise_terminated)
            # None means a handler installed outside Python, which cannot be reinstalled; restore the default.
            previous[signum] = signal.SIG_DFL if old is None else old
    return previous

def check_workers(actor_processes, failures, threads=(), training_over=lambda: False):
    """While training runs, fails it loudly if a learner thread failed or stopped, or an actor process died.

    A learner leaves its loop once frames reach --total_frames, so a stopped thread is an error only if
    training_over() is still False when it is seen stopped (read after the thread check, so a learner
    that has just finished the last update is never mistaken for a failed one).
    """
    if failures:
        raise RuntimeError('a learner thread failed:\n' + '\n'.join(failures))
    for actor in actor_processes:
        if not actor.is_alive():
            raise RuntimeError('actor process {} exited with code {}'.format(actor.pid, actor.exitcode))
    for thread in threads:
        if not thread.is_alive() and not training_over():
            raise RuntimeError('learner thread {} stopped before training ended'.format(thread.name))

def wake_learners(full_queues, num_threads, threads, timeout):
    """Puts num_threads None sentinels on each full queue, then waits up to timeout s for the threads to stop.

    Returns the names of the threads still running. The puts run in daemon helper threads, one per
    queue: a queue's write lock is shared with the actors, and an actor killed while holding it (an
    out-of-memory kill, say) would block a put for ever. A learner that is never woken stays blocked
    and ends with the process.
    """
    def send(queue):
        for _ in range(num_threads):
            queue.put(None)
    for queue in full_queues:
        threading.Thread(target=send, args=(queue,), name='stop-sentinels', daemon=True).start()
    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    return [thread.name for thread in threads if thread.is_alive()]

def stop_actors(actor_processes):
    """Terminates the actors. DouZero never stopped them, so a finished run would hang on exit."""
    for actor in actor_processes:
        if actor.is_alive():
            actor.terminate()
    for actor in actor_processes:
        actor.join(timeout=10)
        if actor.is_alive():
            actor.kill()
            actor.join()

def metrics_row(elapsed, frames, fps, updates, stats):
    """One metrics.csv row: wall time, frames and throughput, then per seat its update count and latest learner stats."""
    row = dict(time_s=round(elapsed, 1), frames=frames, fps=round(fps, 1))
    for position in POSITIONS:
        row['updates_' + position] = updates[position]
        for name in STAT_NAMES:
            row[name + '_' + position] = stats.get(name + '_' + position, float('nan'))
    return row

def train(flags):
    """
    This is the main funtion for training. It will first
    initilize everything, such as buffers, optimizers, etc.
    Then it will start subprocesses as actors. Then, it will call
    learning function with  multiple threads.

    For the hand-belief study it also validates everything before creating anything; records the
    run in <savedir>/<xpid>/ (config.json, attempts.jsonl, metrics.csv, summary.md, checkpoints/);
    loads the arm's frozen predictors; starts the actors from the learner's weights; and stops every
    worker, saving a final checkpoint and summary, whether training ends, fails, is interrupted or
    receives SIGTERM or SIGHUP. Returns the final status, 'completed' on success.
    """
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError("CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`")

    # Validate everything before creating anything, so a refused start leaves no run directory behind.
    check_flags(flags)
    run_path = os.path.join(flags.savedir, flags.xpid)
    run_dir.check_run_path(run_path, resume=flags.load_model)
    checkpointpath = run_dir.checkpoint_path(run_path)
    checkpoint_states = read_checkpoint(checkpointpath) if flags.load_model else None
    if flags.seed is None:  # a resumed run keeps its seed; a new run draws one and records it
        flags.seed = checkpoint_states['flags']['seed'] if checkpoint_states is not None else run_dir.fresh_seed()
    np.random.seed(flags.seed)
    torch.manual_seed(flags.seed)
    predictors = load_predictors(flags.hand_model, flags.predictor_dir,
                                 require_trained=not flags.allow_untrained_predictors)
    metadata = run_metadata(flags, predictors)
    run_dir.check_git(metadata['git'], flags.allow_dirty)
    fork_states = None
    if checkpoint_states is not None:
        code_overridden = check_resume(checkpoint_states, metadata, flags, checkpointpath)
        metadata['warm_start'] = checkpoint_states['cs230']['warm_start']  # where the run originally started
        metadata['fork'] = checkpoint_states['cs230'].get('fork')  # the parent it was forked from, if any
        flags.lineage = (metadata['fork'] or {}).get('actor_lineage', 0)  # a resumed fork keeps its own streams
        metadata['invalid_for_final_results'] = (  # sticky: a run marked invalid stays invalid
            checkpoint_states['cs230'].get('invalid_for_final_results', False) or code_overridden)
        if code_overridden:
            log.warning('--unsafe_resume_across_code_change: resuming %s across a code change; the run is invalid '
                        'for final results', run_path)
    elif flags.fork_from_checkpoint:
        fork_states, metadata['fork'] = read_fork_parent(flags.fork_from_checkpoint, metadata, flags)
        new = os.path.realpath(run_path)
        for parent in {os.path.realpath(metadata['fork']['parent_run_id']['run_dir']),
                       os.path.dirname(os.path.dirname(metadata['fork']['parent_checkpoint']['path']))}:
            if new == parent or new.startswith(parent + os.sep):
                raise ValueError('a fork must not be created inside its parent run folder {}; choose another '
                                 '--savedir'.format(parent))
        metadata['warm_start'] = metadata['fork']['parent_warm_start']  # where the lineage started
        flags.lineage = metadata['fork']['actor_lineage']
        metadata['invalid_for_final_results'] = fork_states['cs230'].get('invalid_for_final_results', False)

    T = flags.unroll_length
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'

    # Initialize the learner and actor models, with identical weights, and the actors' frozen predictors
    start_states = checkpoint_states if checkpoint_states is not None else fork_states  # the learner's first weights
    models, learner_model, optimizers = initial_models(flags, device_iterator, start_states)
    actor_predictors = {device: frozen_copies(predictors, device) for device in device_iterator}

    # Initialize buffers
    buffers = create_buffers(flags, device_iterator)

    # Everything that can refuse the start has run; only now is the run recorded.
    run = run_dir.RunDir(run_path, resume=flags.load_model)
    if checkpoint_states is not None:
        config = run.read_config()
    else:
        config = run_dir.run_config(flags, metadata)
        run.write_config(config)
    flags.attempt = len(run.attempts())  # gives a resumed run's actors fresh random streams
    if checkpoint_states is not None:
        run.truncate_metrics(checkpoint_states['frames'])  # rows after the checkpoint describe lost progress
    run.record_attempt(dict(started=run_dir.now(), attempt=flags.attempt, seed=flags.seed, git=metadata['git'],
                            resumed_from_frames=None if checkpoint_states is None else checkpoint_states['frames'],
                            invalid_for_final_results=metadata['invalid_for_final_results'], flags=vars(flags)))
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )

    # Initialize queues
    actor_processes = []
    ctx = mp.get_context('spawn')
    free_queue = {}
    full_queue = {}

    for device in device_iterator:
        _free_queue = {'landlord': ctx.SimpleQueue(), 'landlord_up': ctx.SimpleQueue(), 'landlord_down': ctx.SimpleQueue()}
        _full_queue = {'landlord': ctx.SimpleQueue(), 'landlord_up': ctx.SimpleQueue(), 'landlord_down': ctx.SimpleQueue()}
        free_queue[device] = _free_queue
        full_queue[device] = _full_queue

    # Stat Keys
    keys = stat_keys()
    frames, stats = 0, {k: float('nan') for k in keys}  # nan until a seat's first update
    position_frames = {'landlord':0, 'landlord_up':0, 'landlord_down':0}
    updates = {p: 0 for p in POSITIONS}
    elapsed_before = 0.0

    if checkpoint_states is not None:
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        updates = checkpoint_states.get("updates", updates)
        elapsed_before = checkpoint_states.get("elapsed_s", 0.0)
        log.info(f"Resuming preempted job, current stats:\n{stats}")

    stopping = threading.Event()
    failures = []
    stats_lock = threading.Lock()

    def batch_and_learn(i, device, position, local_lock, position_lock):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, stats
        try:
            while frames < flags.total_frames and not stopping.is_set():
                batch = get_batch(free_queue[device][position], full_queue[device][position], buffers[device][position], flags, local_lock)
                if frames >= flags.total_frames or stopping.is_set():  # training ended while this thread waited
                    break
                with position_lock:  # an update and its counters change together; checkpoints hold this lock
                    _stats = learn(position, models, learner_model.get_model(position), batch,
                        optimizers[position], flags, contextlib.nullcontext())
                    with stats_lock:
                        for k in _stats:
                            stats[k] = _stats[k]
                        to_log = dict(frames=frames)
                        to_log.update({k: stats[k] for k in keys})
                        plogger.log(to_log)
                        frames += T * B
                        position_frames[position] += T * B
                        updates[position] += 1
        except StopLearning:
            pass
        except Exception:
            failures.append('{}: {}'.format(threading.current_thread().name, traceback.format_exc()))
            raise

    for device in device_iterator:
        for m in range(flags.num_buffers):
            free_queue[device]['landlord'].put(m)
            free_queue[device]['landlord_up'].put(m)
            free_queue[device]['landlord_down'].put(m)

    threads = []
    locks = {}
    for device in device_iterator:
        locks[device] = {'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock()}
    position_locks = {'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock()}

    timer = timeit.default_timer
    run_start = timer() - elapsed_before

    def checkpoint(lock_timeout=None):
        # No frames parameter: progress() must read train()'s counter under the locks, together with
        # the weights. A value passed in would be read before the locks, while a learner may still finish
        # an update that the saved weights then contain.
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        snapshot = save_checkpoint(
            checkpointpath, learner_model, optimizers, flags, metadata,
            [position_locks[p] for p in POSITIONS] + [stats_lock],
            progress=lambda: dict(stats=stats, frames=frames, position_frames=position_frames, updates=updates,
                                  elapsed_s=timer() - run_start),
            lock_timeout=lock_timeout)
        save_weights(run.weights_tar_path(snapshot['frames']), snapshot, metadata)
        if flags.hand_model == 'baseline':  # plain per-seat files, for DouZero's evaluate.py
            for position in POSITIONS:
                _atomic_save(snapshot['model_state_dict'][position], run.weights_path(position, snapshot['frames']))

    def stop_learners():
        """Lets an update in progress finish, then wakes every learner waiting for a batch with None."""
        stopping.set()
        stuck = wake_learners([full_queue[d][p] for d in device_iterator for p in POSITIONS], flags.num_threads,
                              threads, timeout=STOP_TIMEOUT_S)
        if stuck:  # recorded in the status; the shutdown goes on without them
            raise RuntimeError('still running {} s after the stop: {}'.format(STOP_TIMEOUT_S, ', '.join(stuck)))

    fps_log = []
    status, failure = 'completed', None
    last_logged_frames = None
    previous_handlers = {}
    try:
        previous_handlers = install_stop_handlers()  # inside the try, so the finally always restores them
        # Starting actor processes
        for device in device_iterator:
            num_actors = flags.num_actors
            for i in range(flags.num_actors):
                actor = ctx.Process(
                    target=act,
                    args=(i, device, free_queue[device], full_queue[device], models[device], buffers[device], flags,
                          actor_predictors[device]),
                    daemon=True)
                actor.start()
                actor_processes.append(actor)

        for device in device_iterator:
            for i in range(flags.num_threads):
                for position in POSITIONS:
                    thread = threading.Thread(
                        target=batch_and_learn, name='learner-{}-{}-{}'.format(device, position, i),
                        args=(i,device,position,locks[device][position],position_locks[position]), daemon=True)
                    thread.start()
                    threads.append(thread)

        last_checkpoint_time = timer() - flags.save_interval * 60
        while frames < flags.total_frames:
            start_frames = frames
            position_start_frames = {k: position_frames[k] for k in position_frames}
            start_time = timer()
            time.sleep(5)
            check_workers(actor_processes, failures, threads, training_over=lambda: frames >= flags.total_frames)

            if timer() - last_checkpoint_time > flags.save_interval * 60:
                checkpoint()
                last_checkpoint_time = timer()
            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            position_fps = {k:(position_frames[k]-position_start_frames[k])/(end_time-start_time) for k in position_frames}
            log.info('After %i (L:%i U:%i D:%i) frames: @ %.1f fps (avg@ %.1f fps) (L:%.1f U:%.1f D:%.1f) Stats:\n%s',
                     frames,
                     position_frames['landlord'],
                     position_frames['landlord_up'],
                     position_frames['landlord_down'],
                     fps,
                     fps_avg,
                     position_fps['landlord'],
                     position_fps['landlord_up'],
                     position_fps['landlord_down'],
                     pprint.pformat(stats))
            run.log_metrics(metrics_row(timer() - run_start, frames, fps, updates, stats))
            last_logged_frames = frames
        # A learner can fail, or an actor die, in the last tick, after another learner reached total_frames.
        check_workers(actor_processes, failures)

    except KeyboardInterrupt:
        status = 'interrupted'
    except Terminated as e:
        status = 'terminated ({})'.format(e)
    except Exception as e:
        status, failure = 'failed: {!r}'.format(e), traceback.format_exc()
        raise
    finally:
        problems = []
        for name, step in (('stopping the learners', stop_learners),
                           ('the final checkpoint', lambda: checkpoint(lock_timeout=STOP_TIMEOUT_S)),
                           ('stopping the actors', lambda: stop_actors(actor_processes))):
            try:
                step()
            except BaseException as e:  # keep shutting down, whatever went wrong
                problems.append('{} failed: {!r}'.format(name, e))
                log.error('%s failed', name, exc_info=True)
        if problems:
            status += '; ' + '; '.join(problems)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        elapsed = timer() - run_start
        if frames != last_logged_frames:
            run.log_metrics(metrics_row(elapsed, frames, frames / max(elapsed, 1e-9), updates, stats))
        run.write_summary(run_dir.summary_markdown(config, run.attempts(), status, elapsed, frames, updates, stats,
                                                   checkpointpath, failure))
        plogger.close(successful=status == 'completed')
    if status == 'completed':
        log.info('Learning finished after %d frames.', frames)
    else:
        log.warning('Training stopped after %d frames: %s', frames, status)
    return status
