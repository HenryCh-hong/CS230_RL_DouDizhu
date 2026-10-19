"""Opt-in end-to-end tests of train(): one actor process, two learner threads per seat, a few hundred
frames per run, about a minute and a half in all.

They start real processes, so they are skipped unless CS230_INTEGRATION=1:

    CS230_INTEGRATION=1 ../.venv/bin/python -m pytest tests/test_integration.py -p no:cacheprovider -q
"""
import csv
import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from douzero.dmc import models
from douzero.evaluation import simulation
from helpers import save_predictors

pytestmark = pytest.mark.skipif(os.environ.get('CS230_INTEGRATION') != '1',
                                reason='set CS230_INTEGRATION=1 to run train() end to end')

REPO = Path(__file__).resolve().parents[1]
ADP = REPO / 'baselines' / 'douzero_ADP'
ENV = dict(os.environ, OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
FRAMES_PER_UPDATE = 8 * 2  # --unroll_length x --batch_size below


def train_command(tmp_path, xpid, frames=200, *extra, resume=False, warm_start=ADP):
    predictors = tmp_path / 'predictors'
    if not predictors.exists():
        save_predictors(predictors, 1, trained=False)
    # A resume restarts from the run's own checkpoint, so it takes --load_model instead of --warm_start_dir.
    start = ['--load_model'] if resume else ['--warm_start_dir', str(warm_start)]
    return ([sys.executable, str(REPO / 'train.py'), '--hand_model', 'derived_both', '--predictor_dir', str(predictors),
             '--allow_untrained_predictors'] + start +
            ['--actor_device_cpu', '--training_device', 'cpu', '--num_actors', '1', '--num_threads', '2',
             '--num_buffers', '6', '--unroll_length', '8', '--batch_size', '2', '--total_frames', str(frames),
             '--save_interval', '1', '--seed', '0', '--savedir', str(tmp_path / 'runs'), '--xpid', xpid,
             '--allow_dirty'] + list(extra))


def run(command):
    return subprocess.run(command, cwd=REPO, env=ENV, capture_output=True, text=True, timeout=180)


def start(tmp_path, xpid, stderr_path):
    """Starts a run that would train for ever.

    Its stderr goes to a file, not a pipe: nobody reads a pipe until the end, and once it is full
    (512 bytes on macOS) the run blocks on its next log line, before it can install its signal handlers.
    """
    with open(stderr_path, 'w') as stderr_file:
        return subprocess.Popen(train_command(tmp_path, xpid, 10 ** 9), cwd=REPO, env=ENV,
                                stdout=subprocess.DEVNULL, stderr=stderr_file)


def stop(process):
    """Kills a run a test has given up on, actors first, so nothing outlives the test."""
    if process.poll() is None:
        for pid in actor_pids(process):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.kill()
        process.wait()


def metric_frames(run_path):
    with open(run_path / 'metrics.csv', newline='') as f:
        return [int(row['frames']) for row in csv.DictReader(f)]


def wait_for_metrics(process, run_path, rows, timeout=120, past_first_save=False):
    """Waits until metrics.csv has this many rows (one per 5 s training tick), and, with past_first_save, until
    it reports more frames than the first periodic checkpoint; False if the run never gets there."""
    deadline = time.time() + timeout
    while time.time() < deadline and process.poll() is None:
        if (run_path / 'metrics.csv').exists() and len(metric_frames(run_path)) >= rows:
            if not past_first_save or metric_frames(run_path)[-1] > min(saved_frames(run_path), default=float('inf')):
                return True
        time.sleep(0.5)
    return False


def saved_frames(run_path):
    """The frame counts of the weights_<frames>.tar files, one per checkpoint, in order."""
    return sorted(int(p.stem.split('_')[1]) for p in (run_path / 'checkpoints').glob('weights_*.tar'))


def actor_pids(process):
    found = subprocess.run(['pgrep', '-P', str(process.pid), '-f', 'spawn_main'], capture_output=True, text=True)
    return [int(pid) for pid in found.stdout.split()]


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def checked_checkpoint(run_path):
    """model.tar's saved state, after checking that its frame count matches the updates in its weights."""
    states = torch.load(run_path / 'checkpoints' / 'model.tar', map_location='cpu', weights_only=True)
    assert states['frames'] == FRAMES_PER_UPDATE * sum(states['updates'].values()) == sum(
        states['position_frames'].values())
    return states


def test_a_tiny_run_completes_resumes_and_refuses_a_changed_flag(tmp_path):
    first = run(train_command(tmp_path, 'm2c'))
    assert first.returncode == 0, first.stderr[-2000:]
    run_path = tmp_path / 'runs' / 'm2c'
    assert '**Status:** completed' in (run_path / 'summary.md').read_text()
    frames = metric_frames(run_path)
    assert frames == sorted(frames) and frames[-1] >= 200
    assert checked_checkpoint(run_path)['frames'] == frames[-1]
    arm = models.load_arm_checkpoint(str(run_path / 'checkpoints' / 'model.tar'), allow_untrained=True)
    assert arm['hand_model'] == 'derived_both'
    weights = sorted((run_path / 'checkpoints').glob('weights_*.tar'))
    assert weights and models.load_arm_checkpoint(str(weights[-1]), allow_untrained=True)['hand_model'] == 'derived_both'
    assert not list((run_path / 'checkpoints').glob('*_weights_*.ckpt'))  # per-seat DouZero files: baseline only

    resumed = run(train_command(tmp_path, 'm2c', 400, resume=True))
    assert resumed.returncode == 0, resumed.stderr[-2000:]
    frames = metric_frames(run_path)
    assert frames == sorted(frames) and frames[-1] >= 400
    assert checked_checkpoint(run_path)['frames'] == frames[-1]
    attempts = [json.loads(line) for line in (run_path / 'attempts.jsonl').read_text().splitlines()]
    assert [a['flags']['total_frames'] for a in attempts] == [200, 400]  # each attempt records its own flags
    assert attempts[1]['resumed_from_frames'] >= 200

    refused = run(train_command(tmp_path, 'm2c', 600, '--learning_rate', '0.001', resume=True))
    assert refused.returncode != 0 and 'learning_rate' in refused.stderr
    assert len((run_path / 'attempts.jsonl').read_text().splitlines()) == 2  # refused before anything was recorded


@pytest.mark.parametrize('signum, status', [(signal.SIGTERM, 'terminated (SIGTERM)'),
                                            (signal.SIGHUP, 'terminated (SIGHUP)'),
                                            (signal.SIGINT, 'interrupted')])
def test_a_stop_signal_still_saves_a_final_checkpoint_and_a_summary(tmp_path, signum, status):
    if signal.getsignal(signum) is signal.SIG_IGN:  # the run would inherit it and keep ignoring the signal
        pytest.skip('{} is ignored in this shell'.format(signal.Signals(signum).name))
    run_path, stderr_path = tmp_path / 'runs' / 'stop', tmp_path / 'stderr.log'
    process = start(tmp_path, 'stop', stderr_path)
    try:
        # The periodic checkpoint is written at the first training tick; wait until the learners have trained
        # past it, so the final checkpoint can be told apart from it.
        started = wait_for_metrics(process, run_path, rows=2, past_first_save=True)
        actors = actor_pids(process)
        process.send_signal(signum)
        process.wait(timeout=120)
    finally:
        stop(process)
    stderr = stderr_path.read_text()
    assert started and actors, 'the run never reached its training loop:\n' + stderr[-2000:]
    assert process.returncode == 1, stderr[-2000:]  # train.py's exit code for a run that did not complete
    assert '**Status:** {}\n'.format(status) in (run_path / 'summary.md').read_text()
    final, saved = checked_checkpoint(run_path)['frames'], saved_frames(run_path)
    assert final == metric_frames(run_path)[-1] == saved[-1] and saved[0] < final  # saved after the signal
    assert not [pid for pid in actors if alive(pid)]  # no actor outlives the run


def test_a_dead_actor_fails_the_run_loudly_and_still_saves(tmp_path):
    run_path, stderr_path = tmp_path / 'runs' / 'dead', tmp_path / 'stderr.log'
    process = start(tmp_path, 'dead', stderr_path)
    try:
        started = wait_for_metrics(process, run_path, rows=1)
        actors = actor_pids(process)
        for pid in actors:
            os.kill(pid, signal.SIGKILL)
        process.wait(timeout=120)
    finally:
        stop(process)
    stderr = stderr_path.read_text()
    assert started and actors, 'the run never reached its training loop:\n' + stderr[-2000:]
    assert process.returncode == 1, stderr[-2000:]  # the watchdog's RuntimeError ends train.py
    summary = (run_path / 'summary.md').read_text()
    assert '**Status:** failed: RuntimeError' in summary and 'exited with code -9' in summary
    assert '## Failure' in summary
    checked_checkpoint(run_path)


def test_a_refused_start_leaves_no_run_directory(tmp_path):
    command = train_command(tmp_path, 'bad')
    command[command.index('--predictor_dir') + 1] = str(tmp_path / 'no_such_dir')
    refused = run(command)
    assert refused.returncode != 0 and 'no predictor checkpoint' in refused.stderr
    assert not (tmp_path / 'runs' / 'bad').exists()  # so the same --xpid can be used once the flag is fixed


def test_a_warm_start_that_does_not_fit_is_refused_before_the_run_directory_exists(tmp_path):
    swapped = tmp_path / 'swapped'
    swapped.mkdir()
    for seat, source in (('landlord', 'landlord_up'), ('landlord_up', 'landlord'), ('landlord_down', 'landlord_down')):
        shutil.copy(ADP / (source + '.ckpt'), swapped / (seat + '.ckpt'))
    refused = run(train_command(tmp_path, 'swapped', warm_start=swapped))
    assert refused.returncode != 0 and 'size mismatch' in refused.stderr
    assert not (tmp_path / 'runs' / 'swapped').exists()


def test_evaluation_fails_loudly_when_a_worker_fails(tmp_path):
    deck = [k for k in range(3, 15) for _ in range(4)] + [17] * 4 + [20, 30]
    deal = {'landlord': sorted(deck[:20]), 'landlord_up': sorted(deck[20:37]), 'landlord_down': sorted(deck[37:54]),
            'three_landlord_cards': sorted(deck[17:20])}
    with open(tmp_path / 'deals.pkl', 'wb') as f:
        pickle.dump([deal], f)
    missing = str(tmp_path / 'missing.ckpt')
    with pytest.raises(RuntimeError):  # used to hang forever waiting for the dead worker's result
        simulation.evaluate(missing, missing, missing, str(tmp_path / 'deals.pkl'), 1)
