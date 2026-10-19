"""Run directory of one training run: config.json, attempts.jsonl, metrics.csv, summary.md, checkpoints/, traces/.

Everything needed to reproduce and audit a run lives here. config.json is written once, at the
first start, and holds the flags, the arm, the seed, the exact checkpoints used (with sha256),
the git state and the machine. Every start, including each resume, appends a line to
attempts.jsonl. Neither file records environment variables, which can hold credentials.
"""
import csv
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from douzero.dmc.models import file_sha256

REPO = Path(__file__).resolve().parents[2]


def checkpoint_path(run_path):
    return str(Path(run_path).resolve() / 'checkpoints' / 'model.tar')


def check_run_path(path, resume=False):
    """Refuses a run folder that cannot be used, without creating anything.

    A new run needs an empty or missing folder; a resume needs the folder of a run that was started.
    """
    path = Path(path).resolve()
    if resume:
        if not (path / 'config.json').is_file():
            raise FileNotFoundError('cannot resume: {} has no config.json'.format(path))
    elif path.exists() and any(path.iterdir()):
        if (path / 'checkpoints' / 'model.tar').is_file():
            raise FileExistsError('{} already holds a run; choose another --xpid, or resume it with --load_model'
                                  .format(path))
        raise FileExistsError('{} holds a run that stopped before its first checkpoint, so it cannot be resumed; '
                              'choose another --xpid, or delete the folder once you have read its files'.format(path))


class RunDir:
    def __init__(self, path, resume=False):
        self.path = Path(path).resolve()
        check_run_path(self.path, resume)
        for sub in ('checkpoints', 'traces'):
            (self.path / sub).mkdir(parents=True, exist_ok=True)
        self._metrics_fields = None

    @property
    def checkpoint_path(self):
        return checkpoint_path(self.path)

    def weights_path(self, position, frames):
        """A plain per-seat state dict, readable by DouZero's evaluate.py (baseline runs only)."""
        return str(self.path / 'checkpoints' / '{}_weights_{}.ckpt'.format(position, frames))

    def weights_tar_path(self, frames):
        """All seats' weights at one point, with the run metadata, readable by load_arm_checkpoint."""
        return str(self.path / 'checkpoints' / 'weights_{}.tar'.format(frames))

    def write_config(self, config):
        with open(self.path / 'config.json', 'x') as f:  # 'x': never overwrite another run's record
            json.dump(config, f, indent=2, sort_keys=True)

    def read_config(self):
        with open(self.path / 'config.json') as f:
            return json.load(f)

    def record_attempt(self, record):
        with open(self.path / 'attempts.jsonl', 'a') as f:
            f.write(json.dumps(record, sort_keys=True) + '\n')

    def attempts(self):
        path = self.path / 'attempts.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def log_metrics(self, row):
        """Appends one row to metrics.csv. The first row fixes the columns; later rows must match them."""
        path = self.path / 'metrics.csv'
        if self._metrics_fields is None:
            if path.exists():  # a resumed run continues the file
                with open(path, newline='') as f:
                    self._metrics_fields = next(csv.reader(f))
            else:
                self._metrics_fields = list(row)
                with open(path, 'w', newline='') as f:
                    csv.writer(f, lineterminator='\n').writerow(self._metrics_fields)
        if list(row) != self._metrics_fields:
            raise ValueError('metrics row has columns {}, expected {}'.format(list(row), self._metrics_fields))
        with open(path, 'a', newline='') as f:
            csv.writer(f, lineterminator='\n').writerow([row[k] for k in self._metrics_fields])

    def truncate_metrics(self, frames):
        """Drops the rows logged after the checkpoint a resumed run restarts from: that progress was lost."""
        path = self.path / 'metrics.csv'
        if not path.exists():
            return
        with open(path, newline='') as f:
            rows = list(csv.reader(f))
        kept = rows[:1] + [row for row in rows[1:] if float(row[rows[0].index('frames')]) <= frames]
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'w', newline='') as f:
            csv.writer(f, lineterminator='\n').writerows(kept)
        os.replace(tmp, path)

    def write_summary(self, text):
        (self.path / 'summary.md').write_text(text)


def fresh_seed():
    """A random seed for a run started without --seed. It is recorded, so a rerun can use it.

    The seed fixes initialization and each actor's random stream, not the whole run: actors and
    learners are asynchronous, so the learning trajectory still varies between reruns.
    """
    return int.from_bytes(os.urandom(4), 'little') % (2 ** 31)


def git_info(path=REPO):
    """The code's git state: commit, branch (None when detached), and whether anything differs from the commit.

    Only the code folder (path, DouZero/) counts: the project repository also holds notes, docs and
    the proposal, which do not change what is trained. untracked maps every untracked, non-ignored
    file under path to its sha256, and code_sha256 hashes the tracked diff together with those files,
    so a run from a dirty tree still pins its exact code. dirty is None when the state cannot be read
    (no git, or not a repository).
    """
    def git(*args):
        return subprocess.run(['git', '-C', str(path)] + list(args), capture_output=True, text=True,
                              check=True).stdout
    try:
        status = git('status', '--porcelain', '--', '.').splitlines()
        branch = git('rev-parse', '--abbrev-ref', 'HEAD').strip()
        diff = git('diff', 'HEAD', '--', '.')
        untracked = sorted(p for p in git('ls-files', '--others', '--exclude-standard', '-z').split('\0') if p)
        untracked_sha256 = {p: file_sha256(os.path.join(str(path), p)) for p in untracked}
        code = hashlib.sha256((diff + json.dumps(untracked_sha256, sort_keys=True)).encode()).hexdigest()
        return dict(commit=git('rev-parse', 'HEAD').strip(), branch=None if branch == 'HEAD' else branch,
                    dirty=bool(status), untracked=untracked_sha256,
                    diff_sha256=hashlib.sha256(diff.encode()).hexdigest(), code_sha256=code)
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, branch=None, dirty=None, untracked={}, diff_sha256=None, code_sha256=None)


def check_git(git, allow_dirty):
    """Refuses to train unless the code is a clean commit, or --allow_dirty accepts a recorded dirty tree."""
    if allow_dirty:
        return
    if git['dirty'] is None:
        raise RuntimeError('cannot read the git state of the code; pass --allow_dirty to train anyway')
    if git['dirty']:
        raise RuntimeError('the git tree has uncommitted changes; commit them, or pass --allow_dirty')


def machine_info():
    return dict(platform=platform.platform(), machine=platform.machine(), cpus=os.cpu_count(),
                python=sys.version.split()[0], torch=str(torch.__version__), numpy=np.__version__,
                cuda=torch.cuda.is_available(),
                gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])


def now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def run_config(flags, metadata):
    """config.json: everything needed to reproduce the run, and nothing secret."""
    return dict(started=now(), flags=vars(flags), metadata=metadata, machine=machine_info())


def summary_markdown(config, attempts, status, elapsed, frames, updates, stats, checkpoint_path, failure=None):
    """summary.md: a human-readable account of one run."""
    meta, flags = config['metadata'], config['flags']
    gits = [attempt['git'] for attempt in attempts if 'git' in attempt] or [meta['git']]  # every attempt's code
    start = meta['warm_start']
    predictor = meta['predictor']
    warnings = []
    if status != 'completed':
        warnings.append('the run did not complete ({})'.format(status))
    if any(git['dirty'] is None for git in gits):
        warnings.append('the git state of the code could not be read')
    dirty = sorted({git['code_sha256'] for git in gits if git['dirty']})
    if dirty:
        warnings.append('trained from a git tree with uncommitted changes (code sha256 {})'.format(', '.join(dirty)))
    if len({(git['commit'], git['code_sha256']) for git in gits}) > 1:
        warnings.append('the attempts ran different code; see the git record of each in attempts.jsonl')
    if predictor and not all(f['provenance'].get('trained') for f in predictor['files'].values()):
        warnings.append('UNTRAINED predictors: beliefs are noise; a smoke run, not an experiment')
    lines = ['# Run {}'.format(flags['xpid']), '',
             '- **Status:** {}'.format(status),
             '- **Arm:** {} (`--hand_model {}`), belief input {}'.format(meta['arm'], meta['hand_model'],
                                                                        meta['belief_dim'] or 'none'),
             '- **Seed:** {} (it seeds initialization and each actor\'s stream of deals and exploration; '
             'asynchronous actors and learners still make the learning trajectory vary)'.format(meta['seed']),
             '- **Attempts:** {} (first start plus resumes; see attempts.jsonl)'.format(len(attempts)),
             '- **Starting checkpoint:** {}'.format(start['dir'] if start else 'random initialization'),
             '- **Predictor:** {}'.format('none' if predictor is None else '{}, K={}, frozen'.format(
                 predictor['dir'], predictor['num_targets'])),
             '- **Training:** {} frames; learner updates landlord / landlord_up / landlord_down = {} / {} / {}; '
             'training wall-clock {:.1f} s'.format(frames, updates['landlord'], updates['landlord_up'],
                                                    updates['landlord_down'], elapsed),
             '- **Final learner stats** (last update of each seat; nan = no update yet):', '',
             '| seat | loss | Q mean | Q std | grad norm |', '|---|---|---|---|---|']
    for position in ('landlord', 'landlord_up', 'landlord_down'):
        lines.append('| {} | {} |'.format(position, ' | '.join(
            '{:.4g}'.format(stats.get('{}_{}'.format(name, position), float('nan')))
            for name in ('loss', 'q_mean', 'q_std', 'grad_norm'))))
    lines += ['',
              '- **Validation CE / ADP / WP:** not computed in this run.',
              '- **Checkpoints:** `{}`, plus `weights_<frames>.tar` per save beside it.'.format(checkpoint_path),
              '- **Warnings:** {}'.format('; '.join(warnings) if warnings else 'none'),
              '- **Machine:** {} | {} CPUs | Python {} | torch {} | GPUs: {}'.format(
                  config['machine']['platform'], config['machine']['cpus'], config['machine']['python'],
                  config['machine']['torch'], ', '.join(config['machine']['gpus']) or 'none'), '']
    if failure:
        lines += ['## Failure', '', '```', failure.strip(), '```', '']
    return '\n'.join(lines)
