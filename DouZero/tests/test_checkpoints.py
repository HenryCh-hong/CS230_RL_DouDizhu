"""Stage 3: checkpoints, run directories and strict loading.

T18  checkpoint round trip for every arm (weights, optimizer state, predictors, metadata)
T19  a missing or wrong predictor, or a partial or mismatched checkpoint, fails loudly
Also: actors start from exactly the learner's warm-start weights, flags that contradict each
other are rejected, run directories are never silently reused, and run metadata holds no
environment variables.
"""
import copy
import json
import os
import multiprocessing as mp
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from douzero.dmc import belief, dmc, file_writer, models, run_dir
from douzero.dmc.arguments import parser
from douzero.evaluation.belief_agent import BeliefDeepAgent
from douzero.evaluation.deep_agent import DeepAgent
from douzero.evaluation.oracle_agent import OracleAgent
from helpers import POSITIONS, arm_flags, save_predictors, saved_run

ARMS = ('baseline', 'none', 'next', 'derived_both', 'learned_both', 'oracle')
ADP_DIR = Path(__file__).resolve().parents[1] / 'baselines' / 'douzero_ADP'


# ---------------------------------------------------------------- T18: round trip

@pytest.mark.parametrize('hand_model', ARMS)
def test_checkpoint_round_trip_for_every_arm(hand_model, tmp_path):
    path, flags, predictors, metadata, learner, optimizers = saved_run(hand_model, tmp_path)
    assert not os.path.exists(path + '.tmp')  # written atomically
    arm = models.load_arm_checkpoint(path)
    assert arm['hand_model'] == hand_model and arm['metadata']['seed'] == 3
    assert arm['metadata']['arm'] == belief.ARM_NAMES[hand_model]
    num_targets = belief.PREDICTOR_TARGETS.get(hand_model, 0)
    for position in POSITIONS:
        saved, loaded = learner.get_model(position).state_dict(), arm['q_networks'][position].state_dict()
        assert saved.keys() == loaded.keys() and all(torch.equal(saved[k], loaded[k]) for k in saved)
        predictor = arm['predictors'][position]
        if num_targets:
            assert (predictor.position, predictor.num_targets) == (position, num_targets)
            assert not predictor.training and not any(p.requires_grad for p in predictor.parameters())
            original = predictors[position].state_dict()
            assert all(torch.equal(original[k], v) for k, v in predictor.state_dict().items())
        else:
            assert predictor is None
    record = arm['metadata']['predictor']
    if num_targets:
        assert record['frozen'] is True and record['num_targets'] == num_targets
        for position in POSITIONS:
            assert record['files'][position]['sha256'] == models.file_sha256(
                os.path.join(flags.predictor_dir, position + '.ckpt'))
    else:
        assert record is None
    states = dmc.load_checkpoint_states(path, metadata, flags)  # a resume restores the optimizer state too
    _, _, resumed = dmc.initial_models(flags, ['cpu'], states)
    for position in POSITIONS:
        before, after = optimizers[position].state_dict()['state'], resumed[position].state_dict()['state']
        assert before.keys() == after.keys()
        assert all(torch.equal(before[i][name], after[i][name]) for i in before for name in before[i])
    assert states['frames'] == 3200 and states['updates'] == {p: 1 for p in POSITIONS}


def test_actors_start_from_exactly_the_learners_warm_start_weights(tmp_path):
    if not (ADP_DIR / 'landlord.ckpt').exists():
        pytest.skip('DouZero-ADP checkpoints not in baselines/douzero_ADP')
    for hand_model in ('baseline', 'derived_both'):
        flags = arm_flags(hand_model, tmp_path, '--warm_start_dir', str(ADP_DIR))
        actors, learner, _ = dmc.initial_models(flags, ['cpu'])
        for position in POSITIONS:
            adp = torch.load(ADP_DIR / (position + '.ckpt'), map_location='cpu', weights_only=True)
            actor, ours = actors['cpu'].get_model(position).state_dict(), learner.get_model(position).state_dict()
            assert all(torch.equal(actor[k], ours[k]) for k in ours)
            for key, value in adp.items():
                assert torch.equal(ours[key][:, :value.shape[1]] if key == 'dense1.weight' else ours[key], value)
            if hand_model != 'baseline':
                assert torch.equal(ours['dense1.weight'][:, -belief.BELIEF_DIM:], torch.zeros(512, belief.BELIEF_DIM))


def test_belief_agent_reads_the_arm_from_the_checkpoint(tmp_path, random_play_states):
    torch.manual_seed(5)
    state = next(s for s in random_play_states if s['infoset'].player_position == 'landlord_down'
                 and len(s['infoset'].legal_actions) > 1)
    beliefs = {}
    for hand_model in ('next', 'derived_both'):  # both runs save the same seeded predictor files
        path = saved_run(hand_model, tmp_path / hand_model)[0]
        agent = BeliefDeepAgent('landlord_down', path)
        assert agent.hand_model == hand_model
        beliefs[hand_model] = agent.belief(state['infoset'])
        assert agent.values(state['infoset']).shape == (len(state['infoset'].legal_actions),)
        assert agent.act(state['infoset']) in state['infoset'].legal_actions
    assert torch.equal(beliefs['next'][:, :75], beliefs['derived_both'][:, :75])  # same next-hand belief
    assert torch.equal(beliefs['next'][:, 75:], torch.zeros(1, 75))  # M1 leaves the second slot empty
    assert beliefs['derived_both'][:, 75:].sum() == pytest.approx(15.0)  # M2c fills it by reflection


# ---------------------------------------------------------------- T19: loud failures

def test_predictor_loading_fails_loudly(tmp_path):
    k1, k2 = save_predictors(tmp_path / 'k1', 1), save_predictors(tmp_path / 'k2', 2)
    untrained = save_predictors(tmp_path / 'untrained', 1, trained=False)
    with pytest.raises(FileNotFoundError):
        models.load_predictors('next', str(tmp_path / 'missing'))
    for hand_model, directory in (('next', k2), ('derived_both', k2), ('learned_both', k1),  # wrong K
                                  ('none', k1), ('oracle', k1), ('baseline', k1),           # no predictor allowed
                                  ('next', None), ('next', untrained)):                     # none given; untrained
        with pytest.raises(ValueError):
            models.load_predictors(hand_model, directory)
    assert models.load_predictors('next', untrained, require_trained=False)['landlord'].num_targets == 1
    wrong_seat = tmp_path / 'wrong_seat'
    shutil.copytree(k1, wrong_seat)
    shutil.copy(wrong_seat / 'landlord.ckpt', wrong_seat / 'landlord_up.ckpt')
    with pytest.raises(ValueError):
        models.load_predictors('next', str(wrong_seat))


def test_checkpoint_loading_fails_loudly(tmp_path):
    path, flags, _, metadata, _, _ = saved_run('derived_both', tmp_path)
    with pytest.raises(FileNotFoundError):
        models.load_arm_checkpoint(str(tmp_path / 'missing.tar'))
    plain = tmp_path / 'plain.ckpt'
    torch.save(models.model_dict['landlord']().state_dict(), plain)
    with pytest.raises(ValueError):  # a DouZero weights file has no run metadata
        models.load_arm_checkpoint(str(plain))

    checkpoint = torch.load(path, weights_only=True)
    partial = dict(checkpoint, model_state_dict=dict(checkpoint['model_state_dict']))
    partial['model_state_dict']['landlord'] = {k: v for k, v in checkpoint['model_state_dict']['landlord'].items()
                                               if k != 'dense6.bias'}
    torch.save(partial, tmp_path / 'partial.tar')
    with pytest.raises(RuntimeError):
        models.load_arm_checkpoint(str(tmp_path / 'partial.tar'))
    mismatched = dict(checkpoint, cs230=dict(checkpoint['cs230'], belief_dim=0))
    torch.save(mismatched, tmp_path / 'mismatched.tar')
    with pytest.raises(RuntimeError):
        models.load_arm_checkpoint(str(tmp_path / 'mismatched.tar'))

    models.save_hand_predictor(models.HandPredictor('landlord', 1),
                               os.path.join(flags.predictor_dir, 'landlord.ckpt'), provenance=dict(trained=True))
    with pytest.raises(ValueError):  # the predictor file changed after training (sha256)
        models.load_arm_checkpoint(path)

    with pytest.raises(ValueError):
        dmc.load_checkpoint_states(path, dict(metadata, hand_model='next'), flags)  # resuming as another arm
    with pytest.raises(FileNotFoundError):
        dmc.load_checkpoint_states(str(tmp_path / 'missing.tar'), metadata, flags)


def test_resume_refuses_changed_training_flags_but_allows_a_longer_run(tmp_path):
    path, flags, _, metadata, _, _ = saved_run('next', tmp_path)
    longer = copy.copy(flags)
    longer.total_frames, longer.save_interval = flags.total_frames * 2, 5
    assert dmc.load_checkpoint_states(path, metadata, longer)['frames'] == 3200
    for key, value in (('learning_rate', 1e-3), ('unroll_length', 50), ('batch_size', 16), ('objective', 'wp'),
                       ('exp_epsilon', 0.0), ('seed', 99)):
        changed = copy.copy(flags)
        setattr(changed, key, value)
        with pytest.raises(ValueError, match=key):
            dmc.load_checkpoint_states(path, metadata, changed)


def test_resume_matches_predictors_by_hash_not_by_path(tmp_path):
    path, flags, _, metadata, _, _ = saved_run('next', tmp_path)
    moved = tmp_path / 'moved_predictors'
    shutil.copytree(flags.predictor_dir, moved)
    moved_flags = copy.copy(flags)
    moved_flags.predictor_dir = str(moved)
    moved_metadata = dmc.run_metadata(moved_flags, models.load_predictors('next', str(moved)))
    assert dmc.load_checkpoint_states(path, moved_metadata, moved_flags)['frames'] == 3200


def test_evaluation_refuses_untrained_predictors_unless_allowed(tmp_path):
    os.makedirs(tmp_path, exist_ok=True)
    flags = parser.parse_args(['--hand_model', 'next', '--actor_device_cpu', '--training_device', 'cpu', '--seed', '1',
                               '--predictor_dir', save_predictors(tmp_path / 'untrained', 1, trained=False)])
    predictors = models.load_predictors('next', flags.predictor_dir, require_trained=False)
    _, learner, optimizers = dmc.initial_models(flags, ['cpu'])
    path = str(tmp_path / 'smoke.tar')
    dmc.save_checkpoint(path, learner, optimizers, flags, dmc.run_metadata(flags, predictors), [],
                        stats={}, frames=0, position_frames={}, updates={})
    with pytest.raises(ValueError):
        models.load_arm_checkpoint(path)
    with pytest.raises(ValueError):
        BeliefDeepAgent('landlord', path)
    assert models.load_arm_checkpoint(path, allow_untrained=True)['hand_model'] == 'next'


def test_intermediate_weights_files_carry_the_arm_and_reload_strictly(tmp_path):
    path, flags, _, metadata, learner, _ = saved_run('derived_both', tmp_path)
    weights = str(tmp_path / 'weights_3200.tar')
    dmc.save_weights(weights, torch.load(path, weights_only=True), metadata)
    arm = models.load_arm_checkpoint(weights)
    assert arm['hand_model'] == 'derived_both'
    loaded = arm['q_networks']['landlord_up'].state_dict()
    assert all(torch.equal(loaded[k], v) for k, v in learner.get_model('landlord_up').state_dict().items())


def test_douzero_deep_agent_refuses_a_run_checkpoint_and_partial_weights(tmp_path):
    path = saved_run('baseline', tmp_path)[0]
    with pytest.raises(ValueError):  # a model.tar used to load as a random network without any error
        DeepAgent('landlord', path)
    partial = {k: v for k, v in models.model_dict['landlord']().state_dict().items() if k != 'dense6.bias'}
    torch.save(partial, tmp_path / 'partial.ckpt')
    with pytest.raises(RuntimeError):
        DeepAgent('landlord', str(tmp_path / 'partial.ckpt'))
    if (ADP_DIR / 'landlord.ckpt').exists():
        DeepAgent('landlord', str(ADP_DIR / 'landlord.ckpt'))  # the released weights still load


def test_agents_refuse_the_other_kind_of_checkpoint(tmp_path):
    oracle_path = saved_run('oracle', tmp_path / 'oracle')[0]
    m1_path = saved_run('next', tmp_path / 'm1')[0]
    with pytest.raises(ValueError):
        BeliefDeepAgent('landlord', oracle_path)
    with pytest.raises(ValueError):
        OracleAgent('landlord', m1_path)
    assert OracleAgent('landlord', oracle_path).hand_model == 'oracle'


def test_contradictory_flags_are_rejected(tmp_path):
    k1 = save_predictors(tmp_path / 'k1', 1)
    base = ['--actor_device_cpu', '--training_device', 'cpu']
    for args in (['--hand_model', 'next', '--predictor_dir', k1, '--no-freeze_predictor'],
                 ['--hand_model', 'none', '--predictor_dir', k1],
                 ['--hand_model', 'next'],
                 ['--hand_model', 'none', '--load_model', '--warm_start_dir', str(ADP_DIR)],
                 ['--hand_model', 'none', '--num_buffers', '4', '--batch_size', '8']):
        with pytest.raises((ValueError, NotImplementedError)):
            dmc.check_flags(parser.parse_args(base + args))


# ---------------------------------------------------------------- run directories and metadata

def test_run_directory_is_never_silently_reused(tmp_path):
    run = run_dir.RunDir(tmp_path / 'run')
    run.write_config({'arm': 'M0c'})
    with pytest.raises(FileExistsError):
        run.write_config({'arm': 'M1'})
    with pytest.raises(FileExistsError):
        run_dir.RunDir(tmp_path / 'run')
    assert run_dir.RunDir(tmp_path / 'run', resume=True).read_config() == {'arm': 'M0c'}
    with pytest.raises(FileNotFoundError):
        run_dir.RunDir(tmp_path / 'never_started', resume=True)
    assert os.path.isdir(run.path / 'checkpoints') and os.path.isdir(run.path / 'traces')


def test_a_run_folder_without_a_checkpoint_says_it_cannot_be_resumed(tmp_path):
    run_dir.RunDir(tmp_path / 'run').write_config({'arm': 'M0c'})  # a start that failed before its first save
    with pytest.raises(FileExistsError, match='before its first checkpoint'):
        run_dir.RunDir(tmp_path / 'run')
    (tmp_path / 'run' / 'checkpoints' / 'model.tar').write_bytes(b'')
    with pytest.raises(FileExistsError, match='--load_model'):
        run_dir.RunDir(tmp_path / 'run')


def test_periodic_checkpoints_read_the_live_frame_counter():
    # train()'s checkpoint() must take frames from train()'s own counter, read under the checkpoint locks.
    # A parameter named frames would shadow it with a value read before the locks, while a learner can still
    # finish an update that the saved weights then contain but the saved frame count does not.
    nested = {code.co_name: code for code in dmc.train.__code__.co_consts if hasattr(code, 'co_name')}
    assert 'frames' in nested['checkpoint'].co_freevars and 'frames' not in nested['checkpoint'].co_varnames


def test_stop_handlers_leave_an_ignored_signal_ignored():
    # nohup starts a run with SIGHUP ignored so that it survives a closed terminal; train() must keep that.
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        previous = dmc.install_stop_handlers()
        assert signal.getsignal(signal.SIGTERM) is dmc._raise_terminated
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN and signal.SIGHUP not in previous
        for signum, handler in previous.items():  # what train() does on exit
            signal.signal(signum, handler)
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
        signal.signal(signal.SIGHUP, signal.SIG_DFL)
        dmc.install_stop_handlers()
        assert signal.getsignal(signal.SIGHUP) is dmc._raise_terminated
    finally:
        for signum, handler in saved.items():
            signal.signal(signum, handler)


def test_stop_handlers_can_restore_a_handler_python_did_not_install(monkeypatch):
    # signal.signal returns None for a handler installed outside Python, and signal.signal(s, None) raises.
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    real_signal = signal.signal
    try:
        monkeypatch.setattr(signal, 'signal', lambda signum, handler: (real_signal(signum, handler), None)[1])
        previous = dmc.install_stop_handlers()
        assert previous and all(handler is signal.SIG_DFL for handler in previous.values())
    finally:
        monkeypatch.undo()
        for signum, handler in saved.items():
            signal.signal(signum, handler)


def test_the_summary_warns_about_every_attempt_not_only_the_latest(tmp_path):
    flags = arm_flags('none', tmp_path)
    config = run_dir.run_config(flags, dmc.run_metadata(flags, models.load_predictors('none', None)))
    clean = dict(commit='a' * 40, branch='main', dirty=False, untracked={}, diff_sha256='d0', code_sha256='c0')
    dirty = dict(clean, dirty=True, code_sha256='c1')
    stats, updates = {}, {p: 0 for p in POSITIONS}

    def warnings(*gits):
        text = run_dir.summary_markdown(config, [dict(git=g) for g in gits], 'completed', 1.0, 0, updates, stats, 'x')
        return next(line for line in text.splitlines() if line.startswith('- **Warnings:**'))

    assert warnings(clean, clean) == '- **Warnings:** none'
    assert 'uncommitted changes' in warnings(dirty, clean)  # the dirty attempt was the first one
    assert 'different code' in warnings(dirty, clean) and 'different code' in warnings(clean, dict(clean, commit='b' * 40))
    assert 'could not be read' in warnings(dict(clean, dirty=None, commit=None, code_sha256=None), clean)


def test_a_resumed_run_drops_metrics_from_lost_progress_and_records_each_attempt(tmp_path):
    run = run_dir.RunDir(tmp_path / 'run')
    run.write_config({'arm': 'M1'})
    run.record_attempt({'seed': 0})
    for frames in (0, 3200, 6400, 9600):
        run.log_metrics({'frames': frames, 'loss_landlord': 1.0})
    resumed = run_dir.RunDir(tmp_path / 'run', resume=True)
    resumed.truncate_metrics(6400)  # the checkpoint holds 6,400 frames; the row at 9,600 was never saved
    resumed.record_attempt({'seed': 0, 'resumed_from_frames': 6400})
    resumed.log_metrics({'frames': 9600, 'loss_landlord': 0.5})
    frames = [int(line.split(',')[0]) for line in (run.path / 'metrics.csv').read_text().splitlines()[1:]]
    assert frames == [0, 3200, 6400, 9600]
    assert [a['seed'] for a in resumed.attempts()] == [0, 0] and resumed.attempts()[1]['resumed_from_frames'] == 6400


def test_unupdated_seats_are_logged_as_missing_not_zero():
    stats = {k: float('nan') for k in dmc.stat_keys()}
    stats['loss_landlord'] = 0.25
    row = dmc.metrics_row(5.0, 3200, 640.0, {p: 0 for p in POSITIONS}, stats)
    assert row['loss_landlord'] == 0.25 and row['loss_landlord_up'] != row['loss_landlord_up']  # NaN


def test_the_watchdog_reports_dead_actors_and_failed_learners():
    alive = SimpleNamespace(is_alive=lambda: True, pid=1, exitcode=None)
    dead = SimpleNamespace(is_alive=lambda: False, pid=2, exitcode=-9)
    learner = SimpleNamespace(is_alive=lambda: True, name='learner-cpu-landlord-0')
    stopped = SimpleNamespace(is_alive=lambda: False, name='learner-cpu-landlord_up-0')
    dmc.check_workers([alive], [], [learner])
    with pytest.raises(RuntimeError, match='exit'):
        dmc.check_workers([alive, dead], [], [learner])
    with pytest.raises(RuntimeError, match='boom'):
        dmc.check_workers([alive], ['learner-cpu-landlord-0: boom'], [learner])
    with pytest.raises(RuntimeError, match='landlord_up'):
        dmc.check_workers([alive], [], [learner, stopped])
    # A learner leaves its loop once frames reach --total_frames; that is the end of training, not a failure.
    dmc.check_workers([alive], [], [learner, stopped], training_over=lambda: True)


def test_waking_the_learners_cannot_hang_on_a_queue_lock_a_dead_actor_held():
    blocked, free = mp.get_context('spawn').SimpleQueue(), mp.get_context('spawn').SimpleQueue()

    def learner(queue):  # what a learner waiting in get_batch does with a None
        while queue.get() is not None:
            pass

    threads = [threading.Thread(target=learner, args=(queue,), name=name, daemon=True)
               for queue, name in ((blocked, 'learner-blocked'), (free, 'learner-free'))]
    for thread in threads:
        thread.start()
    # _wlock is SimpleQueue's write lock (a CPython detail); holding it is what a dead actor would leave behind.
    blocked._wlock.acquire()  # as if an actor had been SIGKILLed in the middle of a put on this queue
    try:
        stuck = within(10, dmc.wake_learners, [blocked, free], 1, threads, timeout=1)
    finally:
        blocked._wlock.release()  # lets the abandoned sentinel through, so the test leaves no thread behind
    assert stuck == ['learner-blocked']
    threads[0].join(5)
    assert not threads[0].is_alive()


def test_the_final_checkpoint_gives_up_on_a_lock_a_stuck_learner_holds(tmp_path):
    _, flags, _, metadata, learner, optimizers = saved_run('none', tmp_path)
    free, stuck = threading.Lock(), threading.Lock()
    stuck.acquire()
    with pytest.raises(TimeoutError):
        within(10, dmc.save_checkpoint, str(tmp_path / 'final.tar'), learner, optimizers, flags, metadata,
               [free, stuck], lock_timeout=0.1, stats={}, frames=0, position_frames={}, updates={})
    assert not os.path.exists(tmp_path / 'final.tar')
    assert free.acquire(blocking=False)  # the lock it had already taken was released


def within(seconds, function, *args, **kwargs):
    """function(*args, **kwargs), failing the test instead of hanging the suite if it has not returned in time."""
    outcome = {}

    def call():
        try:
            outcome['value'] = function(*args, **kwargs)
        except BaseException as error:  # handed to the test below
            outcome['error'] = error

    caller = threading.Thread(target=call, daemon=True)
    caller.start()
    caller.join(seconds)
    assert not caller.is_alive(), '{} still blocked after {} s'.format(function.__name__, seconds)
    if 'error' in outcome:
        raise outcome['error']
    return outcome['value']


def _git(repo, *args):
    subprocess.run(['git', '-C', str(repo)] + list(args), check=True, capture_output=True)


def test_git_provenance_pins_untracked_code_and_fails_closed(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q')
    (repo / 'tracked.py').write_text('x = 1\n')
    _git(repo, 'add', 'tracked.py')
    _git(repo, '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'commit', '-q', '-m', 'init')
    clean = run_dir.git_info(repo)
    assert clean['dirty'] is False and clean['untracked'] == {}
    run_dir.check_git(clean, allow_dirty=False)
    (repo / 'new_module.py').write_text('y = 2\n')
    first = run_dir.git_info(repo)
    (repo / 'new_module.py').write_text('y = 3\n')
    second = run_dir.git_info(repo)
    assert first['dirty'] and list(first['untracked']) == ['new_module.py']
    assert first['code_sha256'] != second['code_sha256']  # the untracked file's content is pinned
    with pytest.raises(RuntimeError):
        run_dir.check_git(first, allow_dirty=False)
    run_dir.check_git(first, allow_dirty=True)
    unknown = run_dir.git_info(tmp_path / 'not_a_repo')
    assert unknown['dirty'] is None
    with pytest.raises(RuntimeError):  # an unknown state is not a clean one
        run_dir.check_git(unknown, allow_dirty=False)


def test_git_provenance_covers_only_the_code_folder(tmp_path):
    # DouZero/ lives inside the project repository, next to notes and docs that do not change what is trained.
    repo = tmp_path / 'project'
    (repo / 'code').mkdir(parents=True)
    _git(repo, 'init', '-q')
    (repo / 'code' / 'train.py').write_text('x = 1\n')
    (repo / 'notes.md').write_text('notes\n')
    _git(repo, 'add', '.')
    _git(repo, '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'commit', '-q', '-m', 'init')
    (repo / 'notes.md').write_text('more notes\n')
    (repo / 'draft.md').write_text('draft\n')
    clean = run_dir.git_info(repo / 'code')
    assert clean['dirty'] is False and clean['untracked'] == {}
    (repo / 'code' / 'train.py').write_text('x = 2\n')
    assert run_dir.git_info(repo / 'code')['dirty'] is True


def test_metrics_file_keeps_one_schema(tmp_path):
    run = run_dir.RunDir(tmp_path / 'run')
    run.log_metrics({'frames': 0, 'loss_landlord': 1.0})
    run.log_metrics({'frames': 3200, 'loss_landlord': 0.5})
    with pytest.raises(ValueError):
        run.log_metrics({'frames': 6400})
    lines = (run.path / 'metrics.csv').read_text().splitlines()
    assert lines == ['frames,loss_landlord', '0,1.0', '3200,0.5']


def test_run_metadata_holds_no_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv('CS230_TEST_SECRET', 'do-not-record')
    flags = arm_flags('derived_both', tmp_path)
    config = run_dir.run_config(flags, dmc.run_metadata(flags, models.load_predictors('derived_both',
                                                                                       flags.predictor_dir)))
    assert 'do-not-record' not in json.dumps(config)
    assert config['metadata']['git']['commit'] and 'dirty' in config['metadata']['git']
    writer = file_writer.FileWriter(xpid='meta_test', xp_args={}, rootdir=str(tmp_path))
    meta = json.loads((tmp_path / 'meta_test' / 'meta.json').read_text())
    assert 'do-not-record' not in json.dumps(meta)
    assert set(meta['env']) <= {'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS'}
    writer.close()


# ---------------------------------------------------------------- code identity, unsafe override and forks

def tree(path):
    """Every file (with its sha256) and folder under path: to check that nothing in it changed, not even a new folder."""
    return {str(p.relative_to(path)): models.file_sha256(p) if p.is_file() else 'folder'
            for p in sorted(Path(path).rglob('*'))}


def other_code(metadata, **changes):
    """The run metadata as if the code had been different: changes replace fields of its git record."""
    return dict(metadata, git=dict(metadata['git'], **changes))


def parent_run(tmp_path, hand_model='next'):
    """A saved run laid out as a run folder: config.json, checkpoints/model.tar and a weights_3200.tar."""
    path, _, _, metadata, _, _ = saved_run(hand_model, tmp_path / 'scratch')
    run = run_dir.RunDir(tmp_path / 'runs' / 'parent')
    run.write_config({'arm': metadata['arm']})
    metadata = dict(metadata, run=dict(xpid='parent', run_dir=str(run.path)))  # as train() records it
    states = dmc.read_checkpoint(path)
    torch.save(dict(states, cs230=metadata), run.checkpoint_path)
    dmc.save_weights(str(run.path / 'checkpoints' / 'weights_3200.tar'), states, metadata)
    return run.path


def fork_flags(hand_model, tmp_path, parent_checkpoint, *extra):
    return arm_flags(hand_model, tmp_path, '--fork_from_checkpoint', str(parent_checkpoint),
                     '--fork_reason', '  probe the belief  ', *extra)


def test_resume_refuses_a_code_change_and_points_to_the_fork(tmp_path):
    path, flags, _, metadata, _, _ = saved_run('next', tmp_path)
    assert dmc.check_resume(dmc.read_checkpoint(path), metadata, flags, path) is False  # same code: no override used
    for changes, field in ((dict(commit='b' * 40), 'commit'), (dict(code_sha256='f' * 64), 'code_sha256')):
        with pytest.raises(ValueError, match='--fork_from_checkpoint') as error:
            dmc.load_checkpoint_states(path, other_code(metadata, **changes), flags)
        assert field in str(error.value)  # the message says what differs


def test_resume_refuses_an_unknown_git_state_on_either_side(tmp_path):
    path, flags, _, metadata, _, _ = saved_run('next', tmp_path)
    unknown = dict(commit=None, branch=None, dirty=None, untracked={}, diff_sha256=None, code_sha256=None)
    with pytest.raises(ValueError, match='unknown'):
        dmc.check_resume(dmc.read_checkpoint(path), dict(metadata, git=unknown), flags, path)
    states = dmc.read_checkpoint(path)
    states['cs230'] = dict(states['cs230'], git=unknown)
    with pytest.raises(ValueError, match='unknown'):
        dmc.check_resume(states, metadata, flags, path)


def test_unsafe_override_resumes_across_a_code_change_but_keeps_the_other_checks(tmp_path):
    path, flags, _, metadata, _, _ = saved_run('next', tmp_path)
    states = dmc.read_checkpoint(path)
    flags.unsafe_resume_across_code_change = True
    assert dmc.check_resume(states, other_code(metadata, code_sha256='f' * 64), flags, path) is True  # accepted, reported
    assert dmc.check_resume(states, metadata, flags, path) is False  # nothing to override: the code is the same
    changed_lr = copy.copy(flags)
    changed_lr.learning_rate = 1e-3
    with pytest.raises(ValueError, match='learning_rate'):
        dmc.check_resume(states, other_code(metadata, code_sha256='f' * 64), changed_lr, path)
    with pytest.raises(ValueError, match='unknown'):  # an unknown code identity is refused even with the override
        dmc.check_resume(states, other_code(metadata, dirty=None, commit=None), flags, path)


def test_load_arm_checkpoint_exposes_the_invalid_flag(tmp_path):
    path = saved_run('next', tmp_path)[0]
    assert models.load_arm_checkpoint(path)['metadata']['invalid_for_final_results'] is False
    checkpoint = torch.load(path, weights_only=True)
    torch.save(dict(checkpoint, cs230=dict(checkpoint['cs230'], invalid_for_final_results=True)), tmp_path / 'bad.tar')
    assert models.load_arm_checkpoint(str(tmp_path / 'bad.tar'))['metadata']['invalid_for_final_results'] is True
    older = {k: v for k, v in checkpoint['cs230'].items() if k not in ('invalid_for_final_results', 'fork')}
    torch.save(dict(checkpoint, cs230=older), tmp_path / 'older.tar')  # written before the flag existed
    assert models.load_arm_checkpoint(str(tmp_path / 'older.tar'))['metadata']['invalid_for_final_results'] is False


def test_fork_flags_are_checked(tmp_path):
    parent = str(parent_run(tmp_path) / 'checkpoints' / 'model.tar')
    base = ['--actor_device_cpu', '--training_device', 'cpu', '--hand_model', 'none']
    dmc.check_flags(parser.parse_args(base + ['--fork_from_checkpoint', parent, '--fork_reason', 'x']))
    for args in (['--fork_from_checkpoint', parent],                                  # no reason
                 ['--fork_from_checkpoint', parent, '--fork_reason', '   '],         # a blank reason
                 ['--fork_reason', 'x'],                                             # a reason without a fork
                 ['--fork_from_checkpoint', parent, '--fork_reason', 'x', '--load_model'],
                 ['--fork_from_checkpoint', parent, '--fork_reason', 'x', '--warm_start_dir', str(ADP_DIR)],
                 ['--unsafe_resume_across_code_change']):                            # only with --load_model
        with pytest.raises(ValueError):
            dmc.check_flags(parser.parse_args(base + args))
    dmc.check_flags(parser.parse_args(base + ['--load_model', '--unsafe_resume_across_code_change']))


def test_fork_record_names_the_parent_and_its_code(tmp_path):
    parent = parent_run(tmp_path)
    checkpoint = parent / 'checkpoints' / 'model.tar'
    flags = fork_flags('derived_both', tmp_path, checkpoint)
    metadata = dmc.run_metadata(flags, models.load_predictors('derived_both', flags.predictor_dir))
    states, record = dmc.read_fork_parent(flags.fork_from_checkpoint, metadata, flags)
    assert record['parent_run_id'] == dict(xpid='parent', run_dir=str(parent.resolve()))
    assert record['parent_checkpoint'] == dict(path=str(checkpoint.resolve()), sha256=models.file_sha256(checkpoint))
    assert record['parent_frames'] == 3200 and record['parent_hand_model'] == 'next'
    assert record['parent_code'] == dict(commit=metadata['git']['commit'], code_sha256=metadata['git']['code_sha256'])
    assert record['reason'] == 'probe the belief' and record['optimizer_state_restored'] is True
    assert record['parent_fork'] is None and record['actor_lineage'] == int(record['parent_checkpoint']['sha256'][:8], 16) | 1
    record = dmc.read_fork_parent(flags.fork_from_checkpoint, other_code(metadata, code_sha256='f' * 64), flags)[1]
    assert record['current_code']['code_sha256'] == 'f' * 64 and record['parent_code']['code_sha256'] != 'f' * 64
    flags.fork_from_checkpoint = str(parent / 'checkpoints' / 'weights_3200.tar')
    assert dmc.read_fork_parent(flags.fork_from_checkpoint, metadata, flags)[1]['optimizer_state_restored'] is False


def test_a_copied_parent_checkpoint_still_names_its_own_run(tmp_path):
    parent = parent_run(tmp_path)
    copy = tmp_path / 'elsewhere' / 'model.tar'
    copy.parent.mkdir()
    shutil.copy(parent / 'checkpoints' / 'model.tar', copy)
    flags = fork_flags('derived_both', tmp_path, copy)
    metadata = dmc.run_metadata(flags, models.load_predictors('derived_both', flags.predictor_dir))
    record = dmc.read_fork_parent(flags.fork_from_checkpoint, metadata, flags)[1]
    assert record['parent_run_id'] == dict(xpid='parent', run_dir=str(parent.resolve()))  # from the file, not its folder
    assert record['parent_checkpoint']['path'] == str(copy.resolve())


def test_a_fork_of_a_fork_keeps_its_ancestry_and_its_own_actor_streams(tmp_path):
    parent = parent_run(tmp_path)
    flags = fork_flags('derived_both', tmp_path, parent / 'checkpoints' / 'model.tar')
    metadata = dmc.run_metadata(flags, models.load_predictors('derived_both', flags.predictor_dir))
    first = dmc.read_fork_parent(flags.fork_from_checkpoint, metadata, flags)[1]
    states = dmc.read_checkpoint(flags.fork_from_checkpoint)
    child = tmp_path / 'child.tar'  # a checkpoint of the first fork
    torch.save(dict(states, cs230=dict(states['cs230'], fork=first, warm_start=dict(dir='/adp', files={}))), child)
    second = dmc.read_fork_parent(str(child), metadata, flags)[1]
    assert second['parent_fork'] == first and second['parent_warm_start'] == dict(dir='/adp', files={})
    assert second['actor_lineage'] != first['actor_lineage'] != 0


def test_a_fork_is_refused_inside_its_parent_folder(tmp_path):
    parent = parent_run(tmp_path)
    before = tree(parent)
    flags = fork_flags('next', tmp_path, parent / 'checkpoints' / 'model.tar', '--savedir', str(parent / 'forks'))
    flags.allow_dirty = True
    with pytest.raises(ValueError, match='inside'):
        dmc.train(flags)
    assert tree(parent) == before


def test_a_fork_needs_a_parent_with_the_same_network_shape(tmp_path):
    flags = fork_flags('derived_both', tmp_path, parent_run(tmp_path, 'baseline') / 'checkpoints' / 'model.tar')
    metadata = dmc.run_metadata(flags, models.load_predictors('derived_both', flags.predictor_dir))
    with pytest.raises(ValueError, match='belief_dim'):  # a DouZero parent has no belief input
        dmc.read_fork_parent(flags.fork_from_checkpoint, metadata, flags)
    oracle = arm_flags('oracle', tmp_path / 'oracle')
    oracle_metadata = dmc.run_metadata(oracle, {})
    oracle.fork_reason = 'compare arms'
    oracle.fork_from_checkpoint = str(parent_run(tmp_path / 'next_parent') / 'checkpoints' / 'model.tar')
    record = dmc.read_fork_parent(oracle.fork_from_checkpoint, oracle_metadata, oracle)[1]
    assert record['parent_hand_model'] == 'next'  # another arm with the same shape is allowed; the record says so


def test_a_refused_fork_writes_nothing_into_the_parent_or_the_new_folder(tmp_path):
    parent = parent_run(tmp_path)
    before = tree(parent)
    flags = fork_flags('derived_both', tmp_path, parent / 'checkpoints' / 'model.tar', '--xpid', 'child')
    flags.allow_dirty = True
    run = run_dir.RunDir(tmp_path / 'runs' / 'in_use')  # a folder in use is refused for a fork, as for any new run
    run.write_config({'arm': 'M0c'})
    in_use = tree(run.path)
    flags.xpid = 'in_use'
    with pytest.raises(FileExistsError):
        dmc.train(flags)
    flags.xpid = 'child'
    flags.hand_model, flags.predictor_dir = 'baseline', None  # no belief input: not the parent M1's network shape
    with pytest.raises(ValueError, match='belief_dim'):  # a parent with another network shape, from a new folder
        dmc.train(flags)
    assert not (tmp_path / 'runs' / 'child').exists()
    assert tree(parent) == before and tree(run.path) == in_use  # neither folder gained or lost a file or folder


def test_the_summary_shows_the_fork_and_the_invalid_flag(tmp_path):
    flags = arm_flags('none', tmp_path)
    metadata = dmc.run_metadata(flags, models.load_predictors('none', None))
    clean = dict(commit='a' * 40, branch='main', dirty=False, untracked={}, diff_sha256='d0', code_sha256='c0')
    fork = dict(parent_run_id=dict(xpid='parent', run_dir='/runs/parent'),
                parent_checkpoint=dict(path='/runs/parent/checkpoints/model.tar', sha256='p0'), parent_frames=3200,
                parent_hand_model='next', parent_code=dict(commit='b' * 40, code_sha256='c0'),
                current_code=dict(commit='a' * 40, code_sha256='c1'), reason='probe the belief',
                optimizer_state_restored=True)
    config = run_dir.run_config(flags, dict(metadata, fork=fork))
    stats, updates = {}, {p: 0 for p in POSITIONS}

    def summary(attempts, invalid=False):
        config['metadata']['invalid_for_final_results'] = invalid
        return run_dir.summary_markdown(config, attempts, 'completed', 1.0, 0, updates, stats, 'x')

    text = summary([dict(git=clean)])
    forked = next(line for line in text.splitlines() if line.startswith('- **Forked from:**'))
    for part in ('/runs/parent', '/runs/parent/checkpoints/model.tar', 'p0', '3200', 'probe the belief', 'b' * 40, 'c1'):
        assert part in forked
    warnings = next(line for line in text.splitlines() if line.startswith('- **Warnings:**'))
    assert 'forked from a `--hand_model next` run' in warnings and 'INVALID' not in warnings
    flagged = summary([dict(git=clean, invalid_for_final_results=True)])
    warnings = next(line for line in flagged.splitlines() if line.startswith('- **Warnings:**'))
    assert warnings.startswith('- **Warnings:** INVALID FOR FINAL RESULTS:')  # sticky: a later attempt says so too
    assert 'INVALID' in next(line for line in summary([dict(git=clean)], invalid=True).splitlines()
                             if line.startswith('- **Warnings:**'))


def test_a_resume_across_a_code_change_is_refused_before_anything_is_written(tmp_path, monkeypatch):
    path, _, _, metadata, _, _ = saved_run('next', tmp_path)
    flags = arm_flags('next', tmp_path, '--load_model')
    flags.allow_dirty = True
    run = run_dir.RunDir(tmp_path / 'runs' / 'run_next')
    run.write_config({'arm': 'M1'})
    shutil.copy(path, run.checkpoint_path)
    monkeypatch.setattr(run_dir, 'git_info', lambda *args: dict(metadata['git'], code_sha256='f' * 64))
    before = tree(run.path)
    with pytest.raises(ValueError, match='code_sha256'):
        dmc.train(flags)
    assert tree(run.path) == before  # no attempt, metrics or summary was written


def test_a_fork_starts_its_learner_and_actors_from_the_parent_weights(tmp_path):
    path, _, _, _, learner, optimizers = saved_run('next', tmp_path)
    states = dmc.read_checkpoint(path)
    flags = arm_flags('next', tmp_path / 'fork', '--learning_rate', '0.001')
    actors, forked, forked_optimizers = dmc.initial_models(flags, ['cpu'], states)
    for position in POSITIONS:
        original, start = learner.get_model(position).state_dict(), forked.get_model(position).state_dict()
        assert all(torch.equal(original[k], start[k]) for k in original)
        actor = actors['cpu'].get_model(position).state_dict()
        assert all(torch.equal(original[k], actor[k]) for k in original)
        before, after = optimizers[position].state_dict()['state'], forked_optimizers[position].state_dict()['state']
        assert before.keys() == after.keys() and all(torch.equal(before[i][n], after[i][n]) for i in before for n in before[i])
        assert forked_optimizers[position].param_groups[0]['lr'] == 0.001  # the fork's own learning rate, not the parent's
    weights_only = {k: v for k, v in states.items() if k != 'optimizer_state_dict'}  # a weights_<frames>.tar fork
    _, _, fresh = dmc.initial_models(flags, ['cpu'], weights_only)
    assert all(fresh[position].state_dict()['state'] == {} for position in POSITIONS)
