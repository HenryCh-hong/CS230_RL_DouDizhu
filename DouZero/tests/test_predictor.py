"""Stage 4: the offline predictor dataset, the two baselines, predictor training and its run outputs.

Everything runs on a tiny seeded dataset with random-weight networks: these tests check
correctness (data, splits, labels, masks, losses, checkpoints, logs, plots), not predictor quality.
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from douzero.dmc import belief, models, utils
from douzero.env import hand_belief as hb
from douzero.predictor import data, metrics, report, train

POSITIONS = ('landlord', 'landlord_down', 'landlord_up')
ROOT = Path(__file__).resolve().parents[2]
GAMES = dict(train=3, val=2, test=2)


@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    model, source = data.douzero_model(checkpoint_dir='')  # random DouZero weights: fast, no checkpoint needed
    out = tmp_path_factory.mktemp('data')
    manifest = data.generate(out, GAMES, model=model, model_source=source, exp_epsilon=0.05, shard_games=2)
    return dict(dir=out, manifest=manifest, model=model, source=source)


def batch(tiny, seat, split='train', size=32):
    arrays = data.load_split(tiny['dir'], split, seat)
    return {key: value[:size] for key, value in arrays.items()}


# ---------------------------------------------------------------- 1-5: generation, determinism, splits, reload, labels

def test_every_split_and_seat_gets_samples(tiny):
    manifest = tiny['manifest']
    assert manifest['schema'] == data.SCHEMA and set(manifest['splits']) == set(GAMES)
    for split, games in GAMES.items():
        assert manifest['splits'][split]['games'] == games
        for seat in POSITIONS:
            assert manifest['splits'][split]['samples'][seat] > 0
            assert len(data.load_split(tiny['dir'], split, seat)['labels']) == manifest['splits'][split]['samples'][seat]
    data.verify(tiny['dir'])  # every file matches its recorded sha256


def test_generation_is_deterministic(tiny, tmp_path):
    again = data.generate(tmp_path, GAMES, model=tiny['model'], model_source=tiny['source'], exp_epsilon=0.05,
                          shard_games=2)
    assert again['files'] == tiny['manifest']['files']  # byte-identical shards
    assert data.game_seeds('train', 0) == data.game_seeds('train', 0) != data.game_seeds('val', 0)


def test_splits_are_isolated_from_each_other_and_from_the_evaluation_deals(tiny):
    hashes = {split: {g['deal_hash'] for g in tiny['manifest']['splits'][split]['game_list']} for split in GAMES}
    assert all(len(h) == GAMES[s] for s, h in hashes.items())
    assert not (hashes['train'] & hashes['val']) and not (hashes['train'] & hashes['test']) \
        and not (hashes['val'] & hashes['test'])
    spec = importlib.util.spec_from_file_location('dup_eval', ROOT / 'tools' / 'dup_eval.py')
    dup_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dup_eval)
    evaluation = {data.deal_hash(deal) for seed in range(3) for deal in dup_eval.make_deals(50, seed)}
    assert not (set().union(*hashes.values()) & evaluation)
    seeds = {data.game_seeds(split, i) for split, n in GAMES.items() for i in range(n)}
    assert len(seeds) == sum(GAMES.values())


def test_a_split_reloads_exactly_what_was_played(tiny):
    samples, game = data.play_game(tiny['model'], 'train', 0, exp_epsilon=0.05)
    assert game['deal_hash'] == tiny['manifest']['splits']['train']['game_list'][0]['deal_hash']
    for seat in POSITIONS:
        stored = data.load_split(tiny['dir'], 'train', seat)
        for key, value in samples[seat].items():
            assert np.array_equal(stored[key][:len(value)], value), (seat, key)


def test_labels_conserve_the_unseen_cards_and_match_the_public_observation(tiny):
    for split in GAMES:
        for seat in POSITIONS:
            arrays = data.load_split(tiny['dir'], split, seat)
            labels, union = arrays['labels'].astype(int), arrays['union'].astype(int)
            assert np.array_equal(labels.sum(axis=1), union)  # next + other = unseen, kind by kind
            assert (labels <= union[:, None]).all()  # every true count is feasible
            assert np.array_equal(union, hb.union_counts_from_obs(arrays['x_no_action']))
            assert np.array_equal(arrays['sizes'], labels.sum(axis=2))
            assert np.array_equal(arrays['sizes'], hb.hidden_sizes_from_obs(seat, arrays['x_no_action']))
            assert arrays['x_no_action'].dtype == np.int8 and arrays['z'].shape[1:] == (5, 162)


def test_phases_follow_the_prespecified_decision_progress_rule(tiny):
    arrays = data.load_split(tiny['dir'], 'train', 'landlord')
    progress = arrays['t'] / np.maximum(arrays['game_length'] - 1, 1)
    assert np.array_equal(arrays['phase'], np.digitize(progress, [1 / 3, 2 / 3]))
    assert data.phase_of(0, 60) == 0 and data.phase_of(20, 61) == 1 and data.phase_of(59, 60) == 2


# ---------------------------------------------------------------- 6: the baselines and legal masks

def test_the_legal_uniform_baseline_spreads_mass_over_feasible_counts_only():
    union = np.array([[2, 0, 4] + [1] * 12])
    probs = metrics.uniform_probs(union)
    assert np.allclose(probs[0, 0], [1 / 3, 1 / 3, 1 / 3, 0, 0]) and np.array_equal(probs[0, 1], [1, 0, 0, 0, 0])
    assert np.allclose(probs[0, 2], [0.2] * 5)


def test_the_batched_hypergeometric_baseline_matches_card_counting(tiny):
    arrays = batch(tiny, 'landlord_down', size=40)
    probs = metrics.hypergeometric_probs(arrays['union'], arrays['sizes'][:, 0])
    for i in range(len(probs)):
        assert np.allclose(probs[i], hb.hypergeometric_belief(arrays['union'][i], arrays['sizes'][i, 0]), atol=1e-12)
    assert np.allclose(probs.sum(axis=-1), 1) and not (probs * ~hb.legal_count_mask(arrays['union'])).any()


def test_baseline_metrics_are_finite_and_scored_like_the_model(tiny):
    arrays = batch(tiny, 'landlord')
    report = metrics.baseline_report(arrays)
    for name in ('uniform', 'hypergeometric'):
        for hand in ('next', 'other'):
            assert np.isfinite(report[name][hand]['ce']) and report[name][hand]['ce'] > 0
            assert set(report[name][hand]['ce_by_phase']) == set(data.PHASES)
    nll = metrics.nll(metrics.uniform_probs(arrays['union']), arrays['labels'][:, 0])
    assert nll.mean() == pytest.approx(report['uniform']['next']['ce'])


# ---------------------------------------------------------------- 7-10: loader, forward, backward, overfitting

def test_batches_cover_the_split_once_per_epoch(tiny):
    arrays = data.load_split(tiny['dir'], 'train', 'landlord_up')
    seen = []
    for chunk in data.batches(arrays, batch_size=7, seed=3):
        assert chunk['z'].dtype == torch.float32 and chunk['z'].shape[1:] == (5, 162)
        assert chunk['labels'].dtype == torch.long and chunk['labels'].shape[1:] == (2, 15)
        seen.extend(chunk['index'].tolist())
    assert sorted(seen) == list(range(len(arrays['labels'])))


@pytest.mark.parametrize('num_targets', [1, 2])
def test_forward_and_backward_pass(tiny, num_targets):
    torch.manual_seed(0)
    predictor = models.HandPredictor('landlord', num_targets)
    chunk = next(data.batches(batch(tiny, 'landlord'), batch_size=16, seed=0))
    out = predictor(chunk['z'], chunk['x_no_action'])
    assert out['probs'].shape == (16, num_targets, 15, 5)
    assert not (out['probs'] * ~belief.legal_count_mask(chunk['union']).unsqueeze(1)).any()
    loss = train.loss(out['logits'], chunk['labels'], num_targets)
    loss.backward()
    grads = {name: p.grad for name, p in predictor.named_parameters()}
    assert all(torch.isfinite(g).all() for g in grads.values())
    assert grads['dense1.weight'].abs().sum() > 0 and grads['lstm.weight_ih_l0'].abs().sum() > 0
    assert all(grads['heads.{}.0.0.weight'.format(t)].abs().sum() > 0 for t in range(num_targets))


@pytest.mark.parametrize('num_targets', [1, 2])
def test_a_small_batch_can_be_overfit(tiny, num_targets):
    first, last = train.overfit_one_batch(batch(tiny, 'landlord_down', size=24), 'landlord_down', num_targets,
                                          steps=150, seed=0)
    assert last < 0.25 * first and last < 0.1


# ---------------------------------------------------------------- 11-14: checkpoints

def test_a_checkpoint_reloads_with_bit_identical_predictions(tiny, tmp_path):
    torch.manual_seed(1)
    predictor = models.HandPredictor('landlord_up', 2)
    path = tmp_path / 'landlord_up.ckpt'
    models.save_hand_predictor(predictor, str(path), provenance=dict(trained=True, data='tiny test'))
    reloaded = models.load_hand_predictor(str(path), position='landlord_up', num_targets=2, require_trained=True)
    chunk = next(data.batches(batch(tiny, 'landlord_up'), batch_size=32, seed=0))
    predictor.eval()
    with torch.no_grad():
        assert torch.equal(predictor(chunk['z'], chunk['x_no_action'])['probs'],
                           reloaded(chunk['z'], chunk['x_no_action'])['probs'])


# ---------------------------------------------------------------- 15-16 and 14: a whole training run

@pytest.fixture(scope='module')
def runs(tiny, tmp_path_factory):
    out = tmp_path_factory.mktemp('runs')
    config = dict(batch_size=16, lr=1e-3, max_steps=24, eval_every=6, patience=100, seed=0, examples=4)
    return {k: train.train_predictor(tiny['dir'], k, out / 'k{}'.format(k), **config) for k in (1, 2)}


def test_a_training_run_selects_on_validation_and_scores_the_test_split_once(runs):
    for num_targets, run in runs.items():
        run = Path(run)
        config = json.loads((run / 'config.json').read_text())
        assert 'lowest validation CE' in config['selection_rule'] and config['num_targets'] == num_targets
        with open(run / 'metrics.csv', newline='') as f:
            rows = list(csv.DictReader(f))
        tests = json.loads((run / 'test_metrics.json').read_text())
        for seat in POSITIONS:
            evaluated = [r for r in rows if r['seat'] == seat]
            best = min(evaluated, key=lambda r: (float(r['val_ce']), int(r['step'])))
            predictor = models.load_hand_predictor(str(run / 'checkpoints' / (seat + '.ckpt')), position=seat,
                                                   num_targets=num_targets, require_trained=True)
            assert predictor.provenance['selected_step'] == int(best['step'])
            assert predictor.provenance['split_used_for_selection'] == 'val'
            assert set(tests[seat]) >= {'ce', 'ce_uniform', 'ce_hypergeometric', 'mae'}
        assert {'step', 'seat', 'train_ce', 'val_ce', 'val_ce_uniform', 'val_ce_hypergeometric', 'val_mae',
                'val_ce_early', 'val_ce_mid', 'val_ce_late', 'lr', 'time_s'} <= set(rows[0])
        if num_targets == 2:
            assert {'val_ce_next', 'val_ce_other', 'val_conservation', 'val_reflect_tv'} <= set(rows[0])
        summary = (run / 'summary.md').read_text()
        assert 'legal-uniform' in summary and 'hypergeometric' in summary and 'test split' in summary


def test_one_k1_checkpoint_serves_both_m1_and_m2c(runs, tiny):
    directory = str(Path(runs[1]) / 'checkpoints')
    m1 = models.load_predictors('next', directory)
    m2c = models.load_predictors('derived_both', directory)
    chunk = batch(tiny, 'landlord', size=1)
    x, z = torch.from_numpy(chunk['x_no_action'][0]), torch.from_numpy(chunk['z'][0])
    one = utils.actor_belief('next', m1['landlord'], 'landlord', x, z)
    two = utils.actor_belief('derived_both', m2c['landlord'], 'landlord', x, z)
    assert torch.equal(one[0, :75], two[0, :75]) and not one[0, 75:].any() and two[0, 75:].any()
    assert models.file_sha256(directory + '/landlord.ckpt') == models.predictor_record(directory, m1)['files'][
        'landlord']['sha256']


def test_the_run_page_plots_curves_against_both_baselines_and_shows_examples(runs):
    for num_targets, run in runs.items():
        page = (Path(run) / 'plots' / 'index.html').read_text()
        assert page.startswith('<!doctype html>') and page.count('<svg') >= 5
        for label in ('train CE', 'validation CE', 'legal-uniform', 'hypergeometric', 'early', 'late'):
            assert label in page
        if num_targets == 2:
            assert 'CE_next' in page and 'CE_other' in page and 'conservation' in page
        examples = (Path(run) / 'plots' / 'examples.html').read_text()
        assert 'true count' in examples and 'legal-uniform' in examples and 'learned' in examples


# ---------------------------------------------------------------- review fixes: seeds, deal checks, uncertainty, stopping

def test_game_seeds_use_the_full_seed_sequence_state():
    numpy_seed, torch_seed = data.game_seeds('train', 0)
    assert isinstance(numpy_seed, tuple) and len(numpy_seed) == 4 and 0 <= torch_seed < 2 ** 64
    assert len({data.game_seeds('train', i)[0] for i in range(3000)}) == 3000  # 128 bits: no birthday collisions


def test_a_missing_douzero_checkpoint_is_refused_not_replaced_by_random_weights(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.douzero_model(str(tmp_path / 'nowhere'))


def test_a_dataset_meeting_an_evaluation_deal_is_never_written(tiny, tmp_path):
    forbidden = {tiny['manifest']['splits']['val']['game_list'][1]['deal_hash']}
    with pytest.raises(ValueError, match='evaluation'):
        data.generate(tmp_path / 'bad', GAMES, model=tiny['model'], model_source=tiny['source'], exp_epsilon=0.05,
                      shard_games=2, forbidden_deals=forbidden, forbidden_source='a test set')
    assert not (tmp_path / 'bad' / 'manifest.json').exists()
    assert tiny['manifest']['deal_check']['unique'] is True


def test_training_refuses_a_dataset_whose_deals_were_not_checked(tiny, tmp_path):
    import shutil
    copy = tmp_path / 'unchecked'
    shutil.copytree(tiny['dir'], copy)
    manifest = json.loads((copy / 'manifest.json').read_text())
    del manifest['deal_check']
    (copy / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='deal check'):
        train.train_predictor(copy, 1, tmp_path / 'run', max_steps=2, eval_every=1)
    assert not (tmp_path / 'run').exists()


def test_test_comparisons_carry_game_clustered_confidence_intervals(runs):
    for run in runs.values():
        tests = json.loads((Path(run) / 'test_metrics.json').read_text())
        for seat in POSITIONS:
            assert tests[seat]['games'] == GAMES['test']
            for key in ('uniform_minus_model', 'hypergeometric_minus_model'):
                low, high = tests[seat]['ci95'][key]
                assert low <= tests[seat][key] <= high


def test_run_pages_show_paths_inside_douzero_relative_to_it():
    douzero = ROOT / 'DouZero'
    assert report.shown(str(douzero / 'artifacts' / 'data')) == 'artifacts/data'
    assert report.shown('DouZero-ADP from {}'.format(douzero / 'baselines' / 'douzero_ADP')) == \
        'DouZero-ADP from baselines/douzero_ADP'
    assert report.shown('/elsewhere/data') == '/elsewhere/data'


def test_early_stopping_ends_a_seat_after_patience_evaluations_without_improvement(tiny, tmp_path):
    run = train.train_predictor(tiny['dir'], 1, tmp_path / 'stop', batch_size=8, lr=1e-2, max_steps=400,
                                eval_every=4, patience=1, seed=0, examples=1)
    with open(Path(run) / 'metrics.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    for seat in POSITIONS:
        evaluated = [r for r in rows if r['seat'] == seat]
        assert len(evaluated) < 100 and evaluated[-1]['best'] == '0'  # stopped by patience, not by max_steps
