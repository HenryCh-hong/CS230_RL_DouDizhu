"""Offline predictor training: supervised learning of the hidden hands from public observations.

One HandPredictor per seat (the seats see different observations): K=1 predicts the next hidden
hand and serves both M1 and M2c; K=2 predicts (next, other) and serves M2. The loss is the
masked cross-entropy of the true counts. The checkpoint rule is fixed before any real training
(SELECTION_RULE) and uses the validation split only; the test split is scored once, at the end,
on the selected checkpoints. A run writes <run>/config.json, metrics.csv, baselines.json,
test_metrics.json, summary.md, checkpoints/<seat>.ckpt (the selected, frozen-ready predictors)
and plots/.
"""
import json
import math
import os
import time

import numpy as np
import torch
from torch import nn

from douzero.dmc import models
from douzero.dmc.belief import ARM_NAMES, hand_cross_entropy
from douzero.dmc.run_dir import RunDir, git_info, machine_info, now
from douzero.predictor import data, metrics, report

POSITIONS = data.POSITIONS
SELECTION_RULE = ('for each seat separately, the checkpoint with the lowest validation CE (the mean over the '
                  'predicted hands and the 15 card kinds), evaluated on the whole validation split every '
                  'eval_every steps; the earliest on ties; training stops after patience evaluations without '
                  'improvement or at max_steps. The test split is scored once, on the selected checkpoints, after '
                  'training; it never influences the choice.')
ROW = ('step', 'epoch', 'seat', 'samples', 'train_ce', 'val_ce', 'val_ce_next', 'val_ce_other', 'val_ce_early',
       'val_ce_mid', 'val_ce_late', 'val_ce_uniform', 'val_ce_hypergeometric', 'val_mae', 'val_conservation',
       'val_reflect_tv', 'grad_norm', 'lr', 'time_s', 'best')


def loss(logits, labels, num_targets):
    """The training loss: the masked CE of the true counts, averaged over the predicted hands."""
    return hand_cross_entropy(logits, labels[:, :num_targets]).mean()


def seat_seed(seed, seat):
    return int(np.random.SeedSequence([seed, POSITIONS.index(seat)]).generate_state(1)[0])


def overfit_one_batch(arrays, seat, num_targets, steps=150, seed=0, lr=1e-3):
    """Trains a fresh predictor on one small batch only and returns (first CE, last CE): a pipeline check."""
    torch.manual_seed(seed)
    predictor = models.HandPredictor(seat, num_targets)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    chunk = next(data.batches(arrays, batch_size=len(arrays['labels']), seed=seed, shuffle=False))
    history = []
    for _ in range(steps):
        value = loss(predictor(chunk['z'], chunk['x_no_action'])['logits'], chunk['labels'], num_targets)
        optimizer.zero_grad()
        value.backward()
        optimizer.step()
        history.append(value.item())
    return history[0], history[-1]


def _mean(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(values)) if values else float('nan')


def _hands(num_targets):
    return metrics.HANDS[:num_targets]


def evaluation_row(num_targets, report_, baselines):
    """The validation columns of a metrics row: the model's report against the two baselines."""
    hands = _hands(num_targets)
    row = dict(val_ce=_mean([report_[h]['ce'] for h in hands]),
               val_ce_next=report_['next']['ce'],
               val_ce_other=report_['other']['ce'] if num_targets == 2 else float('nan'),
               val_mae=_mean([report_[h]['mae'] for h in hands]),
               val_ce_uniform=_mean([baselines['uniform'][h]['ce'] for h in hands]),
               val_ce_hypergeometric=_mean([baselines['hypergeometric'][h]['ce'] for h in hands]),
               val_conservation=report_.get('conservation', float('nan')),
               val_reflect_tv=report_.get('reflect_tv', float('nan')))
    for phase in metrics.PHASES:
        row['val_ce_' + phase] = _mean([report_[h]['ce_by_phase'][phase] for h in hands])
    return row


def _save(predictor, path, provenance):
    tmp = path + '.tmp'
    models.save_hand_predictor(predictor, tmp, provenance=provenance)
    os.replace(tmp, path)


def train_seat(seat, num_targets, train_arrays, val_arrays, val_baselines, run, provenance, batch_size, lr,
               max_steps, eval_every, patience, seed, max_grad_norm, start_time):
    """Trains one seat's predictor and keeps, in checkpoints/<seat>.ckpt, the one the selection rule picks."""
    if not len(train_arrays['labels']) or not len(val_arrays['labels']):
        raise ValueError('{}: the train and validation splits need samples'.format(seat))
    torch.manual_seed(seat_seed(seed, seat))
    predictor = models.HandPredictor(seat, num_targets)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    path = str(run.path / 'checkpoints' / (seat + '.ckpt'))
    step = epoch = since_best = 0
    best, losses, norms = float('inf'), [], []
    while step < max_steps and since_best < patience:
        for chunk in data.batches(train_arrays, batch_size, seed=seat_seed(seed + 1 + epoch, seat)):
            predictor.train()
            value = loss(predictor(chunk['z'], chunk['x_no_action'])['logits'], chunk['labels'], num_targets)
            optimizer.zero_grad()
            value.backward()
            norms.append(nn.utils.clip_grad_norm_(predictor.parameters(), max_grad_norm).item())
            optimizer.step()
            losses.append(value.item())
            step += 1
            if not math.isfinite(losses[-1]):
                raise RuntimeError('{}: non-finite training loss at step {}'.format(seat, step))
            if step % eval_every == 0 or step == max_steps:
                predictor.eval()
                row = dict(step=step, epoch=epoch, seat=seat, samples=step * batch_size, train_ce=_mean(losses),
                           grad_norm=_mean(norms), lr=lr, time_s=round(time.time() - start_time, 1))
                row.update(evaluation_row(num_targets, metrics.model_report(predictor, val_arrays), val_baselines))
                if math.isnan(row['val_ce']):  # fail now, not after every seat has trained
                    raise RuntimeError('{}: validation CE is not a number at step {}'.format(seat, step))
                improved = row['val_ce'] < best
                row['best'] = int(improved)
                if improved:
                    best, since_best = row['val_ce'], 0
                    _save(predictor, path, dict(provenance, selected_step=step, selected_val_ce=row['val_ce'],
                                                epochs_seen=epoch, train_samples=int(len(train_arrays['labels'])),
                                                saved=now()))
                else:
                    since_best += 1
                run.log_metrics({key: row[key] for key in ROW})
                losses, norms = [], []
                if since_best >= patience or step >= max_steps:
                    break
        epoch += 1
    return best


def test_report(seat, num_targets, path, test_arrays, test_baselines):
    """The selected predictor on the test split, next to both baselines (computed once, after training)."""
    predictor = models.load_hand_predictor(path, position=seat, num_targets=num_targets, require_trained=True)
    result = metrics.model_report(predictor, test_arrays)
    per_sample = metrics.sample_ce(predictor, test_arrays)
    hands = _hands(num_targets)
    out = dict(samples=int(len(test_arrays['labels'])), selected_step=predictor.provenance['selected_step'],
               ce=_mean([result[h]['ce'] for h in hands]), mae=_mean([result[h]['mae'] for h in hands]),
               ce_uniform=_mean([test_baselines['uniform'][h]['ce'] for h in hands]),
               ce_hypergeometric=_mean([test_baselines['hypergeometric'][h]['ce'] for h in hands]),
               by_phase={phase: dict(model=_mean([result[h]['ce_by_phase'][phase] for h in hands]),
                                     uniform=_mean([test_baselines['uniform'][h]['ce_by_phase'][phase] for h in hands]),
                                     hypergeometric=_mean([test_baselines['hypergeometric'][h]['ce_by_phase'][phase]
                                                           for h in hands]))
                         for phase in metrics.PHASES},
               hands={h: result[h] for h in hands})
    out['uniform_minus_model'] = out['ce_uniform'] - out['ce']
    out['hypergeometric_minus_model'] = out['ce_hypergeometric'] - out['ce']
    games = test_arrays['game']
    out['games'] = int(len(np.unique(games)))
    out['ci95'] = dict(  # resampling whole games, because decisions of one game are correlated
        ce=metrics.clustered_ci(per_sample['model'], games),
        uniform_minus_model=metrics.clustered_ci(per_sample['uniform'] - per_sample['model'], games),
        hypergeometric_minus_model=metrics.clustered_ci(per_sample['hypergeometric'] - per_sample['model'], games))
    for key in ('conservation', 'reflect_tv'):
        if key in result:
            out[key] = result[key]
    return out


def train_predictor(data_dir, num_targets, run_path, batch_size=256, lr=1e-3, max_steps=20000, eval_every=500,
                    patience=5, seed=0, examples=12, max_grad_norm=10.0, seats=POSITIONS):
    """Trains and selects one predictor per seat, scores them on the test split once, and writes the run folder."""
    data.verify(data_dir)
    manifest = data.read_manifest(data_dir)
    if not manifest.get('deal_check', {}).get('unique'):
        raise ValueError('{} has no passed deal check (repeated or evaluation deals); regenerate it'.format(data_dir))
    run = RunDir(run_path)  # refuses a folder that already holds a run
    hyper = dict(batch_size=batch_size, lr=lr, optimizer='Adam', max_steps=max_steps, eval_every=eval_every,
                 patience=patience, seed=seed, max_grad_norm=max_grad_norm)
    arms = [ARM_NAMES[h] for h in (('next', 'derived_both') if num_targets == 1 else ('learned_both',))]
    manifest_sha256 = models.file_sha256(os.path.join(str(data_dir), 'manifest.json'))
    config = dict(kind='offline hidden-hand predictor (Stage 4)', run_id=run.path.name, num_targets=num_targets,
                  for_arms=arms, seats=list(seats), hyper=hyper, selection_rule=SELECTION_RULE,
                  phase_rule=manifest['phase_rule'],
                  data=dict(dir=os.path.abspath(str(data_dir)), manifest_sha256=manifest_sha256,
                            model=manifest['model'], exp_epsilon=manifest['exp_epsilon'],
                            samples={s: manifest['splits'][s]['samples'] for s in manifest['splits']}),
                  git=git_info(), machine=machine_info(), started=now())
    run.write_config(config)
    (run.path / 'plots').mkdir(exist_ok=True)
    provenance = dict(trained=True, kind='offline predictor (Stage 4)', run_id=run.path.name, num_targets=num_targets,
                      for_arms=arms, data_manifest_sha256=manifest_sha256, data_dir=config['data']['dir'],
                      data_model=manifest['model'], split_used_for_selection='val', selection_rule=SELECTION_RULE,
                      hyper=hyper,
                      git_commit=config['git']['commit'], code_sha256=config['git']['code_sha256'])
    baselines = {'val': {seat: metrics.baseline_report(data.load_split(data_dir, 'val', seat)) for seat in seats}}
    start_time = time.time()
    for seat in seats:
        train_seat(seat, num_targets, data.load_split(data_dir, 'train', seat), data.load_split(data_dir, 'val', seat),
                   baselines['val'][seat], run, dict(provenance, seat=seat), batch_size, lr, max_steps, eval_every,
                   patience, seed, max_grad_norm, start_time)
    # The test split is read only now, after every seat's checkpoint has been chosen.
    baselines['test'] = {seat: metrics.baseline_report(data.load_split(data_dir, 'test', seat)) for seat in seats}
    with open(run.path / 'baselines.json', 'w') as f:
        json.dump(baselines, f, indent=1)
    tests = {seat: test_report(seat, num_targets, str(run.path / 'checkpoints' / (seat + '.ckpt')),
                               data.load_split(data_dir, 'test', seat), baselines['test'][seat]) for seat in seats}
    with open(run.path / 'test_metrics.json', 'w') as f:
        json.dump(tests, f, indent=1)
    report.write_run_pages(run.path, config, tests, data_dir, examples)
    run.write_summary(report.summary_markdown(config, tests, manifest, time.time() - start_time))
    return str(run.path)
