"""The pages of a predictor run: summary.md, plots/index.html (curves) and plots/examples.html.

The raw numbers stay the source of truth (metrics.csv, baselines.json, test_metrics.json); the
charts are drawn from them as inline SVG, with no plotting library, each with a legend, hover
tooltips on every mark and a table of its numbers. Colours follow the entity: the model blue,
hypergeometric card counting orange, legal-uniform aqua (validated together for colour-blind
separation); CE_next blue and CE_other magenta.
"""
import csv
import math
import os

import numpy as np
import torch

from douzero.dmc import models
from douzero.pages import belief_focus_html, e, page, top_kinds
from douzero.predictor import data, metrics

SERIES = dict(model='var(--series-1)', hypergeometric='var(--series-2)', uniform='var(--series-3)',
              train='var(--series-1)', next='var(--series-1)', other='var(--series-5)')
SEAT_SERIES = dict(landlord='var(--series-1)', landlord_down='var(--series-2)', landlord_up='var(--series-3)')
W, H, PAD_L, PAD_R, PAD_T, PAD_B = 520, 230, 52, 118, 14, 34
DOUZERO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def shown(text):
    """Paths inside DouZero/ relative to it, so pages do not carry this machine's folders; config.json keeps them whole."""
    return text.replace(DOUZERO + os.sep, '')


def _ci(interval):
    return '[{:.4f}, {:.4f}]'.format(*interval)


def _fmt(value):
    return '–' if value is None or (isinstance(value, float) and math.isnan(value)) else '{:.4f}'.format(value)


def _ticks(low, high, count=4):
    if high <= low:
        high = low + 1e-6
    return [low + (high - low) * i / count for i in range(count + 1)]


def _key(items):
    return '<div class="key">{}</div>'.format(''.join(
        '<i style="background:{}{}"></i>{}'.format(color, ';height:0;border-top:2px dashed ' + color if dashed else '',
                                                   e(name)) for name, color, dashed in items))


def _table(header, rows):
    return '<details><summary class="small">the numbers</summary><table class="q">{}{}</table></details>'.format(
        '<tr>{}</tr>'.format(''.join('<th>{}</th>'.format(e(h)) for h in header)),
        ''.join('<tr>{}</tr>'.format(''.join('<td>{}</td>'.format(e(c)) for c in row)) for row in rows))


def line_chart(title, series, refs=(), x_label='step', y_label='CE (nats)', marker=None):
    """series: (name, colour, dashed, [(x, y)]); refs: (name, colour, y) dashed reference lines, labelled at the right."""
    points = [p for _, _, _, pts in series for p in pts if not math.isnan(p[1])]
    if not points:
        return ''
    xs = [p[0] for p in points]
    ys = [p[1] for p in points] + [r[2] for r in refs if not math.isnan(r[2])]
    x0, x1 = min(xs), max(max(xs), min(xs) + 1)
    y0, y1 = min(ys), max(ys)
    span = (y1 - y0) or abs(y1) or 1.0
    y0, y1 = max(0.0, y0 - 0.08 * span) if y0 >= 0 else y0 - 0.08 * span, y1 + 0.08 * span  # CE is never negative
    sx = lambda x: PAD_L + (x - x0) / (x1 - x0) * (W - PAD_L - PAD_R)
    sy = lambda y: PAD_T + (y1 - y) / (y1 - y0) * (H - PAD_T - PAD_B)
    svg = ['<svg viewBox="0 0 {} {}" role="img" aria-label="{}">'.format(W, H, e(title))]
    for tick in _ticks(y0, y1):
        svg.append('<line class="axis" x1="{0}" x2="{1}" y1="{2:.1f}" y2="{2:.1f}"/><text x="{3}" y="{4:.1f}" '
                   'text-anchor="end">{5:.3f}</text>'.format(PAD_L, W - PAD_R, sy(tick), PAD_L - 4, sy(tick) + 4, tick))
    for tick in _ticks(x0, x1):
        svg.append('<text x="{:.1f}" y="{}" text-anchor="middle">{:.0f}</text>'.format(sx(tick), H - PAD_B + 14, tick))
    svg.append('<text x="{}" y="{}" text-anchor="middle">{}</text>'.format((PAD_L + W - PAD_R) / 2, H - 4, e(x_label)))
    if marker is not None:  # labelled just above the x axis, clear of the tick labels
        svg.append('<line x1="{0:.1f}" x2="{0:.1f}" y1="{1}" y2="{2}" stroke="var(--ink)" stroke-dasharray="2 3"/>'
                   '<text class="lbl" x="{0:.1f}" y="{3}" dx="3">selected</text>'.format(
                       sx(marker), PAD_T, H - PAD_B, H - PAD_B - 4))
    shown = sorted(((sy(value), name, color, value) for name, color, value in refs if not math.isnan(value)),
                   key=lambda item: item[0])
    label_y = []
    for y, _, _, _ in shown:  # push labels apart so that close reference lines stay readable
        label_y.append(max(y + 4, label_y[-1] + 12) if label_y else y + 4)
    for (y, name, color, value), text_y in zip(shown, label_y):
        svg.append('<line x1="{}" x2="{}" y1="{y:.1f}" y2="{y:.1f}" stroke="{}" stroke-width="2" stroke-dasharray="6 4">'
                   '<title>{}: {:.4f}</title></line><text class="lbl" x="{}" y="{:.1f}">{}</text>'.format(
                       PAD_L, W - PAD_R, color, e(name), value, W - PAD_R + 6, text_y, e(name), y=y))
    for name, color, dashed, pts in series:
        pts = [p for p in pts if not math.isnan(p[1])]
        if not pts:
            continue
        svg.append('<polyline fill="none" stroke="{}" stroke-width="2"{} points="{}"/>'.format(
            color, ' stroke-dasharray="3 3"' if dashed else '', ' '.join('{:.1f},{:.1f}'.format(sx(x), sy(y))
                                                                       for x, y in pts)))
        for x, y in pts:  # hover targets larger than the marks
            svg.append('<circle cx="{:.1f}" cy="{:.1f}" r="4" fill="{}"/><circle cx="{:.1f}" cy="{:.1f}" r="9" '
                       'fill="transparent"><title>{} at {} {:.0f}: {:.4f}</title></circle>'.format(
                           sx(x), sy(y), color, sx(x), sy(y), e(name), e(x_label), x, y))
        last = pts[-1]
        svg.append('<text class="lbl" x="{:.1f}" y="{:.1f}" dx="6">{}</text>'.format(sx(last[0]), sy(last[1]) - 6,
                                                                                  e(name)))
    svg.append('</svg>')
    rows = sorted({x for _, _, _, pts in series for x, _ in pts})
    table = _table([x_label] + [s[0] for s in series],
                   [[int(x)] + [_fmt(dict(s[3]).get(x)) for s in series] for x in rows])
    keys = [(s[0], s[1], s[2]) for s in series] + [(r[0], r[1], True) for r in refs]
    return '<div class="chart"><h3>{}</h3>{}{}{}</div>'.format(e(title), _key(keys), ''.join(svg), table)


def bar_chart(title, groups, bars, values, y_label='CE (nats)'):
    """Grouped bars: groups along x, one bar per (name, colour) in bars, values[group][bar] (lower CE is better)."""
    flat = [v for row in values for v in row if not math.isnan(v)]
    if not flat:
        return ''
    top = max(flat) * 1.12
    width = (W - PAD_L - 20) / len(groups)
    bar = min(28.0, (width - 16) / len(bars))
    sy = lambda y: PAD_T + (top - y) / top * (H - PAD_T - PAD_B)
    svg = ['<svg viewBox="0 0 {} {}" role="img" aria-label="{}">'.format(W, H, e(title))]
    for tick in _ticks(0, top):
        svg.append('<line class="axis" x1="{0}" x2="{1}" y1="{2:.1f}" y2="{2:.1f}"/><text x="{3}" y="{4:.1f}" '
                   'text-anchor="end">{5:.2f}</text>'.format(PAD_L, W - 20, sy(tick), PAD_L - 4, sy(tick) + 4, tick))
    for g, group in enumerate(groups):
        left = PAD_L + g * width + (width - bar * len(bars)) / 2
        for b, (name, color) in enumerate(bars):
            value = values[g][b]
            if math.isnan(value):
                continue
            x = left + b * bar
            svg.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" fill="{}" stroke="var(--chart-surface)" '
                       'stroke-width="2"><title>{}, {}: {:.4f}</title></rect>'.format(
                           x, sy(value), bar, sy(0) - sy(value), color, e(group), e(name), value))
        svg.append('<text class="lbl" x="{:.1f}" y="{}" text-anchor="middle">{}</text>'.format(
            PAD_L + g * width + width / 2, H - PAD_B + 16, e(group)))
    svg.append('<text x="12" y="{}" transform="rotate(-90 12 {})" text-anchor="middle">{}</text></svg>'.format(
        H / 2, H / 2, e(y_label)))
    table = _table(['group'] + [b[0] for b in bars], [[g] + [_fmt(v) for v in row] for g, row in zip(groups, values)])
    return '<div class="chart"><h3>{}</h3>{}{}{}</div>'.format(e(title), _key([(n, c, False) for n, c in bars]),
                                                             ''.join(svg), table)


def read_metrics(run_path):
    with open(os.path.join(str(run_path), 'metrics.csv'), newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            if key != 'seat':
                row[key] = float(value) if value not in ('', 'nan') else float('nan')
    return rows


def write_run_pages(run_path, config, tests, data_dir, examples):
    """plots/index.html (the curves and comparisons) and plots/examples.html (beliefs on validation samples)."""
    rows = read_metrics(run_path)
    seats = config['seats']
    num_targets = config['num_targets']
    selected = {}
    for seat in seats:
        evaluated = [r for r in rows if r['seat'] == seat]
        selected[seat] = min(evaluated, key=lambda r: (r['val_ce'], r['step']))
    body = ['<header><h1>Predictor run {}</h1>'.format(e(config['run_id'])),
            '<p>K = {} ({}), for {}. Data: {} (ε = {}). Raw numbers: <code>metrics.csv</code>, <code>baselines.json</code>, '
            '<code>test_metrics.json</code> in the run folder.</p>'.format(
                num_targets, 'the next hidden hand' if num_targets == 1 else 'both hidden hands',
                ' and '.join(config['for_arms']), e(shown(config['data']['model'])), config['data']['exp_epsilon']),
            '<p class="small"><b>Selection rule:</b> {}</p>'.format(e(config['selection_rule'])),
            '<p class="small"><b>Phases:</b> {}</p></header>'.format(e(config['phase_rule']))]
    body.append('<h2>Learning curves (lower CE is better)</h2><div class="charts">')
    for seat in seats:
        evaluated = [r for r in rows if r['seat'] == seat]
        body.append(line_chart(
            '{}: train CE and validation CE'.format(seat),
            [('validation CE', SERIES['model'], False, [(r['step'], r['val_ce']) for r in evaluated]),
             ('train CE', SERIES['train'], True, [(r['step'], r['train_ce']) for r in evaluated])],
            refs=[('hypergeometric', SERIES['hypergeometric'], evaluated[0]['val_ce_hypergeometric']),
                  ('legal-uniform', SERIES['uniform'], evaluated[0]['val_ce_uniform'])],
            marker=selected[seat]['step']))
    body.append('</div><h2>Selected predictors against both baselines (validation split)</h2><div class="charts">')
    bars = [('model', SERIES['model']), ('hypergeometric', SERIES['hypergeometric']), ('legal-uniform', SERIES['uniform'])]
    body.append(bar_chart('Validation CE by seat', seats, bars,
                          [[selected[s]['val_ce'], selected[s]['val_ce_hypergeometric'], selected[s]['val_ce_uniform']]
                           for s in seats]))
    baselines = _load_baselines(run_path)
    hands = metrics.HANDS[:num_targets]
    for seat in seats:
        base = baselines['val'][seat]
        body.append(bar_chart('{}: validation CE by game phase (early, mid, late)'.format(seat), list(metrics.PHASES), bars, [
            [selected[seat]['val_ce_' + phase],
             np.mean([base['hypergeometric'][h]['ce_by_phase'][phase] for h in hands]),
             np.mean([base['uniform'][h]['ce_by_phase'][phase] for h in hands])] for phase in metrics.PHASES]))
    body.append('</div>')
    if num_targets == 2:
        body.append('<h2>Two hands (M2): CE_next, CE_other and consistency</h2><div class="charts">')
        for seat in seats:
            evaluated = [r for r in rows if r['seat'] == seat]
            body.append(line_chart('{}: validation CE_next and CE_other'.format(seat), [
                ('CE_next', SERIES['next'], False, [(r['step'], r['val_ce_next']) for r in evaluated]),
                ('CE_other', SERIES['other'], False, [(r['step'], r['val_ce_other']) for r in evaluated])],
                marker=selected[seat]['step']))
        body.append(line_chart('Expected conservation violation |E[next] + E[other] − unseen| (validation)', [
            (seat, SEAT_SERIES[seat], False, [(r['step'], r['val_conservation']) for r in rows if r['seat'] == seat])
            for seat in seats], y_label='cards per kind'))
        body.append(line_chart('Distance between learned p_other and reflect(p_next) (validation)', [
            (seat, SEAT_SERIES[seat], False, [(r['step'], r['val_reflect_tv']) for r in rows if r['seat'] == seat])
            for seat in seats], y_label='total variation'))
        body.append('</div><p class="small">M2 is not forced to be consistent: these curves measure how far its two '
                    'learned hands are from obeying next + other = unseen.</p>')
    body.append('<h2>Test split (scored once, after selection)</h2><p class="small">Differences are baseline − model '
                '(positive: the model is better), with 95% intervals from a bootstrap over whole test games.</p>' + _table(
        ['seat', 'games', 'selected step', 'test CE', 'hypergeometric', 'legal-uniform',
         'hypergeometric − model [95% CI]', 'uniform − model [95% CI]', 'MAE'],
        [[s, tests[s]['games'], int(tests[s]['selected_step']), _fmt(tests[s]['ce']),
          _fmt(tests[s]['ce_hypergeometric']), _fmt(tests[s]['ce_uniform']),
          '{} {}'.format(_fmt(tests[s]['hypergeometric_minus_model']), _ci(tests[s]['ci95']['hypergeometric_minus_model'])),
          '{} {}'.format(_fmt(tests[s]['uniform_minus_model']), _ci(tests[s]['ci95']['uniform_minus_model'])),
          _fmt(tests[s]['mae'])] for s in seats]).replace('<details>', '<details open>'))
    body.append('<p><a href="examples.html">Beliefs on validation samples →</a></p>')
    plots = os.path.join(str(run_path), 'plots')
    with open(os.path.join(plots, 'index.html'), 'w') as f:
        f.write(page('Predictor run {}'.format(config['run_id']), '<main class="stack">{}</main>'.format(''.join(body))))
    with open(os.path.join(plots, 'examples.html'), 'w') as f:
        f.write(examples_page(run_path, config, data_dir, examples))


def _load_baselines(run_path):
    import json
    with open(os.path.join(str(run_path), 'baselines.json')) as f:
        return json.load(f)


def examples_page(run_path, config, data_dir, examples):
    """A few validation decisions: the true counts against legal-uniform, card counting and the learned belief."""
    num_targets = config['num_targets']
    body = ['<header><h1>What did the predictor learn beyond counting?</h1>',
            '<p>Validation decisions, spread evenly through the split. For the kinds with the most unseen copies, '
            'each row shows P(0, 1, … u copies); the dashed outline marks the true count. CE is over all 15 kinds '
            'of that one decision (lower is better).</p></header>']
    per_seat = max(1, math.ceil(examples / len(config['seats'])))
    for seat in config['seats']:
        arrays = data.load_split(data_dir, 'val', seat)
        count = len(arrays['labels'])
        if not count:
            continue
        predictor = models.load_hand_predictor(os.path.join(str(run_path), 'checkpoints', seat + '.ckpt'),
                                               position=seat, num_targets=num_targets, require_trained=True)
        for i in np.linspace(0, count - 1, per_seat).round().astype(int):
            sample = {key: value[i:i + 1] for key, value in arrays.items()}
            with torch.no_grad():
                learned = predictor(torch.from_numpy(sample['z']).float(),
                                    torch.from_numpy(sample['x_no_action']).float())['probs'][0].double().numpy()
            union, sizes, labels = sample['union'][0].astype(int), sample['sizes'][0], sample['labels'][0].astype(int)
            body.append('<div class="panel"><h3>{} · decision t = {} of {} · {} phase · {} unseen cards</h3>'.format(
                e(seat), int(sample['t'][0]), int(sample['game_length'][0]), metrics.PHASES[int(sample['phase'][0])],
                int(union.sum())))
            for h, hand in enumerate(metrics.HANDS):
                rows = [dict(label='legal-uniform', css='uniform', probs=metrics.uniform_probs(union)),
                        dict(label='hypergeometric (card counting)', css='counting',
                             probs=metrics.hypergeometric_probs(union[None], sizes[h:h + 1])[0])]
                if h < num_targets:
                    rows.append(dict(label='learned (K={} predictor)'.format(num_targets),
                                     css='learned' if num_targets == 2 else 'predicted', probs=learned[h]))
                for row in rows:
                    row['ce'] = float(metrics.nll(row['probs'][None], labels[h][None]).mean())
                    row['probs'] = np.asarray(row['probs']).tolist()
                body.append('<div class="label">{} hand ({} cards)</div>'.format(hand, int(sizes[h])))
                body.append(belief_focus_html(union.tolist(), rows, top_kinds(union.tolist()), labels[h].tolist()))
            body.append('</div>')
    return page('Predictor examples {}'.format(config['run_id']), '<main class="stack">{}</main>'.format(''.join(body)),
                mode='teaching')


def summary_markdown(config, tests, manifest, elapsed):
    """summary.md of a predictor run."""
    seats = config['seats']
    samples = config['data']['samples']
    warnings = []
    if manifest.get('deal_check', {}).get('evaluation') == 'not checked':
        warnings.append('the dataset was not checked against the evaluation deals')
    if 'RANDOM' in str(manifest['model']):
        warnings.append('the data were generated with RANDOM DouZero weights: a test run, not a predictor')
    if min(samples['train'][s] for s in seats) < 10000:
        warnings.append('tiny dataset (fewer than 10,000 training samples per seat): a pipeline check, not a result')
    if config['git']['dirty'] is None:
        warnings.append('the git state of the code could not be read')
    elif config['git']['dirty']:
        warnings.append('trained from a git tree with uncommitted changes (code sha256 {})'.format(
            config['git']['code_sha256']))
    lines = ['# Predictor run {}'.format(config['run_id']), '',
             '- **Predictor:** K={} ({}), one per seat, for {}.'.format(
                 config['num_targets'], 'next hidden hand' if config['num_targets'] == 1 else 'next and other hand',
                 ' and '.join(config['for_arms'])),
             '- **Data:** `{}` (manifest sha256 {}), from {}, ε = {}.'.format(
                 shown(config['data']['dir']), config['data']['manifest_sha256'][:16], shown(config['data']['model']),
                 config['data']['exp_epsilon']),
             '- **Samples per seat** (train / val / test): ' + '; '.join('{} {} / {} / {}'.format(
                 s, samples['train'][s], samples['val'][s], samples['test'][s]) for s in seats),
             '- **Training:** {}; {:.0f} s wall-clock.'.format(
                 ', '.join('{}={}'.format(k, v) for k, v in config['hyper'].items()), elapsed),
             '- **Selection rule:** ' + config['selection_rule'],
             '- **Phases:** ' + config['phase_rule'], '',
             '## Test split (scored once, after selection)', '',
             'Differences are baseline − model (positive: the model is better), with 95% intervals from a bootstrap '
             'over whole test games, because decisions of one game are correlated.', '',
             '| seat | games | selected step | CE model | CE hypergeometric | CE legal-uniform | hypergeometric − model '
             '[95% CI] | uniform − model [95% CI] | MAE |', '|---|---|---|---|---|---|---|---|---|']
    for s in seats:
        t = tests[s]
        lines.append('| {} | {} | {} | {} | {} | {} | {} {} | {} {} | {} |'.format(
            s, t['games'], t['selected_step'], _fmt(t['ce']), _fmt(t['ce_hypergeometric']), _fmt(t['ce_uniform']),
            _fmt(t['hypergeometric_minus_model']), _ci(t['ci95']['hypergeometric_minus_model']),
            _fmt(t['uniform_minus_model']), _ci(t['ci95']['uniform_minus_model']), _fmt(t['mae'])))
    lines += ['', '## Test CE by game phase (model / hypergeometric / legal-uniform)', '',
              '| seat | early | mid | late |', '|---|---|---|---|']
    for s in seats:
        lines.append('| {} | {} |'.format(s, ' | '.join('{} / {} / {}'.format(
            _fmt(tests[s]['by_phase'][p]['model']), _fmt(tests[s]['by_phase'][p]['hypergeometric']),
            _fmt(tests[s]['by_phase'][p]['uniform'])) for p in metrics.PHASES)))
    if config['num_targets'] == 2:
        lines += ['', '## Two hands (test split)', '', '| seat | CE_next | CE_other | conservation violation | '
                  'reflect TV |', '|---|---|---|---|---|']
        for s in seats:
            t = tests[s]
            lines.append('| {} | {} | {} | {} | {} |'.format(
                s, _fmt(t['hands']['next']['ce']), _fmt(t['hands']['other']['ce']), _fmt(t['conservation']),
                _fmt(t['reflect_tv'])))
    lines += ['', '- **Warnings:** ' + ('; '.join(warnings) if warnings else 'none'),
              '- **Files:** `checkpoints/<seat>.ckpt` (the selected predictors), `metrics.csv`, `baselines.json`, '
              '`test_metrics.json`, `plots/index.html`, `plots/examples.html`.', '']
    return '\n'.join(lines)
