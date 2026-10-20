"""Shared look of every generated HTML page: a light retro, pixel-style card-table theme.

The replay and comparison pages and the predictor run pages (douzero/predictor) use this module,
so they look alike without importing each other. The theme is a style, not a
code: blocky 2-pixel frames with hard drop shadows, square cards and chips, monospace capitals
for labels. Readability comes first, so the body text stays in the system font, and every
information class keeps its own colour and a second cue that is not colour (a solid or dashed
border, a texture, a label): the actor's own cards solid green, public cards solid blue, hidden
truth dashed orange on a hatched ground, and the belief sources as labelled chips. Teaching and
normal pages carry different banners. Pages are single static files and use no network.
"""
from html import escape

from douzero.env import hand_belief as hb

FACES = dict(zip(hb.KINDS, ('3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', 'BJ', 'RJ')))
MODE_BANNER = {'teaching': 'TEACHING MODE · hidden truth revealed, drawn dashed orange on a hatched ground',
               'normal': 'NORMAL MODE · shows only what the acting player knows'}


def e(value):
    return escape(str(value))


def cards_html(cards, kind):
    """Card chips. kind is 'own' (the actor's private hand), 'public' or 'hidden' (ground truth, teaching only)."""
    if not cards:
        return '<span class="cards {} empty">none</span>'.format(kind)
    return '<span class="cards {}">{}</span>'.format(kind, ''.join(
        '<span class="card{}">{}</span>'.format(' joker' if card >= 20 else '', FACES[card]) for card in cards))


def backs_html(count):
    return '<span class="cards backs" title="{} cards, hidden">{}</span>'.format(
        count, '<span class="card back"></span>' * count)


def page(title, body, script='', mode=None):
    """One self-contained page; mode ('teaching' or 'normal') adds its banner and frame."""
    banner = '<div class="banner {}">{}</div>'.format(mode, MODE_BANNER[mode]) if mode else ''
    return ''.join(['<!doctype html>\n<html lang="en"><head><meta charset="utf-8">',
                    '<meta name="viewport" content="width=device-width, initial-scale=1">',
                    '<title>{}</title><style>{}</style></head>'.format(e(title), CSS),
                    '<body class="{}">'.format('mode-' + mode if mode else ''), banner, body,
                    '<script>{}</script>'.format(script) if script else '', '</body></html>\n'])


def top_kinds(unseen, count=4):
    """The kinds with the most unseen copies (ties in card order): where beliefs have the most room to differ."""
    return sorted((k for k in range(hb.NUM_KINDS) if unseen[k] >= 1), key=lambda k: (-unseen[k], k))[:count]


def _bars(probs, u, truth):
    cells = []
    for k in range(u + 1):
        mark = ' truth' if truth is not None and k == truth else ''
        cells.append('<i class="bar{}" style="--p:{:.3f}" title="P({} copies) = {:.3f}{}"></i>'.format(
            mark, probs[k], k, probs[k], ', the true count' if mark else ''))
    return '<span class="bars">{}</span>'.format(''.join(cells))


def belief_focus_html(unseen, rows, kinds, true_counts=None):
    """Beliefs side by side for a few card kinds: P(0..u copies) as bars, with the expected count.

    rows: dicts with label, css (uniform, counting, predicted, derived, learned, oracle), probs
    (15 x 5) and, optionally, ce (over all 15 kinds, against the truth). true_counts, given only in
    teaching mode, adds the true row and outlines the true count in every belief's bars.
    """
    head = ''.join('<th>{} <span class="muted">({} unseen)</span></th>'.format(FACES[hb.KINDS[k]], unseen[k])
                   for k in kinds)
    body = []
    if true_counts is not None:
        body.append('<tr class="truth"><th>true count (hidden)</th>{}<td></td></tr>'.format(''.join(
            '<td><b>{}</b></td>'.format(true_counts[k]) for k in kinds)))
    for row in rows:
        cells = []
        for k in kinds:
            probs = row['probs'][k]
            mean = sum(i * p for i, p in enumerate(probs))
            cells.append('<td>{}<div class="small">E = {:.2f}</div></td>'.format(
                _bars(probs, unseen[k], None if true_counts is None else true_counts[k]), mean))
        ce = row.get('ce')
        body.append('<tr class="{}"><th>{}</th>{}<td class="ce">{}</td></tr>'.format(
            e(row['css']), row['label'], ''.join(cells), '–' if ce is None else '{:.3f}'.format(ce)))
    return ('<div class="scroll"><table class="focus"><tr><th>belief</th>{}<th>CE (nats)</th></tr>{}</table></div>'
            .format(head, ''.join(body)))


CSS = """
:root{--bg:#e9e3d0;--panel:#fffaf0;--ink:#1b1a17;--muted:#5d5a52;--line:#cfc6ad;--frame:#2b2a26;
--own:#1f7a4d;--public:#2a5db0;--hidden:#b4581d;--predicted:#6b46b8;--derived:#0f7f86;--learned:#a23b72;
--oracle:#c0392b;--zero:#8a9099;--uniform:#7a7466;--counting:#8a6d1f;--sel:#ffe08a;--card:#ffffff;
--seat-landlord:#c99a06;--seat-landlord_down:#2a5db0;--seat-landlord_up:#1f7a4d;
--chart-surface:#fcfcfb;--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-5:#e87ba4}
@media (prefers-color-scheme:dark){:root{--bg:#14161c;--panel:#1f232d;--ink:#ecebe6;--muted:#a8a69c;--line:#3a3f4c;
--frame:#05060a;--own:#5cc28f;--public:#7aa7f0;--hidden:#f0a060;--predicted:#b195f0;--derived:#5cc8ce;
--learned:#e07fb0;--oracle:#f0857a;--zero:#80868e;--uniform:#b0aa9a;--counting:#cdb15a;--sel:#5a4c1c;
--card:#2a2f3b;--seat-landlord:#e0b53a;
--chart-surface:#1a1a19;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-5:#d55181}}
*{box-sizing:border-box}
body{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
h1,h2,h3,.label,.badge,.controls button,.chip{font-family:ui-monospace,Menlo,Consolas,monospace}
h1{font-size:18px;margin:0 0 8px;text-transform:uppercase;letter-spacing:.06em;
padding-bottom:6px;border-bottom:4px solid var(--frame)}
h2{font-size:15px;margin:18px 0 8px;text-transform:uppercase;letter-spacing:.04em}
h3{font-size:13px;margin:0 0 6px;text-transform:uppercase;letter-spacing:.03em}
p{margin:6px 0}.muted{color:var(--muted)}.small{font-size:12px}
.label{margin-top:8px;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.banner{margin:-16px -16px 14px;padding:8px 16px;font:700 12px ui-monospace,Menlo,monospace;letter-spacing:.06em;
color:#fff;border-bottom:4px solid var(--frame)}
.banner.teaching{background:repeating-linear-gradient(135deg,#b4581d 0 14px,#7a3a12 14px 28px)}
.banner.normal{background:#2a5db0}
header .facts th{text-align:left;padding-right:10px;color:var(--muted);font-weight:500;vertical-align:top}
.warn{color:var(--oracle);font-weight:600}.legend{font-size:12px}
.badge{display:inline-block;padding:1px 6px;font-size:11px;font-weight:700;border:2px solid;text-transform:uppercase}
.badge.acting{color:var(--own)}.badge.teaching{color:var(--hidden)}.badge.normal{color:var(--public)}
.seat,.center,.panel,.facts-box{background:var(--panel);border:2px solid var(--frame);box-shadow:4px 4px 0 var(--frame);
padding:10px;min-width:0}
.controls{position:sticky;top:0;z-index:2;background:var(--bg);padding:8px 0;border-bottom:2px solid var(--frame);
display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.controls button{padding:4px 10px;border:2px solid var(--frame);box-shadow:3px 3px 0 var(--frame);
background:var(--panel);color:var(--ink);font-size:12px;font-weight:700;text-transform:uppercase;cursor:pointer}
.controls button:active{transform:translate(3px,3px);box-shadow:none}
.controls input[type=range]{flex:1 1 200px;accent-color:var(--frame)}
.ticks{flex:1 1 100%;display:flex;flex-wrap:wrap;gap:2px}.tick{width:10px;height:14px;background:var(--line);
border:1px solid var(--frame)}
.tick.landlord{background:var(--seat-landlord)}.tick.landlord_down{background:var(--seat-landlord_down)}
.tick.landlord_up{background:var(--seat-landlord_up)}.tick.explored{outline:2px solid var(--hidden);outline-offset:1px}
.tick.end{background:var(--ink)}.tick.current{transform:scaleY(1.5)}
.table{display:grid;grid-template-columns:1fr 1.35fr 1fr;grid-template-areas:". up ." "left center right";gap:14px}
.seat.landlord_up{grid-area:up}.seat.landlord{grid-area:left}.seat.landlord_down{grid-area:right}.center{grid-area:center}
.seat.acting{border:3px solid var(--own);box-shadow:4px 4px 0 var(--own)}
.seat-key{font:12px ui-monospace,Menlo,monospace;color:var(--muted)}
.note{font-size:11px;color:var(--muted)}.hidden-note,.truth-text{color:var(--hidden)}
.cards{display:inline-flex;flex-wrap:wrap;gap:3px}.cards.empty{color:var(--muted);font-size:12px}
.card{display:inline-block;min-width:22px;padding:1px 3px;border:2px solid var(--line);background:var(--card);
text-align:center;font:700 13px/1.3 ui-monospace,Menlo,monospace;box-shadow:2px 2px 0 var(--frame)}
.card.joker{font-style:italic}.cards.own .card{border-color:var(--own)}
.cards.public .card{border-color:var(--public)}
.cards.hidden .card{border:2px dashed var(--hidden);color:var(--hidden);box-shadow:none;
background:repeating-linear-gradient(135deg,var(--card) 0 4px,color-mix(in srgb,var(--hidden) 20%,var(--card)) 4px 6px)}
.card.back{min-width:10px;height:18px;padding:0;box-shadow:none;border-color:var(--frame);
background:repeating-conic-gradient(var(--frame) 0 25%,var(--line) 0 50%) 0 0/4px 4px}
.pass{font-style:italic;color:var(--muted)}
.chip{display:inline-block;padding:0 6px;font-size:11px;border:2px solid currentColor}
.chip.predicted{color:var(--predicted)}.chip.derived{color:var(--derived)}.chip.learned{color:var(--learned)}
.chip.oracle{color:var(--oracle)}.chip.oracle::before{content:"! "}.chip.zero{color:var(--zero)}
.chip.uniform{color:var(--uniform)}.chip.counting{color:var(--counting)}
.decision{display:grid;grid-template-columns:1.6fr 1fr;gap:14px;margin-top:14px}
table{border-collapse:collapse}.kinds td,.kinds th{padding:2px 4px;text-align:center;font-size:12px;
border-bottom:1px solid var(--line)}.kinds th:first-child{text-align:left;font-weight:500;white-space:nowrap}
.beliefs td{background:color-mix(in srgb,var(--c,var(--public)) calc(var(--v,0)*55%),transparent)}
.beliefs tr.counting{--c:var(--counting)}.beliefs tr.uniform{--c:var(--uniform)}.beliefs tr.predicted{--c:var(--predicted)}
.beliefs tr.derived{--c:var(--derived)}.beliefs tr.learned{--c:var(--learned)}.beliefs tr.oracle{--c:var(--oracle)}
.beliefs tr.zero{--c:var(--zero)}.beliefs tr.truth td,.beliefs tr.truth th{color:var(--hidden);font-weight:700}
.beliefs tr.group th{padding-top:8px;text-transform:uppercase;font-size:11px;color:var(--muted)}
.q td,.q th{padding:3px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
.q tr.sel td{background:var(--sel)}.explore{color:var(--hidden);font-weight:700}
.transition{border-left:4px solid var(--public);padding-left:8px}
ol.history{margin:4px 0;padding-left:26px}.q-value{color:var(--muted)}
.comparison .panel,.stack>.panel{margin-top:14px}
.focus th,.focus td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:center;vertical-align:bottom}
.focus th:first-child{text-align:left;white-space:nowrap}.focus .ce{font:700 12px ui-monospace,Menlo,monospace}
.focus tr.truth td,.focus tr.truth th{color:var(--hidden)}
.bars{display:inline-flex;gap:2px;align-items:flex-end;height:34px}
.bar{display:inline-block;width:9px;height:calc(var(--p)*32px + 2px);background:var(--c,var(--public));
border:1px solid var(--frame)}
.bar.truth{outline:2px dashed var(--hidden);outline-offset:1px}
.focus tr.uniform{--c:var(--uniform)}.focus tr.counting{--c:var(--counting)}.focus tr.predicted{--c:var(--predicted)}
.focus tr.derived{--c:var(--derived)}.focus tr.learned{--c:var(--learned)}.focus tr.oracle{--c:var(--oracle)}
.chart{background:var(--chart-surface);border:2px solid var(--frame);box-shadow:4px 4px 0 var(--frame);padding:8px;
margin:10px 0}.chart svg{width:100%;height:auto;display:block}.chart .axis{stroke:var(--line)}
.chart text{fill:var(--muted);font:11px ui-monospace,Menlo,monospace}.chart .lbl{fill:var(--ink)}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.key{font-size:12px;margin:2px 0 4px}.key i{display:inline-block;width:14px;height:4px;margin:0 4px 2px 10px;
vertical-align:middle}
body.paged section.step[hidden]{display:none}section.step{scroll-margin-top:80px}.scroll{overflow-x:auto}
body:not(.paged) .controls>:not(.ticks){display:none}
@media (max-width:760px){.table{grid-template-columns:1fr;grid-template-areas:"up" "left" "center" "right"}
.decision{grid-template-columns:1fr}.kinds{display:block;overflow-x:auto}}
"""
