"""Renders visualization records (record.py) as self-contained HTML pages.

One file per page and no network access: inline CSS, a few lines of JavaScript to step through a
game, and every step still readable, one after another, without JavaScript. NORMAL mode shows
only what the acting player knows; TEACHING mode (reveal) adds the hidden ground truth, drawn
with dashed orange borders so it can never be mistaken for public information.
"""
from html import escape

from douzero.env import hand_belief as hb
from douzero.viz.record import strip_comparison, strip_hidden

FACES = dict(zip(hb.KINDS, ('3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', 'BJ', 'RJ')))
ROLE = {'landlord': 'Landlord', 'landlord_down': 'Peasant', 'landlord_up': 'Peasant'}
SEAT_NOTE = {'landlord': 'plays first and started with 20 cards',
             'landlord_down': 'plays right after the landlord',
             'landlord_up': 'plays right before the landlord'}
LAYOUT = ('landlord_up', 'landlord', 'landlord_down')  # top, left, right; the public table sits in the middle
ARM_ORDER = ('none', 'next', 'derived_both', 'learned_both', 'oracle')


def e(value):
    return escape(str(value))


# ---------------------------------------------------------------- small pieces

def cards_html(cards, kind):
    """Card chips. kind is 'own' (the actor's private hand), 'public' or 'hidden' (ground truth, teaching only)."""
    if not cards:
        return '<span class="cards {} empty">none</span>'.format(kind)
    return '<span class="cards {}">{}</span>'.format(kind, ''.join(
        '<span class="card{}">{}</span>'.format(' joker' if card >= 20 else '', FACES[card]) for card in cards))


def backs_html(count):
    return '<span class="cards backs" title="{} cards, hidden">{}</span>'.format(
        count, '<span class="card back"></span>' * count)


def move_html(cards, move_type):
    if not cards:
        return '<span class="pass">pass</span>'
    return '{} <span class="muted">{}</span>'.format(cards_html(cards, 'public'), e(move_type))


def seat_name(position):
    return '{} <span class="seat-key">{}</span>'.format(ROLE[position], e(position))


def source_chip(source):
    return '<span class="chip {}">{}</span>'.format(e(source['kind']), e(source['text']))


def fmt(value, digits=2):
    return '–' if value is None else '{:.{}f}'.format(value, digits)


def kinds_header():
    return '<tr><th></th>{}</tr>'.format(''.join('<th>{}</th>'.format(FACES[k]) for k in hb.KINDS))


def count_cells(counts, unseen):
    return ''.join('<td>{}</td>'.format(n if u else '·') for n, u in zip(counts, unseen))


def belief_cells(summary, unseen):
    """Expected counts, shaded by size, with the full distribution P(count = 0..u) on hover."""
    cells = []
    for expected, probs, u in zip(summary['expected'], summary['probs'], unseen):
        if not u:
            cells.append('<td>·</td>')
            continue
        tip = 'P(count = 0..{}) = {}'.format(u, ' '.join('{:.2f}'.format(p) for p in probs[:u + 1]))
        cells.append('<td style="--v:{:.2f}" title="{}">{:.1f}</td>'.format(min(expected / 4, 1), e(tip), expected))
    return ''.join(cells)


def history_html(history, last=6):
    """The public sequence of moves, the most recent last."""
    if not history:
        return '<p class="muted">No moves yet: this is the first decision of the game.</p>'
    items = ['<li><b>{}</b> {}</li>'.format(e(h['seat']), move_html(h['action'], h['type'])) for h in history]
    recent = '<ol class="history" start="{}">{}</ol>'.format(max(len(items) - last, 0) + 1, ''.join(items[-last:]))
    if len(items) <= last:
        return recent
    return recent + '<details><summary>all {} moves</summary><ol class="history">{}</ol></details>'.format(
        len(items), ''.join(items))


# ---------------------------------------------------------------- the table: three seats around the public area

def seat_html(position, actor, public, own_hand, hands=None, beliefs=None, sources=None, ce=None):
    """One seat. hands (teaching mode) holds every true hand; beliefs and sources are the actor's belief input."""
    next_player, other_player = public['hidden_seats']
    badge = ''
    if position == actor:
        badge = ' <span class="badge acting">acting now</span>'
    parts = ['<div class="seat-head"><b>{}</b>{}</div>'.format(seat_name(position), badge),
             '<div class="muted small">{}</div>'.format(SEAT_NOTE[position]),
             '<div>Cards left: <b>{}</b></div>'.format(public['cards_left'][position]),
             '<div class="label">Hand</div>']
    if position == actor:
        parts += [cards_html(own_hand, 'own'), '<div class="note">its own hand: private, known only to this player</div>']
    elif hands is not None:
        parts += [cards_html(hands[position], 'hidden'),
                  '<div class="note hidden-note">true hand, hidden from the actor (teaching mode)</div>']
    else:
        parts += [backs_html(public['cards_left'][position]), '<div class="note">hidden: only the count is public</div>']
    parts += ['<div class="label">Played so far (public)</div>', cards_html(public['played'][position], 'public')]
    if beliefs is not None and position in (next_player, other_player):
        slot = 'next' if position == next_player else 'other'
        summary = beliefs[slot]
        source = sources[0 if slot == 'next' else 1]
        top = sorted(zip(summary['expected'], hb.KINDS), reverse=True)[:3]
        parts += ['<div class="label">What the actor believes about this hand</div>',
                  '<div>{} {}</div>'.format('next hand (slot 1)' if slot == 'next' else 'other hand (slot 2)',
                                             source_chip(source))]
        if source['kind'] != 'zero':
            size, expected_size = public['cards_left'][position], sum(summary['expected'])
            parts.append('<div class="small">expected size {:.1f} (the real size, {}, is public{}); most expected: '
                         '{}</div>'.format(expected_size, size, '; this belief does not use it'
                                           if abs(expected_size - size) > 0.5 else '',
                                           ', '.join('{} ×{:.1f}'.format(FACES[kind], value) for value, kind in top)))
        if ce is not None:
            parts.append('<div class="small truth-text">CE against the true hand: belief {}, card counting {}, '
                         'uniform {} nats</div>'.format(fmt(ce['belief'][slot] if ce['belief'] else None),
                                                        fmt(ce['card_counting'][slot]), fmt(ce['uniform'][slot])))
    classes = 'seat {}{}'.format(e(position), ' acting' if position == actor else '')
    return '<div class="{}">{}</div>'.format(classes, ''.join(parts))


def center_html(public, actor, history, legal_count):
    to_beat = public['to_beat']
    if to_beat:
        if to_beat['type'] == 'rocket':
            rule = 'Nothing beats the rocket, so {} must pass.'.format(actor)
        elif to_beat['type'] == 'bomb':
            rule = '{} must play a higher bomb or the rocket, or pass.'.format(actor)
        else:
            rule = '{} must play a higher {} of the same shape, a bomb or the rocket, or pass.'.format(
                actor, e(to_beat['type']))
        beat = 'To beat: {} by <b>{}</b>. {}'.format(move_html(to_beat['cards'], to_beat['type']), e(to_beat['by']),
                                                     rule)
    else:
        beat = 'Free lead: {} may play any legal combination.'.format(actor)
    next_player, other_player = public['hidden_seats']
    unseen = public['unseen']
    return ''.join([
        '<div class="center">', '<div class="label">Public table</div>',
        '<p><b>{}</b> ({}) to act, with {} legal moves.</p>'.format(e(actor), ROLE[actor], legal_count),
        '<p>{}</p>'.format(beat),
        '<p>Bombs and rockets so far: <b>{}</b>, so this game is worth 2^{} = {}.</p>'.format(
            public['bombs'], public['bombs'], 2 ** public['bombs']),
        '<p>Bottom cards the landlord must still hold: {} <span class="small muted">(everyone saw the three '
        'bottom cards; any not yet matched by a card of the same rank the landlord played are still in its hand. '
        'DouZero\'s observation does not include them, so neither the arms nor card counting use this public '
        'deduction.)</span></p>'.format(cards_html(public['three_landlord_cards'], 'public')),
        '<div class="label">Unseen cards from {}\'s point of view</div>'.format(e(actor)),
        '<div class="scroll"><table class="kinds">{}<tr>{}</tr></table></div>'.format(
            kinds_header(), '<th>copies</th>' + ''.join('<td>{}</td>'.format(n or '·') for n in unseen)),
        '<p class="small">54 cards minus its own hand and everything played: {} cards, which {} ({}) and {} ({}) '
        'hold between them. The two sizes are public; this count is the actor\'s own, because only it knows its '
        'hand; who holds which card is hidden.</p>'.format(sum(unseen), e(next_player), public['hidden_sizes'][0],
                                                           e(other_player), public['hidden_sizes'][1]),
        '<div class="label">Moves so far (public)</div>', history_html(history), '</div>'])


def table_html(actor, public, own_hand, history, legal_count, hands=None, beliefs=None, sources=None, ce=None):
    seats = {p: seat_html(p, actor, public, own_hand, hands, beliefs, sources, ce) for p in LAYOUT}
    return '<div class="table">{}{}{}{}</div>'.format(
        seats['landlord_up'], seats['landlord'], center_html(public, actor, history, legal_count),
        seats['landlord_down'])


def belief_table(rows, public):
    """rows: (label, css class, cells html) for a 15-kind table headed by the public unseen counts."""
    body = [kinds_header(),
            '<tr class="public"><th>unseen copies (actor\'s view)</th>{}</tr>'.format(
                count_cells(public['unseen'], public['unseen']))]
    body += ['<tr class="{}"><th>{}</th>{}</tr>'.format(e(css), label, cells) for label, css, cells in rows]
    return '<div class="scroll"><table class="kinds beliefs">{}</table></div>'.format(''.join(body))


# ---------------------------------------------------------------- one decision of the replay

def belief_panel(record, step, hidden_step):
    public = step['public']
    heading = '<h3>Belief input of {} (what its Q-network is told about the two hidden hands)</h3>'.format(
        e(record['arm']))
    if step['belief'] is None:
        reason = ('masked in normal mode: the Oracle\'s belief input is the true hidden hands'
                  if step.get('oracle_masked') else 'the original DouZero network has no belief input')
        return heading + '<p class="muted">None: {}.</p>'.format(reason)
    sources = record['slot_sources']
    rows = []
    for index, slot in enumerate(('next', 'other')):
        seat = public['hidden_seats'][index]
        rows.append(('<span class="group">{} hand: {}, {} cards</span>'.format(
            'next' if slot == 'next' else 'other', seat, public['hidden_sizes'][index]), 'group', '<td colspan="15"></td>'))
        rows.append(('card counting (reference)', 'counting', belief_cells(step['card_counting'][slot], public['unseen'])))
        rows.append(('belief: {}'.format(e(sources[index]['text'])), sources[index]['kind'],
                     belief_cells(step['belief'][slot], public['unseen'])))
        if hidden_step:
            rows.append(('true count (hidden)', 'truth', count_cells(hidden_step['true_counts'][slot], public['unseen'])))
    notes = ['<p class="small">Each cell is the expected number of copies of that card the hand holds; hover for '
             'the full distribution. Card counting assumes the hidden cards were dealt at random.</p>']
    if hidden_step:
        ce = hidden_step['cross_entropy']
        notes.append('<p class="small truth-text">Cross-entropy against the true hands, next / other, in nats '
                     '(lower is better; Stage 4\'s metric for one decision): belief {} / {}, card counting {} / {}, '
                     'uniform over the feasible counts {} / {}.</p>'.format(
                         fmt(ce['belief']['next']), fmt(ce['belief']['other']),
                         fmt(ce['card_counting']['next']), fmt(ce['card_counting']['other']),
                         fmt(ce['uniform']['next']), fmt(ce['uniform']['other'])))
    return heading + belief_table(rows, public) + ''.join(notes)


def choice_panel(step):
    selected = step['selected']
    if step['top_actions'] is None:  # the Oracle in normal mode: its Q-values come from the true hands
        return ('<h3>Choice among {} legal moves</h3><p>It played {} (public).</p><p class="muted">The Q-values and '
                'ranking are masked in normal mode: the Oracle scores moves with the true hidden hands.</p>').format(
                    step['legal_count'], move_html(selected['cards'], selected['type']))
    rows = []
    for rank, action in enumerate(step['top_actions'], 1):
        rows.append('<tr{}><td>{}</td><td>{}</td><td>{:+.3f}</td></tr>'.format(
            ' class="sel"' if rank == selected['rank'] else '', rank, move_html(action['cards'], action['type']),
            action['q']))
    if selected['rank'] > len(step['top_actions']):
        rows.append('<tr class="gap"><td>…</td><td></td><td></td></tr><tr class="sel"><td>{}</td><td>{}</td>'
                    '<td>{:+.3f}</td></tr>'.format(selected['rank'], move_html(selected['cards'], selected['type']),
                                                   selected['q']))
    exploration = step['exploration']
    if exploration['draw'] is None:
        how = 'Epsilon is 0, so the move is always the highest-Q one (greedy).'
    elif exploration['explored']:
        how = ('Exploration: drew {:.3f} &lt; ε = {}, so the move was drawn uniformly at random from the {} legal '
               'moves, ignoring Q. {}'.format(
                   exploration['draw'], exploration['epsilon'], step['legal_count'],
                   'This time the draw landed on the best-ranked move anyway.' if selected['rank'] == 1 else
                   'It is ranked {} of {} by Q, so this is a deliberately non-greedy move.'.format(
                       selected['rank'], step['legal_count'])))
    else:
        how = 'Drew {:.3f} ≥ ε = {}, so the move is greedy: the highest-Q one.'.format(
            exploration['draw'], exploration['epsilon'])
    if step['legal_count'] == 1:
        how += ' (Only one legal move here, so the choice was forced.)'
    return ''.join([
        '<h3>Choice: top {} of {} legal moves by Q</h3>'.format(len(step['top_actions']), step['legal_count']),
        '<table class="q"><tr><th>rank</th><th>move</th><th>Q</th></tr>{}</table>'.format(''.join(rows)),
        '<p class="small">Q is the network\'s estimate of this seat\'s final return if it makes the move '
        '(training units, ±2^bombs). The highlighted row was played.</p>',
        '<p class="{}">{}</p>'.format('explore' if exploration['explored'] else '', how)])


def transition_html(record, t):
    steps = record['steps']
    step = steps[t]
    actor, public, action = step['seat'], step['public'], step['action']
    to_beat = public['to_beat']
    if not action:
        what = '<b>{}</b> passes'.format(e(actor))
        if to_beat:
            what += ' rather than beat {} from {}'.format(move_html(to_beat['cards'], to_beat['type']),
                                                          e(to_beat['by']))
    else:
        what = '<b>{}</b> plays {}'.format(e(actor), move_html(action, step['selected']['type']))
        what += ', beating {} from {}'.format(move_html(to_beat['cards'], to_beat['type']), e(to_beat['by'])) \
            if to_beat else ', leading a new trick'
        what += ' (cards left {} → {})'.format(public['cards_left'][actor], public['cards_left'][actor] - len(action))
        if step['selected']['type'] in ('bomb', 'rocket'):
            what += '. That doubles the stakes'
    if t + 1 < len(steps):
        following = steps[t + 1]
        if following['public']['to_beat'] is None:
            after = 'The trick is over: <b>{}</b> leads next.'.format(e(following['seat']))
        else:
            after = 'Next: <b>{}</b> must beat it or pass.'.format(e(following['seat']))
    else:
        after = 'That was {}\'s last card: the game ends.'.format(e(actor))
    return '<p class="transition">After this decision: {}. {}</p>'.format(what, after)


def step_html(record, t, hidden):
    step = record['steps'][t]
    hidden_step = hidden['steps'][t] if hidden else None
    history = [dict(seat=s['seat'], action=s['action'], type=s['selected']['type']) for s in record['steps'][:t]]
    return ''.join([
        '<section class="step" id="t{}">'.format(t),
        '<h2>Decision {} of {} <span class="muted">(t = {})</span>: {} to act</h2>'.format(
            t + 1, len(record['steps']), t, seat_name(step['seat'])),
        table_html(step['seat'], step['public'], step['own_hand'], history, step['legal_count'],
                   hidden_step['hands'] if hidden_step else None, step['belief'], record['slot_sources'],
                   hidden_step['cross_entropy'] if hidden_step else None),
        '<div class="decision"><div class="panel">{}</div><div class="panel">{}</div></div>'.format(
            belief_panel(record, step, hidden_step), choice_panel(step)),
        transition_html(record, t), '</section>'])


def end_html(record, hidden):
    result = record['result']
    returns = result['returns']
    rows = ''.join('<tr><td>{}</td><td>{:+g}</td><td>{}</td></tr>'.format(
        seat_name(p), returns[p], result['decisions_per_seat'][p]) for p in ('landlord', 'landlord_down', 'landlord_up'))
    winner = 'The landlord wins.' if result['winner'] == 'landlord' else 'The peasants win.'
    parts = ['<section class="step" id="end"><h2>Game over</h2>',
             '<p><b>{}</b> Bombs and rockets played: {}, so the stake is 2^{} = {}.</p>'.format(
                 winner, result['bombs'], result['bombs'], 2 ** result['bombs']),
             '<table class="q"><tr><th>seat</th><th>return G</th><th>decisions</th></tr>{}</table>'.format(rows),
             '<p class="small">These are the training returns of the ADP objective: the landlord gets ±2^bombs and '
             'each peasant the opposite. Every decision a seat made is trained toward that seat\'s G: a Monte Carlo '
             'target, with no discount and no bootstrapping. Evaluation reports ADP in other units, where the '
             'landlord side scores ±2·2^bombs, so state the units with every number.</p>']
    if hidden:
        parts.append('<div class="label">Cards left in each hand at the end (hidden during play)</div><ul>{}</ul>'.format(
            ''.join('<li>{}: {}</li>'.format(seat_name(p), cards_html(hidden['final_hands'][p], 'hidden'))
                    for p in ('landlord', 'landlord_down', 'landlord_up'))))
    parts.append('</section>')
    return ''.join(parts)


# ---------------------------------------------------------------- pages

def header_html(title, facts, reveal):
    mode = ('<span class="badge teaching">TEACHING mode: hidden hands revealed</span>' if reveal else
            '<span class="badge normal">NORMAL mode: only what the acting player knows</span>')
    rows = ''.join('<tr><th>{}</th><td{}>{}</td></tr>'.format(label, ' class="warn"' if 'UNTRAINED' in str(value) else '',
                                                           value) for label, value in facts)
    legend = ('<span class="cards own"><span class="card">K</span></span> the actor\'s own hand '
              '<span class="cards public"><span class="card">K</span></span> public '
              '<span class="cards hidden"><span class="card">K</span></span> hidden truth (teaching only) '
              '<span class="chip predicted">predicted</span> <span class="chip derived">derived by conservation</span> '
              '<span class="chip learned">learned</span> <span class="chip oracle">oracle</span> '
              '<span class="chip zero">zero</span> <span class="muted">· BJ/RJ = black/red joker</span>')
    terms = ('<b>Terms.</b> Slot 1 is the belief about the <i>next</i> hidden player (the one who plays right after '
             'the actor), slot 2 about the <i>other</i>. E[count] is the expected number of copies of a card in a '
             'hand. Q is the network\'s estimate of the final return of a move. ε is the exploration rate. CE is '
             'cross-entropy, the average of −log P(true count) over the 15 kinds, so lower is better; a dash means '
             'undefined: a zero slot gives the truth no probability.')
    return ('<header><h1>{}</h1>{}<table class="facts">{}</table><p class="legend">{}</p>'
            '<p class="legend">{}</p></header>').format(e(title), mode, rows, legend, terms)


def nav_html(record):
    steps = record['steps']
    marks = ''.join('<a class="tick {}{}" href="#t{}" title="t={} {}: {}"></a>'.format(
        s['seat'], ' explored' if s['exploration']['explored'] else '', s['t'], s['t'], s['seat'],
        e(s['selected']['type'])) for s in steps)
    return ('<nav class="controls"><button id="prev" type="button">◀ prev</button>'
            '<input id="slider" type="range" min="0" max="{}" value="0" aria-label="decision">'
            '<button id="next" type="button">next ▶</button> <span id="pos"></span>'
            '<label><input type="checkbox" id="all"> show every step</label>'
            '<span class="muted small">keys ← → · one tick per decision, coloured by seat; ringed ticks explored</span>'
            '<div class="ticks">{}<a class="tick end" href="#end" title="game over"></a></div></nav>').format(
                len(steps), marks)


def page(title, body, script=''):
    return ''.join(['<!doctype html>\n<html lang="en"><head><meta charset="utf-8">',
                    '<meta name="viewport" content="width=device-width, initial-scale=1">',
                    '<title>{}</title><style>{}</style></head><body>'.format(e(title), CSS), body,
                    '<script>{}</script>'.format(SCRIPT) if script else '', '</body></html>\n'])


def render_replay(record, reveal=None):
    """The whole game as one HTML page: a section per decision plus the end of the game."""
    reveal = record['revealed'] if reveal is None else reveal
    if reveal and 'hidden' not in record:
        raise ValueError('this record holds no hidden information to reveal; record it with reveal=True')
    if not reveal and record['revealed']:
        record = strip_hidden(record)  # a NORMAL page reads only what a NORMAL record holds
    hidden = record['hidden'] if reveal else None
    sources = record['slot_sources']
    arm = '{} (--hand_model {}), playing all three seats (self-play)'.format(record['arm'], record['hand_model'])
    facts = [('game', e(record['game_id'])), ('arm', e(arm)),
             ('belief slots', ('slot 1 (next hand): {} · slot 2 (other hand): {}'.format(
                 source_chip(sources[0]), source_chip(sources[1])) if sources else 'no belief input')),
             ('Q-network', e(record['q_network'])), ('predictors', e(record['predictors'])),
             ('exploration', 'ε = {} (the training actors\' rate)'.format(e(record['exp_epsilon'])))]
    body = header_html('Episode replay: {} seed {}'.format(record['arm'], record['seed']), facts, reveal)
    body += nav_html(record) + '<main>{}{}</main>'.format(
        ''.join(step_html(record, t, hidden) for t in range(len(record['steps']))), end_html(record, hidden))
    return page('Replay {} seed {}'.format(record['arm'], record['seed']), body, script=SCRIPT)


def render_comparison(comparison, reveal=None):
    """One decision point as each of the five arms sees and scores it."""
    reveal = comparison['revealed'] if reveal is None else reveal
    if reveal and 'hidden' not in comparison:
        raise ValueError('this comparison holds no hidden information to reveal; make it with reveal=True')
    if not reveal and comparison['revealed']:
        comparison = strip_comparison(comparison)
    hidden = comparison['hidden'] if reveal else None
    state, public = comparison['state'], comparison['state']['public']
    arms = {arm['hand_model']: arm for arm in comparison['arms']}
    facts = [('state', 'game {}, decision t = {}, from a game played by {}'.format(
                 e(comparison['game_id']), comparison['t'], e(comparison['played_by']))),
             ('Q-networks', e(comparison['q_network'])),
             ('K=1 predictor (M1 and M2c share it)', e(comparison['predictors']['k1'])),
             ('K=2 predictor (M2)', e(comparison['predictors']['k2']))]
    body = [header_html('State comparison: one decision, five arms', facts, reveal),
            '<main class="comparison">',
            table_html(state['seat'], public, state['own_hand'], state['history'], state['legal_count'],
                       hidden['hands'] if hidden else None)]
    for index, slot in enumerate(('next', 'other')):
        rows = [('card counting (reference)', 'counting', belief_cells(state['card_counting'][slot], public['unseen']))]
        for hand_model in ARM_ORDER:
            arm, source = arms[hand_model], arms[hand_model]['slot_sources'][index]
            label = '<b>{}</b> {}'.format(e(arm['arm']), source_chip(source))
            if arm['belief'] is None:
                cells = '<td colspan="15" class="muted">masked in normal mode: it is the true hidden hand</td>'
            else:
                cells = belief_cells(arm['belief'][slot], public['unseen'])
            rows.append((label, source['kind'], cells))
        if hidden:
            rows.append(('true count (hidden)', 'truth', count_cells(hidden['true_counts'][slot], public['unseen'])))
        body.append('<div class="panel"><h3>Slot {}: belief about the {} hand ({}, {} cards)</h3>{}</div>'.format(
            index + 1, slot, public['hidden_seats'][index], public['hidden_sizes'][index], belief_table(rows, public)))
    # Ranks down, arms across, so a difference between two arms' rankings is visible side by side.
    masked = '<td class="muted">masked: scored with the true hands</td>'
    q_rows = ['<tr><th>rank</th>{}</tr>'.format(''.join('<th>{}</th>'.format(e(arms[h]['arm'])) for h in ARM_ORDER))]
    for rank in range(max(len(arms[h]['top_actions'] or []) for h in ARM_ORDER)):
        cells = []
        for hand_model in ARM_ORDER:
            top = arms[hand_model]['top_actions']
            if top is None:
                cells.append(masked)
            elif rank < len(top):
                cells.append('<td>{} <span class="q-value">{:+.3f}</span></td>'.format(
                    move_html(top[rank]['cards'], top[rank]['type']), top[rank]['q']))
            else:
                cells.append('<td></td>')
        q_rows.append('<tr><th>{}</th>{}</tr>'.format(rank + 1, ''.join(cells)))
    q_rows.append('<tr class="sel"><th>greedy choice</th>{}</tr>'.format(''.join(
        masked if arms[h]['greedy_action'] is None else
        '<td>{}</td>'.format(move_html(arms[h]['greedy_action']['cards'], arms[h]['greedy_action']['type']))
        for h in ARM_ORDER)))
    if hidden:
        q_rows.append('<tr class="truth-text"><th>CE next / other (nats)</th>{}</tr>'.format(''.join(
            '<td>{} / {}</td>'.format(fmt(hidden['cross_entropy']['by_arm'][h]['next']),
                                      fmt(hidden['cross_entropy']['by_arm'][h]['other'])) for h in ARM_ORDER)))
    same = ('M0c, M1, M2c and M2 rank the legal moves identically here. They all use the same Q-network ({}), '
            'whose belief weights are zero, so the belief cannot change Q yet. Rankings start to differ only after '
            'RL training, with a trained checkpoint per arm.'.format(e(comparison['q_network']))
            if comparison['identical_rankings'] else
            'The public arms rank the legal moves differently: their beliefs, through trained belief weights, change Q.')
    if hidden:
        same += ' The Oracle ranks them {} the others.'.format(
            'the same as' if hidden['oracle_ranks_identically'] else 'differently from')
    body.append('<div class="panel"><h3>How each arm scores the {} legal moves (greedy, ε = 0)</h3>'
                '<div class="scroll"><table class="q arms">{}</table></div><p>{}</p></div>'.format(
                    state['legal_count'], ''.join(q_rows), same))
    body.append('</main>')
    return page('State comparison {} t={}'.format(comparison['game_id'], comparison['t']), ''.join(body))


CSS = """
:root{--bg:#f7f6f2;--panel:#fff;--ink:#1d1f23;--muted:#5f6670;--line:#d9dce1;--own:#1f7a4d;--public:#2a5db0;
--hidden:#b4581d;--predicted:#6b46b8;--derived:#0f7f86;--learned:#a23b72;--oracle:#c0392b;--zero:#8a9099;
--sel:#f6e3a1;--seat-landlord:#c0392b;--seat-landlord_down:#2a5db0;--seat-landlord_up:#1f7a4d}
@media (prefers-color-scheme:dark){:root{--bg:#15171a;--panel:#1e2125;--ink:#e8e9eb;--muted:#a0a6ae;--line:#3a3f46;
--own:#5cc28f;--public:#7aa7f0;--hidden:#f0a060;--predicted:#b195f0;--derived:#5cc8ce;--learned:#e07fb0;
--oracle:#f0857a;--zero:#80868e;--sel:#4d4420}}
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--bg);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
h1{font-size:20px;margin:0 0 6px}h2{font-size:17px;margin:18px 0 8px}h3{font-size:14px;margin:0 0 6px}
.muted{color:var(--muted)}.small{font-size:12px}.label{margin-top:8px;font-size:11px;text-transform:uppercase;
letter-spacing:.04em;color:var(--muted)}p{margin:6px 0}
header .facts th{text-align:left;padding-right:10px;color:var(--muted);font-weight:500;vertical-align:top}
.warn{color:var(--oracle);font-weight:600}.legend{font-size:12px}
.badge{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;font-weight:600;border:1px solid}
.badge.acting{color:var(--own);border-color:var(--own)}.badge.teaching{color:var(--hidden);border-color:var(--hidden)}
.badge.normal{color:var(--public);border-color:var(--public)}
.controls{position:sticky;top:0;z-index:2;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);
display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.controls button{font:inherit;padding:3px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel);
color:var(--ink);cursor:pointer}.controls input[type=range]{flex:1 1 200px}
.ticks{flex:1 1 100%;display:flex;flex-wrap:wrap;gap:2px}.tick{width:9px;height:14px;border-radius:2px;
background:var(--line)}.tick.landlord{background:var(--seat-landlord)}.tick.landlord_down{background:var(--seat-landlord_down)}
.tick.landlord_up{background:var(--seat-landlord_up)}.tick.explored{outline:2px solid var(--hidden);outline-offset:1px}
.tick.end{background:var(--ink)}.tick.current{transform:scaleY(1.5)}
.table{display:grid;grid-template-columns:1fr 1.35fr 1fr;grid-template-areas:". up ." "left center right";gap:10px}
.seat,.center,.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;min-width:0}
.seat.landlord_up{grid-area:up}.seat.landlord{grid-area:left}.seat.landlord_down{grid-area:right}.center{grid-area:center}
.seat.acting{border:2px solid var(--own)}.seat-key{font:12px ui-monospace,Menlo,monospace;color:var(--muted)}
.note{font-size:11px;color:var(--muted)}.hidden-note,.truth-text{color:var(--hidden)}
.cards{display:inline-flex;flex-wrap:wrap;gap:2px}.cards.empty{color:var(--muted);font-size:12px}
.card{display:inline-block;min-width:22px;padding:1px 3px;border:1px solid var(--line);border-radius:4px;
background:var(--panel);text-align:center;font:600 13px/1.3 ui-monospace,Menlo,monospace}
.card.joker{color:var(--oracle)}.cards.own .card{border-color:var(--own);border-width:2px}
.cards.public .card{border-color:var(--public)}.cards.hidden .card{border:2px dashed var(--hidden);color:var(--hidden)}
.card.back{min-width:10px;height:18px;padding:0;background:repeating-linear-gradient(45deg,var(--line) 0 3px,
transparent 3px 6px)}.pass{font-style:italic;color:var(--muted)}
.chip{display:inline-block;padding:0 6px;border-radius:9px;font-size:11px;border:1px solid currentColor}
.chip.predicted{color:var(--predicted)}.chip.derived{color:var(--derived)}.chip.learned{color:var(--learned)}
.chip.oracle{color:var(--oracle)}.chip.zero{color:var(--zero)}
.decision{display:grid;grid-template-columns:1.6fr 1fr;gap:10px;margin-top:10px}
table{border-collapse:collapse}.kinds td,.kinds th{padding:2px 4px;text-align:center;font-size:12px;
border-bottom:1px solid var(--line)}.kinds th:first-child{text-align:left;font-weight:500;white-space:nowrap}
.beliefs td{background:color-mix(in srgb,var(--c,var(--public)) calc(var(--v,0)*55%),transparent)}
.beliefs tr.counting{--c:var(--public)}.beliefs tr.predicted{--c:var(--predicted)}.beliefs tr.derived{--c:var(--derived)}
.beliefs tr.learned{--c:var(--learned)}.beliefs tr.oracle{--c:var(--oracle)}.beliefs tr.zero{--c:var(--zero)}
.beliefs tr.truth td,.beliefs tr.truth th{color:var(--hidden);font-weight:600}
.beliefs tr.group th{padding-top:8px;text-transform:uppercase;font-size:11px;color:var(--muted)}
.q td,.q th{padding:3px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
.q tr.sel td{background:var(--sel)}.explore{color:var(--hidden);font-weight:600}
.transition{border-left:3px solid var(--public);padding-left:8px}
ol.history{margin:4px 0;padding-left:26px}ol.top{margin:0;padding-left:18px}.q-value{color:var(--muted)}
.comparison .panel{margin-top:10px}
body.paged section.step[hidden]{display:none}section.step{scroll-margin-top:80px}.scroll{overflow-x:auto}
body:not(.paged) .controls>:not(.ticks){display:none}
@media (max-width:760px){.table{grid-template-columns:1fr;grid-template-areas:"up" "left" "center" "right"}
.decision{grid-template-columns:1fr}.kinds{display:block;overflow-x:auto}}
"""

SCRIPT = """
(function(){
  var steps=[].slice.call(document.querySelectorAll('section.step'));
  if(!steps.length)return;
  var slider=document.getElementById('slider'),pos=document.getElementById('pos'),all=document.getElementById('all');
  var ticks=[].slice.call(document.querySelectorAll('.tick')),i=0;
  document.body.classList.add('paged');
  function show(k,scroll){
    i=Math.max(0,Math.min(steps.length-1,k));
    steps.forEach(function(s,j){s.hidden=!all.checked&&j!==i;});
    ticks.forEach(function(t,j){t.classList.toggle('current',j===i);});
    slider.value=i;pos.textContent=(i===steps.length-1?'end':'decision '+(i+1)+' of '+(steps.length-1));
    if(history.replaceState)history.replaceState(null,'','#'+steps[i].id);
    if(scroll)steps[i].scrollIntoView({block:'start'});
  }
  document.getElementById('prev').onclick=function(){show(i-1,true);};
  document.getElementById('next').onclick=function(){show(i+1,true);};
  slider.oninput=function(){show(+slider.value,true);};
  all.onchange=function(){show(i,true);};
  ticks.forEach(function(t,j){t.onclick=function(ev){ev.preventDefault();show(j,true);};});
  document.addEventListener('keydown',function(ev){
    if(ev.target.tagName==='INPUT'&&ev.target.type!=='checkbox')return;
    if(ev.key==='ArrowLeft')show(i-1,true);if(ev.key==='ArrowRight')show(i+1,true);});
  var m=location.hash.match(/^#t(\\d+)$/);show(location.hash==='#end'?steps.length-1:(m?+m[1]:0));
})();
"""
