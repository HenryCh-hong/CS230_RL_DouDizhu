"""The one-decision trace (trace_decision.py) is compact and shows every stage of a decision."""
import trace_decision

SECTIONS = ('ACTING PLAYER', 'OBSERVATION', 'TRUE HIDDEN HANDS', 'PREDICTORS', 'BELIEF INPUT', 'Q-NETWORK',
            'Top 5 legal actions by Q', 'Epsilon-greedy')


def test_trace_shows_every_stage_of_one_decision_compactly():
    text = trace_decision.trace_decision(seed=7, num_random_moves=12)
    for section in SECTIONS:
        assert section in text, section
    for arm in ('M0c', 'M1', 'M2c', 'M2', 'Oracle'):
        assert arm in text, arm
    assert 'identical for all five arms: True' in text
    assert len(text.splitlines()) <= 60


def test_trace_runs_without_checkpoints_from_the_first_move():
    text = trace_decision.trace_decision(seed=3, num_random_moves=0, checkpoint_dir=None)
    assert 'no checkpoint' in text and 'identical for all five arms: True' in text
