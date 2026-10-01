import pytest

from app.reasoning.dwell import DwellTracker


def test_dwell_starts_when_track_enters_zone():
    tracker = DwellTracker()

    state = tracker.observe(
        track_id=4,
        inside_zone=True,
        now_seconds=10.0,
        threshold_seconds=15.0,
    )

    assert state.elapsed_seconds == 0.0
    assert not state.is_loitering
    assert not state.just_triggered


def test_dwell_accumulates_while_track_remains_inside():
    tracker = DwellTracker()

    tracker.observe(4, True, 10.0, 15.0)
    state = tracker.observe(4, True, 18.0, 15.0)

    assert state.elapsed_seconds == 8.0
    assert not state.is_loitering


def test_dwell_triggers_at_threshold():
    tracker = DwellTracker()

    tracker.observe(4, True, 10.0, 15.0)
    state = tracker.observe(4, True, 25.0, 15.0)

    assert state.elapsed_seconds == 15.0
    assert state.is_loitering
    assert state.just_triggered


def test_dwell_event_triggers_only_once_per_entry():
    tracker = DwellTracker()

    tracker.observe(4, True, 10.0, 15.0)
    tracker.observe(4, True, 25.0, 15.0)

    state = tracker.observe(4, True, 30.0, 15.0)

    assert state.is_loitering
    assert not state.just_triggered


def test_leaving_zone_resets_dwell():
    tracker = DwellTracker()

    tracker.observe(4, True, 10.0, 15.0)
    tracker.observe(4, True, 20.0, 15.0)
    tracker.observe(4, False, 21.0, 15.0)

    state = tracker.observe(4, True, 30.0, 15.0)

    assert state.elapsed_seconds == 0.0
    assert not state.is_loitering


def test_threshold_must_be_positive():
    tracker = DwellTracker()

    with pytest.raises(ValueError):
        tracker.observe(4, True, 10.0, 0.0)
