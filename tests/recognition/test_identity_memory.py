from app.recognition.identity_memory import TrackIdentityMemory


def test_confirms_identity_after_required_strong_observations():
    memory = TrackIdentityMemory()

    assert memory.update(0.0, {12}, {12: "alice"}) == {}
    assert memory.confirmation_progress(12, "alice") == (1, 3)
    assert memory.update(0.5, {12}, {12: "alice"}) == {}
    assert memory.confirmation_progress(12, "alice") == (2, 3)
    assert memory.update(1.0, {12}, {12: "alice"}) == {12: "alice"}
    assert memory.confirmation_progress(12, "alice") is None


def test_retains_confirmed_identity_while_track_is_active_without_face():
    memory = TrackIdentityMemory()
    memory.update(0.0, {12}, {12: "alice"})
    memory.update(0.5, {12}, {12: "alice"})
    memory.update(1.0, {12}, {12: "alice"})

    result = memory.update(1.5, {12}, {})

    assert result == {12: "alice"}


def test_unconfirmed_evidence_expires_outside_the_evidence_window():
    memory = TrackIdentityMemory()
    memory.update(0.0, {12}, {12: "alice"})
    memory.update(0.5, {12}, {12: "alice"})

    result = memory.update(2.1, {12}, {12: "alice"})

    assert result == {}


def test_expires_identity_after_track_is_missing_past_ttl():
    memory = TrackIdentityMemory()
    memory.update(0.0, {12}, {12: "alice"})
    memory.update(0.5, {12}, {12: "alice"})
    memory.update(1.0, {12}, {12: "alice"})
    memory.update(3.1, set(), {})

    result = memory.update(3.2, {12}, {})

    assert result == {}


def test_keeps_track_unknown_when_no_identity_reaches_quorum():
    memory = TrackIdentityMemory()

    observations = [
        (0.0, "alice"),
        (0.2, "alice"),
        (0.4, "bob"),
        (0.6, "bob"),
    ]

    result = {}
    for now_seconds, identity in observations:
        result = memory.update(now_seconds, {12}, {12: identity})

    assert result == {}


def test_does_not_switch_confirmed_identity_for_one_conflicting_match():
    memory = TrackIdentityMemory()
    memory.update(0.0, {12}, {12: "alice"})
    memory.update(0.5, {12}, {12: "alice"})
    memory.update(1.0, {12}, {12: "alice"})

    result = memory.update(1.5, {12}, {12: "bob"})

    assert result == {12: "alice"}


def test_reset_clears_pending_and_confirmed_state():
    memory = TrackIdentityMemory()
    memory.update(0.0, {12}, {12: "alice"})
    memory.update(0.5, {12}, {12: "alice"})
    memory.update(1.0, {12}, {12: "alice"})

    memory.reset()

    assert memory.update(1.1, {12}, {}) == {}
