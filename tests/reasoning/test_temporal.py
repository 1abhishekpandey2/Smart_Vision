import pytest

from app.reasoning.temporal import TemporalPersistence


def test_temporal_persistence_starts_unconfirmed():
    temporal = TemporalPersistence(
        window_size=5,
        required_positive_frames=3,
    )

    assert temporal.is_confirmed is False


def test_temporal_persistence_does_not_confirm_before_window_is_full():
    temporal = TemporalPersistence(
        window_size=5,
        required_positive_frames=3,
    )

    temporal.observe(True)
    temporal.observe(True)
    temporal.observe(True)

    assert temporal.is_confirmed is False


def test_temporal_persistence_confirms_when_threshold_is_reached():
    temporal = TemporalPersistence(
        window_size=5,
        required_positive_frames=3,
    )

    observations = [
        True,
        False,
        True,
        False,
        True,
    ]

    for observation in observations:
        temporal.observe(observation)

    assert temporal.is_confirmed is True


def test_temporal_persistence_does_not_confirm_below_threshold():
    temporal = TemporalPersistence(
        window_size=5,
        required_positive_frames=3,
    )

    observations = [
        True,
        False,
        True,
        False,
        False,
    ]

    for observation in observations:
        temporal.observe(observation)

    assert temporal.is_confirmed is False


def test_temporal_persistence_uses_sliding_window():
    temporal = TemporalPersistence(
        window_size=5,
        required_positive_frames=3,
    )

    for observation in [
        True,
        True,
        True,
        False,
        False,
    ]:
        temporal.observe(observation)

    assert temporal.is_confirmed is True

    temporal.observe(False)

    assert temporal.is_confirmed is False


def test_temporal_persistence_reset_clears_confirmation():
    temporal = TemporalPersistence(
        window_size=5,
        required_positive_frames=3,
    )

    for observation in [
        True,
        True,
        True,
        False,
        False,
    ]:
        temporal.observe(observation)

    assert temporal.is_confirmed is True

    temporal.reset()

    assert temporal.is_confirmed is False


@pytest.mark.parametrize(
    ("window_size", "required_positive_frames"),
    [
        (0, 1),
        (-1, 1),
        (5, 0),
        (5, -1),
        (5, 6),
    ],
)
def test_temporal_persistence_rejects_invalid_configuration(
    window_size,
    required_positive_frames,
):
    with pytest.raises(ValueError):
        TemporalPersistence(
            window_size=window_size,
            required_positive_frames=required_positive_frames,
        )


@pytest.mark.parametrize(
    ("window_size", "required_positive_frames"),
    [
        (True, 1),
        (5, True),
        (5.0, 3),
        (5, 3.0),
        ("5", 3),
        (5, "3"),
    ],
)
def test_temporal_persistence_rejects_invalid_types(
    window_size,
    required_positive_frames,
):
    with pytest.raises(TypeError):
        TemporalPersistence(
            window_size=window_size,
            required_positive_frames=required_positive_frames,
        )


def test_observe_rejects_non_boolean_values():
    temporal = TemporalPersistence(
        window_size=5,
        required_positive_frames=3,
    )

    with pytest.raises(TypeError):
        temporal.observe(1)

def test_positive_count_reports_current_positive_observations():
    temporal = TemporalPersistence(
        window_size=5,
        required_positive_frames=3,
    )

    temporal.observe(True)
    temporal.observe(False)
    temporal.observe(True)

    assert temporal.positive_count == 2


def test_observation_count_reports_current_window_population():
    temporal = TemporalPersistence(
        window_size=5,
        required_positive_frames=3,
    )

    temporal.observe(True)
    temporal.observe(False)

    assert temporal.observation_count == 2


def test_window_size_is_exposed_read_only():
    temporal = TemporalPersistence(
        window_size=5,
        required_positive_frames=3,
    )

    assert temporal.window_size == 5