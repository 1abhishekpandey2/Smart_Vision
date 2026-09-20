from collections import deque


class TemporalPersistence:
    """
    Tracks recent positive/negative observations and determines whether
    enough temporal evidence exists to confirm a condition.
    """

    def __init__(
        self,
        window_size: int,
        required_positive_frames: int,
    ):
        if not isinstance(window_size, int) or isinstance(window_size, bool):
            raise TypeError("window_size must be an integer")

        if not isinstance(required_positive_frames, int) or isinstance(
            required_positive_frames, bool
        ):
            raise TypeError("required_positive_frames must be an integer")

        if window_size <= 0:
            raise ValueError("window_size must be greater than 0")

        if required_positive_frames <= 0:
            raise ValueError("required_positive_frames must be greater than 0")

        if required_positive_frames > window_size:
            raise ValueError(
                "required_positive_frames cannot be greater than window_size"
            )

        self._window_size = window_size
        self._required_positive_frames = required_positive_frames

        self._history: deque[bool] = deque(maxlen=window_size)

    def observe(self, positive: bool) -> None:
        if not isinstance(positive, bool):
            raise TypeError("positive must be a boolean")

        self._history.append(positive)

    @property
    def is_confirmed(self) -> bool:
        if len(self._history) < self._window_size:
            return False

        return self.positive_count >= self._required_positive_frames

    @property
    def positive_count(self) -> int:
        return sum(self._history)

    @property
    def observation_count(self) -> int:
        return len(self._history)

    @property
    def window_size(self) -> int:
        return self._window_size

    def reset(self) -> None:
        self._history.clear()
