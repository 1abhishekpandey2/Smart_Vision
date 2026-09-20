from collections import deque
from math import hypot

Point = tuple[int, int]


class TrackHistoryStore:
    def __init__(
        self,
        max_points: int = 60,
        smoothing_alpha: float = 0.35,
        minimum_movement: float = 3.0,
    ):
        if not isinstance(max_points, int) or isinstance(max_points, bool):
            raise TypeError("max_points must be an integer")

        if max_points <= 0:
            raise ValueError("max_points must be greater than 0")

        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be between 0.0 and 1.0")

        if minimum_movement < 0:
            raise ValueError("minimum_movement cannot be negative")

        self._max_points = max_points
        self._smoothing_alpha = smoothing_alpha
        self._minimum_movement = minimum_movement

        self._histories: dict[
            int,
            deque[Point],
        ] = {}

        self._smoothed_positions: dict[
            int,
            tuple[float, float],
        ] = {}

    def update(
        self,
        track_id: int,
        point: Point,
    ) -> Point:
        raw_x, raw_y = point

        previous_smoothed = self._smoothed_positions.get(track_id)

        if previous_smoothed is None:
            smoothed_x = float(raw_x)
            smoothed_y = float(raw_y)

        else:
            previous_x, previous_y = previous_smoothed

            alpha = self._smoothing_alpha

            smoothed_x = alpha * raw_x + (1.0 - alpha) * previous_x

            smoothed_y = alpha * raw_y + (1.0 - alpha) * previous_y

        self._smoothed_positions[track_id] = (
            smoothed_x,
            smoothed_y,
        )

        smoothed_point = (
            round(smoothed_x),
            round(smoothed_y),
        )

        if track_id not in self._histories:
            self._histories[track_id] = deque(maxlen=self._max_points)

        history = self._histories[track_id]

        if not history:
            history.append(smoothed_point)
            return smoothed_point

        previous_point = history[-1]

        distance = hypot(
            smoothed_point[0] - previous_point[0],
            smoothed_point[1] - previous_point[1],
        )

        if distance >= self._minimum_movement:
            history.append(smoothed_point)

        return smoothed_point

    def get(
        self,
        track_id: int,
    ) -> tuple[Point, ...]:
        history = self._histories.get(track_id)

        if history is None:
            return ()

        return tuple(history)

    def clear_track(
        self,
        track_id: int,
    ) -> None:
        self._histories.pop(
            track_id,
            None,
        )

        self._smoothed_positions.pop(
            track_id,
            None,
        )

    def clear(self) -> None:
        self._histories.clear()
        self._smoothed_positions.clear()
