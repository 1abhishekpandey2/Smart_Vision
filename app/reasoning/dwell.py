from dataclasses import dataclass


@dataclass(frozen=True)
class DwellState:
    elapsed_seconds: float
    is_loitering: bool
    just_triggered: bool


class DwellTracker:
    def __init__(self):
        self._entered_at: dict[int, float] = {}
        self._triggered: set[int] = set()

    def observe(
        self,
        track_id: int,
        inside_zone: bool,
        now_seconds: float,
        threshold_seconds: float,
    ) -> DwellState:
        if threshold_seconds <= 0:
            raise ValueError("threshold_seconds must be greater than zero")

        if not inside_zone:
            self.reset_track(track_id)
            return DwellState(0.0, False, False)

        entered_at = self._entered_at.get(track_id)

        if entered_at is None or now_seconds < entered_at:
            self._entered_at[track_id] = now_seconds
            self._triggered.discard(track_id)
            return DwellState(0.0, False, False)

        elapsed = now_seconds - entered_at
        is_loitering = elapsed >= threshold_seconds
        just_triggered = is_loitering and track_id not in self._triggered

        if just_triggered:
            self._triggered.add(track_id)

        return DwellState(
            elapsed_seconds=elapsed,
            is_loitering=is_loitering,
            just_triggered=just_triggered,
        )

    def reset_track(self, track_id: int) -> None:
        self._entered_at.pop(track_id, None)
        self._triggered.discard(track_id)

    def reset(self) -> None:
        self._entered_at.clear()
        self._triggered.clear()
