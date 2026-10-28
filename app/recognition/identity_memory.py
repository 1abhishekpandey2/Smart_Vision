from collections import Counter, deque
from dataclasses import dataclass, field


@dataclass
class _TrackIdentityState:
    last_seen_seconds: float
    evidence: deque[tuple[float, str]] = field(default_factory=deque)
    confirmed_identity: str | None = None


class TrackIdentityMemory:
    """Confirm and retain face identities for currently tracked people."""

    def __init__(
        self,
        required_observations: int = 3,
        evidence_window_seconds: float = 2.0,
        missing_track_ttl_seconds: float = 2.0,
    ) -> None:
        if required_observations < 1:
            raise ValueError("required_observations must be at least 1")
        if evidence_window_seconds <= 0.0:
            raise ValueError("evidence_window_seconds must be positive")
        if missing_track_ttl_seconds <= 0.0:
            raise ValueError("missing_track_ttl_seconds must be positive")

        self._required_observations = required_observations
        self._evidence_window_seconds = evidence_window_seconds
        self._missing_track_ttl_seconds = missing_track_ttl_seconds
        self._tracks: dict[int, _TrackIdentityState] = {}

    @property
    def required_observations(self) -> int:
        return self._required_observations

    def update(
        self,
        now_seconds: float,
        active_track_ids: set[int],
        strong_identities: dict[int, str],
    ) -> dict[int, str]:
        """Update state and return identities for tracks visible this frame.

        Only identities supplied as strong evidence count toward confirmation.
        A confirmed name is retained while its track remains active, including
        frames where no face is detected. Missing tracks expire after the TTL.
        """

        expired_track_ids = [
            track_id
            for track_id, state in self._tracks.items()
            if (
                track_id not in active_track_ids
                and now_seconds - state.last_seen_seconds
                > self._missing_track_ttl_seconds
            )
        ]

        for track_id in expired_track_ids:
            del self._tracks[track_id]

        for track_id in active_track_ids:
            state = self._tracks.setdefault(
                track_id,
                _TrackIdentityState(last_seen_seconds=now_seconds),
            )
            state.last_seen_seconds = now_seconds

            oldest_valid_time = now_seconds - self._evidence_window_seconds
            while state.evidence and state.evidence[0][0] < oldest_valid_time:
                state.evidence.popleft()

            identity = strong_identities.get(track_id)
            if identity and (
                not state.evidence or state.evidence[-1][0] != now_seconds
            ):
                state.evidence.append((now_seconds, identity))

            if state.confirmed_identity is None:
                counts = Counter(
                    evidence_identity for _, evidence_identity in state.evidence
                )
                qualifying = [
                    evidence_identity
                    for evidence_identity, count in counts.items()
                    if count >= self._required_observations
                ]

                # Conflicting qualifying identities leave the track unknown.
                if len(qualifying) == 1:
                    state.confirmed_identity = qualifying[0]
                    state.evidence.clear()

        return {
            track_id: self._tracks[track_id].confirmed_identity
            for track_id in active_track_ids
            if self._tracks[track_id].confirmed_identity is not None
        }

    def confirmation_progress(
        self,
        track_id: int,
        identity: str,
    ) -> tuple[int, int] | None:
        """Return pending strong-observation count and required count."""

        state = self._tracks.get(track_id)
        if state is None or state.confirmed_identity is not None:
            return None

        observed = sum(
            evidence_identity == identity for _, evidence_identity in state.evidence
        )
        return observed, self._required_observations

    def reset(self) -> None:
        self._tracks.clear()
