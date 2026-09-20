from collections import deque

from app.events.intrusion import IntrusionEvent


class IntrusionEventHistory:
    def __init__(self, max_events: int = 50):
        if not isinstance(max_events, int) or isinstance(max_events, bool):
            raise TypeError("max_events must be an integer")

        if max_events <= 0:
            raise ValueError("max_events must be greater than 0")

        self._events: deque[IntrusionEvent] = deque(maxlen=max_events)

    def add(self, event: IntrusionEvent) -> None:
        self._events.append(event)

    def recent(
        self,
        limit: int = 5,
    ) -> tuple[IntrusionEvent, ...]:
        if limit <= 0:
            return ()

        events = list(self._events)

        return tuple(reversed(events[-limit:]))

    def clear(self) -> None:
        self._events.clear()
