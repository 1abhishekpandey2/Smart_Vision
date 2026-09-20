from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto

Point = tuple[int, int]


class IntrusionEventType(Enum):
    ENTERED = auto()
    EXITED = auto()


@dataclass(frozen=True)
class IntrusionEvent:
    track_id: int
    event_type: IntrusionEventType
    timestamp: datetime
    position: Point
