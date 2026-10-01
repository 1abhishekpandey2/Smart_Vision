from dataclasses import dataclass
from datetime import datetime
from typing import Any

Point = tuple[float, float]
BoundingBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class VisionSettings:
    fire_enabled: bool
    person_enabled: bool
    intrusion_enabled: bool

    loitering_enabled: bool
    dwell_seconds: int

    show_track_paths: bool

    fire_confidence: float
    person_confidence: float


@dataclass(frozen=True)
class VisionRequest:
    frame: Any
    settings: VisionSettings

    # Normalized coordinates:
    # 0.0 <= x <= 1.0
    # 0.0 <= y <= 1.0
    zone_points: tuple[Point, ...]
    source_time_seconds: float
    generation: int


@dataclass(frozen=True)
class OverlayBox:
    bounding_box: BoundingBox
    label: str
    kind: str
    alert: bool = False


@dataclass(frozen=True)
class TrackPath:
    track_id: int

    # These are original video-frame pixel coordinates.
    points: tuple[tuple[int, int], ...]

    alert: bool = False


@dataclass(frozen=True)
class VisionEvent:
    timestamp: datetime
    event_type: str
    description: str


@dataclass(frozen=True)
class VisionResult:
    generation: int

    overlays: tuple[OverlayBox, ...]

    track_paths: tuple[TrackPath, ...]

    people_count: int
    inside_zone_count: int
    loitering_count: int

    fire_evidence: str
    fire_confirmed: bool

    zone_active: bool

    events: tuple[VisionEvent, ...]
