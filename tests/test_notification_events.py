from datetime import datetime

import numpy as np

from app.camera.frame import Frame
from app.detection.detection import Detection
from app.tracking.track import Track
from app.ui.vision_types import VisionRequest, VisionSettings
from app.ui.workers.vision_worker import VisionWorker


class FakeFireDetector:
    def __init__(self, detections=()):
        self.detections = list(detections)

    def detect(self, frame):
        return self.detections


class FakePersonTracker:
    def __init__(self, tracks=()):
        self.tracks = list(tracks)

    def track(self, frame):
        return self.tracks


def make_request(
    *,
    source_time_seconds,
    source_id="recording.mp4",
    fire_enabled=False,
    person_enabled=False,
    intrusion_enabled=False,
    loitering_enabled=False,
    dwell_seconds=15,
    zone_points=(),
):
    return VisionRequest(
        frame=Frame(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            timestamp=datetime(2026, 11, 14, 12, 0, 0),
            camera_id="test-camera",
        ),
        settings=VisionSettings(
            fire_enabled=fire_enabled,
            person_enabled=person_enabled,
            face_recognition_enabled=False,
            identity_confirmation_observations=3,
            intrusion_enabled=intrusion_enabled,
            loitering_enabled=loitering_enabled,
            dwell_seconds=dwell_seconds,
            show_track_paths=False,
            fire_confidence=0.4,
            person_confidence=0.35,
        ),
        zone_points=zone_points,
        source_time_seconds=source_time_seconds,
        generation=1,
        source_id=source_id,
    )


def test_smoke_events_are_emitted_on_detection_edges_with_video_time():
    worker = VisionWorker()
    worker._fire_detector = FakeFireDetector(
        [Detection(1, "smoke", 0.9, (10, 10, 20, 20))]
    )
    results = []
    worker.result_ready.connect(results.append)

    def process_at(seconds):
        worker.process(make_request(source_time_seconds=seconds, fire_enabled=True))

    process_at(1.25)
    process_at(1.50)
    worker._fire_detector.detections = []
    process_at(1.75)

    assert [event.description for event in results[0].events] == ["Smoke detected"]
    assert results[0].events[0].event_type == "SMOKE"
    assert results[0].events[0].source_time_seconds == 1.25
    assert results[0].events[0].source_id == "recording.mp4"
    assert results[1].events == ()
    assert [event.description for event in results[2].events] == [
        "Smoke no longer detected"
    ]
    assert results[2].events[0].source_time_seconds == 1.75


def test_fire_events_use_video_offsets_at_confirmation_and_clear():
    worker = VisionWorker()
    worker._fire_detector = FakeFireDetector(
        [Detection(0, "fire", 0.9, (10, 10, 20, 20))]
    )
    results = []
    worker.result_ready.connect(results.append)

    for index in range(10):
        worker.process(make_request(source_time_seconds=index / 10, fire_enabled=True))

    confirmed = [event for result in results for event in result.events]
    assert len(confirmed) == 1
    assert confirmed[0].description == "Fire confirmed"
    assert confirmed[0].source_time_seconds == 0.9
    assert confirmed[0].source_id == "recording.mp4"

    worker._fire_detector.detections = []
    for index in range(4):
        worker.process(
            make_request(source_time_seconds=1.0 + index / 10, fire_enabled=True)
        )

    events = [event for result in results for event in result.events]
    assert [event.description for event in events] == [
        "Fire confirmed",
        "Fire cleared",
    ]
    assert events[-1].source_time_seconds == 1.3


def test_intrusion_and_loitering_events_have_the_trigger_frame_offset():
    worker = VisionWorker()
    outside = Track(4, 0, "person", 0.9, (40, 75, 60, 99))
    inside = Track(4, 0, "person", 0.9, (40, 55, 60, 75))
    worker._person_tracker = FakePersonTracker([outside])
    results = []
    worker.result_ready.connect(results.append)
    zone = ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))

    def process_at(seconds):
        worker.process(
            make_request(
                source_time_seconds=seconds,
                person_enabled=True,
                intrusion_enabled=True,
                loitering_enabled=True,
                dwell_seconds=1,
                zone_points=zone,
            )
        )

    process_at(0.0)
    worker._person_tracker.tracks = [inside]
    process_at(1.0)
    process_at(2.0)
    process_at(3.1)
    worker._person_tracker.tracks = [outside]
    process_at(4.1)
    process_at(5.1)

    events = [event for result in results for event in result.events]
    assert [event.event_type for event in events] == [
        "INTRUSION",
        "LOITERING",
        "EXIT",
    ]
    assert [event.source_time_seconds for event in events] == [2.0, 3.1, 5.1]
    assert all(event.source_id == "recording.mp4" for event in events)
