from datetime import datetime

import numpy as np

from app.camera.frame import Frame
from app.recognition.association import FaceObservation
from app.recognition.registry import FaceMatch, MatchState
from app.tracking.track import Track
from app.ui.vision_types import VisionRequest, VisionSettings
from app.ui.workers.vision_worker import VisionWorker


class FakePersonTracker:
    def __init__(self, tracks):
        self.tracks = tracks

    def track(self, frame):
        return self.tracks


class FakeFaceEmbedder:
    def __init__(self, observations):
        self.observations = observations

    def detect_faces(self, image):
        return self.observations


class FakeFaceRegistry:
    def match(self, embedding):
        return FaceMatch(
            identity="alice",
            similarity=0.72,
            second_identity="bob",
            second_similarity=0.18,
            margin=0.54,
            state=MatchState.STRONG_MATCH,
        )


def make_request(
    face_recognition_enabled: bool,
    source_time_seconds: float = 0.0,
    identity_confirmation_observations: int = 3,
) -> VisionRequest:
    return VisionRequest(
        frame=Frame(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            timestamp=datetime(2026, 10, 28),
            camera_id="test-camera",
        ),
        settings=VisionSettings(
            fire_enabled=False,
            person_enabled=True,
            face_recognition_enabled=face_recognition_enabled,
            identity_confirmation_observations=(identity_confirmation_observations),
            intrusion_enabled=False,
            loitering_enabled=False,
            dwell_seconds=15,
            show_track_paths=False,
            fire_confidence=0.4,
            person_confidence=0.35,
        ),
        zone_points=(),
        source_time_seconds=source_time_seconds,
        generation=1,
    )


def test_worker_adds_strong_face_match_to_associated_track_label():
    worker = VisionWorker()
    worker._person_tracker = FakePersonTracker(
        [
            Track(
                track_id=12,
                class_id=0,
                class_name="person",
                confidence=0.9,
                bounding_box=(0, 0, 99, 99),
            )
        ]
    )
    worker._face_embedder = FakeFaceEmbedder(
        [
            FaceObservation(
                bounding_box=(40.0, 10.0, 60.0, 30.0),
                embedding=np.ones(512, dtype=np.float32),
            )
        ]
    )
    worker._face_registry = FakeFaceRegistry()

    results = []
    worker.result_ready.connect(results.append)

    worker.process(make_request(face_recognition_enabled=True))

    assert len(results) == 1
    assert results[0].overlays[0].label == (
        "Person #12 90% | Face: alice 0.72 | Confirming 1/3"
    )


def test_worker_does_not_add_face_label_when_recognition_is_disabled():
    worker = VisionWorker()
    worker._person_tracker = FakePersonTracker(
        [
            Track(
                track_id=12,
                class_id=0,
                class_name="person",
                confidence=0.9,
                bounding_box=(0, 0, 99, 99),
            )
        ]
    )

    results = []
    worker.result_ready.connect(results.append)

    worker.process(make_request(face_recognition_enabled=False))

    assert len(results) == 1
    assert results[0].overlays[0].label == "Person #12 90%"


def test_worker_keeps_confirmed_identity_after_face_disappears():
    worker = VisionWorker()
    worker._person_tracker = FakePersonTracker(
        [
            Track(
                track_id=12,
                class_id=0,
                class_name="person",
                confidence=0.9,
                bounding_box=(0, 0, 99, 99),
            )
        ]
    )
    embedder = FakeFaceEmbedder(
        [
            FaceObservation(
                bounding_box=(40.0, 10.0, 60.0, 30.0),
                embedding=np.ones(512, dtype=np.float32),
            )
        ]
    )
    worker._face_embedder = embedder
    worker._face_registry = FakeFaceRegistry()

    results = []
    worker.result_ready.connect(results.append)

    for now_seconds in (0.0, 0.5, 1.0):
        worker.process(
            make_request(
                face_recognition_enabled=True,
                source_time_seconds=now_seconds,
            )
        )

    assert results[-1].overlays[0].label == "Person #12 90% | Identity: alice"

    embedder.observations = []
    worker.process(
        make_request(
            face_recognition_enabled=True,
            source_time_seconds=1.5,
        )
    )

    assert results[-1].overlays[0].label == "Person #12 90% | Identity: alice"


def test_worker_uses_selected_identity_confirmation_count():
    worker = VisionWorker()
    worker._person_tracker = FakePersonTracker(
        [
            Track(
                track_id=12,
                class_id=0,
                class_name="person",
                confidence=0.9,
                bounding_box=(0, 0, 99, 99),
            )
        ]
    )
    worker._face_embedder = FakeFaceEmbedder(
        [
            FaceObservation(
                bounding_box=(40.0, 10.0, 60.0, 30.0),
                embedding=np.ones(512, dtype=np.float32),
            )
        ]
    )
    worker._face_registry = FakeFaceRegistry()

    results = []
    worker.result_ready.connect(results.append)

    worker.process(
        make_request(
            face_recognition_enabled=True,
            identity_confirmation_observations=1,
        )
    )

    assert results[0].overlays[0].label == "Person #12 90% | Identity: alice"
