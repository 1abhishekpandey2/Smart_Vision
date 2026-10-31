import numpy as np

from app.recognition.association import FaceObservation
from app.recognition.registry import FaceRegistry, MatchState
from app.ui.workers.vision_worker import VisionWorker


class FakeEmbedder:
    def embed_image(self, image_path):
        if image_path.name == "bad.jpg":
            raise ValueError("Expected exactly one face")
        if image_path.name.startswith("alice"):
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0], dtype=np.float32)

    def detect_faces(self, image):
        count = int(image[0, 0, 0])
        return [
            FaceObservation(
                bounding_box=(0.0, 0.0, 1.0, 1.0),
                embedding=np.asarray([1.0, float(index)], dtype=np.float32),
            )
            for index in range(count)
        ]


def test_registration_appends_samples_without_removing_existing_identity(tmp_path):
    registry_path = tmp_path / "face_registry.npz"
    existing = FaceRegistry()
    existing.add_sample("alice", np.asarray([1.0, 0.0]))
    existing.save(registry_path)

    worker = VisionWorker(face_registry_path=str(registry_path))
    worker._face_embedder = FakeEmbedder()
    success_messages = []
    failure_messages = []
    worker.face_registration_succeeded.connect(success_messages.append)
    worker.face_registration_failed.connect(failure_messages.append)

    worker.register_face_images("bob", ["bob_1.jpg", "bob_2.jpg"])

    saved = FaceRegistry.load(registry_path)
    assert saved.match(np.asarray([1.0, 0.0])).identity == "alice"
    assert saved.match(np.asarray([0.0, 1.0])).identity == "bob"
    assert saved.match(np.asarray([0.0, 1.0])).state is MatchState.STRONG_MATCH
    assert success_messages == ["Registered 2 image(s) for 'bob'."]
    assert failure_messages == []
    assert not (tmp_path / "face_registry.tmp.npz").exists()


def test_registration_reuses_existing_identity_case_insensitively(tmp_path):
    registry_path = tmp_path / "face_registry.npz"
    existing = FaceRegistry()
    existing.add_sample("Alex", np.asarray([1.0, 0.0]))
    existing.save(registry_path)

    worker = VisionWorker(face_registry_path=str(registry_path))
    worker._face_embedder = FakeEmbedder()
    success_messages = []
    worker.face_registration_succeeded.connect(success_messages.append)

    worker.register_face_images("aLeX", ["alex_1.jpg"])

    with np.load(registry_path, allow_pickle=False) as data:
        identities = data["identities"].tolist()

    saved = FaceRegistry.load(registry_path)
    assert identities == ["Alex", "Alex"]
    assert saved.match(np.asarray([1.0, 0.0])).identity == "Alex"
    assert saved.match(np.asarray([0.0, 1.0])).identity == "Alex"
    assert success_messages == ["Registered 1 image(s) for 'Alex'."]


def test_gallery_reports_sorted_identity_sample_counts():
    registry = FaceRegistry()
    registry.add_sample("Zoe", np.asarray([1.0, 0.0]))
    registry.add_sample("alex", np.asarray([0.0, 1.0]))
    registry.add_sample("ALEX", np.asarray([0.1, 0.9]))

    assert registry.identity_sample_counts == (("alex", 2), ("Zoe", 1))


def test_worker_loads_gallery_sample_counts_without_loading_embedder(tmp_path):
    registry_path = tmp_path / "face_registry.npz"
    registry = FaceRegistry()
    registry.add_sample("Alex", np.asarray([1.0, 0.0]))
    registry.add_sample("aLeX", np.asarray([0.9, 0.1]))
    registry.add_sample("Zoe", np.asarray([0.0, 1.0]))
    registry.save(registry_path)

    worker = VisionWorker(face_registry_path=str(registry_path))
    gallery_results = []
    worker.face_gallery_ready.connect(gallery_results.append)

    worker.load_face_gallery()

    assert gallery_results == [(("Alex", 2), ("Zoe", 1))]
    assert worker._face_embedder is None


def test_worker_reports_empty_gallery_when_registry_is_missing(tmp_path):
    worker = VisionWorker(face_registry_path=str(tmp_path / "missing.npz"))
    gallery_results = []
    worker.face_gallery_ready.connect(gallery_results.append)

    worker.load_face_gallery()

    assert gallery_results == [()]
    assert worker._face_embedder is None


def test_failed_image_leaves_existing_registry_file_unchanged(tmp_path):
    registry_path = tmp_path / "face_registry.npz"
    existing = FaceRegistry()
    existing.add_sample("alice", np.asarray([1.0, 0.0]))
    existing.save(registry_path)
    original_bytes = registry_path.read_bytes()

    worker = VisionWorker(face_registry_path=str(registry_path))
    worker._face_embedder = FakeEmbedder()
    success_messages = []
    failure_messages = []
    worker.face_registration_succeeded.connect(success_messages.append)
    worker.face_registration_failed.connect(failure_messages.append)

    worker.register_face_images("bob", ["bob_1.jpg", "bad.jpg"])

    assert registry_path.read_bytes() == original_bytes
    assert failure_messages == ["Expected exactly one face"]
    assert success_messages == []


def test_registration_creates_gallery_when_no_registry_exists(tmp_path):
    registry_path = tmp_path / "new" / "face_registry.npz"
    worker = VisionWorker(face_registry_path=str(registry_path))
    worker._face_embedder = FakeEmbedder()
    success_messages = []
    worker.face_registration_succeeded.connect(success_messages.append)

    worker.register_face_images("alice", ["alice_1.jpg"])

    saved = FaceRegistry.load(registry_path)
    assert saved.match(np.asarray([1.0, 0.0])).identity == "alice"
    assert success_messages == ["Registered 1 image(s) for 'alice'."]


def test_live_capture_returns_embedding_when_exactly_one_face_is_detected():
    worker = VisionWorker()
    worker._face_embedder = FakeEmbedder()
    samples = []
    failures = []
    worker.face_capture_sample_ready.connect(samples.append)
    worker.face_capture_sample_failed.connect(failures.append)

    worker.capture_face_embedding(np.ones((2, 2, 3), dtype=np.uint8))

    assert len(samples) == 1
    np.testing.assert_allclose(samples[0], [1.0, 0.0])
    assert failures == []


def test_live_capture_rejects_frames_without_exactly_one_face():
    worker = VisionWorker()
    worker._face_embedder = FakeEmbedder()
    samples = []
    failures = []
    worker.face_capture_sample_ready.connect(samples.append)
    worker.face_capture_sample_failed.connect(failures.append)

    worker.capture_face_embedding(np.zeros((2, 2, 3), dtype=np.uint8))

    assert samples == []
    assert failures == ["Keep exactly one face in the camera view before capturing."]


def test_live_embeddings_append_to_existing_identity_and_save_npz(tmp_path):
    registry_path = tmp_path / "face_registry.npz"
    existing = FaceRegistry()
    existing.add_sample("Alex", np.asarray([1.0, 0.0]))
    existing.save(registry_path)

    worker = VisionWorker(face_registry_path=str(registry_path))
    success_messages = []
    worker.face_registration_succeeded.connect(success_messages.append)

    worker.register_face_embeddings(
        "aLeX",
        [np.asarray([0.9, 0.1]), np.asarray([0.8, 0.2])],
    )

    saved = FaceRegistry.load(registry_path)
    assert saved.identity_sample_counts == (("Alex", 3),)
    assert success_messages == ["Registered 2 live sample(s) for 'Alex'."]
    assert not (tmp_path / "face_registry.tmp.npz").exists()
