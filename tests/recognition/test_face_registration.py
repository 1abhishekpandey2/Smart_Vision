import numpy as np

from app.recognition.registry import FaceRegistry, MatchState
from app.ui.workers.vision_worker import VisionWorker


class FakeEmbedder:
    def embed_image(self, image_path):
        if image_path.name == "bad.jpg":
            raise ValueError("Expected exactly one face")
        if image_path.name.startswith("alice"):
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0], dtype=np.float32)


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
