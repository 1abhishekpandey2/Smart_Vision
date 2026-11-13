import numpy as np

from app.recognition.face_pose import (
    FacePose,
    FacePoseCheck,
    LiveFaceSample,
)
from app.recognition.registry import FaceRegistry, MatchState
from app.ui.workers.vision_worker import VisionWorker


class FakeEmbedder:
    def embed_image(self, image_path):
        if image_path.name == "bad.jpg":
            raise ValueError("Expected exactly one face")
        if image_path.name.startswith("alice"):
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0], dtype=np.float32)

    def inspect_live_capture(self, image):
        count = int(image[0, 0, 0])
        if count != 1:
            return count, None
        return (
            1,
            LiveFaceSample(
                pose=FacePose(0.0, 0.5),
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            ),
        )


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


def test_registry_removes_every_sample_for_case_insensitive_identity(tmp_path):
    registry_path = tmp_path / "face_registry.npz"
    registry = FaceRegistry()
    registry.add_sample("Alex", np.asarray([1.0, 0.0]))
    registry.add_sample("aLeX", np.asarray([0.9, 0.1]))
    registry.add_sample("Zoe", np.asarray([0.0, 1.0]))

    assert registry.remove_identity("alex") == "Alex"
    assert registry.identity_sample_counts == (("Zoe", 1),)
    assert registry.remove_identity("Unknown") is None

    registry.save(registry_path)
    saved = FaceRegistry.load(registry_path)
    assert saved.match(np.asarray([0.0, 1.0])).identity == "Zoe"


def test_empty_registry_can_be_saved_and_loaded(tmp_path):
    registry = FaceRegistry()
    assert registry.remove_identity("Nobody") is None

    registry_path = tmp_path / "empty_registry.npz"
    registry.save(registry_path)
    saved = FaceRegistry.load(registry_path)

    assert saved.identity_sample_counts == ()
    assert saved.match(np.asarray([1.0, 0.0])).state is MatchState.NO_MATCH


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


def test_worker_removes_person_atomically_and_keeps_other_identities(tmp_path):
    registry_path = tmp_path / "face_registry.npz"
    existing = FaceRegistry()
    existing.add_sample("Alex", np.asarray([1.0, 0.0]))
    existing.add_sample("aLeX", np.asarray([0.9, 0.1]))
    existing.add_sample("Zoe", np.asarray([0.0, 1.0]))
    existing.save(registry_path)

    worker = VisionWorker(face_registry_path=str(registry_path))
    removed = []
    failures = []
    worker.face_identity_removal_succeeded.connect(removed.append)
    worker.face_identity_removal_failed.connect(failures.append)
    worker._identity_memory.update(0.0, {7}, {7: "Alex"})
    assert worker._identity_memory.confirmation_progress(7, "Alex") == (1, 3)

    worker.remove_face_identity("alex")

    saved = FaceRegistry.load(registry_path)
    assert removed == ["Alex"]
    assert failures == []
    assert saved.identity_sample_counts == (("Zoe", 1),)
    assert worker._identity_memory.confirmation_progress(7, "Alex") is None
    assert not (tmp_path / "face_registry.tmp.npz").exists()

    worker.remove_face_identity("ZOE")

    empty = FaceRegistry.load(registry_path)
    assert removed == ["Alex", "Zoe"]
    assert empty.identity_sample_counts == ()
    assert empty.match(np.asarray([1.0, 0.0])).state is MatchState.NO_MATCH

    worker.register_face_embeddings("Maya", [np.asarray([1.0, 0.0])])
    assert FaceRegistry.load(registry_path).identity_sample_counts == (("Maya", 1),)


def test_worker_failed_removal_leaves_registry_file_unchanged(tmp_path):
    registry_path = tmp_path / "face_registry.npz"
    existing = FaceRegistry()
    existing.add_sample("Alex", np.asarray([1.0, 0.0]))
    existing.save(registry_path)
    original_bytes = registry_path.read_bytes()

    worker = VisionWorker(face_registry_path=str(registry_path))
    removed = []
    failures = []
    worker.face_identity_removal_succeeded.connect(removed.append)
    worker.face_identity_removal_failed.connect(failures.append)

    worker.remove_face_identity("Unknown")

    assert registry_path.read_bytes() == original_bytes
    assert removed == []
    assert failures == ["Registered person 'Unknown' was not found."]


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


def test_live_pose_check_returns_embedding_when_exactly_one_face_is_detected():
    worker = VisionWorker()
    worker._face_embedder = FakeEmbedder()
    results = []
    worker.face_pose_check_ready.connect(results.append)

    worker.inspect_face_pose_frame(np.ones((2, 2, 3), dtype=np.uint8), True)

    assert len(results) == 1
    assert isinstance(results[0], FacePoseCheck)
    assert results[0].face_count == 1
    assert results[0].manual_capture
    np.testing.assert_allclose(results[0].sample.embedding, [1.0, 0.0])


def test_live_pose_check_rejects_frames_without_exactly_one_face():
    worker = VisionWorker()
    worker._face_embedder = FakeEmbedder()
    results = []
    worker.face_pose_check_ready.connect(results.append)

    worker.inspect_face_pose_frame(np.zeros((2, 2, 3), dtype=np.uint8), False)

    assert len(results) == 1
    assert results[0].face_count == 0
    assert results[0].sample is None
    assert not results[0].manual_capture


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
