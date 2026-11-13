from types import SimpleNamespace

import numpy as np

from app.recognition.insightface_embedder import InsightFaceEmbedder


class FakeFaceAnalyzer:
    def __init__(self, faces):
        self.faces = faces

    def get(self, image):
        return self.faces


def test_detect_faces_returns_boxes_and_float32_embeddings():
    analyzer = FakeFaceAnalyzer(
        [
            SimpleNamespace(
                bbox=np.asarray([10.5, 20.0, 40.5, 60.0]),
                normed_embedding=np.ones(512, dtype=np.float64),
            )
        ]
    )
    embedder = object.__new__(InsightFaceEmbedder)
    embedder._analyzer = analyzer

    observations = embedder.detect_faces(np.zeros((100, 100, 3)))

    assert len(observations) == 1
    assert observations[0].bounding_box == (10.5, 20.0, 40.5, 60.0)
    assert observations[0].embedding.dtype == np.float32
    assert observations[0].embedding.shape == (512,)


def test_detect_faces_skips_face_without_embedding():
    analyzer = FakeFaceAnalyzer(
        [
            SimpleNamespace(
                bbox=np.asarray([10.0, 20.0, 40.0, 60.0]),
                normed_embedding=None,
            )
        ]
    )
    embedder = object.__new__(InsightFaceEmbedder)
    embedder._analyzer = analyzer

    observations = embedder.detect_faces(np.zeros((100, 100, 3)))

    assert observations == []


def test_inspect_live_capture_returns_embedding_and_landmark_pose():
    face = SimpleNamespace(
        normed_embedding=np.ones(512, dtype=np.float64),
        kps=np.asarray(
            [
                [40.0, 40.0],
                [60.0, 40.0],
                [50.0, 55.0],
                [45.0, 70.0],
                [55.0, 70.0],
            ],
            dtype=np.float32,
        ),
    )
    embedder = object.__new__(InsightFaceEmbedder)
    embedder._analyzer = FakeFaceAnalyzer([face])

    face_count, sample = embedder.inspect_live_capture(np.zeros((100, 100, 3)))

    assert face_count == 1
    assert sample.pose.nose_horizontal_offset == 0.0
    assert sample.pose.nose_vertical_ratio == 0.5
    assert sample.embedding.dtype == np.float32
    assert sample.embedding.shape == (512,)


def test_inspect_live_capture_requires_exactly_one_face():
    embedder = object.__new__(InsightFaceEmbedder)
    embedder._analyzer = FakeFaceAnalyzer([object(), object()])

    face_count, sample = embedder.inspect_live_capture(np.zeros((100, 100, 3)))

    assert face_count == 2
    assert sample is None
