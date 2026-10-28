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
