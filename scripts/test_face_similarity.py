import argparse
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def load_face_embedding(
    analyzer: FaceAnalysis,
    image_path: Path,
) -> np.ndarray:
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = analyzer.get(image)

    if not faces:
        raise ValueError(f"No face detected in: {image_path}")

    if len(faces) != 1:
        raise ValueError(
            f"Expected exactly one face in {image_path}, " f"but detected {len(faces)}."
        )

    face = faces[0]

    embedding = face.normed_embedding

    if embedding is None:
        raise ValueError(f"No recognition embedding produced for: {image_path}")

    print(f"\n{image_path.name}")
    print(f"  detection confidence: {float(face.det_score):.3f}")
    print(f"  embedding shape: {embedding.shape}")

    return np.asarray(
        embedding,
        dtype=np.float32,
    )


def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    return float(np.dot(first, second))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two face images using InsightFace embeddings."
    )

    parser.add_argument(
        "image_a",
        type=Path,
    )

    parser.add_argument(
        "image_b",
        type=Path,
    )

    args = parser.parse_args()

    analyzer = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )

    analyzer.prepare(
        ctx_id=-1,
        det_size=(640, 640),
    )

    embedding_a = load_face_embedding(
        analyzer,
        args.image_a,
    )

    embedding_b = load_face_embedding(
        analyzer,
        args.image_b,
    )

    similarity = cosine_similarity(
        embedding_a,
        embedding_b,
    )

    print("\nRESULT")
    print(f"Cosine similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()
