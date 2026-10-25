from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


class InsightFaceEmbedder:
    def __init__(self) -> None:
        self._analyzer = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )

        self._analyzer.prepare(
            ctx_id=-1,
            det_size=(640, 640),
        )

    def embed_image(
        self,
        image_path: Path,
    ) -> np.ndarray:
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        faces = self._analyzer.get(image)

        if not faces:
            raise ValueError(f"No face detected in: {image_path}")

        if len(faces) != 1:
            raise ValueError(
                f"Expected exactly one face in {image_path}, "
                f"but detected {len(faces)}."
            )

        embedding = faces[0].normed_embedding

        if embedding is None:
            raise ValueError(f"No face embedding produced for: {image_path}")

        return np.asarray(
            embedding,
            dtype=np.float32,
        )
