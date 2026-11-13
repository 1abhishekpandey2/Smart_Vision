from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.recognition.association import FaceObservation
from app.recognition.face_pose import LiveFaceSample, estimate_face_pose


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
                f"Expected exactly one face in {image_path}, but detected {len(faces)}."
            )

        embedding = faces[0].normed_embedding

        if embedding is None:
            raise ValueError(f"No face embedding produced for: {image_path}")

        return np.asarray(
            embedding,
            dtype=np.float32,
        )

    def detect_faces(
        self,
        image: np.ndarray,
    ) -> list[FaceObservation]:
        """Detect every face in a BGR frame and return its box and embedding."""

        observations = []

        for face in self._analyzer.get(image):
            embedding = face.normed_embedding

            if embedding is None:
                continue

            x1, y1, x2, y2 = (float(value) for value in face.bbox[:4])

            observations.append(
                FaceObservation(
                    bounding_box=(x1, y1, x2, y2),
                    embedding=np.asarray(
                        embedding,
                        dtype=np.float32,
                    ),
                )
            )

        return observations

    def inspect_live_capture(
        self, image: np.ndarray
    ) -> tuple[int, LiveFaceSample | None]:
        """Return a pose and embedding only when exactly one face is present."""

        faces = self._analyzer.get(image)
        if len(faces) != 1:
            return len(faces), None

        face = faces[0]
        if face.normed_embedding is None:
            raise ValueError("No face embedding produced for the live frame.")
        if face.kps is None:
            raise ValueError("Face landmarks were not produced for the live frame.")

        return 1, LiveFaceSample(
            pose=estimate_face_pose(face.kps),
            embedding=np.asarray(face.normed_embedding, dtype=np.float32),
        )
