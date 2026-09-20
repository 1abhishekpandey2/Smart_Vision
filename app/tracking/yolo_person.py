from pathlib import Path
from typing import Any

from ultralytics import YOLO

from app.camera.frame import Frame
from app.tracking.track import Track


class YOLOPersonTracker:
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.25,
        tracker_config: str = "bytetrack.yaml",
        model: Any | None = None,
    ):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")

        self._model_path = Path(model_path)
        self._confidence_threshold = confidence_threshold
        self._tracker_config = tracker_config

        self._model = model or YOLO(self._model_path)

    def track(self, frame: Frame) -> list[Track]:
        results = self._model.track(
            source=frame.image,
            persist=True,
            classes=[self.PERSON_CLASS_ID],
            conf=self._confidence_threshold,
            tracker=self._tracker_config,
            verbose=False,
        )

        if not results:
            return []

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return []

        if boxes.id is None:
            return []

        tracks = []

        for xyxy, confidence, class_id, track_id in zip(
            boxes.xyxy,
            boxes.conf,
            boxes.cls,
            boxes.id,
        ):
            class_id_value = int(class_id.item())
            track_id_value = int(track_id.item())
            confidence_value = float(confidence.item())

            x1, y1, x2, y2 = (int(value) for value in xyxy.tolist())

            class_name = self._model.names[class_id_value].lower()

            track = Track(
                track_id=track_id_value,
                class_id=class_id_value,
                class_name=class_name,
                confidence=confidence_value,
                bounding_box=(x1, y1, x2, y2),
            )

            tracks.append(track)

        return tracks
