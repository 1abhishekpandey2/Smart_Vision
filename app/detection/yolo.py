from pathlib import Path
from typing import Any

from ultralytics import YOLO

from app.camera.frame import Frame
from app.detection.detection import Detection
from app.detection.detector import Detector


class YOLODetector(Detector):
    """
    YOLO-based implementation of the Smart_Vision detector interface.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.25,
        model: Any | None = None,
    ):
        self._model_path = Path(model_path)
        self._confidence_threshold = confidence_threshold

        self._model = model or YOLO(self._model_path)

    def detect(self, frame: Frame) -> list[Detection]:
        """
        Run YOLO inference on a frame and convert the results
        into Smart_Vision Detection objects.
        """

        results = self._model.predict(
            source=frame.image,
            conf=self._confidence_threshold,
            verbose=False,
        )

        detections: list[Detection] = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self._model.names[class_id].lower()
                confidence = float(box.conf[0])

                coordinates = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coordinates)

                detection = Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bounding_box=(x1, y1, x2, y2),
                )

                detections.append(detection)

        return detections
