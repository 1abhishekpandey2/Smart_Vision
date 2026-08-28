import numpy as np
from datetime import datetime

from app.camera.frame import Frame
from app.detection.yolo import YOLODetector
from tests.detection.fakes import FakeYOLOModel


def test_yolo_detector_converts_model_output():
    fake_model = FakeYOLOModel()

    detector = YOLODetector(
        model_path="models/firedetect-11s.pt",
        model=fake_model,
    )

    frame = Frame(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        timestamp=datetime.now(),
        camera_id="test-camera",
    )

    detections = detector.detect(frame)

    assert len(detections) == 2

    assert detections[0].class_id == 0
    assert detections[0].class_name == "fire"
    assert detections[0].confidence == 0.91
    assert detections[0].bounding_box == (100, 50, 400, 300)

    assert detections[1].class_id == 1
    assert detections[1].class_name == "smoke"


if __name__ == "__main__":
    test_yolo_detector_converts_model_output()
    print("All YOLO detector tests passed.")
