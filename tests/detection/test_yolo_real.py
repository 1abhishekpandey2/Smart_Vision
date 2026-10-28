from datetime import datetime
from pathlib import Path

import cv2
import pytest

from app.camera.frame import Frame
from app.detection.yolo import YOLODetector

MODEL_PATH = Path("models") / "firedetect-11s.pt"
IMAGE_PATH = Path("data") / "samples" / "train_1024.jpg"


@pytest.mark.integration
def test_real_fire_model_inference():
    if not MODEL_PATH.exists() or not IMAGE_PATH.exists():
        pytest.skip(
            "Requires local models/firedetect-11s.pt and " "data/samples/train_1024.jpg"
        )

    detector = YOLODetector(
        model_path=MODEL_PATH,
        confidence_threshold=0.25,
    )

    image = cv2.imread(str(IMAGE_PATH))
    assert image is not None, f"Unable to load image: {IMAGE_PATH}"

    frame = Frame(
        image=image,
        timestamp=datetime.now(),
        camera_id="test-image",
    )

    detections = detector.detect(frame)

    assert all(0.0 <= detection.confidence <= 1.0 for detection in detections)

    for detection in detections:
        assert detection.class_name
        assert len(detection.bounding_box) == 4
