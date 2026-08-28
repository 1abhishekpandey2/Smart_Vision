from datetime import datetime
from pathlib import Path

import cv2

from app.camera.frame import Frame
from app.detection.yolo import YOLODetector

MODEL_PATH = Path("models") / "firedetect-11s.pt"
IMAGE_PATH = Path("data") / "samples" / "train_1024.jpg"


detector = YOLODetector(
    model_path=MODEL_PATH,
    confidence_threshold=0.25,
)

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise RuntimeError(f"Unable to load image: {IMAGE_PATH}")

frame = Frame(
    image=image,
    timestamp=datetime.now(),
    camera_id="test-image",
)

detections = detector.detect(frame)

print(f"Detections found: {len(detections)}")

for detection in detections:
    print(
        f"Class: {detection.class_name} | "
        f"Confidence: {detection.confidence:.3f} | "
        f"Box: {detection.bounding_box}"
    )
