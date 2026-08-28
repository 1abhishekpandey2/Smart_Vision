import numpy as np


class FakeBox:
    def __init__(
        self,
        class_id: int,
        confidence: float,
        coordinates: list[float],
    ):
        self.cls = np.array([class_id])
        self.conf = np.array([confidence])
        self.xyxy = np.array([coordinates])


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYOLOModel:
    def __init__(self):
        self.names = {
            0: "Fire",
            1: "Smoke",
        }

    def predict(self, source, conf, verbose):
        return [
            FakeResult(
                boxes=[
                    FakeBox(
                        class_id=0,
                        confidence=0.91,
                        coordinates=[100.0, 50.0, 400.0, 300.0],
                    ),
                    FakeBox(
                        class_id=1,
                        confidence=0.73,
                        coordinates=[200.0, 100.0, 500.0, 350.0],
                    ),
                ]
            )
        ]
