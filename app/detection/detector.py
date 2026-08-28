from abc import ABC, abstractmethod

from app.camera.frame import Frame
from app.detection.detection import Detection


class Detector(ABC):
    """
    Abstract interface for vision detectors in Smart_Vision.
    """

    @abstractmethod
    def detect(self, frame: Frame) -> list[Detection]:
        """
        Detect objects in a frame.

        Args:
            frame: Frame to analyze.

        Returns:
            A list of detections found in the frame.
        """
        pass
