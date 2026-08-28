from abc import ABC, abstractmethod

from app.camera.frame import Frame
from app.camera.status import CameraStatus


class Camera(ABC):
    """
    Base class for every camera implementation in Smart_Vision.

    Every camera must be able to:
        • start
        • stop
        • read frames

    Every camera also owns its own connection status.
    """

    def __init__(self, camera_id: str):
        self._id = camera_id
        self._status = CameraStatus.STOPPED
        self._last_error = None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> CameraStatus:
        """
        Current status of the camera.
        Read-only for external modules.
        """
        return self._status

    @abstractmethod
    def start(self) -> None:
        """
        Start the camera.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the camera.
        """
        pass

    @abstractmethod
    def read_frame(self) -> Frame | None:
        """
        Return the latest frame if available.

        Returns:
            Frame if successful.
            None if no frame could be read.
        """
        pass
