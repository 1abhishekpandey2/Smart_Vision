from app.core import logger
import cv2
from datetime import datetime

from app.camera.base import Camera
from app.camera.frame import Frame
from app.camera.status import CameraStatus


class WebcamCamera(Camera):

    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(
        self,
        camera_id: str,
        device_index: int = 0,
        capture_factory=cv2.VideoCapture,
    ):
        super().__init__(camera_id)

        self._device_index = device_index
        self._capture_factory = capture_factory

        self._capture = None
        self._consecutive_failures = 0
        self._last_frame: Frame | None = None

    def _set_disconnected(self, error: str) -> None:
        """
        Update the camera state after a connection failure.
        """

        self._status = CameraStatus.DISCONNECTED
        self._last_error = error

        logger.error(
            "[%s] %s",
            self.id,
            error,
        )

    def start(self) -> None:

        # Camera already runningS
        if self.status == CameraStatus.CONNECTED:
            return

        self._status = CameraStatus.CONNECTING
        self._last_error = None

        try:
            self._capture = self._capture_factory(self._device_index)

            if self._capture.isOpened():
                self._status = CameraStatus.CONNECTED

                logger.info(
                    "[%s] Camera connected successfully.",
                    self.id,
                )

            else:
                self._set_disconnected("Unable to open camera.")

        except Exception as e:
            self._set_disconnected(str(e))

    def read_frame(self) -> Frame | None:

        if self.status != CameraStatus.CONNECTED:
            return None

        success, image = self._capture.read()

        if not success:

            self._consecutive_failures += 1

            if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:

                self._set_disconnected("Failed to read frame.")

            return None

        self._consecutive_failures = 0
        frame = Frame(
            image=image,
            timestamp=datetime.now(),
            camera_id=self.id,
        )
        self._last_frame = frame

        return frame

    def stop(self) -> None:
        """
        Stop the camera and release its resources.
        """

        if self._capture is None:
            self._status = CameraStatus.STOPPED
            return

        try:
            self._capture.release()
            self._capture = None
            self._status = CameraStatus.STOPPED
            self._last_error = None

            logger.info(
                "[%s] Camera stopped.",
                self.id,
            )

        except Exception as e:
            self._last_error = str(e)

            logger.exception(
                "[%s] Unexpected error while stopping camera.",
                self.id,
            )

    @property
    def last_frame(self) -> Frame | None:
        return self._last_frame
