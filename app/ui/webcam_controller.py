import sys

import cv2

from PyQt5.QtCore import (
    QObject,
    QTimer,
    Qt,
    pyqtSignal,
)

from app.camera.status import CameraStatus
from app.camera.webcam import WebcamCamera


class WebcamController(QObject):
    frame_ready = pyqtSignal(object)

    stream_state_changed = pyqtSignal(bool)

    error_occurred = pyqtSignal(str)

    status_changed = pyqtSignal(str)

    def __init__(
        self,
        parent=None,
        interval_ms: int = 33,
    ):
        super().__init__(parent)

        self._camera = None
        self._device_index = None

        self._interval_ms = interval_ms

        self._timer = QTimer(self)

        self._timer.setTimerType(Qt.PreciseTimer)

        self._timer.setInterval(self._interval_ms)

        self._timer.timeout.connect(self._read_next_frame)

    # ==========================================================
    # PROPERTIES
    # ==========================================================

    @property
    def has_source(self) -> bool:
        return self._camera is not None

    @property
    def is_running(self) -> bool:
        return self._timer.isActive()

    @property
    def device_index(self):
        return self._device_index

    # ==========================================================
    # CAPTURE FACTORY
    # ==========================================================

    @staticmethod
    def _create_capture(
        device_index: int,
    ):
        """
        Create the OpenCV capture object.

        Windows:
            Prefer DirectShow because some webcams open
            successfully through MSMF but fail to provide frames.

        Other platforms:
            Allow OpenCV to select the normal backend.
        """

        if sys.platform.startswith("win"):
            return cv2.VideoCapture(
                device_index,
                cv2.CAP_DSHOW,
            )

        return cv2.VideoCapture(device_index)

    # ==========================================================
    # OPEN CAMERA
    # ==========================================================

    def open_camera(
        self,
        device_index: int = 0,
    ) -> bool:
        self.close_camera()

        self.status_changed.emit(f"Opening Webcam {device_index}...")

        camera = WebcamCamera(
            camera_id=(f"webcam-{device_index}"),
            device_index=device_index,
            capture_factory=(self._create_capture),
        )

        camera.start()

        if camera.status != CameraStatus.CONNECTED:
            message = camera.last_error or (f"Unable to open " f"Webcam {device_index}")

            camera.stop()

            self.error_occurred.emit(message)

            return False

        # ------------------------------------------------------
        # IMPORTANT:
        # Confirm that the camera provides a real frame.
        # Opening successfully is not enough.
        # ------------------------------------------------------

        first_frame = camera.read_frame()

        if first_frame is None:
            message = (
                f"Webcam {device_index} opened, " "but no video frame was received."
            )

            camera.stop()

            self.error_occurred.emit(message)

            return False

        self._camera = camera
        self._device_index = device_index

        # Immediately send a first image to the UI.
        self.frame_ready.emit(first_frame)

        self.status_changed.emit(f"Webcam {device_index} connected")

        self.start()

        return True

    # ==========================================================
    # START / RESUME
    # ==========================================================

    def start(self) -> None:
        if self._camera is None:
            return

        if self._camera.status != CameraStatus.CONNECTED:
            return

        if not self._timer.isActive():
            self._timer.start()

        self.stream_state_changed.emit(True)

    # ==========================================================
    # PAUSE
    # ==========================================================

    def pause(self) -> None:
        if self._timer.isActive():
            self._timer.stop()

        self.stream_state_changed.emit(False)

    # ==========================================================
    # CLOSE
    # ==========================================================

    def close_camera(self) -> None:
        if self._timer.isActive():
            self._timer.stop()

        if self._camera is not None:
            self._camera.stop()

        self._camera = None
        self._device_index = None

        self.stream_state_changed.emit(False)

    # ==========================================================
    # READ LIVE FRAME
    # ==========================================================

    def _read_next_frame(self) -> None:
        if self._camera is None:
            self.pause()
            return

        frame = self._camera.read_frame()

        if frame is None:

            if self._camera.status == CameraStatus.DISCONNECTED:
                message = self._camera.last_error or "Webcam disconnected"

                self.pause()

                self.error_occurred.emit(message)

            return

        self.frame_ready.emit(frame)
