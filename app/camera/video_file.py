from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from app.camera.base import Camera
from app.camera.frame import Frame
from app.camera.status import CameraStatus
from app.core.logger import logger


class VideoFileCamera(Camera):
    def __init__(
        self,
        camera_id: str,
        video_path: str | Path,
        capture_factory: Any = cv2.VideoCapture,
    ):
        super().__init__(camera_id)

        self._video_path = Path(video_path)
        self._capture_factory = capture_factory

        self._capture = None
        self._finished = False

        self._fps = 0.0
        self._frame_count = 0

    @property
    def video_path(self) -> Path:
        return self._video_path

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration_seconds(self) -> float:
        if self._fps <= 0:
            return 0.0

        return self._frame_count / self._fps

    @property
    def current_frame_index(self) -> int:
        if self._capture is None:
            return 0

        return int(self._capture.get(cv2.CAP_PROP_POS_FRAMES))

    @property
    def current_time_seconds(self) -> float:
        if self._fps <= 0:
            return 0.0

        # CAP_PROP_POS_FRAMES normally points to the
        # next frame after a successful read().
        displayed_frame_index = max(
            self.current_frame_index - 1,
            0,
        )

        return displayed_frame_index / self._fps

    def start(self) -> None:
        if self.status == CameraStatus.CONNECTED:
            return

        self._status = CameraStatus.CONNECTING
        self._last_error = None
        self._finished = False

        if not self._video_path.exists():
            self._status = CameraStatus.DISCONNECTED

            self._last_error = f"Video file does not exist: " f"{self._video_path}"

            logger.error(
                "[%s] %s",
                self.id,
                self._last_error,
            )

            return

        if self._capture is not None:
            self._capture.release()

        self._capture = self._capture_factory(str(self._video_path))

        if not self._capture.isOpened():
            self._status = CameraStatus.DISCONNECTED

            self._last_error = f"Could not open video: " f"{self._video_path}"

            logger.error(
                "[%s] %s",
                self.id,
                self._last_error,
            )

            self._capture.release()
            self._capture = None

            return

        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS))

        self._frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self._status = CameraStatus.CONNECTED

        logger.info(
            "[%s] Video opened: %s | " "FPS: %.2f | Frames: %s",
            self.id,
            self._video_path,
            self._fps,
            self._frame_count,
        )

    def read_frame(self) -> Frame | None:
        if self._capture is None:
            return None

        success, image = self._capture.read()

        if not success:
            self._finished = True
            return None

        return Frame(
            image=image,
            timestamp=datetime.now(),
            camera_id=self.id,
        )

    def seek_frame(
        self,
        frame_index: int,
    ) -> bool:
        if self._capture is None:
            return False

        if self._frame_count <= 0:
            return False

        frame_index = max(
            0,
            min(
                frame_index,
                self._frame_count - 1,
            ),
        )

        success = self._capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        if success:
            self._finished = False

        return bool(success)

    def restart(self) -> bool:
        return self.seek_frame(0)

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()

        self._capture = None
        self._finished = False

        self._status = CameraStatus.STOPPED
        self._last_error = None
