from pathlib import Path

from PyQt5.QtCore import (
    QObject,
    QTimer,
    Qt,
    pyqtSignal,
)

from app.camera.status import CameraStatus
from app.camera.video_file import VideoFileCamera


class VideoPlaybackController(QObject):
    frame_ready = pyqtSignal(object)

    position_changed = pyqtSignal(float)
    duration_changed = pyqtSignal(float)

    playback_state_changed = pyqtSignal(bool)

    playback_finished = pyqtSignal()

    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        parent=None,
    ):
        super().__init__(parent)

        self._camera = None
        self._speed = 1.0

        self._timer = QTimer(self)

        self._timer.setTimerType(Qt.PreciseTimer)

        self._timer.timeout.connect(self._read_next_frame)

    @property
    def has_source(self) -> bool:
        return self._camera is not None

    @property
    def is_playing(self) -> bool:
        return self._timer.isActive()

    @property
    def duration_seconds(
        self,
    ) -> float:
        if self._camera is None:
            return 0.0

        return self._camera.duration_seconds

    # ==========================================================
    # OPEN
    # ==========================================================

    def open_video(
        self,
        path: str | Path,
    ) -> bool:
        self.close_video()

        camera = VideoFileCamera(
            camera_id="ui-video",
            video_path=path,
        )

        camera.start()

        if camera.status != CameraStatus.CONNECTED:
            self.error_occurred.emit(camera.last_error or "Could not open video")

            return False

        self._camera = camera

        self.duration_changed.emit(camera.duration_seconds)

        # Show first frame immediately.
        frame = camera.read_frame()

        if frame is not None:
            self.frame_ready.emit(frame)

        # But playback itself starts
        # from the beginning.
        camera.restart()

        self.position_changed.emit(0.0)

        self.playback_state_changed.emit(False)

        return True

    # ==========================================================
    # PLAYBACK
    # ==========================================================

    def play(self) -> None:
        if self._camera is None:
            return

        if self._camera.finished:
            self._camera.restart()

            self.position_changed.emit(0.0)

        self._update_timer_interval()

        self._timer.start()

        self.playback_state_changed.emit(True)

    def pause(self) -> None:
        if self._timer.isActive():
            self._timer.stop()

        self.playback_state_changed.emit(False)

    def restart(self) -> None:
        if self._camera is None:
            return

        self._camera.restart()

        self.position_changed.emit(0.0)

        self._show_current_frame()

    # ==========================================================
    # SEEK
    # ==========================================================

    def seek_ratio(
        self,
        ratio: float,
    ) -> None:
        if self._camera is None:
            return

        if self._camera.frame_count <= 0:
            return

        ratio = max(
            0.0,
            min(
                ratio,
                1.0,
            ),
        )

        target_frame = round(ratio * (self._camera.frame_count - 1))

        success = self._camera.seek_frame(target_frame)

        if not success:
            return

        self._show_current_frame()

    # ==========================================================
    # SPEED
    # ==========================================================

    def set_speed(
        self,
        speed: float,
    ) -> None:
        if speed <= 0:
            raise ValueError("Playback speed must be " "greater than zero.")

        self._speed = speed

        if self._timer.isActive():
            self._update_timer_interval()

    # ==========================================================
    # CLOSE
    # ==========================================================

    def close_video(self) -> None:
        self.pause()

        if self._camera is not None:
            self._camera.stop()

        self._camera = None

        self.duration_changed.emit(0.0)

        self.position_changed.emit(0.0)

    # ==========================================================
    # INTERNAL
    # ==========================================================

    def _show_current_frame(
        self,
    ) -> None:
        if self._camera is None:
            return

        frame = self._camera.read_frame()

        if frame is None:
            return

        self.frame_ready.emit(frame)

        self.position_changed.emit(self._camera.current_time_seconds)

    def _read_next_frame(
        self,
    ) -> None:
        if self._camera is None:
            self.pause()
            return

        frame = self._camera.read_frame()

        if frame is None:
            if self._camera.finished:
                self.pause()

                self.position_changed.emit(self._camera.duration_seconds)

                self.playback_finished.emit()

            return

        self.frame_ready.emit(frame)

        self.position_changed.emit(self._camera.current_time_seconds)

    def _update_timer_interval(
        self,
    ) -> None:
        if self._camera is None:
            return

        fps = self._camera.fps

        if fps <= 0:
            fps = 30.0

        effective_fps = fps * self._speed

        interval_ms = max(
            1,
            round(1000.0 / effective_fps),
        )

        self._timer.setInterval(interval_ms)
