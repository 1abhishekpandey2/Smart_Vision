"""Guided live face-sample capture dialog."""

import cv2
import numpy as np

from PyQt5.QtCore import QPointF, QRectF, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.recognition.face_pose import (
    FacePose,
    FacePoseCheck,
    PoseTarget,
    pose_matches_target,
)


class CircularCameraPreview(QLabel):
    """Show a circular live crop with a light pose silhouette guide."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pose_target = PoseTarget.FRONT
        self.setFixedSize(360, 360)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "QLabel { background-color: #111923; "
            "border: 1px solid #364354; border-radius: 180px; }"
        )

    def set_pose_target(self, target: PoseTarget) -> None:
        self._pose_target = target
        self.update()

    def set_frame(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        image = image[top : top + side, left : left + side]
        image = cv2.resize(image, (348, 348), interpolation=cv2.INTER_AREA)
        height, width = image.shape[:2]
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (width // 2, height // 2), width // 2, 255, thickness=-1)
        rgba[:, :, 3] = mask
        qimage = QImage(
            rgba.data,
            width,
            height,
            rgba.strides[0],
            QImage.Format_RGBA8888,
        ).copy()
        self.setPixmap(QPixmap.fromImage(qimage))

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(163, 238, 224, 115), 2))

        # A subtle head-and-shoulders silhouette gives the user a steady visual target.
        head = QRectF(119, 68, 122, 158)
        painter.setBrush(QColor(163, 238, 224, 22))
        painter.drawEllipse(head)
        painter.setBrush(Qt.NoBrush)
        shoulders = QPainterPath()
        shoulders.moveTo(91, 294)
        shoulders.cubicTo(96, 250, 126, 229, 180, 229)
        shoulders.cubicTo(234, 229, 264, 250, 269, 294)
        painter.drawPath(shoulders)
        painter.setPen(QPen(QColor(163, 238, 224, 70), 1, Qt.DashLine))
        painter.drawLine(180, 92, 180, 218)
        painter.drawLine(132, 139, 228, 139)

        if self._pose_target is not PoseTarget.FRONT:
            directions = {
                PoseTarget.LEFT: (-1, 0),
                PoseTarget.RIGHT: (1, 0),
                PoseTarget.UP: (0, -1),
                PoseTarget.DOWN: (0, 1),
            }
            dx, dy = directions[self._pose_target]
            start = QPointF(180, 180)
            end = QPointF(180 + dx * 46, 180 + dy * 46)
            painter.setPen(
                QPen(QColor(95, 232, 195, 220), 3, Qt.SolidLine, Qt.RoundCap)
            )
            painter.drawLine(start, end)
            arrow = QPainterPath()
            arrow.moveTo(end)
            arrow.lineTo(
                QPointF(end.x() - dx * 11 - dy * 7, end.y() - dy * 11 + dx * 7)
            )
            arrow.moveTo(end)
            arrow.lineTo(
                QPointF(end.x() - dx * 11 + dy * 7, end.y() - dy * 11 - dx * 7)
            )
            painter.drawPath(arrow)
        painter.end()


class CircularProgressRing(QWidget):
    """Paint a cumulative five-view progress arc around the camera preview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._progress = 0.0
        self.setFixedSize(400, 400)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)

    def set_progress(self, progress: float) -> None:
        self._progress = min(1.0, max(0.0, progress))
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        arc = QRectF(8, 8, 384, 384)
        painter.setPen(QPen(QColor("#344153"), 8, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(arc, 90 * 16, -360 * 16)
        if self._progress > 0:
            painter.setPen(QPen(QColor("#5de2bf"), 8, Qt.SolidLine, Qt.RoundCap))
            painter.drawArc(arc, 90 * 16, -round(360 * 16 * self._progress))
        painter.end()


class FaceRegistrationDialog(QDialog):
    """Capture face embeddings from a frontal reference and four guided poses."""

    pose_check_requested = pyqtSignal(object, bool)
    registration_requested = pyqtSignal(str, object)

    _POSES = (
        (PoseTarget.FRONT, "Look straight at the camera", "Front"),
        (PoseTarget.LEFT, "Turn toward the left side of the preview", "Left"),
        (PoseTarget.RIGHT, "Turn toward the right side of the preview", "Right"),
        (PoseTarget.UP, "Tilt your chin slightly up", "Up"),
        (PoseTarget.DOWN, "Tilt your chin slightly down", "Down"),
    )

    def __init__(self, identity: str, parent=None):
        super().__init__(parent)
        self.identity = identity
        self._samples = []
        self._current_frame = None
        self._capture_pending = False
        self._saving = False
        self._baseline_pose: FacePose | None = None
        self._stable_pose_observations = 0

        self.setWindowTitle(f"Live face registration — {identity}")
        self.setModal(False)
        self.setMinimumWidth(470)
        self.setStyleSheet(
            "QDialog { background: #0e131a; color: #f5f7fb; "
            "font-family: 'Segoe UI'; font-size: 14px; }"
            "QLabel { color: #f5f7fb; background: transparent; }"
            "QLabel#Title { font-size: 21px; font-weight: 650; }"
            "QLabel#Identity { color: #a7b2c2; font-size: 13px; }"
            "QLabel#Instruction { font-size: 17px; font-weight: 600; }"
            "QLabel#Progress { color: #71e7c6; font-size: 12px; "
            "font-weight: 700; padding: 6px 12px; background: #142c2b; "
            "border: 1px solid #25443e; border-radius: 12px; }"
            "QLabel#Helper { color: #9aa7b8; font-size: 12px; }"
            "QPushButton { min-height: 38px; padding: 0 16px; "
            "font-size: 13px; font-weight: 600; border-radius: 10px; }"
            "QPushButton#Capture { color: #06251f; background: #61dfbb; "
            "border: 1px solid #61dfbb; }"
            "QPushButton#Capture:hover { background: #7aebcb; }"
            "QPushButton#Capture:disabled { color: #687b78; "
            "background: #263633; border-color: #263633; }"
            "QPushButton#Cancel { color: #d5dce6; background: #1a222d; "
            "border: 1px solid #303b4b; }"
            "QPushButton#Cancel:hover { background: #242e3b; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 22)
        layout.setSpacing(12)

        title = QLabel("Live face registration", self)
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        identity_label = QLabel(f"Adding samples for  |  {identity}", self)
        identity_label.setObjectName("Identity")
        identity_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(identity_label)

        stage = QWidget(self)
        stage_layout = QGridLayout(stage)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        self.progress_ring = CircularProgressRing(stage)
        self.preview = CircularCameraPreview(stage)
        stage_layout.addWidget(self.progress_ring, 0, 0, alignment=Qt.AlignCenter)
        stage_layout.addWidget(self.preview, 0, 0, alignment=Qt.AlignCenter)
        layout.addWidget(stage, alignment=Qt.AlignCenter)

        self.instruction = QLabel(self)
        self.instruction.setObjectName("Instruction")
        self.instruction.setAlignment(Qt.AlignCenter)
        self.instruction.setWordWrap(True)
        layout.addWidget(self.instruction)

        self.progress = QLabel(self)
        self.progress.setObjectName("Progress")
        self.progress.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress, alignment=Qt.AlignCenter)

        self.helper = QLabel(self)
        self.helper.setObjectName("Helper")
        self.helper.setAlignment(Qt.AlignCenter)
        self.helper.setWordWrap(True)
        layout.addWidget(self.helper)

        buttons = QHBoxLayout()
        self.capture_button = QPushButton("Capture frontal reference", self)
        self.capture_button.setObjectName("Capture")
        self.capture_button.setToolTip("Capture the current view manually")
        self.capture_button.clicked.connect(self._capture_or_retry)
        buttons.addWidget(self.capture_button)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setObjectName("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)

        self._pose_timer = QTimer(self)
        self._pose_timer.setInterval(250)
        self._pose_timer.timeout.connect(self._request_pose_check)
        self._pose_timer.start()
        self.finished.connect(lambda _result: self._pose_timer.stop())
        self._update_prompt()

    def set_frame(self, image: np.ndarray) -> None:
        self._current_frame = image.copy()
        self.preview.set_frame(image)
        if not self._capture_pending and not self._saving:
            self.capture_button.setEnabled(True)

    def add_sample(self, embedding) -> None:
        self._capture_pending = False
        self._stable_pose_observations = 0
        self._samples.append(np.asarray(embedding, dtype=np.float32))
        self._sync_progress()
        if len(self._samples) == len(self._POSES):
            self._saving = True
            self.instruction.setText("All five views captured. Saving face vectors…")
            self.capture_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self.registration_requested.emit(self.identity, self._samples)
        else:
            self._update_prompt()

    def handle_pose_check(self, result: FacePoseCheck) -> None:
        self._capture_pending = False
        if result.error:
            self._stable_pose_observations = 0
            self._sync_progress()
            self.instruction.setText(result.error)
            self.capture_button.setEnabled(self._current_frame is not None)
            return

        if result.face_count != 1 or result.sample is None:
            self._stable_pose_observations = 0
            self._sync_progress()
            self.instruction.setText(
                "Keep exactly one face visible in the circular preview."
                if result.face_count == 0
                else "Only one person should be visible during registration."
            )
            self.capture_button.setEnabled(self._current_frame is not None)
            return

        if result.manual_capture:
            if len(self._samples) == 0:
                self._baseline_pose = result.sample.pose
            else:
                self.instruction.setText(
                    "Manual capture accepted; the pose guide was overridden."
                )
            self.add_sample(result.sample.embedding)
            return

        target, prompt, _label = self._POSES[len(self._samples)]
        if target is PoseTarget.FRONT:
            return

        if self._baseline_pose is None:
            self._stable_pose_observations = 0
            self._sync_progress()
            self.instruction.setText("Capture the frontal reference first.")
            return

        if pose_matches_target(target, self._baseline_pose, result.sample.pose):
            self._stable_pose_observations += 1
            self._sync_progress()
            if self._stable_pose_observations >= 3:
                self.add_sample(result.sample.embedding)
                return
            self.instruction.setText(
                f"Pose detected. Hold still "
                f"({self._stable_pose_observations}/3 checks)…"
            )
        else:
            self._stable_pose_observations = 0
            self._sync_progress()
            self.instruction.setText(prompt)

        self.capture_button.setEnabled(self._current_frame is not None)

    def show_capture_error(self, message: str) -> None:
        self._capture_pending = False
        self.instruction.setText(message)
        self.capture_button.setEnabled(self._current_frame is not None)

    def registration_succeeded(self, message: str) -> None:
        self.instruction.setText(message)
        self.accept()

    def registration_failed(self, message: str) -> None:
        self._saving = False
        self.instruction.setText(f"Could not save samples: {message}")
        self.capture_button.setText("Retry saving samples")
        self.capture_button.setEnabled(True)
        self.cancel_button.setEnabled(True)

    def _capture_or_retry(self) -> None:
        if len(self._samples) == len(self._POSES):
            self._saving = True
            self.capture_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self.instruction.setText("Saving captured face vectors…")
            self.registration_requested.emit(self.identity, self._samples)
            return
        if self._current_frame is None or self._capture_pending:
            return
        self._capture_pending = True
        self.capture_button.setEnabled(False)
        self.instruction.setText("Checking the current view…")
        self.pose_check_requested.emit(self._current_frame.copy(), True)

    def _request_pose_check(self) -> None:
        if (
            len(self._samples) == 0
            or self._current_frame is None
            or self._capture_pending
            or self._saving
        ):
            return
        self._capture_pending = True
        self.capture_button.setEnabled(False)
        self.pose_check_requested.emit(self._current_frame.copy(), False)

    def _update_prompt(self) -> None:
        target, direction, label = self._POSES[len(self._samples)]
        self.preview.set_pose_target(target)
        self.instruction.setText(direction)
        self.progress.setText(
            f"VIEW {len(self._samples) + 1} OF {len(self._POSES)}  |  {label.upper()}"
        )
        self.helper.setText(
            "Center your face in the guide, then capture the frontal reference."
            if not self._samples
            else "Follow the arrow and hold the pose until the ring advances."
        )
        self.capture_button.setText(
            "Capture frontal reference" if not self._samples else "Capture manually"
        )
        self.capture_button.setEnabled(self._current_frame is not None)
        self._sync_progress()

    def _sync_progress(self) -> None:
        completed = len(self._samples)
        fraction = min(
            1.0,
            (completed + self._stable_pose_observations / 3) / len(self._POSES),
        )
        self.progress_ring.set_progress(fraction)
