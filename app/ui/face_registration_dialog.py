"""Guided live face-sample capture dialog."""

import cv2
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class CircularCameraPreview(QLabel):
    """Show a live BGR frame inside a circular crop."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(360, 360)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "QLabel { background-color: #161b24; "
            "border: 4px solid #64d6a3; border-radius: 180px; }"
        )

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


class FaceRegistrationDialog(QDialog):
    """Capture face embeddings directly from five guided live views."""

    capture_requested = pyqtSignal(object)
    registration_requested = pyqtSignal(str, object)

    _POSES = (
        ("Look straight at the camera", "Front"),
        ("Turn your face slightly left", "Left"),
        ("Turn your face slightly right", "Right"),
        ("Tilt your chin slightly up", "Up"),
        ("Tilt your chin slightly down", "Down"),
    )

    def __init__(self, identity: str, parent=None):
        super().__init__(parent)
        self.identity = identity
        self._samples = []
        self._current_frame = None
        self._capture_pending = False
        self._saving = False

        self.setWindowTitle(f"Live face registration — {identity}")
        self.setModal(False)
        self.setMinimumWidth(430)
        self.setStyleSheet(
            "QDialog { background: #10141b; color: #f5f7fb; }"
            "QLabel { color: #f5f7fb; }"
            "QPushButton { padding: 8px 14px; }"
        )

        layout = QVBoxLayout(self)
        self.preview = CircularCameraPreview(self)
        layout.addWidget(self.preview, alignment=Qt.AlignCenter)

        self.instruction = QLabel(self)
        self.instruction.setAlignment(Qt.AlignCenter)
        self.instruction.setWordWrap(True)
        layout.addWidget(self.instruction)

        self.progress = QLabel(self)
        self.progress.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress)

        buttons = QHBoxLayout()
        self.capture_button = QPushButton("Capture this view", self)
        self.capture_button.clicked.connect(self._capture_or_retry)
        buttons.addWidget(self.capture_button)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        self._update_prompt()

    def set_frame(self, image: np.ndarray) -> None:
        self._current_frame = image.copy()
        self.preview.set_frame(image)
        if not self._capture_pending and not self._saving:
            self.capture_button.setEnabled(True)

    def add_sample(self, embedding) -> None:
        self._capture_pending = False
        self._samples.append(np.asarray(embedding, dtype=np.float32))
        if len(self._samples) == len(self._POSES):
            self._saving = True
            self.instruction.setText("All five views captured. Saving face vectors…")
            self.capture_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self.registration_requested.emit(self.identity, self._samples)
        else:
            self._update_prompt()

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
        self._capture_current_view()

    def _capture_current_view(self) -> None:
        if self._current_frame is None or self._capture_pending or self._saving:
            return
        self._capture_pending = True
        self.capture_button.setEnabled(False)
        self.instruction.setText("Checking for exactly one face…")
        self.capture_requested.emit(self._current_frame.copy())

    def _update_prompt(self) -> None:
        direction, label = self._POSES[len(self._samples)]
        self.instruction.setText(direction)
        self.progress.setText(
            f"View {len(self._samples) + 1} of {len(self._POSES)} · {label}"
        )
        self.capture_button.setEnabled(self._current_frame is not None)
