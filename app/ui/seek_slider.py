from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QSlider, QStyle


class SeekSlider(QSlider):
    """
    QSlider that supports both:

    - dragging the handle
    - clicking directly on the timeline
    """

    seek_requested = pyqtSignal(int)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.orientation() == Qt.Horizontal:
            value = QStyle.sliderValueFromPosition(
                self.minimum(),
                self.maximum(),
                event.x(),
                max(self.width(), 1),
            )

            self.setValue(value)

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)

        if event.button() == Qt.LeftButton:
            self.seek_requested.emit(self.value())
