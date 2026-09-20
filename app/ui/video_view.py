import cv2
import numpy as np

from PyQt5.QtCore import (
    QPoint,
    QRect,
    Qt,
    pyqtSignal,
)

from PyQt5.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QPolygon,
)

from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class VideoCanvas(QWidget):
    zone_completed = pyqtSignal(object)
    zone_cleared = pyqtSignal()
    message_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumSize(
            320,
            180,
        )

        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        self.setFocusPolicy(Qt.StrongFocus)

        self.setMouseTracking(True)

        self._pixmap: QPixmap | None = None

        self._source_width = 0
        self._source_height = 0

        self._message = "Select a camera or video source"

        # Final saved polygon.
        self._zone_points: tuple[tuple[float, float], ...] = ()

        # Temporary polygon while drawing/editing.
        self._draft_zone_points: list[tuple[float, float]] = []

        self._drawing_zone = False
        self._zone_active = False

        self._overlays = ()
        self._track_paths = ()

    def _draw_track_paths(
        self,
        painter: QPainter,
    ) -> None:
        if self._source_width <= 0 or self._source_height <= 0:
            return

        for track_path in self._track_paths:

            if len(track_path.points) < 2:
                continue

            widget_points = [
                self._source_to_widget(
                    x,
                    y,
                )
                for x, y in track_path.points
            ]

            polygon = QPolygon(widget_points)

            if track_path.alert:
                color = QColor(
                    255,
                    80,
                    80,
                )

            else:
                color = QColor(
                    70,
                    210,
                    255,
                )

            pen = QPen(color)

            pen.setWidth(2)

            pen.setCapStyle(Qt.RoundCap)

            pen.setJoinStyle(Qt.RoundJoin)

            painter.setPen(pen)

            painter.setBrush(Qt.NoBrush)

            painter.drawPolyline(polygon)

            # Current end of trajectory.
            current_point = widget_points[-1]

            painter.setBrush(color)

            painter.drawEllipse(
                current_point,
                4,
                4,
            )

    def set_track_paths(
        self,
        track_paths,
    ) -> None:
        self._track_paths = tuple(track_paths)

        self.update()

    # ==========================================================
    # VIDEO
    # ==========================================================

    def set_frame(
        self,
        image: np.ndarray,
    ) -> None:
        if image is None:
            return

        if image.ndim == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            rgb = np.ascontiguousarray(rgb)

            height, width, channels = rgb.shape

            bytes_per_line = channels * width

            qimage = QImage(
                rgb.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888,
            ).copy()

        elif image.ndim == 2:
            gray = np.ascontiguousarray(image)

            height, width = gray.shape

            qimage = QImage(
                gray.data,
                width,
                height,
                width,
                QImage.Format_Grayscale8,
            ).copy()

        else:
            return

        self._source_width = width
        self._source_height = height

        self._pixmap = QPixmap.fromImage(qimage)

        self._message = ""

        self.update()

    def set_message(
        self,
        message: str,
    ) -> None:
        self._pixmap = None
        self._message = message
        self._overlays = ()

        self.update()

    # ==========================================================
    # AI OVERLAYS
    # ==========================================================

    def set_overlays(
        self,
        overlays,
    ) -> None:
        self._overlays = tuple(overlays)

        self.update()

    def clear_overlays(self) -> None:
        self._overlays = ()
        self._track_paths = ()

        self._zone_active = False

        self.update()

    # ==========================================================
    # ZONE
    # ==========================================================

    @property
    def zone_points(
        self,
    ) -> tuple[tuple[float, float], ...]:
        return self._zone_points

    @property
    def drawing_zone(self) -> bool:
        return self._drawing_zone

    def set_zone_active(
        self,
        active: bool,
    ) -> None:
        self._zone_active = active
        self.update()

    def begin_zone_drawing(self) -> None:
        if self._pixmap is None:
            self.message_requested.emit("Open a video before drawing a zone.")
            return

        self._drawing_zone = True
        self._draft_zone_points = []

        self.setFocus()

        self.message_requested.emit(
            "Zone drawing: left-click points, "
            "right-click undo, Enter to finish, "
            "Esc to cancel."
        )

        self.update()

    def edit_zone(self) -> None:
        if not self._zone_points:
            self.begin_zone_drawing()
            return

        self._drawing_zone = True

        self._draft_zone_points = list(self._zone_points)

        self.setFocus()

        self.message_requested.emit(
            "Editing zone. Add points with left-click, "
            "remove the last point with right-click, "
            "Enter to save."
        )

        self.update()

    def finish_zone_drawing(self) -> bool:
        if not self._drawing_zone:
            return False

        if len(self._draft_zone_points) < 3:
            self.message_requested.emit("A polygon requires at least 3 points.")
            return False

        self._zone_points = tuple(self._draft_zone_points)

        self._draft_zone_points.clear()
        self._drawing_zone = False

        self.zone_completed.emit(self._zone_points)

        self.message_requested.emit("Restricted zone saved.")

        self.update()

        return True

    def cancel_zone_drawing(self) -> None:
        if not self._drawing_zone:
            return

        self._draft_zone_points.clear()
        self._drawing_zone = False

        self.message_requested.emit("Zone drawing cancelled.")

        self.update()

    def clear_zone(self) -> None:
        self._zone_points = ()
        self._draft_zone_points.clear()

        self._drawing_zone = False
        self._zone_active = False

        self.zone_cleared.emit()

        self.update()

    # ==========================================================
    # COORDINATE MAPPING
    # ==========================================================

    def _video_rect(self) -> QRect:
        if self._pixmap is None:
            return QRect()

        source_width = self._pixmap.width()

        source_height = self._pixmap.height()

        if source_width <= 0 or source_height <= 0:
            return QRect()

        scale = min(
            self.width() / source_width,
            self.height() / source_height,
        )

        display_width = max(
            1,
            round(source_width * scale),
        )

        display_height = max(
            1,
            round(source_height * scale),
        )

        x = (self.width() - display_width) // 2

        y = (self.height() - display_height) // 2

        return QRect(
            x,
            y,
            display_width,
            display_height,
        )

    def _widget_to_normalized(
        self,
        point: QPoint,
    ) -> tuple[float, float] | None:
        video_rect = self._video_rect()

        if video_rect.isNull() or not video_rect.contains(point):
            return None

        normalized_x = (point.x() - video_rect.x()) / max(
            video_rect.width(),
            1,
        )

        normalized_y = (point.y() - video_rect.y()) / max(
            video_rect.height(),
            1,
        )

        normalized_x = max(
            0.0,
            min(
                normalized_x,
                1.0,
            ),
        )

        normalized_y = max(
            0.0,
            min(
                normalized_y,
                1.0,
            ),
        )

        return (
            normalized_x,
            normalized_y,
        )

    def _normalized_to_widget(
        self,
        point: tuple[float, float],
    ) -> QPoint:
        video_rect = self._video_rect()

        x = video_rect.x() + round(point[0] * video_rect.width())

        y = video_rect.y() + round(point[1] * video_rect.height())

        return QPoint(
            x,
            y,
        )

    def _source_to_widget(
        self,
        x: int,
        y: int,
    ) -> QPoint:
        if self._source_width <= 0 or self._source_height <= 0:
            return QPoint()

        return self._normalized_to_widget(
            (
                x / self._source_width,
                y / self._source_height,
            )
        )

    # ==========================================================
    # MOUSE / KEYBOARD
    # ==========================================================

    def mousePressEvent(
        self,
        event,
    ) -> None:
        if not self._drawing_zone:
            super().mousePressEvent(event)
            return

        normalized = self._widget_to_normalized(event.pos())

        if event.button() == Qt.LeftButton:
            if normalized is None:
                return

            self._draft_zone_points.append(normalized)

            self.update()

        elif event.button() == Qt.RightButton:
            if self._draft_zone_points:
                self._draft_zone_points.pop()
                self.update()

    def keyPressEvent(
        self,
        event,
    ) -> None:
        if self._drawing_zone:
            if event.key() in (
                Qt.Key_Return,
                Qt.Key_Enter,
            ):
                self.finish_zone_drawing()
                return

            if event.key() == Qt.Key_Escape:
                self.cancel_zone_drawing()
                return

        super().keyPressEvent(event)

    # ==========================================================
    # PAINTING
    # ==========================================================

    def paintEvent(
        self,
        event,
    ) -> None:
        painter = QPainter(self)

        painter.setRenderHint(QPainter.Antialiasing)

        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        painter.fillRect(
            self.rect(),
            QColor(
                5,
                6,
                9,
            ),
        )

        if self._pixmap is None:
            painter.setPen(
                QColor(
                    90,
                    100,
                    115,
                )
            )

            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                self._message,
            )

            return

        video_rect = self._video_rect()

        painter.drawPixmap(
            video_rect,
            self._pixmap,
        )

        self._draw_zone(painter)

        self._draw_track_paths(painter)

        self._draw_overlays(painter)

    def _draw_zone(
        self,
        painter: QPainter,
    ) -> None:
        if self._drawing_zone:
            points = self._draft_zone_points
        else:
            points = self._zone_points

        if not points:
            return

        widget_points = [self._normalized_to_widget(point) for point in points]

        polygon = QPolygon(widget_points)

        if self._zone_active:
            border_color = QColor(
                255,
                70,
                70,
            )

            fill_color = QColor(
                255,
                60,
                60,
                38,
            )

        else:
            border_color = QColor(
                255,
                190,
                40,
            )

            fill_color = QColor(
                255,
                190,
                40,
                30,
            )

        pen = QPen(border_color)

        pen.setWidth(2)

        painter.setPen(pen)

        if len(points) >= 3:
            painter.setBrush(fill_color)

            painter.drawPolygon(polygon)

        elif len(points) >= 2:
            painter.setBrush(Qt.NoBrush)

            painter.drawPolyline(polygon)

        painter.setBrush(border_color)

        for point in widget_points:
            painter.drawEllipse(
                point,
                4,
                4,
            )

        if self._drawing_zone:
            painter.setPen(
                QColor(
                    235,
                    235,
                    235,
                )
            )

            painter.drawText(
                15,
                25,
                (
                    "DRAWING ZONE  |  "
                    "Left click: add  |  "
                    "Right click: undo  |  "
                    "Enter: save"
                ),
            )

    def _draw_overlays(
        self,
        painter: QPainter,
    ) -> None:
        if self._source_width <= 0 or self._source_height <= 0:
            return

        for overlay in self._overlays:
            x1, y1, x2, y2 = overlay.bounding_box

            top_left = self._source_to_widget(
                x1,
                y1,
            )

            bottom_right = self._source_to_widget(
                x2,
                y2,
            )

            if overlay.alert:
                color = QColor(
                    255,
                    70,
                    70,
                )

            elif overlay.kind == "fire":
                color = QColor(
                    255,
                    70,
                    60,
                )

            elif overlay.kind == "smoke":
                color = QColor(
                    190,
                    195,
                    205,
                )

            else:
                color = QColor(
                    60,
                    220,
                    120,
                )

            pen = QPen(color)

            pen.setWidth(2)

            painter.setPen(pen)

            painter.setBrush(Qt.NoBrush)

            painter.drawRect(
                QRect(
                    top_left,
                    bottom_right,
                )
            )

            text_y = max(
                top_left.y() - 7,
                18,
            )

            painter.drawText(
                top_left.x(),
                text_y,
                overlay.label,
            )


class VideoView(QFrame):
    zone_completed = pyqtSignal(object)
    zone_cleared = pyqtSignal()
    message_requested = pyqtSignal(str)

    def __init__(
        self,
        parent=None,
    ):
        super().__init__(parent)

        self.setObjectName("VideoFrame")

        self.setMinimumSize(
            320,
            180,
        )

        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        self._source_badge = QLabel("NO SOURCE")

        self._source_badge.setObjectName("SourceBadge")

        self._canvas = VideoCanvas()

        self._canvas.zone_completed.connect(self.zone_completed.emit)

        self._canvas.zone_cleared.connect(self.zone_cleared.emit)

        self._canvas.message_requested.connect(self.message_requested.emit)

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            12,
            12,
            12,
            12,
        )

        layout.setSpacing(8)

        top_row = QHBoxLayout()

        top_row.addWidget(self._source_badge)

        top_row.addStretch()

        layout.addLayout(top_row)

        layout.addWidget(
            self._canvas,
            1,
        )

    @property
    def zone_points(self):
        return self._canvas.zone_points

    @property
    def drawing_zone(self):
        return self._canvas.drawing_zone

    def set_source_name(
        self,
        name: str,
    ) -> None:
        self._source_badge.setText(name)

    def set_message(
        self,
        message: str,
    ) -> None:
        self._canvas.set_message(message)

    def set_frame(
        self,
        image: np.ndarray,
    ) -> None:
        self._canvas.set_frame(image)

    def set_overlays(
        self,
        overlays,
    ) -> None:
        self._canvas.set_overlays(overlays)

    def clear_overlays(self) -> None:
        self._canvas.clear_overlays()

    def set_zone_active(
        self,
        active: bool,
    ) -> None:
        self._canvas.set_zone_active(active)

    def begin_zone_drawing(self) -> None:
        self._canvas.begin_zone_drawing()

    def edit_zone(self) -> None:
        self._canvas.edit_zone()

    def finish_zone_drawing(self) -> bool:
        return self._canvas.finish_zone_drawing()

    def cancel_zone_drawing(self) -> None:
        self._canvas.cancel_zone_drawing()

    def clear_zone(self) -> None:
        self._canvas.clear_zone()

    def set_track_paths(
        self,
        track_paths,
    ) -> None:
        self._canvas.set_track_paths(track_paths)
