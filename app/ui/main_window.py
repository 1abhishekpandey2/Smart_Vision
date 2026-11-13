from pathlib import Path
from time import monotonic

from PyQt5.QtCore import (
    QThread,
    Qt,
    pyqtSignal,
)

from PyQt5.QtGui import (
    QKeySequence,
)

from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.ui.face_registration_dialog import FaceRegistrationDialog
from app.ui.webcam_controller import WebcamController

from app.ui.seek_slider import SeekSlider
from app.ui.video_player import (
    VideoPlaybackController,
)
from app.ui.video_view import VideoView

from app.ui.vision_types import (
    VisionRequest,
    VisionSettings,
)

from app.ui.workers.vision_worker import (
    VisionWorker,
)


class MainWindow(QMainWindow):
    vision_request = pyqtSignal(object)

    vision_reset_request = pyqtSignal()

    face_registration_request = pyqtSignal(str, object)

    face_gallery_request = pyqtSignal()

    face_identity_removal_request = pyqtSignal(str)

    face_pose_check_request = pyqtSignal(object, bool)

    face_embedding_registration_request = pyqtSignal(str, object)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Smart Vision")

        self.setMinimumSize(
            1100,
            700,
        )

        # ======================================================
        # RUNTIME STATE
        # ======================================================

        self._selected_video_path = None

        self._timeline_dragging = False
        self._video_duration = 0.0
        self._video_position_seconds = 0.0

        self._last_frame = None

        # Used to reject stale AI results.
        self._analysis_generation = 0

        # Latest-frame processing.
        self._analysis_busy = False
        self._pending_analysis = None

        self._active_source_kind = None
        self._face_gallery_dialog = None
        self._face_gallery_list = None
        self._face_gallery_add_buttons = []
        self._face_gallery_remove_buttons = []
        self._face_registration_busy = False
        self._live_face_dialog = None
        self._live_face_session_active = False
        self._event_keys = set()

        # Create before building widgets because
        # QComboBox signals can fire during construction.
        self.video_player = VideoPlaybackController(self)
        self.webcam_controller = WebcamController(self)

        self._build_ui()

        self._connect_video_player()

        self._connect_webcam_controller()

        self._setup_vision_worker()

        self._setup_shortcuts()

        self.video_view.zone_completed.connect(self._zone_completed)

        self.video_view.zone_cleared.connect(self._zone_cleared)

        self.video_view.message_requested.connect(self.statusBar().showMessage)

        self.statusBar().showMessage("Smart Vision ready")

    # ==========================================================
    # MAIN UI
    # ==========================================================

    def _build_ui(self):
        central = QWidget()

        root = QVBoxLayout(central)

        root.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        root.setSpacing(0)

        root.addWidget(self._build_top_bar())

        self.main_splitter = QSplitter(Qt.Horizontal)

        self.main_splitter.setChildrenCollapsible(False)

        self.main_splitter.setHandleWidth(7)

        source_panel = self._build_source_panel()

        center_panel = self._build_center_panel()

        intelligence_panel = self._build_intelligence_panel()

        source_panel.setMinimumWidth(180)

        source_panel.setMaximumWidth(340)

        center_panel.setMinimumWidth(320)

        center_panel.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        intelligence_panel.setMinimumWidth(260)

        intelligence_panel.setMaximumWidth(400)

        self.main_splitter.addWidget(source_panel)

        self.main_splitter.addWidget(center_panel)

        self.main_splitter.addWidget(intelligence_panel)

        self.main_splitter.setStretchFactor(
            0,
            0,
        )

        self.main_splitter.setStretchFactor(
            1,
            1,
        )

        self.main_splitter.setStretchFactor(
            2,
            0,
        )

        self.main_splitter.setSizes(
            [
                220,
                850,
                310,
            ]
        )

        root.addWidget(
            self.main_splitter,
            1,
        )

        self.setCentralWidget(central)

    # ==========================================================
    # TOP BAR
    # ==========================================================

    def _build_top_bar(self):
        frame = QFrame()

        frame.setObjectName("TopBar")

        layout = QHBoxLayout(frame)

        layout.setContentsMargins(
            18,
            10,
            18,
            10,
        )

        title_layout = QVBoxLayout()

        title_layout.setSpacing(1)

        title = QLabel("SMART VISION")

        title.setObjectName("AppTitle")

        subtitle = QLabel("Intelligent Video Monitoring")

        subtitle.setObjectName("AppSubtitle")

        title_layout.addWidget(title)

        title_layout.addWidget(subtitle)

        layout.addLayout(title_layout)

        layout.addStretch()

        self.start_button = QPushButton("Start Monitoring")

        self.start_button.setObjectName("PrimaryButton")

        self.stop_button = QPushButton("Stop")

        self.stop_button.setObjectName("DangerButton")

        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self._start_clicked)

        self.stop_button.clicked.connect(self._stop_clicked)

        layout.addWidget(self.start_button)

        layout.addWidget(self.stop_button)

        return frame

    # ==========================================================
    # SOURCE PANEL
    # ==========================================================

    def _build_source_panel(self):
        panel = QFrame()

        panel.setObjectName("Panel")

        layout = QVBoxLayout(panel)

        layout.setContentsMargins(
            14,
            15,
            14,
            15,
        )

        layout.setSpacing(9)

        layout.addWidget(self._section_title("SOURCES"))

        self.source_list = QListWidget()

        self.source_list.addItem("Webcam 0")

        self.source_list.itemClicked.connect(self._source_clicked)

        layout.addWidget(
            self.source_list,
            1,
        )

        self.open_video_button = QPushButton("Open Video...")

        self.open_video_button.clicked.connect(self._open_video)

        layout.addWidget(self.open_video_button)

        layout.addSpacing(14)

        layout.addWidget(self._section_title("SOURCE INFO"))

        self.source_info = QLabel("No source selected")

        self.source_info.setWordWrap(True)

        self.source_info.setStyleSheet("color: #8994a4;")

        layout.addWidget(self.source_info)

        layout.addSpacing(14)

        layout.addWidget(self._section_title("FACE REGISTRATION"))

        self.face_identity_input = QLineEdit()

        self.face_identity_input.setPlaceholderText("Identity name")
        self.face_identity_input.setStyleSheet(
            "QLineEdit { color: #000000; background-color: #ffffff; }"
        )

        layout.addWidget(self.face_identity_input)

        self.register_faces_button = QPushButton("Add Images to Gallery")

        self.register_faces_button.clicked.connect(self._register_face_images_clicked)

        layout.addWidget(self.register_faces_button)

        self.live_face_registration_button = QPushButton("Register from Camera")
        self.live_face_registration_button.clicked.connect(
            self._start_live_face_registration
        )
        layout.addWidget(self.live_face_registration_button)

        self.view_face_gallery_button = QPushButton("View Registered People")
        self.view_face_gallery_button.clicked.connect(self._show_face_gallery_clicked)

        layout.addWidget(self.view_face_gallery_button)

        self.face_registration_status = QLabel(
            "Choose photos with exactly one face each."
        )

        self.face_registration_status.setWordWrap(True)

        layout.addWidget(self.face_registration_status)

        layout.addStretch()

        return panel

    # ==========================================================
    # CENTER PANEL
    # ==========================================================

    def _build_center_panel(self):
        widget = QWidget()

        layout = QVBoxLayout(widget)

        layout.setContentsMargins(
            10,
            10,
            10,
            10,
        )

        layout.setSpacing(9)

        self.video_view = VideoView()

        layout.addWidget(
            self.video_view,
            1,
        )

        layout.addWidget(self._build_playback_panel())

        layout.addWidget(self._build_event_panel())

        return widget

    # ==========================================================
    # PLAYBACK
    # ==========================================================

    def _build_playback_panel(self):
        panel = QFrame()

        panel.setObjectName("Panel")

        layout = QVBoxLayout(panel)

        layout.setContentsMargins(
            12,
            10,
            12,
            10,
        )

        controls = QHBoxLayout()

        controls.setSpacing(8)

        self.restart_button = QPushButton("Restart")

        self.play_button = QPushButton("Play")

        self.pause_button = QPushButton("Pause")

        self.restart_button.setEnabled(False)

        self.play_button.setEnabled(False)

        self.pause_button.setEnabled(False)

        self.restart_button.clicked.connect(self._restart_video)

        self.play_button.clicked.connect(self._play_video)

        self.pause_button.clicked.connect(self._pause_video)

        controls.addWidget(self.restart_button)

        controls.addWidget(self.play_button)

        controls.addWidget(self.pause_button)

        controls.addSpacing(10)

        self.current_time_label = QLabel("00:00")

        controls.addWidget(self.current_time_label)

        self.timeline_slider = SeekSlider(Qt.Horizontal)

        self.timeline_slider.setRange(
            0,
            1000,
        )

        self.timeline_slider.setEnabled(False)

        self.timeline_slider.sliderPressed.connect(self._timeline_pressed)

        self.timeline_slider.sliderMoved.connect(self._timeline_moved)

        self.timeline_slider.seek_requested.connect(self._timeline_seek_requested)

        controls.addWidget(
            self.timeline_slider,
            1,
        )

        self.duration_label = QLabel("00:00")

        controls.addWidget(self.duration_label)

        self.speed_combo = QComboBox()

        self.speed_combo.addItems(
            [
                "0.5x",
                "1x",
                "2x",
            ]
        )

        self.speed_combo.setCurrentText("1x")

        self.speed_combo.setEnabled(False)

        self.speed_combo.currentTextChanged.connect(self._speed_changed)

        controls.addWidget(self.speed_combo)

        layout.addLayout(controls)

        return panel

    # ==========================================================
    # EVENTS
    # ==========================================================

    def _build_event_panel(self):
        panel = QFrame()

        panel.setObjectName("Panel")

        panel.setMinimumHeight(170)

        layout = QVBoxLayout(panel)

        layout.setContentsMargins(
            12,
            10,
            12,
            10,
        )

        header = QHBoxLayout()

        header.addWidget(self._section_title("ALERTS & EVENTS"))

        header.addStretch()

        clear_button = QPushButton("Clear")

        clear_button.clicked.connect(self._clear_events)

        header.addWidget(clear_button)

        layout.addLayout(header)

        self.event_table = QTableWidget(
            0,
            3,
        )

        self.event_table.setHorizontalHeaderLabels(
            [
                "Time / offset",
                "Type",
                "Event",
            ]
        )

        self.event_table.verticalHeader().setVisible(False)

        self.event_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.event_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.event_table.setToolTip(
            "Click an event from a recording to jump to its video timestamp."
        )
        self.event_table.setCursor(Qt.PointingHandCursor)
        self.event_table.cellClicked.connect(self._event_row_clicked)

        header_view = self.event_table.horizontalHeader()

        header_view.setSectionResizeMode(
            0,
            QHeaderView.ResizeToContents,
        )

        header_view.setSectionResizeMode(
            1,
            QHeaderView.ResizeToContents,
        )

        header_view.setSectionResizeMode(
            2,
            QHeaderView.Stretch,
        )

        layout.addWidget(self.event_table)

        return panel

    # ==========================================================
    # INTELLIGENCE
    # ==========================================================

    def _build_intelligence_panel(self):
        panel = QFrame()

        panel.setObjectName("Panel")

        layout = QVBoxLayout(panel)

        layout.setContentsMargins(
            15,
            15,
            15,
            15,
        )

        layout.setSpacing(10)

        layout.addWidget(self._section_title("INTELLIGENCE"))

        self.fire_enabled = QCheckBox("Fire / Smoke Detection")

        self.fire_enabled.setChecked(True)

        self.person_enabled = QCheckBox("Person Tracking")

        self.person_enabled.setChecked(True)

        self.face_recognition_enabled = QCheckBox("Face Recognition")

        self.face_recognition_enabled.setChecked(False)

        self.identity_confirmation_observations = QSpinBox()

        self.identity_confirmation_observations.setRange(1, 10)

        self.identity_confirmation_observations.setValue(3)

        identity_confirmation_row = QHBoxLayout()

        identity_confirmation_row.addWidget(QLabel("Strong matches to confirm"))

        identity_confirmation_row.addStretch()

        identity_confirmation_row.addWidget(self.identity_confirmation_observations)

        self.track_paths_enabled = QCheckBox("Show Track Paths")

        self.track_paths_enabled.setChecked(True)

        self.track_paths_enabled.stateChanged.connect(
            self._track_path_visibility_changed
        )

        self.person_enabled.toggled.connect(self.track_paths_enabled.setEnabled)

        self.intrusion_enabled = QCheckBox("Intrusion Monitoring")

        self.intrusion_enabled.setChecked(True)

        self.loitering_enabled = QCheckBox("Loitering / Dwell")

        self.loitering_enabled.setChecked(False)

        layout.addWidget(self.fire_enabled)

        layout.addWidget(self.person_enabled)

        layout.addWidget(self.face_recognition_enabled)

        layout.addLayout(identity_confirmation_row)

        layout.addWidget(self.intrusion_enabled)

        layout.addWidget(self.loitering_enabled)

        layout.addWidget(self.track_paths_enabled)

        self.fire_enabled.stateChanged.connect(self._analysis_configuration_changed)

        self.person_enabled.stateChanged.connect(self._analysis_configuration_changed)

        self.face_recognition_enabled.stateChanged.connect(
            self._analysis_configuration_changed
        )

        self.identity_confirmation_observations.valueChanged.connect(
            self._analysis_configuration_changed
        )

        self.intrusion_enabled.stateChanged.connect(
            self._analysis_configuration_changed
        )

        self.loitering_enabled.stateChanged.connect(
            self._analysis_configuration_changed
        )

        # ======================================================
        # ZONE
        # ======================================================

        layout.addSpacing(10)

        layout.addWidget(self._section_title("RESTRICTED ZONE"))

        self.create_zone_button = QPushButton("Create Zone")

        self.edit_zone_button = QPushButton("Edit Zone")

        self.clear_zone_button = QPushButton("Clear Zone")

        self.clear_zone_button.setObjectName("DangerButton")

        self.create_zone_button.clicked.connect(self._create_zone_clicked)

        self.edit_zone_button.clicked.connect(self._edit_zone_clicked)

        self.clear_zone_button.clicked.connect(self._clear_zone_clicked)

        layout.addWidget(self.create_zone_button)

        layout.addWidget(self.edit_zone_button)

        layout.addWidget(self.clear_zone_button)

        # ======================================================
        # DWELL
        # ======================================================

        layout.addSpacing(10)

        layout.addWidget(self._section_title("DWELL SETTINGS"))

        dwell_row = QHBoxLayout()

        dwell_row.addWidget(QLabel("Allowed dwell"))

        dwell_row.addStretch()

        self.dwell_seconds = QSpinBox()

        self.dwell_seconds.setRange(
            1,
            3600,
        )

        self.dwell_seconds.setValue(15)

        self.dwell_seconds.setSuffix(" sec")

        # Be explicit that this is coming next.
        self.dwell_seconds.setEnabled(False)

        dwell_row.addWidget(self.dwell_seconds)

        layout.addLayout(dwell_row)
        self.loitering_enabled.toggled.connect(self._update_dwell_controls)

        self.person_enabled.toggled.connect(self._update_dwell_controls)

        self.person_enabled.toggled.connect(self._update_face_controls)

        self.face_recognition_enabled.toggled.connect(self._update_face_controls)

        self._update_face_controls()

        self.dwell_seconds.valueChanged.connect(self._analysis_configuration_changed)

        # ======================================================
        # CONFIDENCE
        # ======================================================

        layout.addSpacing(10)

        layout.addWidget(self._section_title("DETECTION SETTINGS"))

        fire_row = QHBoxLayout()

        fire_row.addWidget(QLabel("Fire confidence"))

        fire_row.addStretch()

        self.fire_confidence = QDoubleSpinBox()

        self.fire_confidence.setRange(
            0.05,
            1.0,
        )

        self.fire_confidence.setSingleStep(0.05)

        self.fire_confidence.setValue(0.40)

        fire_row.addWidget(self.fire_confidence)

        layout.addLayout(fire_row)

        person_row = QHBoxLayout()

        person_row.addWidget(QLabel("Person confidence"))

        person_row.addStretch()

        self.person_confidence = QDoubleSpinBox()

        self.person_confidence.setRange(
            0.05,
            1.0,
        )

        self.person_confidence.setSingleStep(0.05)

        self.person_confidence.setValue(0.35)

        person_row.addWidget(self.person_confidence)

        layout.addLayout(person_row)

        self.fire_confidence.valueChanged.connect(self._analysis_configuration_changed)

        self.person_confidence.valueChanged.connect(
            self._analysis_configuration_changed
        )

        # ======================================================
        # METRICS
        # ======================================================

        layout.addSpacing(10)

        layout.addWidget(self._section_title("LIVE METRICS"))

        self.people_metric = QLabel("Tracked people: --")

        self.fire_metric = QLabel("Fire evidence: --")

        self.zone_metric = QLabel("Inside zone: --")

        self.loiter_metric = QLabel("Loitering: --")

        layout.addWidget(self.people_metric)

        layout.addWidget(self.fire_metric)

        layout.addWidget(self.zone_metric)

        layout.addWidget(self.loiter_metric)

        layout.addStretch()

        return panel

    def _update_dwell_controls(
        self,
        *args,
    ):
        person_enabled = self.person_enabled.isChecked()

        self.loitering_enabled.setEnabled(person_enabled)

        self.dwell_seconds.setEnabled(
            person_enabled and self.loitering_enabled.isChecked()
        )

    def _update_face_controls(
        self,
        *args,
    ):
        self.face_recognition_enabled.setEnabled(self.person_enabled.isChecked())

        self.identity_confirmation_observations.setEnabled(
            self.person_enabled.isChecked()
            and self.face_recognition_enabled.isChecked()
        )

    # ==========================================================
    # VIDEO PLAYER CONNECTION
    # ==========================================================

    def _connect_video_player(self):
        self.video_player.frame_ready.connect(self._video_frame_ready)

        self.video_player.position_changed.connect(self._video_position_changed)

        self.video_player.duration_changed.connect(self._video_duration_changed)

        self.video_player.playback_state_changed.connect(self._playback_state_changed)

        self.video_player.playback_finished.connect(self._video_finished)

        self.video_player.error_occurred.connect(self._video_error)

    def _connect_webcam_controller(
        self,
    ):
        self.webcam_controller.frame_ready.connect(self._video_frame_ready)

        self.webcam_controller.stream_state_changed.connect(self._webcam_state_changed)

        self.webcam_controller.error_occurred.connect(self._webcam_error)

        self.webcam_controller.status_changed.connect(self.statusBar().showMessage)

    # ==========================================================
    # VISION WORKER
    # ==========================================================

    def _setup_vision_worker(self):
        self._vision_thread = QThread(self)

        self._vision_worker = VisionWorker()

        self._vision_worker.moveToThread(self._vision_thread)

        self.vision_request.connect(self._vision_worker.process)

        self.face_registration_request.connect(self._vision_worker.register_face_images)

        self.face_gallery_request.connect(self._vision_worker.load_face_gallery)

        self.face_identity_removal_request.connect(
            self._vision_worker.remove_face_identity
        )

        self.face_pose_check_request.connect(
            self._vision_worker.inspect_face_pose_frame
        )

        self.face_embedding_registration_request.connect(
            self._vision_worker.register_face_embeddings
        )

        self.vision_reset_request.connect(self._vision_worker.reset_runtime_state)

        self._vision_worker.result_ready.connect(self._vision_result_ready)

        self._vision_worker.analysis_finished.connect(self._vision_analysis_finished)

        self._vision_worker.error_occurred.connect(self._vision_error)

        self._vision_worker.status_changed.connect(self.statusBar().showMessage)

        self._vision_worker.face_registration_succeeded.connect(
            self._face_registration_succeeded
        )

        self._vision_worker.face_registration_failed.connect(
            self._face_registration_failed
        )

        self._vision_worker.face_gallery_ready.connect(self._face_gallery_loaded)

        self._vision_worker.face_gallery_failed.connect(self._face_gallery_failed)

        self._vision_worker.face_identity_removal_succeeded.connect(
            self._face_identity_removal_succeeded
        )

        self._vision_worker.face_identity_removal_failed.connect(
            self._face_identity_removal_failed
        )

        self._vision_worker.face_pose_check_ready.connect(
            self._live_face_pose_check_ready
        )

        self._vision_thread.start()

    def _register_face_images_clicked(self):
        if self._face_registration_busy or self._live_face_session_active:
            return

        identity = self.face_identity_input.text().strip()

        if not identity:
            QMessageBox.warning(
                self,
                "Face registration",
                "Enter an identity name first.",
            )
            return

        image_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select face photos",
            str(Path("data/face_registry")),
            "Face images (*.jpg *.jpeg *.png)",
        )

        if not image_paths:
            return

        self._begin_face_registration(identity, image_paths)

    def _start_live_face_registration(self) -> None:
        if self._live_face_session_active:
            self._live_face_dialog.raise_()
            return
        if self._face_registration_busy:
            return
        identity = self.face_identity_input.text().strip()
        if not identity:
            QMessageBox.warning(
                self,
                "Face registration",
                "Enter an identity name first.",
            )
            return

        if (
            self._active_source_kind != "webcam"
            or not self.webcam_controller.has_source
        ):
            self._source_clicked(self.source_list.item(0))
        if not self.webcam_controller.has_source:
            return

        dialog = FaceRegistrationDialog(identity, self)
        dialog.pose_check_requested.connect(self._begin_live_pose_check)
        dialog.registration_requested.connect(self._begin_live_embedding_registration)
        self._vision_worker.face_registration_succeeded.connect(
            dialog.registration_succeeded
        )
        self._vision_worker.face_registration_failed.connect(dialog.registration_failed)
        dialog.finished.connect(
            lambda _result, target=dialog: self._disconnect_live_face_dialog(target)
        )
        self._live_face_dialog = dialog
        self._live_face_session_active = True
        self._set_face_registration_controls_enabled(False)
        dialog.show()
        if self._last_frame is not None:
            dialog.set_frame(self._last_frame.image)

    def _begin_live_embedding_registration(self, identity, embeddings) -> None:
        if self._face_registration_busy:
            return
        self._face_registration_busy = True
        self._set_face_registration_controls_enabled(False)
        self.live_face_registration_button.setEnabled(False)
        self.face_registration_status.setText(
            f"Saving {len(embeddings)} live face sample(s) for {identity}..."
        )
        self.face_embedding_registration_request.emit(identity, embeddings)

    def _begin_live_pose_check(self, image, manual_capture: bool) -> None:
        self._face_registration_busy = True
        self._set_face_registration_controls_enabled(False)
        self.live_face_registration_button.setEnabled(False)
        self.face_pose_check_request.emit(image, manual_capture)

    def _live_face_pose_check_ready(self, result) -> None:
        self._face_registration_busy = False
        self._set_face_registration_controls_enabled(True)
        if self._live_face_dialog is not None:
            self._live_face_dialog.handle_pose_check(result)

    def _disconnect_live_face_dialog(self, dialog) -> None:
        try:
            self._vision_worker.face_registration_succeeded.disconnect(
                dialog.registration_succeeded
            )
            self._vision_worker.face_registration_failed.disconnect(
                dialog.registration_failed
            )
        except TypeError:
            pass
        if self._live_face_dialog is dialog:
            self._live_face_dialog = None
            self._live_face_session_active = False
            self._set_face_registration_controls_enabled(
                enabled=not self._face_registration_busy
            )

    def _begin_face_registration(self, identity, image_paths) -> None:
        self._face_registration_busy = True
        self._set_face_registration_controls_enabled(False)
        self.face_registration_status.setText(
            f"Registering {len(image_paths)} image(s) for {identity}..."
        )
        self.face_registration_request.emit(identity, image_paths)

    def _set_face_registration_controls_enabled(self, enabled: bool) -> None:
        enabled = enabled and not self._live_face_session_active
        self.register_faces_button.setEnabled(enabled)
        self.live_face_registration_button.setEnabled(enabled)
        for button in self._face_gallery_add_buttons:
            button.setEnabled(enabled)
        for button in self._face_gallery_remove_buttons:
            button.setEnabled(enabled)

    def _show_face_gallery_clicked(self) -> None:
        if self._face_gallery_dialog is None:
            self._create_face_gallery_dialog()

        self._face_gallery_dialog.show()
        self._face_gallery_dialog.raise_()
        self._refresh_face_gallery()

    def _create_face_gallery_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Registered People")
        dialog.setMinimumSize(420, 320)
        dialog.setStyleSheet("""
            QDialog { background-color: #10141b; color: #e9edf5; }
            QLabel { color: #e9edf5; background-color: transparent; }
            QListWidget {
                background-color: #10141b;
                color: #f5f7fb;
                border: 1px solid #303846;
                outline: none;
            }
            QListWidget::item { padding: 4px; border-radius: 5px; }
            QListWidget::item:selected { background-color: #202938; }
            QListWidget::item:hover { background-color: #181e27; }
            QWidget#FaceGalleryRow {
                background-color: #181e27;
                border: 1px solid #303846;
                border-radius: 5px;
            }
            QPushButton#RemoveFaceIdentity {
                color: #ffd9d9;
                background-color: #43272d;
                border: 1px solid #74414a;
                border-radius: 5px;
                padding: 5px 9px;
            }
            QPushButton#RemoveFaceIdentity:hover { background-color: #5a3038; }
            """)

        layout = QVBoxLayout(dialog)
        description = QLabel(
            "Each saved sample is a face embedding. Add images to an existing "
            "person or remove that person and all of their saved samples."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self._face_gallery_list = QListWidget(dialog)
        layout.addWidget(self._face_gallery_list, 1)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)

        self._face_gallery_dialog = dialog

    def _refresh_face_gallery(self) -> None:
        if self._face_gallery_list is None:
            return

        self._face_gallery_list.clear()
        self._face_gallery_add_buttons = []
        self._face_gallery_remove_buttons = []
        self._face_gallery_list.addItem("Loading registered people...")
        self.face_gallery_request.emit()

    def _face_gallery_loaded(self, identity_sample_counts) -> None:
        if self._face_gallery_list is None:
            return

        self._face_gallery_list.clear()
        self._face_gallery_add_buttons = []
        self._face_gallery_remove_buttons = []

        if not identity_sample_counts:
            self._face_gallery_list.addItem(
                "No people registered yet. Add the first person in the Sources panel."
            )
            return

        for identity, sample_count in identity_sample_counts:
            item = QListWidgetItem()
            row = QWidget(self._face_gallery_list)
            row.setObjectName("FaceGalleryRow")
            row.setAttribute(Qt.WA_StyledBackground)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(8, 4, 8, 4)

            identity_label = QLabel(identity, row)
            identity_label.setStyleSheet(
                "color: #f5f7fb; background-color: transparent; font-weight: 600;"
            )
            count_label = QLabel(f"{sample_count} sample(s)", row)
            count_label.setStyleSheet("color: #b8c1cf; background-color: transparent;")
            add_button = QPushButton("Add Images", row)
            add_button.setEnabled(not self._face_registration_busy)
            add_button.clicked.connect(
                lambda checked=False, name=identity: self._add_images_for_identity(name)
            )
            remove_button = QPushButton("Remove", row)
            remove_button.setObjectName("RemoveFaceIdentity")
            remove_button.setToolTip(
                f"Remove {sample_count} saved face sample(s) for {identity}"
            )
            remove_button.setEnabled(not self._face_registration_busy)
            remove_button.clicked.connect(
                lambda checked=False, name=identity, count=sample_count: (
                    self._remove_face_identity_clicked(name, count)
                )
            )

            row_layout.addWidget(identity_label, 1)
            row_layout.addWidget(count_label)
            row_layout.addWidget(add_button)
            row_layout.addWidget(remove_button)

            item.setSizeHint(row.sizeHint())
            self._face_gallery_list.addItem(item)
            self._face_gallery_list.setItemWidget(item, row)
            self._face_gallery_add_buttons.append(add_button)
            self._face_gallery_remove_buttons.append(remove_button)

    def _face_gallery_failed(self, message: str) -> None:
        if self._face_gallery_list is not None:
            self._face_gallery_list.clear()
            self._face_gallery_list.addItem(f"Could not load gallery: {message}")

    def _add_images_for_identity(self, identity: str) -> None:
        if self._face_registration_busy:
            return

        image_paths, _ = QFileDialog.getOpenFileNames(
            self._face_gallery_dialog or self,
            f"Add face photos for {identity}",
            str(Path("data/face_registry")),
            "Face images (*.jpg *.jpeg *.png)",
        )
        if image_paths:
            self._begin_face_registration(identity, image_paths)

    def _remove_face_identity_clicked(self, identity: str, sample_count: int) -> None:
        if self._face_registration_busy or self._live_face_session_active:
            return

        response = QMessageBox.question(
            self._face_gallery_dialog or self,
            "Remove registered person",
            f"Remove '{identity}' and all {sample_count} saved face sample(s) "
            "from the local gallery?\n\n"
            "This cannot be undone. Original image files are not changed.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if response != QMessageBox.Yes:
            return

        self._face_registration_busy = True
        self._set_face_registration_controls_enabled(False)
        self.face_registration_status.setText(
            f"Removing {identity} from the gallery..."
        )
        self.face_identity_removal_request.emit(identity)

    def _face_identity_removal_succeeded(self, identity: str) -> None:
        self._face_registration_busy = False
        self._set_face_registration_controls_enabled(True)
        message = f"Removed '{identity}' and all of its saved face samples."
        self.face_registration_status.setText(message)
        self.statusBar().showMessage(message, 10000)
        if (
            self._face_gallery_dialog is not None
            and self._face_gallery_dialog.isVisible()
        ):
            self._refresh_face_gallery()

    def _face_identity_removal_failed(self, message: str) -> None:
        self._face_registration_busy = False
        self._set_face_registration_controls_enabled(True)
        self.face_registration_status.setText("Could not remove registered person.")
        QMessageBox.warning(self, "Face removal failed", message)

    def _face_registration_succeeded(
        self,
        message: str,
    ):
        self._face_registration_busy = False
        self._set_face_registration_controls_enabled(True)
        self.face_registration_status.setText(message)
        self.statusBar().showMessage(message, 10000)
        if (
            self._face_gallery_dialog is not None
            and self._face_gallery_dialog.isVisible()
        ):
            self._refresh_face_gallery()

    def _face_registration_failed(
        self,
        message: str,
    ):
        self._face_registration_busy = False
        self._set_face_registration_controls_enabled(True)
        self.face_registration_status.setText("Registration failed.")
        QMessageBox.warning(self, "Face registration failed", message)

    # ==========================================================
    # SHORTCUTS
    # ==========================================================

    def _setup_shortcuts(self):
        self._space_shortcut = QShortcut(
            QKeySequence("Space"),
            self,
        )

        self._space_shortcut.setContext(Qt.WindowShortcut)

        self._space_shortcut.activated.connect(self._toggle_playback)

    # ==========================================================
    # SOURCE
    # ==========================================================

    def _open_video(self):
        # ==========================================================
        # CLOSED RECORDED VIDEO
        # ==========================================================

        self.webcam_controller.close_camera()

        # ==========================================================
        # OPEN VIDEO FILE
        # ==========================================================

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open CCTV Video",
            str(Path("data/videos")),
            ("Video Files (*.mp4 *.avi *.mov *.mkv)"),
        )

        if not path:
            return

        self._reset_vision_runtime(reanalyse=False)

        self._clear_events()

        video_name = Path(path).name

        self.video_view.set_source_name(video_name)

        self._active_source_kind = "video"
        self._selected_video_path = path
        self._video_position_seconds = 0.0

        success = self.video_player.open_video(path)

        if not success:
            self._active_source_kind = None
            self._selected_video_path = None
            return

        self.source_info.setText(f"Recorded video\n\n{path}")

        self.play_button.setEnabled(True)

        self.restart_button.setEnabled(True)

        self.timeline_slider.setEnabled(True)

        self.speed_combo.setEnabled(True)

        self.statusBar().showMessage(f"Loaded video: {video_name}")

    def _webcam_state_changed(
        self,
        running: bool,
    ):
        if self._active_source_kind != "webcam":
            return

        self.start_button.setEnabled(not running)

        self.stop_button.setEnabled(running)

        if running:
            self.current_time_label.setText("LIVE")

    def _webcam_error(
        self,
        message: str,
    ):
        self.statusBar().showMessage(message)

        self.video_view.set_message("Webcam unavailable")

        self.start_button.setEnabled(True)

        self.stop_button.setEnabled(False)

    def _source_clicked(
        self,
        item,
    ):
        source_name = item.text()

        if source_name != "Webcam 0":
            return

        # ==========================================================
        # STOP RECORDED VIDEO
        # ==========================================================

        self._clear_events()
        self.video_player.close_video()

        self._selected_video_path = None

        # ==========================================================
        # RESET OLD AI STATE
        # ==========================================================

        self._reset_vision_runtime(reanalyse=False)

        self._last_frame = None

        # A polygon belongs to a particular camera scene.
        # Do not reuse a polygon from a recorded file on the webcam.
        self.video_view.clear_zone()

        self.video_view.clear_overlays()

        # ==========================================================
        # UI
        # ==========================================================

        self.video_view.set_source_name("Webcam 0")

        self.source_info.setText("Live source\n\nWebcam 0")

        # Playback-only controls do not apply to live CCTV.
        self.play_button.setEnabled(False)

        self.pause_button.setEnabled(False)

        self.restart_button.setEnabled(False)

        self.timeline_slider.setEnabled(False)

        self.timeline_slider.setValue(0)

        self.current_time_label.setText("LIVE")

        self.duration_label.setText("")

        self.speed_combo.setEnabled(False)

        # ==========================================================
        # OPEN WEBCAM
        # ==========================================================
        self._active_source_kind = "webcam"
        success = self.webcam_controller.open_camera(0)

        if not success:
            self._active_source_kind = None

            self.video_view.set_message("Unable to open Webcam 0")

            return

        self.statusBar().showMessage("Webcam 0 connected")

    # ==========================================================
    # VIDEO FRAME
    # ==========================================================

    def _video_frame_ready(
        self,
        frame,
    ):
        self._last_frame = frame

        if self._live_face_dialog is not None and self._live_face_dialog.isVisible():
            self._live_face_dialog.set_frame(frame.image)

        self.video_view.set_frame(frame.image)

        self._submit_analysis(frame)

    # ==========================================================
    # ANALYSIS
    # ==========================================================

    def _current_vision_settings(
        self,
    ):
        return VisionSettings(
            fire_enabled=(self.fire_enabled.isChecked()),
            person_enabled=(self.person_enabled.isChecked()),
            face_recognition_enabled=(
                self.face_recognition_enabled.isChecked()
                and self.person_enabled.isChecked()
            ),
            identity_confirmation_observations=(
                self.identity_confirmation_observations.value()
            ),
            intrusion_enabled=(self.intrusion_enabled.isChecked()),
            loitering_enabled=(
                self.loitering_enabled.isChecked() and self.person_enabled.isChecked()
            ),
            show_track_paths=(self.track_paths_enabled.isChecked()),
            dwell_seconds=(self.dwell_seconds.value()),
            fire_confidence=(self.fire_confidence.value()),
            person_confidence=(self.person_confidence.value()),
        )

    def _submit_analysis(
        self,
        frame,
    ):
        settings = self._current_vision_settings()

        if not (settings.fire_enabled or settings.person_enabled):
            self.video_view.clear_overlays()

            self.update_metrics(
                people=0,
                fire_evidence="OFF",
                inside_zone=0,
                loitering=0,
            )

            return

        source_time_seconds = (
            self._video_position_seconds
            if self._active_source_kind == "video"
            else monotonic()
        )

        request = VisionRequest(
            frame=frame,
            settings=settings,
            zone_points=(tuple(self.video_view.zone_points)),
            source_time_seconds=source_time_seconds,
            generation=(self._analysis_generation),
            source_id=(
                str(self._selected_video_path)
                if self._active_source_kind == "video"
                and self._selected_video_path is not None
                else None
            ),
        )

        # AI already working:
        # replace the pending request rather than
        # creating a large queue.
        if self._analysis_busy:
            self._pending_analysis = request
            return

        self._dispatch_analysis(request)

    def _dispatch_analysis(
        self,
        request,
    ):
        self._analysis_busy = True

        self.vision_request.emit(request)

    def _vision_result_ready(
        self,
        result,
    ):
        # Ignore stale results from before a seek,
        # zone change or reset.
        if result.generation != self._analysis_generation:
            return

        self.video_view.set_overlays(result.overlays)

        self.video_view.set_track_paths(result.track_paths)

        self.video_view.set_zone_active(result.zone_active)

        self.update_metrics(
            people=(result.people_count),
            fire_evidence=(result.fire_evidence),
            inside_zone=(result.inside_zone_count),
            loitering=(result.loitering_count),
        )

        for event in result.events:
            if event.event_type == "IDENTITY" and event.track_id is not None:
                if event.identity is not None:
                    self._annotate_recent_track_events(
                        event.track_id,
                        event.identity,
                        event.source_id,
                        event.source_time_seconds,
                    )
            if event.source_id is not None and event.source_time_seconds is not None:
                time_text = self._format_event_video_time(event.source_time_seconds)
            else:
                time_text = event.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.add_event(
                time_text,
                event.event_type,
                event.description,
                source_id=event.source_id,
                source_time_seconds=event.source_time_seconds,
                track_id=event.track_id,
                identity=event.identity,
            )

    def _vision_analysis_finished(
        self,
    ):
        self._analysis_busy = False

        if self._pending_analysis is None:
            return

        request = self._pending_analysis

        self._pending_analysis = None

        # Ignore pending frames belonging to
        # an old generation.
        if request.generation != self._analysis_generation:
            return

        self._dispatch_analysis(request)

    def _vision_error(
        self,
        message: str,
    ):
        self.statusBar().showMessage(message)

    def _track_path_visibility_changed(
        self,
        *args,
    ):
        visible = self.track_paths_enabled.isChecked()

        if not visible:
            self.video_view.set_track_paths(())
            return

        if self._last_frame is not None and self.person_enabled.isChecked():
            self._submit_analysis(self._last_frame)

    def _analysis_configuration_changed(
        self,
        *args,
    ):
        # During UI construction the worker
        # has not yet been created.
        if not hasattr(
            self,
            "_vision_thread",
        ):
            return

        self._reset_vision_runtime(reanalyse=True)

    def _reset_vision_runtime(
        self,
        reanalyse: bool,
    ):
        self._analysis_generation += 1

        self._pending_analysis = None

        self.video_view.clear_overlays()

        if hasattr(
            self,
            "_vision_thread",
        ):
            self.vision_reset_request.emit()

        if reanalyse and self._last_frame is not None:
            self._submit_analysis(self._last_frame)

    # ==========================================================
    # PLAY / PAUSE
    # ==========================================================

    def _play_video(self):
        if self.video_player.has_source:
            self.video_player.play()

    def _pause_video(self):
        self.video_player.pause()

    def _toggle_playback(self):

        # ==========================================================
        # RECORDED VIDEO
        # ==========================================================

        if self._active_source_kind == "video":
            if not self.video_player.has_source:
                return

            if self.video_player.is_playing:
                self.video_player.pause()

            else:
                self.video_player.play()

            return

        # ==========================================================
        # LIVE CAMERA
        # ==========================================================

        if self._active_source_kind == "webcam":
            if not self.webcam_controller.has_source:
                return

            if self.webcam_controller.is_running:
                self.webcam_controller.pause()

            else:
                self.webcam_controller.start()

    def _restart_video(self):
        self._video_position_seconds = 0.0

        self._reset_vision_runtime(reanalyse=False)

        self.video_player.restart()

    def _start_clicked(self):

        # ==========================================================
        # RECORDED VIDEO
        # ==========================================================

        if self._active_source_kind == "video":
            if self.video_player.has_source:
                self.video_player.play()

            return

        # ==========================================================
        # LIVE WEBCAM
        # ==========================================================

        if self._active_source_kind == "webcam":
            if self.webcam_controller.has_source:
                self.webcam_controller.start()

            return

        # ==========================================================
        # NO SOURCE
        # ==========================================================

        self.statusBar().showMessage("Select a source first.")

    def _stop_clicked(self):

        if self._active_source_kind == "video":
            self.video_player.pause()

        elif self._active_source_kind == "webcam":
            self.webcam_controller.pause()

    def _playback_state_changed(
        self,
        playing: bool,
    ):
        has_source = self.video_player.has_source

        self.play_button.setEnabled(has_source and not playing)

        self.pause_button.setEnabled(has_source and playing)

        self.start_button.setEnabled(not playing)

        self.stop_button.setEnabled(playing)

    # ==========================================================
    # TIMELINE
    # ==========================================================

    def _video_position_changed(
        self,
        seconds: float,
    ):
        self._video_position_seconds = seconds

        self.current_time_label.setText(self._format_time(seconds))

        if self._timeline_dragging or self._video_duration <= 0:
            return

        ratio = seconds / self._video_duration

        value = round(ratio * 1000)

        self.timeline_slider.setValue(
            max(
                0,
                min(
                    value,
                    1000,
                ),
            )
        )

    def _video_duration_changed(
        self,
        seconds: float,
    ):
        self._video_duration = seconds

        self.duration_label.setText(self._format_time(seconds))

    def _timeline_pressed(self):
        self._timeline_dragging = True

    def _timeline_moved(
        self,
        value: int,
    ):
        if self._video_duration <= 0:
            return

        seconds = value / 1000.0 * self._video_duration

        self.current_time_label.setText(self._format_time(seconds))

    def _timeline_seek_requested(
        self,
        value: int,
    ):
        self._timeline_dragging = False

        ratio = value / 1000.0

        self._video_position_seconds = ratio * self._video_duration

        self._reset_vision_runtime(reanalyse=False)

        self.video_player.seek_ratio(ratio)

    # ==========================================================
    # SPEED
    # ==========================================================

    def _speed_changed(
        self,
        text: str,
    ):
        try:
            speed = float(
                text.lower()
                .replace(
                    "x",
                    "",
                )
                .strip()
            )

        except ValueError:
            return

        self.video_player.set_speed(speed)

    # ==========================================================
    # ZONES
    # ==========================================================

    def _create_zone_clicked(self):
        if self.video_view.drawing_zone:
            self.video_view.finish_zone_drawing()
            return

        self.video_view.begin_zone_drawing()

        if self.video_view.drawing_zone:
            self.create_zone_button.setText("Finish Zone")

    def _edit_zone_clicked(self):
        self.video_view.edit_zone()

        if self.video_view.drawing_zone:
            self.create_zone_button.setText("Finish Zone")

    def _clear_zone_clicked(self):
        self.video_view.clear_zone()

    def _zone_completed(
        self,
        points,
    ):
        self.create_zone_button.setText("Create Zone")

        self._reset_vision_runtime(reanalyse=True)

        self.statusBar().showMessage(
            f"Restricted zone saved with {len(points)} points."
        )

    def _zone_cleared(self):
        self.create_zone_button.setText("Create Zone")

        self._reset_vision_runtime(reanalyse=True)

        self.statusBar().showMessage("Restricted zone cleared.")

    # ==========================================================
    # VIDEO END / ERRORS
    # ==========================================================

    def _video_finished(self):
        self.statusBar().showMessage("Video playback completed.")

    def _video_error(
        self,
        message: str,
    ):
        self.video_view.set_message("Unable to open video.")

        self.statusBar().showMessage(message)

    # ==========================================================
    # EVENTS
    # ==========================================================

    def _clear_events(self):
        self.event_table.setRowCount(0)
        self._event_keys.clear()

    def add_event(
        self,
        time_text: str,
        event_type: str,
        description: str,
        source_id: str | None = None,
        source_time_seconds: float | None = None,
        track_id: int | None = None,
        identity: str | None = None,
    ):
        if source_id is not None and source_time_seconds is not None:
            event_key = (
                source_id,
                round(source_time_seconds, 6),
                event_type,
                description,
            )
        else:
            event_key = (time_text, event_type, description)
        if event_key in self._event_keys:
            return
        self._event_keys.add(event_key)

        row = self.event_table.rowCount()

        self.event_table.insertRow(row)

        time_item = QTableWidgetItem(time_text)
        time_item.setData(Qt.UserRole, source_time_seconds)
        time_item.setData(Qt.UserRole + 1, source_id)
        self.event_table.setItem(row, 0, time_item)

        self.event_table.setItem(
            row,
            1,
            QTableWidgetItem(event_type),
        )

        description_item = QTableWidgetItem(
            self._with_identity_name(description, track_id, identity)
        )
        description_item.setData(Qt.UserRole, track_id)
        description_item.setData(Qt.UserRole + 1, identity)
        self.event_table.setItem(row, 2, description_item)

        self.event_table.scrollToBottom()

    @staticmethod
    def _format_event_video_time(seconds: float) -> str:
        milliseconds = round(max(0.0, seconds) * 1000)
        hours, remainder = divmod(milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        whole_seconds, milliseconds = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"

    @staticmethod
    def _with_identity_name(
        description: str,
        track_id: int | None,
        identity: str | None,
    ) -> str:
        if track_id is None or identity is None:
            return description
        marker = f"Person #{track_id}"
        named_marker = f"{identity} ({marker})"
        if named_marker in description or marker not in description:
            return description
        return description.replace(marker, named_marker, 1)

    def _annotate_recent_track_events(
        self,
        track_id: int,
        identity: str,
        source_id: str | None,
        identity_time_seconds: float | None,
    ) -> None:
        if identity_time_seconds is None:
            return

        person_event_types = {"INTRUSION", "EXIT", "LOITERING"}
        for row in range(self.event_table.rowCount()):
            event_type_item = self.event_table.item(row, 1)
            time_item = self.event_table.item(row, 0)
            description_item = self.event_table.item(row, 2)
            if event_type_item is None or time_item is None or description_item is None:
                continue
            if event_type_item.text() not in person_event_types:
                continue
            if description_item.data(Qt.UserRole) != track_id:
                continue
            if description_item.data(Qt.UserRole + 1) is not None:
                continue
            if time_item.data(Qt.UserRole + 1) != source_id:
                continue

            event_time = time_item.data(Qt.UserRole)
            if event_time is None or not 0 <= identity_time_seconds - event_time <= 2.0:
                continue

            description_item.setText(
                self._with_identity_name(description_item.text(), track_id, identity)
            )
            description_item.setData(Qt.UserRole + 1, identity)

    def _event_row_clicked(self, row: int, _column: int) -> None:
        time_item = self.event_table.item(row, 0)
        if time_item is None:
            return

        source_time_seconds = time_item.data(Qt.UserRole)
        source_id = time_item.data(Qt.UserRole + 1)
        if source_id is None or source_time_seconds is None:
            self.statusBar().showMessage(
                f"Live event at {time_item.text()}; there is no recording position to seek to.",
                5000,
            )
            return

        if (
            self._active_source_kind != "video"
            or self._selected_video_path is None
            or str(self._selected_video_path) != source_id
            or not self.video_player.has_source
        ):
            self.statusBar().showMessage(
                "Open the recording that produced this event to jump to it.",
                5000,
            )
            return

        seconds = float(source_time_seconds)
        self.video_player.pause()
        self._reset_vision_runtime(reanalyse=False)
        if not self.video_player.seek_seconds(seconds):
            self.statusBar().showMessage("Could not seek to this event.", 5000)
            return

        self.statusBar().showMessage(
            f"Jumped to event at {self._format_event_video_time(seconds)}.",
            5000,
        )

    # ==========================================================
    # METRICS
    # ==========================================================

    def update_metrics(
        self,
        people: int,
        fire_evidence: str,
        inside_zone: int,
        loitering: int = 0,
    ):
        self.people_metric.setText(f"Tracked people: {people}")

        self.fire_metric.setText(f"Fire evidence: {fire_evidence}")

        self.zone_metric.setText(f"Inside zone: {inside_zone}")

        self.loiter_metric.setText(f"Loitering: {loitering}")

    # ==========================================================
    # HELPERS
    # ==========================================================

    @staticmethod
    def _section_title(
        text: str,
    ):
        label = QLabel(text)

        label.setObjectName("SectionTitle")

        return label

    @staticmethod
    def _format_time(
        seconds: float,
    ) -> str:
        seconds = max(
            0,
            round(seconds),
        )

        hours = seconds // 3600

        minutes = (seconds % 3600) // 60

        remaining = seconds % 60

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{remaining:02d}"

        return f"{minutes:02d}:{remaining:02d}"

    # ==========================================================
    # CLOSE
    # ==========================================================

    def closeEvent(
        self,
        event,
    ):
        self.video_player.close_video()
        self.webcam_controller.close_camera()

        self._analysis_generation += 1
        self._pending_analysis = None

        self._vision_thread.quit()

        # Wait for any current YOLO inference
        # to finish before destroying the thread.
        self._vision_thread.wait()

        super().closeEvent(event)
