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
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMainWindow,
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

        header.addWidget(self._section_title("RECENT EVENTS"))

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
                "Time",
                "Type",
                "Event",
            ]
        )

        self.event_table.verticalHeader().setVisible(False)

        self.event_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.event_table.setSelectionBehavior(QAbstractItemView.SelectRows)

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

        layout.addWidget(self.intrusion_enabled)

        layout.addWidget(self.loitering_enabled)

        layout.addWidget(self.track_paths_enabled)

        self.fire_enabled.stateChanged.connect(self._analysis_configuration_changed)

        self.person_enabled.stateChanged.connect(self._analysis_configuration_changed)

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

        self.vision_reset_request.connect(self._vision_worker.reset_runtime_state)

        self._vision_worker.result_ready.connect(self._vision_result_ready)

        self._vision_worker.analysis_finished.connect(self._vision_analysis_finished)

        self._vision_worker.error_occurred.connect(self._vision_error)

        self._vision_worker.status_changed.connect(self.statusBar().showMessage)

        self._vision_thread.start()

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
            ("Video Files " "(*.mp4 *.avi *.mov *.mkv)"),
        )

        if not path:
            return

        self._reset_vision_runtime(reanalyse=False)

        video_name = Path(path).name

        self.video_view.set_source_name(video_name)

        success = self.video_player.open_video(path)

        if not success:
            return

        self._active_source_kind = "video"

        self._video_position_seconds = 0.0

        self._selected_video_path = path

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
            self.add_event(
                event.timestamp.strftime("%H:%M:%S"),
                event.event_type,
                event.description,
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
            f"Restricted zone saved " f"with {len(points)} points."
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

    def add_event(
        self,
        time_text: str,
        event_type: str,
        description: str,
    ):
        row = self.event_table.rowCount()

        self.event_table.insertRow(row)

        self.event_table.setItem(
            row,
            0,
            QTableWidgetItem(time_text),
        )

        self.event_table.setItem(
            row,
            1,
            QTableWidgetItem(event_type),
        )

        self.event_table.setItem(
            row,
            2,
            QTableWidgetItem(description),
        )

        self.event_table.scrollToBottom()

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
            return f"{hours:02d}:" f"{minutes:02d}:" f"{remaining:02d}"

        return f"{minutes:02d}:" f"{remaining:02d}"

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
