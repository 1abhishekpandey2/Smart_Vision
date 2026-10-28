from pathlib import Path
import os

from PyQt5.QtCore import (
    QObject,
    pyqtSignal,
    pyqtSlot,
)

from app.reasoning.dwell import DwellTracker
from app.reasoning.temporal import TemporalPersistence
from app.recognition.association import associate_faces_to_tracks
from app.recognition.identity_memory import TrackIdentityMemory
from app.tracking.history import TrackHistoryStore
from app.ui.vision_types import (
    OverlayBox,
    TrackPath,
    VisionEvent,
    VisionResult,
)
from app.zones.polygon import PolygonZone
from app.recognition.registry import FaceRegistry, MatchState


class VisionWorker(QObject):
    """
    Runs Smart Vision AI/reasoning outside the Qt UI thread.

    Responsibilities:
    - Fire / smoke detection
    - Temporal fire reasoning
    - Person detection + tracking
    - Track-history maintenance
    - Restricted-zone reasoning
    - Intrusion entry / exit events
    - Intrusion path generation

    It does NOT draw anything.
    Rendering belongs to VideoView.
    """

    result_ready = pyqtSignal(object)

    analysis_finished = pyqtSignal()

    error_occurred = pyqtSignal(str)

    status_changed = pyqtSignal(str)

    face_registration_succeeded = pyqtSignal(str)

    face_registration_failed = pyqtSignal(str)

    def __init__(
        self,
        fire_model_path: str = "models/firedetect-11s.pt",
        person_model_path: str = "models/yolo26s.pt",
        face_registry_path: str = "data/face_registry.npz",
        parent=None,
    ):
        super().__init__(parent)

        # ======================================================
        # MODEL PATHS
        # ======================================================

        self._fire_model_path = fire_model_path

        self._person_model_path = person_model_path
        self._face_registry_path = Path(face_registry_path)

        # Models are loaded lazily.
        #
        # This is especially important on our Windows setup
        # because of the PyQt/PyTorch DLL loading behavior.
        self._fire_detector = None
        self._person_tracker = None
        self._face_embedder = None
        self._face_registry = None
        self._identity_memory = TrackIdentityMemory()

        # Models run at a permissive base threshold.
        # The UI confidence threshold then filters the
        # returned Smart Vision objects.
        self._base_model_confidence = 0.05

        # ======================================================
        # FIRE TEMPORAL REASONING
        # ======================================================

        self._fire_temporal = TemporalPersistence(
            window_size=10,
            required_positive_frames=7,
        )

        self._previous_fire_confirmed = False

        # ======================================================
        # PERSON TRACK HISTORY
        # ======================================================

        self._history_store = TrackHistoryStore(
            max_points=60,
            smoothing_alpha=0.35,
            minimum_movement=3.0,
        )

        # ======================================================
        # ZONE / INTRUSION STATE
        # ======================================================

        # Previous inside/outside state for every track.
        #
        # Example:
        #
        # {
        #     5: False,
        #     7: True,
        # }
        #
        self._previous_inside: dict[
            int,
            bool,
        ] = {}

        # IDs that have ACTUALLY crossed:
        #
        # OUTSIDE -> INSIDE
        #
        # Only these IDs are allowed to display
        # their track paths.
        self._active_intrusion_ids: set[int] = set()

        # Used to detect whether the user changed
        # or redrew the restricted zone.
        self._zone_signature = None

        # ======================================================
        # DWELL / LOITERING STATE
        # ======================================================

        self._dwell_tracker = DwellTracker()

    # ==========================================================
    # MODEL LOADING
    # ==========================================================

    def _ensure_fire_detector(
        self,
    ) -> None:
        """
        Load the fire/smoke model only when it is first needed.
        """

        if self._fire_detector is not None:
            return

        model_path = Path(self._fire_model_path)

        if not model_path.exists():
            raise FileNotFoundError("Fire model not found: " f"{model_path}")

        self.status_changed.emit("Loading Fire/Smoke model...")

        # Lazy import.
        #
        # Do not move this import to the top of the file
        # unless the PyQt/PyTorch DLL problem has been
        # permanently resolved.
        from app.detection.yolo import YOLODetector

        self._fire_detector = YOLODetector(
            model_path=str(model_path),
            confidence_threshold=(self._base_model_confidence),
        )

        self.status_changed.emit("Fire/Smoke model ready")

    def _ensure_person_tracker(
        self,
    ) -> None:
        """
        Load the person tracking model only when first needed.
        """

        if self._person_tracker is not None:
            return

        model_path = Path(self._person_model_path)

        if not model_path.exists():
            raise FileNotFoundError("Person model not found: " f"{model_path}")

        self.status_changed.emit("Loading Person Tracking model...")

        # Lazy import for the same PyTorch/PyQt reason.
        from app.tracking.yolo_person import YOLOPersonTracker

        self._person_tracker = YOLOPersonTracker(
            model_path=str(model_path),
            confidence_threshold=(self._base_model_confidence),
        )

        self.status_changed.emit("Person Tracking model ready")

    def _ensure_face_recognizer(
        self,
    ) -> None:
        """Load face analysis and the saved gallery only when requested."""

        if self._face_embedder is None or self._face_registry is None:
            if not self._face_registry_path.exists():
                raise FileNotFoundError(
                    f"Face registry not found: {self._face_registry_path}"
                )

            self.status_changed.emit("Loading face recognition models...")

            registry = FaceRegistry.load(self._face_registry_path)

            from app.recognition.insightface_embedder import InsightFaceEmbedder

            embedder = InsightFaceEmbedder()

            self._face_registry = registry
            self._face_embedder = embedder

            self.status_changed.emit("Face recognition ready")

    @pyqtSlot(str, object)
    def register_face_images(
        self,
        identity: str,
        image_paths,
    ) -> None:
        """Add one embedding per valid image and atomically save the gallery."""

        identity = identity.strip()
        paths = [Path(path) for path in image_paths]

        if not identity:
            self.face_registration_failed.emit(
                "Enter an identity name before registering images."
            )
            return

        if not paths:
            self.face_registration_failed.emit("Select at least one face image.")
            return

        try:
            embedder = self._face_embedder
            if embedder is None:
                from app.recognition.insightface_embedder import InsightFaceEmbedder

                embedder = InsightFaceEmbedder()

            # Compute every embedding before changing the gallery. If an image
            # is invalid or contains multiple faces, the existing file stays intact.
            embeddings = [embedder.embed_image(path) for path in paths]

            if self._face_registry_path.exists():
                registry = FaceRegistry.load(self._face_registry_path)
            else:
                registry = FaceRegistry()

            stored_identity = identity
            for embedding in embeddings:
                stored_identity = registry.add_sample(identity, embedding)

            temporary_path = self._face_registry_path.with_name(
                f"{self._face_registry_path.stem}.tmp"
                f"{self._face_registry_path.suffix}"
            )

            try:
                registry.save(temporary_path)
                os.replace(temporary_path, self._face_registry_path)
            finally:
                temporary_path.unlink(missing_ok=True)

            self._face_embedder = embedder
            self._face_registry = registry
            self._identity_memory.reset()

            self.face_registration_succeeded.emit(
                f"Registered {len(embeddings)} image(s) for '{stored_identity}'."
            )
        except Exception as exc:
            self.face_registration_failed.emit(str(exc))

    # ==========================================================
    # RUNTIME RESET
    # ==========================================================

    @pyqtSlot()
    def reset_runtime_state(
        self,
    ) -> None:
        """
        Reset temporal/tracking reasoning state.

        Called after things such as:
        - video seek
        - restart
        - zone modification
        - intelligence configuration changes
        """

        # ------------------------------------------------------
        # FIRE
        # ------------------------------------------------------

        self._fire_temporal.reset()

        self._previous_fire_confirmed = False

        # ------------------------------------------------------
        # INTRUSION
        # ------------------------------------------------------

        self._previous_inside.clear()

        self._active_intrusion_ids.clear()

        self._history_store.clear()

        self._zone_signature = None
        self._dwell_tracker.reset()
        self._identity_memory.reset()

        # ------------------------------------------------------
        # BYTETRACK
        # ------------------------------------------------------

        # Ultralytics keeps persistent tracking state when
        # persist=True.
        #
        # If this version exposes tracker.reset(), reset it too.
        if self._person_tracker is not None:

            model = getattr(
                self._person_tracker,
                "_model",
                None,
            )

            predictor = getattr(
                model,
                "predictor",
                None,
            )

            trackers = getattr(
                predictor,
                "trackers",
                None,
            )

            if trackers:

                for tracker in trackers:

                    reset_method = getattr(
                        tracker,
                        "reset",
                        None,
                    )

                    if callable(reset_method):
                        reset_method()

    # ==========================================================
    # FRAME PROCESSING
    # ==========================================================

    @pyqtSlot(object)
    def process(
        self,
        request,
    ) -> None:
        """
        Process one VisionRequest.

        A VisionResult is emitted after processing.
        """

        try:
            frame = request.frame
            settings = request.settings

            # ==================================================
            # RESULT COLLECTION
            # ==================================================

            overlays = []

            track_paths = []

            events = []

            people_count = 0

            inside_zone_count = 0
            loitering_count = 0

            zone_active = False

            # ==================================================
            # FIRE / SMOKE
            # ==================================================

            if settings.fire_enabled:

                self._ensure_fire_detector()

                raw_detections = self._fire_detector.detect(frame)

                # Apply current UI threshold.
                detections = [
                    detection
                    for detection in raw_detections
                    if (detection.confidence >= settings.fire_confidence)
                ]

                # ----------------------------------------------
                # FIRE / SMOKE BOXES
                # ----------------------------------------------

                for detection in detections:

                    overlays.append(
                        OverlayBox(
                            bounding_box=(detection.bounding_box),
                            label=(
                                f"{detection.class_name.upper()} "
                                f"{detection.confidence:.0%}"
                            ),
                            kind=(detection.class_name),
                            alert=(detection.class_name == "fire"),
                        )
                    )

                # ----------------------------------------------
                # TEMPORAL FIRE REASONING
                # ----------------------------------------------

                fire_present = any(
                    detection.class_name == "fire" for detection in detections
                )

                self._fire_temporal.observe(fire_present)

                fire_confirmed = self._fire_temporal.is_confirmed

                # False -> True
                if fire_confirmed and not self._previous_fire_confirmed:

                    events.append(
                        VisionEvent(
                            timestamp=(frame.timestamp),
                            event_type="FIRE",
                            description=("Fire confirmed"),
                        )
                    )

                # True -> False
                elif self._previous_fire_confirmed and not fire_confirmed:

                    events.append(
                        VisionEvent(
                            timestamp=(frame.timestamp),
                            event_type="FIRE",
                            description=("Fire cleared"),
                        )
                    )

                self._previous_fire_confirmed = fire_confirmed

                # ----------------------------------------------
                # FIRE EVIDENCE TEXT
                # ----------------------------------------------

                positive_count = self._fire_temporal.positive_count

                observation_count = self._fire_temporal.observation_count

                window_size = self._fire_temporal.window_size

                if observation_count < window_size:

                    fire_evidence = f"{positive_count}/" f"{window_size} " f"WARMING"

                elif fire_confirmed:

                    fire_evidence = f"{positive_count}/" f"{window_size} " f"CONFIRMED"

                elif positive_count > 0:

                    fire_evidence = f"{positive_count}/" f"{window_size} " f"POSSIBLE"

                else:

                    fire_evidence = f"0/{window_size}"

            else:

                self._fire_temporal.reset()

                self._previous_fire_confirmed = False

                fire_confirmed = False

                fire_evidence = "OFF"

            # ==================================================
            # CONVERT NORMALIZED ZONE TO VIDEO PIXELS
            # ==================================================

            zone_signature = tuple(
                (
                    float(x),
                    float(y),
                )
                for x, y in request.zone_points
            )

            # --------------------------------------------------
            # ZONE CHANGED
            # --------------------------------------------------

            if zone_signature != self._zone_signature:

                # The old inside/outside state no longer has
                # meaning because the geometry changed.
                self._previous_inside.clear()

                self._active_intrusion_ids.clear()

                self._history_store.clear()
                self._dwell_tracker.reset()

                self._zone_signature = zone_signature

            zone = None

            if len(zone_signature) >= 3:

                height, width = frame.image.shape[:2]

                pixel_points = [
                    (
                        round(x * (width - 1)),
                        round(y * (height - 1)),
                    )
                    for x, y in zone_signature
                ]

                zone = PolygonZone(pixel_points)

            # ==================================================
            # PERSON TRACKING
            # ==================================================

            if settings.person_enabled:

                self._ensure_person_tracker()

                raw_tracks = self._person_tracker.track(frame)

                # Apply UI confidence threshold.
                tracks = [
                    track
                    for track in raw_tracks
                    if (track.confidence >= settings.person_confidence)
                ]

                face_matches = {}

                if settings.face_recognition_enabled and tracks:
                    self._ensure_face_recognizer()

                    face_observations = self._face_embedder.detect_faces(frame.image)

                    associated_faces = associate_faces_to_tracks(
                        face_observations,
                        tracks,
                    )

                    face_matches = {
                        track_id: self._face_registry.match(face.embedding)
                        for track_id, face in associated_faces.items()
                    }

                if settings.face_recognition_enabled:
                    if (
                        self._identity_memory.required_observations
                        != settings.identity_confirmation_observations
                    ):
                        self._identity_memory = TrackIdentityMemory(
                            required_observations=(
                                settings.identity_confirmation_observations
                            ),
                        )

                    strong_identities = {
                        track_id: match.identity
                        for track_id, match in face_matches.items()
                        if (
                            match.state is MatchState.STRONG_MATCH
                            and match.identity is not None
                        )
                    }

                    remembered_identities = self._identity_memory.update(
                        now_seconds=request.source_time_seconds,
                        active_track_ids={track.track_id for track in tracks},
                        strong_identities=strong_identities,
                    )
                else:
                    self._identity_memory.reset()
                    remembered_identities = {}

                people_count = len(tracks)
                zone_reasoning_enabled = (
                    settings.intrusion_enabled or settings.loitering_enabled
                )

                if not settings.loitering_enabled:
                    self._dwell_tracker.reset()

                # If intrusion monitoring is disabled,
                # nobody should remain marked as an intruder.
                if not settings.intrusion_enabled:

                    self._previous_inside.clear()

                    self._active_intrusion_ids.clear()

                # ==============================================
                # PROCESS EACH TRACK
                # ==============================================

                for track in tracks:

                    # ------------------------------------------
                    # TRACK POSITION
                    # ------------------------------------------

                    raw_ground_point = track.ground_point

                    smoothed_ground_point = self._history_store.update(
                        track.track_id,
                        raw_ground_point,
                    )

                    ground_point = (
                        smoothed_ground_point
                        if (smoothed_ground_point is not None)
                        else raw_ground_point
                    )

                    inside_zone = False

                    # ==========================================
                    # RESTRICTED-ZONE REASONING
                    # ==========================================

                    if zone is not None and zone_reasoning_enabled:
                        inside_zone = zone.contains(ground_point)

                        if inside_zone:

                            inside_zone_count += 1

                    if zone is not None and settings.intrusion_enabled:

                        previous_inside = self._previous_inside.get(track.track_id)
                        # ======================================
                        # ENTRY
                        #
                        # OUTSIDE -> INSIDE
                        # ======================================

                        if previous_inside is False and inside_zone:

                            # Mark this tracked person as an
                            # ACTIVE intrusion.
                            #
                            # Their already-recorded approach
                            # path now becomes visible.
                            self._active_intrusion_ids.add(track.track_id)

                            events.append(
                                VisionEvent(
                                    timestamp=(frame.timestamp),
                                    event_type=("INTRUSION"),
                                    description=(
                                        f"Person " f"#{track.track_id} " f"entered zone"
                                    ),
                                )
                            )

                        # ======================================
                        # EXIT
                        #
                        # INSIDE -> OUTSIDE
                        # ======================================

                        elif previous_inside is True and not inside_zone:

                            # Stop displaying intrusion path.
                            self._active_intrusion_ids.discard(track.track_id)

                            events.append(
                                VisionEvent(
                                    timestamp=(frame.timestamp),
                                    event_type="EXIT",
                                    description=(
                                        f"Person " f"#{track.track_id} " f"exited zone"
                                    ),
                                )
                            )

                        # Save current state for next frame.
                        self._previous_inside[track.track_id] = inside_zone
                    # ==========================================
                    # DWELL / LOITERING
                    # ==========================================

                    dwell_state = None

                    if settings.loitering_enabled:

                        dwell_state = self._dwell_tracker.observe(
                            track_id=track.track_id,
                            inside_zone=inside_zone,
                            now_seconds=request.source_time_seconds,
                            threshold_seconds=settings.dwell_seconds,
                        )

                        if dwell_state.is_loitering:

                            loitering_count += 1

                        if dwell_state.just_triggered:

                            events.append(
                                VisionEvent(
                                    timestamp=(frame.timestamp),
                                    event_type="LOITERING",
                                    description=(
                                        f"Person "
                                        f"#{track.track_id} "
                                        f"exceeded "
                                        f"{settings.dwell_seconds}s "
                                        f"dwell time"
                                    ),
                                )
                            )

                    # ==========================================
                    # INTRUSION TRACK PATH
                    # ==========================================
                    #
                    # IMPORTANT:
                    #
                    # Track history is recorded continuously
                    # above, but only becomes visible after
                    # OUTSIDE -> INSIDE has actually occurred.
                    #
                    # A person simply walking normally outside
                    # the zone therefore has no visible trail.
                    #

                    if (
                        settings.show_track_paths
                        and track.track_id in self._active_intrusion_ids
                    ):

                        history = self._history_store.get(track.track_id)

                        if len(history) >= 2:

                            track_paths.append(
                                TrackPath(
                                    track_id=(track.track_id),
                                    points=tuple(history),
                                    alert=True,
                                )
                            )

                    # ==========================================
                    # PERSON BOX
                    # ==========================================

                    person_label = (
                        f"Person " f"#{track.track_id} " f"{track.confidence:.0%}"
                    )

                    if dwell_state is not None and inside_zone:

                        person_label += (
                            f" | Dwell " f"{dwell_state.elapsed_seconds:.1f}s"
                        )

                    if dwell_state is not None and dwell_state.is_loitering:

                        person_label += " | LOITERING"

                    face_match = face_matches.get(track.track_id)
                    remembered_identity = remembered_identities.get(track.track_id)

                    if remembered_identity is not None:
                        person_label += f" | Identity: {remembered_identity}"
                    elif face_match is not None:
                        if face_match.state is MatchState.STRONG_MATCH:
                            person_label += (
                                f" | Face: {face_match.identity}"
                                f" {face_match.similarity:.2f}"
                            )

                            if face_match.identity is not None:
                                progress = self._identity_memory.confirmation_progress(
                                    track.track_id,
                                    face_match.identity,
                                )

                                if progress is not None:
                                    observed, required = progress
                                    person_label += (
                                        f" | Confirming {observed}/{required}"
                                    )
                        elif face_match.state is MatchState.CANDIDATE:
                            person_label += (
                                f" | Candidate: {face_match.identity}"
                                f" {face_match.similarity:.2f}"
                            )
                        else:
                            similarity_text = (
                                f" {face_match.similarity:.2f}"
                                if face_match.similarity is not None
                                else ""
                            )
                            person_label += f" | Unknown face{similarity_text}"

                    person_alert = settings.intrusion_enabled and inside_zone

                    if dwell_state is not None and dwell_state.is_loitering:

                        person_alert = True

                    overlays.append(
                        OverlayBox(
                            bounding_box=(track.bounding_box),
                            label=person_label,
                            kind="person",
                            alert=person_alert,
                        )
                    )

                zone_active = (
                    settings.intrusion_enabled and inside_zone_count > 0
                ) or loitering_count > 0

            else:

                # Person tracking switched OFF.
                #
                # Previous person states are no longer valid.
                self._previous_inside.clear()

                self._active_intrusion_ids.clear()

                self._history_store.clear()

                self._dwell_tracker.reset()

                self._identity_memory.reset()

            # ==================================================
            # CREATE RESULT
            # ==================================================

            result = VisionResult(
                generation=(request.generation),
                overlays=tuple(overlays),
                track_paths=tuple(track_paths),
                people_count=(people_count),
                inside_zone_count=(inside_zone_count),
                loitering_count=(loitering_count),
                fire_evidence=(fire_evidence),
                fire_confirmed=(fire_confirmed),
                zone_active=(zone_active),
                events=tuple(events),
            )

            self.result_ready.emit(result)

        # ======================================================
        # ERROR
        # ======================================================

        except Exception as exc:

            self.error_occurred.emit("Vision processing error: " f"{exc}")

        # ======================================================
        # ALWAYS SIGNAL COMPLETION
        # ======================================================

        finally:

            self.analysis_finished.emit()
