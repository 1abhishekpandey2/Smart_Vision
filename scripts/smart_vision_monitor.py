import argparse
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from math import hypot
from time import monotonic

import cv2
import numpy as np

from app.camera.status import CameraStatus
from app.camera.video_file import VideoFileCamera
from app.camera.webcam import WebcamCamera
from app.core.logger import logger
from app.detection.yolo import YOLODetector
from app.reasoning.temporal import TemporalPersistence
from app.tracking.history import TrackHistoryStore
from app.tracking.yolo_person import YOLOPersonTracker
from app.zones.polygon import PolygonZone

WINDOW_NAME = "Smart Vision - Unified Monitor"

TRAIL_DISPLAY_SECONDS = 4.0
MAX_VISIBLE_TRAILS = 3
TIMELINE_ITEMS = 6


@dataclass(frozen=True)
class RecentEvent:
    timestamp: datetime
    kind: str
    message: str


def add_event(
    events: deque[RecentEvent],
    timestamp: datetime,
    kind: str,
    message: str,
) -> None:
    events.append(
        RecentEvent(
            timestamp=timestamp,
            kind=kind,
            message=message,
        )
    )


def event_color(kind: str) -> tuple[int, int, int]:
    colors = {
        "fire": (0, 0, 255),
        "fire_clear": (0, 200, 0),
        "intrusion": (0, 0, 255),
        "exit": (0, 200, 0),
    }

    return colors.get(
        kind,
        (220, 220, 220),
    )


def main():
    parser = argparse.ArgumentParser(
        description=("Smart Vision unified fire/smoke " "and intrusion monitoring.")
    )

    parser.add_argument(
        "--fire-model",
        type=str,
        default="models/firedetect-11s.pt",
        help="Path to the fire/smoke YOLO model.",
    )

    parser.add_argument(
        "--person-model",
        type=str,
        default="models/yolo26s.pt",
        help="Path to the person tracking YOLO model.",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Webcam device index.",
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help=("Recorded video path. " "If supplied, webcam is not used."),
    )

    parser.add_argument(
        "--fire-conf",
        type=float,
        default=0.4,
        help="Fire/smoke confidence threshold.",
    )

    parser.add_argument(
        "--person-conf",
        type=float,
        default=0.35,
        help="Person confidence threshold.",
    )

    parser.add_argument(
        "--temporal-window",
        type=int,
        default=10,
        help=("Number of recent fire observations " "in the temporal window."),
    )

    parser.add_argument(
        "--temporal-required",
        type=int,
        default=7,
        help=("Positive fire observations required " "for confirmation."),
    )

    parser.add_argument(
        "--trail-points",
        type=int,
        default=60,
        help=("Maximum stored trajectory points " "per person."),
    )

    args = parser.parse_args()

    # ==========================================================
    # FIRE / SMOKE DETECTOR
    # ==========================================================

    logger.info(
        "Loading fire/smoke model from %s",
        args.fire_model,
    )

    fire_detector = YOLODetector(
        model_path=args.fire_model,
        confidence_threshold=args.fire_conf,
    )

    fire_temporal = TemporalPersistence(
        window_size=args.temporal_window,
        required_positive_frames=args.temporal_required,
    )

    # ==========================================================
    # PERSON TRACKER
    # ==========================================================

    logger.info(
        "Loading person tracking model from %s",
        args.person_model,
    )

    person_tracker = YOLOPersonTracker(
        model_path=args.person_model,
        confidence_threshold=args.person_conf,
    )

    history_store = TrackHistoryStore(
        max_points=args.trail_points,
        smoothing_alpha=0.35,
        minimum_movement=3.0,
    )

    # ==========================================================
    # VIDEO SOURCE
    # ==========================================================

    if args.video:
        camera = VideoFileCamera(
            camera_id="recorded-video",
            video_path=args.video,
        )

        using_video = True

    else:
        camera = WebcamCamera(
            camera_id=f"webcam-{args.device}",
            device_index=args.device,
        )

        using_video = False

    camera.start()

    if camera.status != CameraStatus.CONNECTED:
        logger.error(
            "Could not connect to source [%s]: %s",
            camera.id,
            camera.last_error,
        )

        sys.exit(1)

    # ==========================================================
    # RESTRICTED-ZONE STATE
    # ==========================================================

    zone: PolygonZone | None = None

    drawing_mode = False

    zone_points: list[tuple[int, int]] = []

    previous_inside: dict[
        int,
        bool,
    ] = {}

    intrusion_trails: dict[
        int,
        tuple[tuple[int, int], ...],
    ] = {}

    entry_points: dict[
        int,
        tuple[int, int],
    ] = {}

    entry_arrow_starts: dict[
        int,
        tuple[int, int],
    ] = {}

    trail_visible_until: dict[
        int,
        float,
    ] = {}

    trail_started_at: dict[
        int,
        float,
    ] = {}

    recent_events: deque[RecentEvent] = deque(maxlen=50)

    previous_fire_confirmed = False

    # ==========================================================
    # HELPER FOR CLEARING ZONE-SPECIFIC RUNTIME STATE
    # ==========================================================

    def reset_zone_runtime_state() -> None:
        previous_inside.clear()

        intrusion_trails.clear()
        entry_points.clear()
        entry_arrow_starts.clear()

        trail_visible_until.clear()
        trail_started_at.clear()

        history_store.clear()

    # ==========================================================
    # MOUSE CALLBACK
    # ==========================================================

    def on_mouse(
        event,
        x,
        y,
        flags,
        param,
    ):
        if not drawing_mode:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            zone_points.append((x, y))

        elif event == cv2.EVENT_RBUTTONDOWN and zone_points:
            zone_points.pop()

    cv2.namedWindow(WINDOW_NAME)

    cv2.setMouseCallback(
        WINDOW_NAME,
        on_mouse,
    )

    logger.info("Unified Smart Vision monitor started.")

    print()
    print("Controls:")
    print("  Z           - draw/re-draw restricted zone")
    print("  Left click  - add zone point")
    print("  Right click - undo last zone point")
    print("  U           - undo last zone point")
    print("  ENTER       - finish zone")
    print("  C           - clear zone")
    print("  Q           - quit")
    print()

    try:
        while True:

            if camera.status != CameraStatus.CONNECTED:
                logger.error(
                    "Source [%s] disconnected: %s",
                    camera.id,
                    camera.last_error,
                )

                break

            frame = camera.read_frame()

            if frame is None:

                if using_video and camera.finished:
                    logger.info("Video playback completed.")

                    break

                continue

            # Always preserve the original frame.
            image = frame.image.copy()

            # ==================================================
            # FIRE / SMOKE INTELLIGENCE
            # ==================================================

            fire_detections = fire_detector.detect(frame)

            fire_present = any(
                detection.class_name == "fire" for detection in fire_detections
            )

            smoke_count = sum(
                detection.class_name == "smoke" for detection in fire_detections
            )

            fire_temporal.observe(fire_present)

            fire_confirmed = fire_temporal.is_confirmed

            # False -> True
            if fire_confirmed and not previous_fire_confirmed:
                add_event(
                    recent_events,
                    frame.timestamp,
                    "fire",
                    "FIRE CONFIRMED",
                )

                logger.warning("FIRE CONFIRMED")

            # True -> False
            elif previous_fire_confirmed and not fire_confirmed:
                add_event(
                    recent_events,
                    frame.timestamp,
                    "fire_clear",
                    "Fire cleared",
                )

                logger.info("Fire state cleared")

            previous_fire_confirmed = fire_confirmed

            # ==================================================
            # PERSON TRACKING
            # ==================================================

            tracks = person_tracker.track(frame)

            current_inside_ids: set[int] = set()

            render_tracks = []

            for track in tracks:

                raw_ground_point = track.ground_point

                maybe_smoothed = history_store.update(
                    track.track_id,
                    raw_ground_point,
                )

                # Allows compatibility if an older
                # TrackHistoryStore returns None.
                ground_point = (
                    maybe_smoothed if maybe_smoothed is not None else raw_ground_point
                )

                inside_zone = False

                # ==============================================
                # ZONE REASONING
                # ==============================================

                if zone is not None:

                    inside_zone = zone.contains(ground_point)

                    if inside_zone:
                        current_inside_ids.add(track.track_id)

                    old_inside = previous_inside.get(track.track_id)

                    # ------------------------------------------
                    # OUTSIDE -> INSIDE
                    # ------------------------------------------

                    if old_inside is False and inside_zone:
                        trail = history_store.get(track.track_id)

                        intrusion_trails[track.track_id] = trail

                        if trail:

                            # If current point was actually added
                            # to history, use the previous point.
                            if trail[-1] == ground_point and len(trail) >= 2:
                                previous_point = trail[-2]

                            else:
                                previous_point = trail[-1]

                            entry_point = zone.boundary_intersection(
                                previous_point,
                                ground_point,
                            )

                            if entry_point is None:
                                entry_point = ground_point

                        else:
                            previous_point = ground_point

                            entry_point = ground_point

                        entry_points[track.track_id] = entry_point

                        entry_arrow_starts[track.track_id] = previous_point

                        now = monotonic()

                        trail_started_at[track.track_id] = now

                        trail_visible_until[track.track_id] = (
                            now + TRAIL_DISPLAY_SECONDS
                        )

                        add_event(
                            recent_events,
                            frame.timestamp,
                            "intrusion",
                            (f"Person " f"#{track.track_id} " f"ENTERED zone"),
                        )

                        logger.warning(
                            "INTRUSION: Person #%s " "entered restricted zone",
                            track.track_id,
                        )

                    # ------------------------------------------
                    # INSIDE -> OUTSIDE
                    # ------------------------------------------

                    elif old_inside is True and not inside_zone:
                        add_event(
                            recent_events,
                            frame.timestamp,
                            "exit",
                            (f"Person " f"#{track.track_id} " f"EXITED zone"),
                        )

                        logger.info(
                            "Person #%s exited " "restricted zone",
                            track.track_id,
                        )

                    previous_inside[track.track_id] = inside_zone

                render_tracks.append(
                    (
                        track,
                        ground_point,
                        inside_zone,
                    )
                )

            # ==================================================
            # REMOVE EXPIRED TRAJECTORY VISUALS
            # ==================================================

            now = monotonic()

            expired_ids = [
                track_id
                for track_id, visible_until in trail_visible_until.items()
                if now >= visible_until
            ]

            for track_id in expired_ids:

                intrusion_trails.pop(
                    track_id,
                    None,
                )

                entry_points.pop(
                    track_id,
                    None,
                )

                entry_arrow_starts.pop(
                    track_id,
                    None,
                )

                trail_visible_until.pop(
                    track_id,
                    None,
                )

                trail_started_at.pop(
                    track_id,
                    None,
                )

            # ==================================================
            # DRAW RESTRICTED ZONE
            # ==================================================

            if zone is not None:

                contour = np.array(
                    zone.points,
                    dtype=np.int32,
                ).reshape((-1, 1, 2))

                overlay = image.copy()

                zone_color = (0, 0, 255) if current_inside_ids else (0, 255, 255)

                cv2.fillPoly(
                    overlay,
                    [contour],
                    zone_color,
                )

                cv2.addWeighted(
                    overlay,
                    0.12,
                    image,
                    0.88,
                    0,
                    image,
                )

                cv2.polylines(
                    image,
                    [contour],
                    True,
                    zone_color,
                    2,
                    cv2.LINE_AA,
                )

                label_point = zone.points[0]

                cv2.putText(
                    image,
                    "RESTRICTED ZONE",
                    (
                        label_point[0],
                        max(
                            label_point[1] - 10,
                            20,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    zone_color,
                    2,
                    cv2.LINE_AA,
                )

            # ==================================================
            # DRAW ZONE WHILE USER IS CREATING IT
            # ==================================================

            if drawing_mode:

                for point in zone_points:

                    cv2.circle(
                        image,
                        point,
                        5,
                        (255, 255, 0),
                        cv2.FILLED,
                        cv2.LINE_AA,
                    )

                if len(zone_points) >= 2:

                    drawing_contour = np.array(
                        zone_points,
                        dtype=np.int32,
                    ).reshape((-1, 1, 2))

                    cv2.polylines(
                        image,
                        [drawing_contour],
                        False,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            # ==================================================
            # DRAW FIRE / SMOKE DETECTIONS
            # ==================================================

            detection_colors = {
                "fire": (0, 0, 255),
                "smoke": (160, 160, 160),
            }

            for detection in fire_detections:

                x1, y1, x2, y2 = detection.bounding_box

                color = detection_colors.get(
                    detection.class_name,
                    (255, 255, 255),
                )

                label = f"{detection.class_name.upper()} " f"{detection.confidence:.0%}"

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )

                cv2.putText(
                    image,
                    label,
                    (
                        x1,
                        max(
                            y1 - 8,
                            24,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # ==================================================
            # DRAW TRACKED PEOPLE
            # ==================================================

            for (
                track,
                ground_point,
                inside_zone,
            ) in render_tracks:

                x1, y1, x2, y2 = track.bounding_box

                track_color = (0, 0, 255) if inside_zone else (0, 255, 0)

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    track_color,
                    2,
                )

                label = f"Person " f"#{track.track_id} " f"{track.confidence:.0%}"

                cv2.putText(
                    image,
                    label,
                    (
                        x1,
                        max(
                            y1 - 10,
                            25,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    track_color,
                    2,
                    cv2.LINE_AA,
                )

                cv2.circle(
                    image,
                    ground_point,
                    4,
                    track_color,
                    cv2.FILLED,
                    cv2.LINE_AA,
                )

            # ==================================================
            # DRAW LIMITED ENTRY TRAILS
            # ==================================================

            visible_trail_ids = list(intrusion_trails.keys())

            visible_trail_ids.sort(
                key=lambda track_id: (
                    trail_started_at.get(
                        track_id,
                        0.0,
                    )
                ),
                reverse=True,
            )

            visible_trail_ids = visible_trail_ids[:MAX_VISIBLE_TRAILS]

            for track_id in visible_trail_ids:

                trail = intrusion_trails.get(
                    track_id,
                    (),
                )

                entry_point = entry_points.get(track_id)

                arrow_start = entry_arrow_starts.get(track_id)

                if len(trail) >= 2:

                    trail_array = np.array(
                        trail,
                        dtype=np.int32,
                    ).reshape((-1, 1, 2))

                    cv2.polylines(
                        image,
                        [trail_array],
                        False,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.circle(
                        image,
                        trail[0],
                        4,
                        (0, 165, 255),
                        cv2.FILLED,
                        cv2.LINE_AA,
                    )

                # Only one directional arrow per trail.
                if entry_point is not None and arrow_start is not None:

                    arrow_distance = hypot(
                        entry_point[0] - arrow_start[0],
                        entry_point[1] - arrow_start[1],
                    )

                    if arrow_distance >= 5:

                        cv2.arrowedLine(
                            image,
                            arrow_start,
                            entry_point,
                            (0, 0, 255),
                            3,
                            cv2.LINE_AA,
                            tipLength=0.28,
                        )

                    cv2.circle(
                        image,
                        entry_point,
                        6,
                        (0, 0, 255),
                        cv2.FILLED,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        image,
                        f"ENTRY #{track_id}",
                        (
                            entry_point[0] + 8,
                            entry_point[1] - 8,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

            # ==================================================
            # FIRE STATE
            # ==================================================

            if fire_temporal.observation_count < fire_temporal.window_size:
                fire_status = "WARMING UP"

                fire_status_color = (
                    255,
                    255,
                    0,
                )

            elif fire_temporal.is_confirmed:

                fire_status = "CONFIRMED"

                fire_status_color = (
                    0,
                    0,
                    255,
                )

            elif fire_temporal.positive_count > 0:
                fire_status = "POSSIBLE"

                fire_status_color = (
                    0,
                    255,
                    255,
                )

            else:
                fire_status = "MONITORING"

                fire_status_color = (
                    0,
                    255,
                    0,
                )

            # ==================================================
            # ZONE STATE
            # ==================================================

            if drawing_mode:

                zone_status = "ZONE: DRAWING"

            elif zone is None:

                zone_status = "ZONE: NOT SET"

            elif current_inside_ids:

                sorted_ids = sorted(current_inside_ids)

                shown_ids = sorted_ids[:4]

                ids_text = ", ".join(f"#{track_id}" for track_id in shown_ids)

                extra_count = len(sorted_ids) - len(shown_ids)

                if extra_count > 0:
                    ids_text += f" +{extra_count} more"

                zone_status = f"ZONE: INTRUSION " f"{ids_text}"

            else:
                zone_status = "ZONE: SAFE"

            # ==================================================
            # LEFT INFORMATION PANEL
            # ==================================================

            panel_overlay = image.copy()

            cv2.rectangle(
                panel_overlay,
                (10, 10),
                (470, 140),
                (25, 25, 25),
                cv2.FILLED,
            )

            cv2.addWeighted(
                panel_overlay,
                0.85,
                image,
                0.15,
                0,
                image,
            )

            cv2.putText(
                image,
                "SMART VISION",
                (25, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                (
                    f"Fire: "
                    f"{fire_status}   "
                    f"Evidence: "
                    f"{fire_temporal.positive_count}"
                    f"/"
                    f"{fire_temporal.window_size}"
                ),
                (25, 67),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                fire_status_color,
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                (
                    f"Smoke detections: "
                    f"{smoke_count}   "
                    f"Tracked people: "
                    f"{len(tracks)}"
                ),
                (25, 94),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            zone_text_color = (0, 0, 255) if current_inside_ids else (220, 220, 220)

            cv2.putText(
                image,
                zone_status,
                (25, 121),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                zone_text_color,
                1,
                cv2.LINE_AA,
            )

            # ==================================================
            # RECENT EVENTS PANEL
            # ==================================================

            events_to_show = list(recent_events)[-TIMELINE_ITEMS:]

            events_to_show.reverse()

            frame_height, frame_width = image.shape[:2]

            timeline_width = min(
                380,
                frame_width - 20,
            )

            if frame_width >= 900:

                timeline_x = frame_width - timeline_width - 10

                timeline_y = 10

            else:
                timeline_x = 10
                timeline_y = 150

            row_height = 23

            timeline_height = (
                42
                + max(
                    len(events_to_show),
                    1,
                )
                * row_height
            )

            timeline_overlay = image.copy()

            cv2.rectangle(
                timeline_overlay,
                (
                    timeline_x,
                    timeline_y,
                ),
                (
                    timeline_x + timeline_width,
                    timeline_y + timeline_height,
                ),
                (25, 25, 25),
                cv2.FILLED,
            )

            cv2.addWeighted(
                timeline_overlay,
                0.82,
                image,
                0.18,
                0,
                image,
            )

            cv2.putText(
                image,
                "RECENT EVENTS",
                (
                    timeline_x + 12,
                    timeline_y + 25,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if not events_to_show:

                cv2.putText(
                    image,
                    "No events yet",
                    (
                        timeline_x + 12,
                        timeline_y + 51,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )

            else:

                for (
                    index,
                    event,
                ) in enumerate(events_to_show):

                    text = f"{event.timestamp:%H:%M:%S}  " f"{event.message}"

                    cv2.putText(
                        image,
                        text,
                        (
                            timeline_x + 12,
                            timeline_y + 51 + index * row_height,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        event_color(event.kind),
                        1,
                        cv2.LINE_AA,
                    )

            # ==================================================
            # CONTROLS
            # ==================================================

            cv2.putText(
                image,
                (
                    "Z: draw zone | "
                    "U/right-click: undo | "
                    "Enter: finish | "
                    "C: clear | "
                    "Q: quit"
                ),
                (
                    10,
                    frame_height - 15,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                WINDOW_NAME,
                image,
            )

            key = cv2.waitKey(1) & 0xFF

            # ==================================================
            # KEYBOARD CONTROLS
            # ==================================================

            if key == ord("q"):
                break

            elif key == ord("z"):

                drawing_mode = True

                zone = None

                zone_points.clear()

                reset_zone_runtime_state()

                logger.info("Restricted-zone " "drawing started")

            elif key == ord("u") and drawing_mode and zone_points:
                zone_points.pop()

            elif key in (10, 13):

                if drawing_mode:

                    if len(zone_points) >= 3:

                        zone = PolygonZone(zone_points)

                        drawing_mode = False

                        reset_zone_runtime_state()

                        logger.info(
                            "Restricted zone " "created with %s points",
                            len(zone_points),
                        )

                    else:
                        logger.warning(
                            "At least 3 points "
                            "are required to create "
                            "a restricted zone"
                        )

            elif key == ord("c"):

                zone = None

                drawing_mode = False

                zone_points.clear()

                reset_zone_runtime_state()

                logger.info("Restricted zone cleared")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    finally:

        logger.info("Stopping source and " "releasing resources...")

        camera.stop()

        cv2.destroyAllWindows()

        logger.info("Done.")


if __name__ == "__main__":
    main()
