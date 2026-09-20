import argparse
import sys

import cv2
import numpy as np

from app.camera.video_file import VideoFileCamera
from app.camera.status import CameraStatus
from app.camera.webcam import WebcamCamera
from app.core.logger import logger
from app.tracking.history import TrackHistoryStore
from app.tracking.yolo_person import YOLOPersonTracker
from app.zones.polygon import PolygonZone

from math import hypot
from time import monotonic

from app.events.history import IntrusionEventHistory
from app.events.intrusion import (
    IntrusionEvent,
    IntrusionEventType,
)

WINDOW_NAME = "Smart Vision - Intrusion Monitoring"

TRAIL_DISPLAY_SECONDS = 4.0
MAX_VISIBLE_TRAILS = 3
TIMELINE_ITEMS = 5


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Smart Vision live person tracking with " "interactive restricted zones."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo26s.pt",
        help="Path to the YOLO person detection model.",
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
        default=r"D:\CS\project\Smart_Vision\data\videos\road1.mp4",
        help=(
            "Path to a recorded video. "
            "If supplied, the video is used instead of the webcam."
        ),
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Person detection confidence threshold.",
    )

    parser.add_argument(
        "--trail-points",
        type=int,
        default=60,
        help="Maximum number of recent positions stored per track.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Tracker
    # --------------------------------------------------------------

    logger.info(
        "Loading person tracking model from %s",
        args.model,
    )

    tracker = YOLOPersonTracker(
        model_path=args.model,
        confidence_threshold=args.conf,
    )

    history_store = TrackHistoryStore(
        max_points=args.trail_points,
        smoothing_alpha=0.35,
        minimum_movement=3.0,
    )

    # --------------------------------------------------------------
    # Camera
    # --------------------------------------------------------------

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
            "Could not connect to camera [%s]: %s",
            camera.id,
            camera.last_error,
        )

        sys.exit(1)

    # --------------------------------------------------------------
    # Zone state
    # --------------------------------------------------------------

    zone: PolygonZone | None = None

    drawing_mode = False

    zone_points: list[tuple[int, int]] = []

    # Stores whether each track was inside the zone on its
    # previous observation.
    previous_inside: dict[int, bool] = {}

    active_intrusions: set[int] = set()

    intrusion_trails: dict[
        int,
        tuple[tuple[int, int], ...],
    ] = {}

    entry_points: dict[
        int,
        tuple[int, int],
    ] = {}

    trail_visible_until: dict[int, float] = {}

    trail_started_at: dict[int, float] = {}

    event_history = IntrusionEventHistory(max_events=50)

    # --------------------------------------------------------------
    # Mouse callback
    # --------------------------------------------------------------

    def on_mouse(
        event,
        x,
        y,
        flags,
        param,
    ):
        if event == cv2.EVENT_LBUTTONDOWN and drawing_mode:
            zone_points.append((x, y))

    cv2.namedWindow(WINDOW_NAME)

    cv2.setMouseCallback(
        WINDOW_NAME,
        on_mouse,
    )

    logger.info("Intrusion monitoring started.")

    print()
    print("Controls:")
    print("  Z     - draw/re-draw restricted zone")
    print("  Click - add polygon point")
    print("  U     - undo last polygon point")
    print("  ENTER - finish polygon")
    print("  C     - clear restricted zone")
    print("  Q     - quit")
    print()

    try:
        while True:
            if camera.status != CameraStatus.CONNECTED:
                logger.error(
                    "Camera [%s] disconnected: %s",
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

            image = frame.image.copy()

            tracks = tracker.track(frame)

            # ======================================================
            # Draw completed restricted zone
            # ======================================================

            if zone is not None:
                contour = np.array(
                    zone.points,
                    dtype=np.int32,
                ).reshape((-1, 1, 2))

                overlay = image.copy()

                if active_intrusions:
                    zone_color = (0, 0, 255)
                else:
                    zone_color = (0, 255, 255)

                cv2.fillPoly(
                    overlay,
                    [contour],
                    zone_color,
                )

                cv2.addWeighted(
                    overlay,
                    0.15,
                    image,
                    0.85,
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

                first_point = zone.points[0]

                cv2.putText(
                    image,
                    "RESTRICTED ZONE",
                    (
                        first_point[0],
                        max(first_point[1] - 10, 20),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    zone_color,
                    2,
                    cv2.LINE_AA,
                )

            # ======================================================
            # Draw polygon currently being created
            # ======================================================

            if drawing_mode:
                for point in zone_points:
                    cv2.circle(
                        image,
                        point,
                        5,
                        (255, 255, 0),
                        cv2.FILLED,
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

            # ======================================================
            # Process each tracked person
            # ======================================================

            for track in tracks:
                raw_ground_point = track.ground_point

                ground_point = history_store.update(
                    track.track_id,
                    raw_ground_point,
                )

                inside_zone = False

                if zone is not None:
                    inside_zone = zone.contains(ground_point)

                    old_inside = previous_inside.get(track.track_id)

                    # ----------------------------------------------
                    # OUTSIDE -> INSIDE
                    # ----------------------------------------------

                    if old_inside is False and inside_zone:
                        trail = history_store.get(track.track_id)

                        intrusion_trails[track.track_id] = trail

                        active_intrusions.add(track.track_id)

                        # --------------------------------------------------
                        # Estimate actual zone-boundary crossing point
                        # --------------------------------------------------

                        if len(trail) >= 2:
                            previous_point = trail[-2]

                            entry_point = zone.boundary_intersection(
                                previous_point,
                                ground_point,
                            )

                            if entry_point is None:
                                entry_point = ground_point

                        else:
                            entry_point = ground_point

                        entry_points[track.track_id] = entry_point

                        # --------------------------------------------------
                        # Temporary trajectory visualization
                        # --------------------------------------------------

                        now = monotonic()

                        trail_started_at[track.track_id] = now

                        trail_visible_until[track.track_id] = (
                            now + TRAIL_DISPLAY_SECONDS
                        )

                        # --------------------------------------------------
                        # Actual domain event
                        # --------------------------------------------------

                        event_history.add(
                            IntrusionEvent(
                                track_id=track.track_id,
                                event_type=IntrusionEventType.ENTERED,
                                timestamp=frame.timestamp,
                                position=entry_point,
                            )
                        )

                        logger.warning(
                            "INTRUSION: Person #%s entered restricted zone",
                            track.track_id,
                        )

                    # ----------------------------------------------
                    # INSIDE -> OUTSIDE
                    # ----------------------------------------------

                    elif old_inside is True and not inside_zone:
                        active_intrusions.discard(track.track_id)

                        event_history.add(
                            IntrusionEvent(
                                track_id=track.track_id,
                                event_type=IntrusionEventType.EXITED,
                                timestamp=frame.timestamp,
                                position=ground_point,
                            )
                        )

                        logger.info(
                            "Person #%s exited restricted zone",
                            track.track_id,
                        )
                    previous_inside[track.track_id] = inside_zone

                # ----------------------------------------------
                # Track box
                # ----------------------------------------------

                x1, y1, x2, y2 = track.bounding_box

                if inside_zone:
                    track_color = (0, 0, 255)
                else:
                    track_color = (0, 255, 0)

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    track_color,
                    2,
                )

                label = f"Person #{track.track_id} " f"{track.confidence:.0%}"

                cv2.putText(
                    image,
                    label,
                    (
                        x1,
                        max(y1 - 10, 25),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    track_color,
                    2,
                    cv2.LINE_AA,
                )

                # Ground-position marker
                cv2.circle(
                    image,
                    ground_point,
                    5,
                    track_color,
                    cv2.FILLED,
                )
            current_time = monotonic()

            expired_trails = [
                track_id
                for track_id, visible_until in trail_visible_until.items()
                if current_time >= visible_until
            ]

            for track_id in expired_trails:
                intrusion_trails.pop(
                    track_id,
                    None,
                )

                entry_points.pop(
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

            # ======================================================
            # Draw intrusion trajectories
            # ======================================================

            visible_trail_ids = [
                track_id
                for track_id in intrusion_trails
                if track_id in trail_visible_until
            ]

            visible_trail_ids.sort(
                key=lambda track_id: trail_started_at.get(
                    track_id,
                    0.0,
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

                if len(trail) >= 2:
                    trail_array = np.array(
                        trail,
                        dtype=np.int32,
                    ).reshape((-1, 1, 2))

                    # Clean anti-aliased trajectory
                    cv2.polylines(
                        image,
                        [trail_array],
                        False,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # Starting marker
                    cv2.circle(
                        image,
                        trail[0],
                        4,
                        (0, 165, 255),
                        cv2.FILLED,
                        cv2.LINE_AA,
                    )

                # --------------------------------------------------
                # Single directional arrow near entry
                # --------------------------------------------------

                if entry_point is not None and len(trail) >= 2:
                    arrow_start = trail[-2]

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
                        0.48,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

            # ======================================================
            # Status panel
            # ======================================================

            cv2.rectangle(
                image,
                (10, 10),
                (430, 120),
                (30, 30, 30),
                cv2.FILLED,
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
                f"Tracked persons: {len(tracks)}",
                (25, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            recent_events = event_history.recent(TIMELINE_ITEMS)

            frame_height, frame_width = image.shape[:2]

            timeline_width = min(
                420,
                frame_width - 20,
            )

            if frame_width >= 900:
                timeline_x = frame_width - timeline_width - 10
                timeline_y = 10
            else:
                timeline_x = 10
                timeline_y = 130

            row_height = 25

            timeline_height = 42 + max(len(recent_events), 1) * row_height

            overlay = image.copy()

            cv2.rectangle(
                overlay,
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
                overlay,
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
                    timeline_y + 26,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if not recent_events:
                cv2.putText(
                    image,
                    "No intrusion events",
                    (
                        timeline_x + 12,
                        timeline_y + 54,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )

            else:
                for index, event in enumerate(recent_events):
                    if event.event_type is IntrusionEventType.ENTERED:
                        event_text = (
                            f"{event.timestamp:%H:%M:%S}  "
                            f"Person #{event.track_id} ENTERED"
                        )

                        event_color = (0, 0, 255)

                    else:
                        event_text = (
                            f"{event.timestamp:%H:%M:%S}  "
                            f"Person #{event.track_id} EXITED"
                        )

                        event_color = (0, 255, 0)

                    text_y = timeline_y + 54 + index * row_height

                    cv2.putText(
                        image,
                        event_text,
                        (
                            timeline_x + 12,
                            text_y,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.43,
                        event_color,
                        1,
                        cv2.LINE_AA,
                    )

            if drawing_mode:
                status_text = "DRAWING ZONE - click points, ENTER to finish"

                status_color = (255, 255, 0)

            elif zone is None:
                status_text = "NO RESTRICTED ZONE - press Z to create"

                status_color = (255, 255, 0)

            elif active_intrusions:
                ids = ", ".join(
                    f"#{track_id}" for track_id in sorted(active_intrusions)
                )

                status_text = f"INTRUSION DETECTED - Person {ids}"

                status_color = (0, 0, 255)

            else:
                status_text = "ZONE STATUS: SAFE"

                status_color = (0, 255, 0)

            cv2.putText(
                image,
                status_text,
                (25, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                status_color,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                "Z: draw | U: undo | Enter: finish | " "C: clear | Q: quit",
                (
                    10,
                    image.shape[0] - 15,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"Active intrusions: {len(active_intrusions)}",
                (25, 124),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                ((0, 0, 255) if active_intrusions else (0, 255, 0)),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                WINDOW_NAME,
                image,
            )

            # ======================================================
            # Keyboard controls
            # ======================================================

            key = cv2.waitKey(1) & 0xFF

            # Q
            if key == ord("q"):
                break

            # Z -> create/re-create polygon
            elif key == ord("z"):
                drawing_mode = True

                zone = None

                zone_points.clear()

                previous_inside.clear()
                active_intrusions.clear()
                intrusion_trails.clear()
                entry_points.clear()

                history_store.clear()

                logger.info("Restricted-zone drawing started")

            # U -> undo most recent polygon point
            elif key == ord("u"):
                if drawing_mode and zone_points:
                    zone_points.pop()

            # ENTER -> finish polygon
            elif key in (10, 13):
                if drawing_mode:
                    if len(zone_points) >= 3:
                        zone = PolygonZone(zone_points)

                        drawing_mode = False

                        previous_inside.clear()
                        active_intrusions.clear()
                        intrusion_trails.clear()
                        entry_points.clear()

                        history_store.clear()

                        logger.info(
                            "Restricted zone created " "with %s points",
                            len(zone_points),
                        )

                    else:
                        logger.warning(
                            "At least 3 points are required "
                            "to create a restricted zone"
                        )

            # C -> clear zone
            elif key == ord("c"):
                zone = None

                drawing_mode = False

                zone_points.clear()

                previous_inside.clear()
                active_intrusions.clear()
                intrusion_trails.clear()
                entry_points.clear()

                history_store.clear()

                logger.info("Restricted zone cleared")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    finally:
        logger.info("Stopping camera and releasing resources...")

        camera.stop()

        cv2.destroyAllWindows()

        logger.info("Done.")


if __name__ == "__main__":
    main()
