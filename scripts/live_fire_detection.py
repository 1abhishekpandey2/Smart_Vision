import argparse
import sys

import cv2

from app.camera.status import CameraStatus
from app.camera.webcam import WebcamCamera
from app.core.logger import logger
from app.detection.yolo import YOLODetector
from app.reasoning.temporal import TemporalPersistence


def main():
    parser = argparse.ArgumentParser(
        description="Run live fire and smoke detection on webcam video stream."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/firedetect-11s.pt",
        help="Path to the YOLO detection model file "
        "(default: models/firedetect-11s.pt)",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold for detections (default: 0.4)",
    )

    parser.add_argument(
        "--temporal-window",
        type=int,
        default=10,
        help="Number of recent frames used for temporal reasoning (default: 10)",
    )

    parser.add_argument(
        "--temporal-required",
        type=int,
        default=7,
        help="Positive fire frames required inside the temporal window " "(default: 7)",
    )

    args = parser.parse_args()

    # Load detector
    logger.info("Loading detector from %s...", args.model)

    detector = YOLODetector(
        model_path=args.model,
        confidence_threshold=args.conf,
    )

    # Create temporal fire reasoner
    try:
        fire_temporal = TemporalPersistence(
            window_size=args.temporal_window,
            required_positive_frames=args.temporal_required,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    # Initialize camera
    camera_id = f"webcam-{args.device}"

    logger.info(
        "Initializing webcam camera ID '%s'...",
        camera_id,
    )

    camera = WebcamCamera(
        camera_id=camera_id,
        device_index=args.device,
    )

    camera.start()

    if camera.status != CameraStatus.CONNECTED:
        logger.error(
            "Could not connect to camera [%s]: %s",
            camera.id,
            camera.last_error,
        )

        sys.exit(1)

    logger.info(
        "Live detection started. Temporal fire rule: %s of %s frames.",
        args.temporal_required,
        args.temporal_window,
    )

    print("Press 'q' key in the video window to quit.")

    # Color mapping for detection classes.
    # OpenCV uses BGR instead of RGB.
    colors = {
        "fire": (0, 0, 255),
        "smoke": (128, 128, 128),
    }

    default_color = (0, 255, 0)

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
                continue

            # Preserve the original raw Frame image.
            image = frame.image.copy()

            # ----------------------------------------------------------
            # Detection
            # ----------------------------------------------------------

            detections = detector.detect(frame)

            # Determine whether this particular frame contains fire.
            fire_present = any(
                detection.class_name == "fire" for detection in detections
            )

            # Feed one temporal observation into our reasoning component.
            fire_temporal.observe(fire_present)

            # ----------------------------------------------------------
            # Draw detections
            # ----------------------------------------------------------

            for detection in detections:
                x1, y1, x2, y2 = detection.bounding_box

                label = f"{detection.class_name} " f"{detection.confidence:.2%}"

                color = colors.get(
                    detection.class_name.lower(),
                    default_color,
                )

                # Bounding box
                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )

                # Calculate text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    1,
                )

                # Prevent the text background from going above the image.
                label_top = max(y1 - text_height - 8, 0)

                cv2.rectangle(
                    image,
                    (x1, label_top),
                    (x1 + text_width + 4, y1),
                    color,
                    cv2.FILLED,
                )

                cv2.putText(
                    image,
                    label,
                    (x1 + 2, max(y1 - 5, text_height)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # ----------------------------------------------------------
            # Temporal state
            # ----------------------------------------------------------

            if fire_temporal.observation_count < fire_temporal.window_size:
                temporal_status = "WARMING UP"
                status_color = (255, 255, 0)

            elif fire_temporal.is_confirmed:
                temporal_status = "FIRE CONFIRMED"
                status_color = (0, 0, 255)

            elif fire_temporal.positive_count > 0:
                temporal_status = "POSSIBLE FIRE"
                status_color = (0, 255, 255)

            else:
                temporal_status = "MONITORING"
                status_color = (0, 255, 0)

            # ----------------------------------------------------------
            # Temporal status panel
            # ----------------------------------------------------------

            panel_x1 = 10
            panel_y1 = 10
            panel_x2 = 430
            panel_y2 = 125

            cv2.rectangle(
                image,
                (panel_x1, panel_y1),
                (panel_x2, panel_y2),
                (30, 30, 30),
                cv2.FILLED,
            )

            cv2.rectangle(
                image,
                (panel_x1, panel_y1),
                (panel_x2, panel_y2),
                status_color,
                2,
            )

            cv2.putText(
                image,
                "SMART VISION",
                (25, 37),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                f"Fire evidence: "
                f"{fire_temporal.positive_count}"
                f"/{fire_temporal.window_size} "
                f"(need {args.temporal_required})",
                (25, 69),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                temporal_status,
                (25, 103),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                status_color,
                2,
                cv2.LINE_AA,
            )

            # Key-control information
            cv2.putText(
                image,
                "Press 'q' to quit",
                (10, image.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Smart Vision - Live Fire Detection",
                image,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    finally:
        logger.info("Stopping camera and releasing resources...")

        camera.stop()
        cv2.destroyAllWindows()

        logger.info("Done.")


if __name__ == "__main__":
    main()
