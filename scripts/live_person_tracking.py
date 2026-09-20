import argparse
import sys

import cv2

from app.camera.status import CameraStatus
from app.camera.webcam import WebcamCamera
from app.core.logger import logger
from app.tracking.yolo_person import YOLOPersonTracker

WINDOW_NAME = "Smart Vision - Person Tracking"


def main():
    parser = argparse.ArgumentParser(
        description="Run Smart Vision live person tracking."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo11n.pt",
        help="Path to the person detection model.",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Webcam device index.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Person detection confidence threshold.",
    )

    args = parser.parse_args()

    logger.info(
        "Loading person tracking model from %s",
        args.model,
    )

    tracker = YOLOPersonTracker(
        model_path=args.model,
        confidence_threshold=args.conf,
    )

    camera = WebcamCamera(
        camera_id=f"webcam-{args.device}",
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

    logger.info("Live person tracking started.")

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

            image = frame.image.copy()

            tracks = tracker.track(frame)

            for track in tracks:
                x1, y1, x2, y2 = track.bounding_box

                label = f"Person #{track.track_id} " f"{track.confidence:.0%}"

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    image,
                    label,
                    (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                ground_x, ground_y = track.ground_point

                cv2.circle(
                    image,
                    (ground_x, ground_y),
                    5,
                    (0, 255, 255),
                    cv2.FILLED,
                )

            # ------------------------------------------------------
            # Information panel
            # ------------------------------------------------------

            cv2.rectangle(
                image,
                (10, 10),
                (300, 85),
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
                (25, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

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
                WINDOW_NAME,
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
