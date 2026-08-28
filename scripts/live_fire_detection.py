import argparse
import sys

import cv2

from app.camera.webcam import WebcamCamera
from app.camera.status import CameraStatus
from app.detection.yolo import YOLODetector
from app.core.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="Run live fire and smoke detection on webcam video stream."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/firedetect-11s.pt",
        help="Path to the YOLO detection model file (default: models/firedetect-11s.pt)",
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
    args = parser.parse_args()

    # Load detector
    logger.info("Loading detector from %s...", args.model)
    detector = YOLODetector(model_path=args.model, confidence_threshold=args.conf)

    # Initialize camera
    camera_id = f"webcam-{args.device}"
    logger.info("Initializing webcam camera ID '%s'...", camera_id)
    camera = WebcamCamera(camera_id=camera_id, device_index=args.device)

    camera.start()
    if camera.status != CameraStatus.CONNECTED:
        logger.error(
            "Could not connect to camera [%s]: %s",
            camera.id,
            camera.last_error,
        )
        sys.exit(1)

    logger.info("Live detection started.")
    print("Press 'q' key in the video window to quit.")

    # Color mapping for classes (BGR format)
    colors = {"fire": (0, 0, 255), "smoke": (128, 128, 128)}  # Red  # Gray
    default_color = (0, 255, 0)  # Green

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

            # Run detection
            detections = detector.detect(frame)

            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = det.bounding_box
                label = f"{det.class_name} {det.confidence:.2%}"
                color = colors.get(det.class_name.lower(), default_color)

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # Draw background for text
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    image,
                    (x1, y1 - text_height - 6),
                    (x1 + text_width, y1),
                    color,
                    cv2.FILLED,
                )

                # Write label text
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Display key controls
            cv2.putText(
                image,
                "Press 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Smart Vision - Live Fire Detection", image)

            # Wait for key press
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
