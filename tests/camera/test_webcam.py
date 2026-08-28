import sys
from pathlib import Path

# Add project root to sys.path to allow running the test script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.camera.webcam import WebcamCamera
from app.camera.status import CameraStatus
from tests.camera.fakes import FakeVideoCapture


def test_successful_frame_is_stored():
    camera = WebcamCamera(
        "test-camera",
        capture_factory=lambda device_index: FakeVideoCapture(
            device_index,
            read_results=[True],
        ),
    )

    camera.start()

    try:
        frame = camera.read_frame()

        assert frame is not None
        assert camera.last_frame is frame
        assert camera.status == CameraStatus.CONNECTED

        print("✓ Successful frame stored correctly")

    finally:
        camera.stop()


def test_consecutive_failures_disconnect_camera():
    camera = WebcamCamera(
        "test-camera",
        capture_factory=lambda device_index: FakeVideoCapture(
            device_index,
            read_results=[True, False, False, False, False, False],
        ),
    )

    camera.start()

    try:
        # First read succeeds and establishes a last known frame.
        frame = camera.read_frame()

        assert frame is not None
        assert camera.last_frame is frame
        assert camera.status == CameraStatus.CONNECTED

        # Five consecutive failures.
        for _ in range(5):
            failed_frame = camera.read_frame()
            assert failed_frame is None

        assert camera.status == CameraStatus.DISCONNECTED

        # The last successful frame must still be available.
        assert camera.last_frame is frame

        print("✓ Five consecutive failures disconnect camera")
        print("✓ Last successful frame preserved")

    finally:
        camera.stop()


if __name__ == "__main__":
    test_successful_frame_is_stored()
    test_consecutive_failures_disconnect_camera()

    print("\nAll camera tests passed.")
