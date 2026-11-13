import os

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from app.recognition.face_pose import (
    FacePose,
    FacePoseCheck,
    LiveFaceSample,
    PoseTarget,
    estimate_face_pose,
    pose_matches_target,
)
from app.ui.face_registration_dialog import FaceRegistrationDialog


@pytest.fixture
def frontal_landmarks():
    return np.asarray(
        [
            [40.0, 40.0],
            [60.0, 40.0],
            [50.0, 55.0],
            [45.0, 70.0],
            [55.0, 70.0],
        ],
        dtype=np.float32,
    )


@pytest.fixture(scope="module")
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def test_pose_metrics_are_normalized_to_eye_and_face_spacing(frontal_landmarks):
    pose = estimate_face_pose(frontal_landmarks)

    assert pose == FacePose(nose_horizontal_offset=0.0, nose_vertical_ratio=0.5)


@pytest.mark.parametrize(
    ("target", "nose_x", "nose_y", "expected"),
    [
        (PoseTarget.LEFT, 47.0, 55.0, True),
        (PoseTarget.RIGHT, 53.0, 55.0, True),
        (PoseTarget.UP, 50.0, 52.0, True),
        (PoseTarget.DOWN, 50.0, 58.0, True),
        (PoseTarget.LEFT, 49.0, 55.0, False),
        (PoseTarget.UP, 50.0, 54.0, False),
    ],
)
def test_pose_target_requires_a_minimum_relative_landmark_shift(
    frontal_landmarks,
    target,
    nose_x,
    nose_y,
    expected,
):
    baseline = estimate_face_pose(frontal_landmarks)
    shifted = frontal_landmarks.copy()
    shifted[2] = [nose_x, nose_y]
    current = estimate_face_pose(shifted)

    assert pose_matches_target(target, baseline, current) is expected


def test_pose_estimation_rejects_wrong_landmark_shape():
    with pytest.raises(ValueError, match="five finite 2D face landmarks"):
        estimate_face_pose(np.zeros((4, 2), dtype=np.float32))


def test_dialog_auto_advances_after_three_matching_pose_checks(qt_app):
    dialog = FaceRegistrationDialog("Synthetic")
    dialog.set_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert dialog.progress_ring._progress == 0.0

    frontal = LiveFaceSample(
        pose=FacePose(0.0, 0.5),
        embedding=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    dialog.handle_pose_check(FacePoseCheck(1, frontal, manual_capture=True))
    assert dialog.progress.text().endswith("LEFT")
    assert dialog.progress_ring._progress == pytest.approx(0.2)

    left = LiveFaceSample(
        pose=FacePose(-0.15, 0.5),
        embedding=np.asarray([0.9, 0.1], dtype=np.float32),
    )
    for expected_progress in (0.2 + 1 / 15, 0.2 + 2 / 15):
        dialog.handle_pose_check(FacePoseCheck(1, left, manual_capture=False))
        assert dialog.progress_ring._progress == pytest.approx(expected_progress)
        assert "Hold still" in dialog.instruction.text()
    dialog.handle_pose_check(FacePoseCheck(1, left, manual_capture=False))

    assert len(dialog._samples) == 2
    assert dialog.progress.text().endswith("RIGHT")
    assert dialog.progress_ring._progress == pytest.approx(0.4)
    assert dialog.preview._pose_target is PoseTarget.RIGHT
    dialog.reject()
