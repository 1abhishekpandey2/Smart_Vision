"""Provisional face-pose cues for guided live registration."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class PoseTarget(StrEnum):
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


@dataclass(frozen=True)
class FacePose:
    """Two normalized measurements from InsightFace's five keypoints."""

    nose_horizontal_offset: float
    nose_vertical_ratio: float


@dataclass(frozen=True)
class LiveFaceSample:
    pose: FacePose
    embedding: np.ndarray


@dataclass(frozen=True)
class FacePoseCheck:
    face_count: int
    sample: LiveFaceSample | None
    manual_capture: bool
    error: str | None = None


YAW_OFFSET_THRESHOLD = 0.12
PITCH_RATIO_THRESHOLD = 0.08


def estimate_face_pose(landmarks: np.ndarray) -> FacePose:
    """Estimate coarse pose cues from five 2D landmarks.

    Landmark order is InsightFace's left eye, right eye, nose, left mouth
    corner, right mouth corner. These normalized ratios are useful as a
    registration guide only; they are not calibrated head-pose angles.
    """

    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("Expected five finite 2D face landmarks.")

    eye_center = (points[0] + points[1]) / 2.0
    mouth_center = (points[3] + points[4]) / 2.0
    eye_span = float(np.linalg.norm(points[1] - points[0]))
    vertical_span = float(mouth_center[1] - eye_center[1])
    if eye_span <= 1e-6 or vertical_span <= 1e-6:
        raise ValueError("Face landmarks do not define usable eye and mouth spans.")

    nose = points[2]
    return FacePose(
        nose_horizontal_offset=float((nose[0] - eye_center[0]) / eye_span),
        nose_vertical_ratio=float((nose[1] - eye_center[1]) / vertical_span),
    )


def pose_matches_target(
    target: PoseTarget,
    baseline: FacePose,
    current: FacePose,
) -> bool:
    """Compare current landmark ratios with the user's frontal reference."""

    yaw_delta = current.nose_horizontal_offset - baseline.nose_horizontal_offset
    pitch_delta = current.nose_vertical_ratio - baseline.nose_vertical_ratio

    if target is PoseTarget.LEFT:
        return yaw_delta <= -YAW_OFFSET_THRESHOLD
    if target is PoseTarget.RIGHT:
        return yaw_delta >= YAW_OFFSET_THRESHOLD
    if target is PoseTarget.UP:
        return pitch_delta <= -PITCH_RATIO_THRESHOLD
    if target is PoseTarget.DOWN:
        return pitch_delta >= PITCH_RATIO_THRESHOLD
    return (
        abs(yaw_delta) < YAW_OFFSET_THRESHOLD
        and abs(pitch_delta) < PITCH_RATIO_THRESHOLD
    )
