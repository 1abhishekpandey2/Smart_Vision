from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.tracking.track import Track


@dataclass(frozen=True)
class FaceObservation:
    """One face detected in the current frame."""

    bounding_box: tuple[float, float, float, float]
    embedding: np.ndarray


def associate_faces_to_tracks(
    faces: Sequence[FaceObservation],
    tracks: Sequence[Track],
) -> dict[int, FaceObservation]:
    """Associate faces by unique face-center containment in person boxes.

    A face is assigned only when its center is inside exactly one person box,
    and that person box contains exactly one face center. Ambiguous and
    unassociated faces are omitted so callers can leave those tracks unknown.
    """

    faces_by_track: dict[int, list[FaceObservation]] = {}

    for face in faces:
        face_x1, face_y1, face_x2, face_y2 = face.bounding_box
        face_center_x = (face_x1 + face_x2) / 2.0
        face_center_y = (face_y1 + face_y2) / 2.0

        containing_tracks = [
            track
            for track in tracks
            if (
                track.bounding_box[0] <= face_center_x <= track.bounding_box[2]
                and track.bounding_box[1] <= face_center_y <= track.bounding_box[3]
            )
        ]

        if len(containing_tracks) != 1:
            continue

        track_id = containing_tracks[0].track_id
        faces_by_track.setdefault(track_id, []).append(face)

    return {
        track_id: track_faces[0]
        for track_id, track_faces in faces_by_track.items()
        if len(track_faces) == 1
    }
