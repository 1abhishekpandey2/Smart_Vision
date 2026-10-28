import numpy as np

from app.recognition.association import (
    FaceObservation,
    associate_faces_to_tracks,
)
from app.tracking.track import Track


def make_track(
    track_id: int,
    bounding_box: tuple[int, int, int, int],
) -> Track:
    return Track(
        track_id=track_id,
        class_id=0,
        class_name="person",
        confidence=0.9,
        bounding_box=bounding_box,
    )


def make_face(
    bounding_box: tuple[float, float, float, float],
) -> FaceObservation:
    return FaceObservation(
        bounding_box=bounding_box,
        embedding=np.ones(512, dtype=np.float32),
    )


def test_associates_face_when_center_is_inside_one_person_box():
    face = make_face((40.0, 20.0, 60.0, 40.0))
    track = make_track(12, (0, 0, 100, 200))

    result = associate_faces_to_tracks([face], [track])

    assert result == {12: face}


def test_leaves_face_unassigned_when_center_is_outside_all_person_boxes():
    face = make_face((140.0, 20.0, 160.0, 40.0))
    track = make_track(12, (0, 0, 100, 200))

    result = associate_faces_to_tracks([face], [track])

    assert result == {}


def test_leaves_face_unassigned_when_center_is_inside_overlapping_boxes():
    face = make_face((40.0, 20.0, 60.0, 40.0))
    tracks = [
        make_track(12, (0, 0, 100, 200)),
        make_track(13, (25, 0, 125, 200)),
    ]

    result = associate_faces_to_tracks([face], tracks)

    assert result == {}


def test_leaves_track_unassigned_when_multiple_faces_map_to_it():
    faces = [
        make_face((20.0, 20.0, 40.0, 40.0)),
        make_face((60.0, 20.0, 80.0, 40.0)),
    ]
    track = make_track(12, (0, 0, 100, 200))

    result = associate_faces_to_tracks(faces, [track])

    assert result == {}


def test_associates_faces_to_distinct_tracks():
    first_face = make_face((20.0, 20.0, 40.0, 40.0))
    second_face = make_face((120.0, 20.0, 140.0, 40.0))
    tracks = [
        make_track(12, (0, 0, 100, 200)),
        make_track(13, (100, 0, 200, 200)),
    ]

    result = associate_faces_to_tracks(
        [first_face, second_face],
        tracks,
    )

    assert result == {12: first_face, 13: second_face}
