import os
from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from app.camera.frame import Frame
from app.ui.main_window import MainWindow
from app.ui.video_player import VideoPlaybackController


@pytest.fixture(scope="module")
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qt_app):
    main_window = MainWindow()
    yield main_window
    main_window.close()
    qt_app.processEvents()


def test_recorded_event_click_seeks_to_stored_video_offset(window):
    player = Mock()
    player.has_source = True
    player.seek_seconds.return_value = True
    original_player = window.video_player
    window.video_player = player
    window._active_source_kind = "video"
    window._selected_video_path = "recording.mp4"
    window.add_event(
        "00:00:02.345",
        "INTRUSION",
        "Person #4 entered zone",
        source_id="recording.mp4",
        source_time_seconds=2.345,
    )

    window._event_row_clicked(0, 2)

    player.pause.assert_called_once_with()
    player.seek_seconds.assert_called_once_with(2.345)
    window.video_player = original_player


def test_live_event_click_does_not_attempt_recording_seek(window):
    player = Mock()
    player.has_source = True
    original_player = window.video_player
    window.video_player = player
    window._active_source_kind = "webcam"
    window.add_event(
        "2026-11-14 12:00:00.125",
        "SMOKE",
        "Smoke detected",
        source_time_seconds=42.0,
    )

    window._event_row_clicked(0, 1)

    player.pause.assert_not_called()
    player.seek_seconds.assert_not_called()
    assert "Live event" in window.statusBar().currentMessage()
    window.video_player = original_player


def test_recorded_event_time_format_and_duplicate_suppression(window):
    assert window._format_event_video_time(3661.2346) == "01:01:01.235"
    window.add_event(
        "00:00:01.000",
        "FIRE",
        "Fire confirmed",
        source_id="recording.mp4",
        source_time_seconds=1.0,
    )
    window.add_event(
        "00:00:01.000",
        "FIRE",
        "Fire confirmed",
        source_id="recording.mp4",
        source_time_seconds=1.0,
    )

    assert window.event_table.rowCount() == 1


def test_confirmed_identity_is_added_to_recent_person_events(window):
    window.add_event(
        "00:00:01.000",
        "INTRUSION",
        "Person #4 entered zone",
        source_id="recording.mp4",
        source_time_seconds=1.0,
        track_id=4,
    )

    window._annotate_recent_track_events(
        track_id=4,
        identity="Alex",
        source_id="recording.mp4",
        identity_time_seconds=1.5,
    )

    assert window.event_table.item(0, 2).text() == "Alex (Person #4) entered zone"

    window.add_event(
        "00:00:02.000",
        "LOITERING",
        "Alex (Person #4) exceeded 15s dwell time",
        source_id="recording.mp4",
        source_time_seconds=2.0,
        track_id=4,
        identity="Alex",
    )
    assert window.event_table.item(1, 2).text() == (
        "Alex (Person #4) exceeded 15s dwell time"
    )


class FakeSeekCamera:
    fps = 30.0
    frame_count = 300

    def __init__(self):
        self.next_frame_index = 0
        self.displayed_frame_index = 0

    @property
    def current_time_seconds(self):
        return self.displayed_frame_index / self.fps

    def seek_frame(self, frame_index):
        self.next_frame_index = max(0, min(frame_index, self.frame_count - 1))
        return True

    def read_frame(self):
        self.displayed_frame_index = self.next_frame_index
        self.next_frame_index += 1
        return Frame(
            image=np.zeros((2, 2, 3), dtype=np.uint8),
            timestamp=datetime(2026, 11, 14),
            camera_id="test-video",
        )

    def stop(self):
        pass


def test_video_player_seek_seconds_targets_nearest_frame_before_frame_signal(qt_app):
    controller = VideoPlaybackController()
    controller._camera = FakeSeekCamera()
    signals = []
    controller.position_changed.connect(
        lambda seconds: signals.append(("time", seconds))
    )
    controller.frame_ready.connect(lambda _frame: signals.append(("frame", None)))

    assert controller.seek_seconds(70 / 30)

    assert controller._camera.displayed_frame_index == 70
    assert signals == [("time", 70 / 30), ("frame", None)]
    controller.close_video()
