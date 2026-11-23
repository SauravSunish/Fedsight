import numpy as np
from fedsight.portable_tracker import MotionIoUTracker


def test_motion_ids_survive_order_changes_and_a_missed_sample():
    tracker = MotionIoUTracker()
    assert list(tracker.update([[0, 0, 20, 40], [100, 0, 120, 40]])) == [1, 2]
    assert list(tracker.update([[98, 0, 118, 40], [2, 0, 22, 40]])) == [2, 1]
    tracker.update([])
    assert list(tracker.update([[6, 0, 26, 40], [96, 0, 116, 40]])) == [1, 2]


def test_expired_or_nonoverlapping_track_is_new():
    tracker = MotionIoUTracker(max_missed=1)
    assert tracker.update([[0, 0, 10, 20]])[0] == 1
    assert tracker.update([[100, 0, 110, 20]])[0] == 2
    tracker.update([])
    assert tracker.update([[0, 0, 10, 20]])[0] == 3
