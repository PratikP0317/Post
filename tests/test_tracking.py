import numpy as np
import pytest


@pytest.mark.parametrize("name", ["bytetrack", "ocsort", "botsort"])
def test_real_boxmot_tracker_contract_when_installed(name):
    pytest.importorskip("boxmot")
    from open_vocab_track.tracking import create_tracker

    tracker = create_tracker(name, 30)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detections = np.asarray([[10, 10, 40, 80, 0.9, 0]], dtype=np.float32)
    tracks = np.asarray(tracker.update(detections, frame))
    assert tracks.shape == (1, 8)
    assert tracks[0, 4] >= 1
