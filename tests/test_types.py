import numpy as np

from open_vocab_track.types import Detection, as_boxmot, sanitize_detections


def test_sanitize_and_boxmot_contract():
    detections = sanitize_detections(
        [Detection((-5, 4, 40, 60), 1.2, "target"), Detection((1, 1, 1.5, 1.5), 0.5, "tiny")],
        width=30,
        height=50,
    )
    array = as_boxmot(detections)
    assert array.shape == (1, 6)
    assert array.dtype == np.float32
    np.testing.assert_allclose(array[0], [0, 4, 29, 49, 1, 0])


def test_empty_boxmot_contract():
    assert as_boxmot([]).shape == (0, 6)
