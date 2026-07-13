import json

import cv2
import numpy as np

from open_vocab_track.benchmark import match_counts, run_refdrone
from open_vocab_track.types import Detection


def test_one_to_one_matching_does_not_double_count():
    predictions = [Detection((0, 0, 10, 10), 0.9, "x"), Detection((1, 1, 10, 10), 0.8, "x")]
    assert match_counts(predictions, [(0, 0, 10, 10)], 0.5) == (1, 1, 0)


def test_refdrone_end_to_end(tmp_path):
    class FakeDetector:
        def detect(self, frame, prompt):
            assert prompt == "person wearing green"
            return [Detection((10, 20, 30, 50), 0.9, prompt)]

    cv2.imwrite(str(tmp_path / "frame.jpg"), np.zeros((80, 100, 3), dtype=np.uint8))
    payload = {
        "images": [{"id": 1, "file_name": "frame.jpg", "caption": "person wearing green"}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [10, 20, 20, 30], "empty": 0}],
    }
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(payload))
    metrics = run_refdrone(annotations, tmp_path, FakeDetector())
    assert metrics["iou_0.5"]["f1"] == 1.0
