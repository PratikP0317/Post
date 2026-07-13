from pathlib import Path

import cv2
import numpy as np

from open_vocab_track.pipeline import run_video
from open_vocab_track.types import Detection


class FakeDetector:
    def detect(self, frame, prompt):
        return [Detection((8, 8, 28, 35), 0.9, prompt)]


class FakeTracker:
    def update(self, dets, frame):
        if not len(dets):
            return np.empty((0, 8), dtype=np.float32)
        return np.asarray([[*dets[0, :4], 7, dets[0, 4], 0, 0]], dtype=np.float32)


def test_video_pipeline_writes_video_and_metadata(tmp_path: Path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    metadata = tmp_path / "tracks.jsonl"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    summary = run_video(
        source, output, FakeDetector(), FakeTracker(), "person in green", metadata_path=metadata
    )
    assert summary.frames == 3
    assert summary.detections == 3
    assert output.stat().st_size > 0
    assert len(metadata.read_text().splitlines()) == 3
