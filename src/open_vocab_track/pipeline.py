from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from open_vocab_track.detectors.base import OpenVocabularyDetector
from open_vocab_track.tracking import Tracker
from open_vocab_track.types import Detection, as_boxmot, sanitize_detections


@dataclass
class RunSummary:
    input: str
    output: str
    prompt: str
    frames: int
    detections: int
    tracks_emitted: int
    detector_calls: int
    detector_seconds: float
    wall_seconds: float
    source_fps: float

    @property
    def processed_fps(self) -> float:
        return self.frames / self.wall_seconds if self.wall_seconds else 0.0


def _detect_resized(
    detector: OpenVocabularyDetector, frame: np.ndarray, prompt: str, max_side: int
) -> list[Detection]:
    height, width = frame.shape[:2]
    if max_side <= 0 or max(height, width) <= max_side:
        return detector.detect(frame, prompt)
    scale = max_side / max(height, width)
    small = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    detections = detector.detect(small, prompt)
    mapped = [
        Detection(tuple(value / scale for value in det.xyxy), det.score, det.label) for det in detections
    ]
    return sanitize_detections(mapped, width, height)


def _draw(frame: np.ndarray, tracks: np.ndarray, prompt: str) -> None:
    for row in tracks:
        if len(row) < 7:
            continue
        x1, y1, x2, y2, track_id, confidence = row[:6]
        color = (37, 210, 90)
        cv2.rectangle(frame, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
        label = f"#{int(track_id)} {prompt} {confidence:.2f}"
        cv2.putText(
            frame, label, (round(x1), max(18, round(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )


def run_video(
    source: str | Path,
    output: str | Path,
    detector: OpenVocabularyDetector,
    tracker: Tracker,
    prompt: str,
    detect_every: int = 1,
    max_inference_side: int = 1280,
    max_frames: int = 0,
    metadata_path: str | Path | None = None,
) -> RunSummary:
    if detect_every < 1:
        raise ValueError("detect_every must be at least 1")
    source, output = str(source), str(output)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output}")

    metadata_file = None
    if metadata_path:
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        metadata_file = open(metadata_path, "w", encoding="utf-8")  # noqa: SIM115

    frames = detection_count = track_count = detector_calls = 0
    detector_seconds = 0.0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok or (max_frames and frames >= max_frames):
                break
            if frames % detect_every == 0:
                before = time.perf_counter()
                detections = _detect_resized(detector, frame, prompt, max_inference_side)
                detector_seconds += time.perf_counter() - before
                detector_calls += 1
            else:
                detections = []
            detection_count += len(detections)
            tracks = tracker.update(as_boxmot(detections), frame)
            tracks = np.asarray(tracks) if tracks is not None else np.empty((0, 8), dtype=np.float32)
            track_count += len(tracks)
            if metadata_file:
                metadata_file.write(json.dumps({"frame": frames, "tracks": tracks.tolist()}) + "\n")
            _draw(frame, tracks, prompt)
            writer.write(frame)
            frames += 1
    finally:
        capture.release()
        writer.release()
        if metadata_file:
            metadata_file.close()

    return RunSummary(
        input=source,
        output=output,
        prompt=prompt,
        frames=frames,
        detections=detection_count,
        tracks_emitted=track_count,
        detector_calls=detector_calls,
        detector_seconds=detector_seconds,
        wall_seconds=time.perf_counter() - started,
        source_fps=fps,
    )


def write_summary(summary: RunSummary, path: str | Path) -> None:
    payload = asdict(summary)
    payload["processed_fps"] = summary.processed_fps
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
