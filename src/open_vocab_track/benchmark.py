from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import cv2

from open_vocab_track.detectors.base import OpenVocabularyDetector
from open_vocab_track.types import Detection


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (
        max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        - intersection
    )
    return intersection / union if union else 0.0


def match_counts(predictions: list[Detection], targets: list[tuple[float, ...]], threshold: float):
    candidates = sorted(
        (
            (iou(pred.xyxy, target), pi, ti)
            for pi, pred in enumerate(predictions)
            for ti, target in enumerate(targets)
        ),
        reverse=True,
    )
    used_predictions, used_targets = set(), set()
    for overlap, pi, ti in candidates:
        if overlap < threshold:
            break
        if pi not in used_predictions and ti not in used_targets:
            used_predictions.add(pi)
            used_targets.add(ti)
    true_positive = len(used_targets)
    return true_positive, len(predictions) - true_positive, len(targets) - true_positive


def run_refdrone(
    annotations_path: str | Path,
    image_root: str | Path,
    detector: OpenVocabularyDetector,
    limit: int = 0,
) -> dict:
    data = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    annotations = defaultdict(list)
    for item in data["annotations"]:
        if item.get("empty", 0):
            continue
        x, y, width, height = item["bbox"]
        annotations[item["image_id"]].append((x, y, x + width, y + height))

    totals = {0.5: [0, 0, 0], 0.75: [0, 0, 0]}
    latency = 0.0
    evaluated = 0
    for image_info in data["images"]:
        if limit and evaluated >= limit:
            break
        path = Path(image_root) / image_info["file_name"]
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"Missing RefDrone image: {path}")
        before = time.perf_counter()
        predictions = detector.detect(frame, image_info["caption"])
        latency += time.perf_counter() - before
        targets = annotations[image_info["id"]]
        for threshold in totals:
            counts = match_counts(predictions, targets, threshold)
            totals[threshold] = [a + b for a, b in zip(totals[threshold], counts)]
        evaluated += 1

    metrics = {
        "images": evaluated,
        "detector_seconds": latency,
        "seconds_per_image": latency / evaluated if evaluated else 0.0,
    }
    for threshold, (tp, fp, fn) in totals.items():
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics[f"iou_{threshold}"] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return metrics
