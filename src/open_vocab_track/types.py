from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One pixel-space axis-aligned detection."""

    xyxy: tuple[float, float, float, float]
    score: float
    label: str


def sanitize_detections(
    detections: list[Detection], width: int, height: int, min_area: float = 4.0
) -> list[Detection]:
    """Clamp boxes to the image and remove malformed/tiny detections."""
    clean: list[Detection] = []
    for det in detections:
        x1, y1, x2, y2 = (float(v) for v in det.xyxy)
        x1, x2 = sorted((max(0.0, min(x1, width - 1.0)), max(0.0, min(x2, width - 1.0))))
        y1, y2 = sorted((max(0.0, min(y1, height - 1.0)), max(0.0, min(y2, height - 1.0))))
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        clean.append(Detection((x1, y1, x2, y2), min(1.0, max(0.0, det.score)), det.label))
    return clean


def as_boxmot(detections: list[Detection], class_id: int = 0) -> np.ndarray:
    """Convert to BoxMOT's Nx6 [x1,y1,x2,y2,confidence,class] contract."""
    if not detections:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray([[*det.xyxy, det.score, float(class_id)] for det in detections], dtype=np.float32)
