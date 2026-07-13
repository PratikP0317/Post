from __future__ import annotations

from typing import Protocol

import numpy as np


class Tracker(Protocol):
    def update(self, dets: np.ndarray, frame: np.ndarray) -> np.ndarray: ...


def create_tracker(name: str, fps: float = 30.0) -> Tracker:
    """Create a motion-first BoxMOT tracker suitable for VLM detections."""
    normalized = name.lower().replace("_", "-")
    frame_rate = max(1, round(fps))
    if normalized == "bytetrack":
        from boxmot.trackers.bytetrack.bytetrack import ByteTrack

        return ByteTrack(frame_rate=frame_rate, track_thresh=0.45, min_conf=0.05, track_buffer=30)
    if normalized == "ocsort":
        from boxmot.trackers.ocsort.ocsort import OcSort

        return OcSort(min_conf=0.05, max_age=30, min_hits=1, use_byte=True)
    if normalized == "botsort":
        import torch
        from boxmot.trackers.botsort.botsort import BotSort

        # ReID is deliberately disabled: arbitrary referring expressions do not
        # share a stable person-class embedding and avoid a large extra model.
        return BotSort(
            reid_weights=None,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            half=torch.cuda.is_available(),
            with_reid=False,
            cmc_method="ecc",
            frame_rate=frame_rate,
            track_low_thresh=0.05,
            track_high_thresh=0.45,
            new_track_thresh=0.45,
        )
    raise ValueError(f"Unknown tracker {name!r}; choose bytetrack, ocsort, or botsort")
