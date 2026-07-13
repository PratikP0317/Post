from __future__ import annotations

from open_vocab_track.detectors.base import OpenVocabularyDetector


def create_detector(name: str, **kwargs) -> OpenVocabularyDetector:
    normalized = name.lower().replace("_", "-")
    if normalized == "moondream":
        from .moondream import MoondreamDetector

        return MoondreamDetector(**kwargs)
    if normalized in {"florence", "florence2", "florence-2"}:
        from .florence2 import Florence2Detector

        return Florence2Detector(**kwargs)
    if normalized in {"locateanything", "locate-anything", "locatinganything"}:
        from .locate_anything import LocateAnythingDetector

        return LocateAnythingDetector(**kwargs)
    raise ValueError(f"Unknown detector {name!r}; choose moondream, locate-anything, or florence2")
