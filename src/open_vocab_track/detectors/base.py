from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from open_vocab_track.types import Detection


class OpenVocabularyDetector(ABC):
    name: str

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray, prompt: str) -> list[Detection]:
        """Detect every region matching prompt in a BGR uint8 frame."""
