from __future__ import annotations

import numpy as np
from PIL import Image

from open_vocab_track.detectors.base import OpenVocabularyDetector
from open_vocab_track.types import Detection, sanitize_detections


class MoondreamDetector(OpenVocabularyDetector):
    """Moondream 2 native open-vocabulary detection adapter."""

    name = "moondream"

    def __init__(
        self,
        model_id: str = "vikhyatk/moondream2",
        revision: str = "2025-06-21",
        device: str = "cuda",
        default_score: float = 0.9,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover - depends on model environment
            raise RuntimeError("Install model dependencies with: pip install -e '.[models]'") from exc

        self.default_score = default_score
        kwargs = {"revision": revision, "trust_remote_code": True}
        if device == "cuda":
            kwargs["device_map"] = {"": "cuda"}
        else:
            kwargs["device_map"] = {"": device}
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()

    @staticmethod
    def parse_objects(
        objects: list[dict], width: int, height: int, prompt: str, default_score: float = 0.9
    ) -> list[Detection]:
        detections = []
        for obj in objects:
            # Moondream emits normalized coordinates in [0, 1].
            box = (
                float(obj["x_min"]) * width,
                float(obj["y_min"]) * height,
                float(obj["x_max"]) * width,
                float(obj["y_max"]) * height,
            )
            detections.append(Detection(box, float(obj.get("score", default_score)), prompt))
        return sanitize_detections(detections, width, height)

    def detect(self, frame_bgr: np.ndarray, prompt: str) -> list[Detection]:
        height, width = frame_bgr.shape[:2]
        image = Image.fromarray(frame_bgr[:, :, ::-1])
        result = self.model.detect(image, prompt)
        return self.parse_objects(result.get("objects", []), width, height, prompt, self.default_score)
