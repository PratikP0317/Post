from __future__ import annotations

import numpy as np
from PIL import Image

from open_vocab_track.detectors.base import OpenVocabularyDetector
from open_vocab_track.types import Detection, sanitize_detections


class Florence2Detector(OpenVocabularyDetector):
    """Florence-2 caption-to-phrase-grounding adapter."""

    name = "florence2"
    task = "<CAPTION_TO_PHRASE_GROUNDING>"

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-base-ft",
        device: str = "cuda",
        default_score: float = 0.9,
        num_beams: int = 3,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install model dependencies with: pip install -e '.[models]'") from exc

        self.torch = torch
        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.default_score = default_score
        self.num_beams = num_beams
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=self.dtype, trust_remote_code=True)
            .to(device)
            .eval()
        )

    @staticmethod
    def parse_result(result: dict, prompt: str, default_score: float = 0.9) -> list[Detection]:
        payload = result.get(Florence2Detector.task, result)
        boxes = payload.get("bboxes", [])
        labels = payload.get("labels", [])
        return [
            Detection(tuple(float(v) for v in box), default_score, labels[i] if i < len(labels) else prompt)
            for i, box in enumerate(boxes)
        ]

    def detect(self, frame_bgr: np.ndarray, prompt: str) -> list[Detection]:
        height, width = frame_bgr.shape[:2]
        image = Image.fromarray(frame_bgr[:, :, ::-1])
        text = self.task + prompt
        inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device, self.dtype)
        with self.torch.inference_mode():
            ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=self.num_beams,
                do_sample=False,
            )
        generated = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        result = self.processor.post_process_generation(generated, task=self.task, image_size=(width, height))
        return sanitize_detections(self.parse_result(result, prompt, self.default_score), width, height)
