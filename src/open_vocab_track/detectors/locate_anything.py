from __future__ import annotations

import re

import numpy as np
from PIL import Image

from open_vocab_track.detectors.base import OpenVocabularyDetector
from open_vocab_track.types import Detection, sanitize_detections


class LocateAnythingDetector(OpenVocabularyDetector):
    """NVIDIA LocateAnything-3B phrase-grounding adapter."""

    name = "locate-anything"

    def __init__(
        self,
        model_id: str = "nvidia/LocateAnything-3B",
        device: str = "cuda",
        default_score: float = 0.9,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 2048,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install model dependencies with: pip install -e '.[models]'") from exc

        self.torch = torch
        self.device = device
        self.dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.default_score = default_score
        self.generation_mode = generation_mode
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = (
            AutoModel.from_pretrained(model_id, torch_dtype=self.dtype, trust_remote_code=True)
            .to(device)
            .eval()
        )

    @staticmethod
    def parse_answer(answer: str, width: int, height: int, prompt: str, score: float = 0.9):
        detections = []
        pattern = r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>"
        for match in re.finditer(pattern, answer):
            x1, y1, x2, y2 = (int(v) for v in match.groups())
            detections.append(
                Detection(
                    (x1 * width / 1000, y1 * height / 1000, x2 * width / 1000, y2 * height / 1000),
                    score,
                    prompt,
                )
            )
        return sanitize_detections(detections, width, height)

    def detect(self, frame_bgr: np.ndarray, prompt: str) -> list[Detection]:
        height, width = frame_bgr.shape[:2]
        image = Image.fromarray(frame_bgr[:, :, ::-1])
        question = f"Locate all the instances that match the following description: {prompt}."
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": question}],
            }
        ]
        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(
            self.device
        )
        with self.torch.inference_mode():
            response = self.model.generate(
                pixel_values=inputs["pixel_values"].to(self.dtype),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                generation_mode=self.generation_mode,
                do_sample=False,
                verbose=False,
            )
        answer = response[0] if isinstance(response, tuple) else response
        if not isinstance(answer, str):
            answer = self.tokenizer.decode(answer, skip_special_tokens=False)
        return self.parse_answer(answer, width, height, prompt, self.default_score)
