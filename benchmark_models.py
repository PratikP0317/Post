"""Model-agnostic detector interface and RefDrone detector adapters.

Heavy model dependencies are imported lazily by ``load_model`` so importing this
module does not initialize every backend or require every model environment.
"""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
  from PIL.Image import Image
else:
  Image = Any


@dataclass(frozen=True)
class Detection:
  """One prediction in absolute xyxy coordinates on the original image."""

  bbox_xyxy: tuple[float, float, float, float]
  score: float | None = None
  label: str | None = None


class DetectorModel(ABC):
  """Common lifecycle for every model benchmarked on RefDrone."""

  model_name = "detector"

  def __init__(self, options: Mapping[str, Any] | None = None, threshold: float = 0.0):
    self.options = dict(options or {})
    self.threshold = float(threshold)

  @abstractmethod
  def load_model(self) -> None:
    """Load weights or initialize the remote API client."""

  @abstractmethod
  def predict(self, image: Image, prompt: str) -> list[Detection]:
    """Return zero or more absolute xyxy detections for ``prompt``."""

  def synchronize(self) -> None:
    """Wait for asynchronous accelerator work before latency is sampled."""

  def close(self) -> None:
    """Release model resources."""


def sanitize_detections(
  detections: list[Detection], image_size: tuple[int, int], threshold: float
) -> list[Detection]:
  """Clamp predictions and discard invalid boxes or low-scoring results."""

  width, height = image_size
  sanitized: list[Detection] = []

  for detection in detections:
    try:
      x1, y1, x2, y2 = (float(value) for value in detection.bbox_xyxy)
    except (TypeError, ValueError):
      continue

    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
      continue

    score = detection.score

    if score is not None:
      try:
        score = float(score)
      except (TypeError, ValueError):
        continue

      if not math.isfinite(score) or score < threshold:
        continue

    x1 = max(0.0, min(x1, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    x2 = max(0.0, min(x2, float(width)))
    y2 = max(0.0, min(y2, float(height)))

    if x2 <= x1 or y2 <= y1:
      continue

    sanitized.append(Detection((x1, y1, x2, y2), score, detection.label))

  return sanitized


def _torch_device_and_dtype(torch: Any, options: Mapping[str, Any]) -> tuple[str, Any]:
  requested_device = str(options.get("device", "auto"))
  device = "cuda:0" if requested_device == "auto" and torch.cuda.is_available() else requested_device

  if device == "auto":
    device = "cpu"

  requested_dtype = str(options.get("dtype", "auto"))

  if requested_dtype == "auto":
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
  else:
    dtype = getattr(torch, requested_dtype, None)

    if dtype is None:
      raise ValueError(f"Unsupported torch dtype: {requested_dtype}")

  return device, dtype


class MoondreamDetector(DetectorModel):
  model_name = "moondream"

  def __init__(self, options: Mapping[str, Any] | None = None, threshold: float = 0.0):
    super().__init__(options, threshold)
    self.model: Any = None

  def load_model(self) -> None:
    import moondream as md

    api_key_env = str(self.options.get("api_key_env", "MOONDREAM_API_KEY"))
    api_key = self.options.get("api_key") or os.environ.get(api_key_env)

    if not api_key:
      raise RuntimeError(f"Moondream API key is not set; configure {api_key_env}")

    self.model = md.vl(api_key=str(api_key))

  def predict(self, image: Image, prompt: str) -> list[Detection]:
    if self.model is None:
      raise RuntimeError("MoondreamDetector.load_model() must be called before predict()")

    width, height = image.size
    result = self.model.detect(image=image, object=prompt)
    detections: list[Detection] = []

    for obj in result.get("objects", []):
      score = obj.get("score", obj.get("confidence"))
      score = float(score) if score is not None else None
      detections.append(
        Detection(
          (
            float(obj["x_min"]) * width,
            float(obj["y_min"]) * height,
            float(obj["x_max"]) * width,
            float(obj["y_max"]) * height,
          ),
          score,
          prompt,
        )
      )

    return sanitize_detections(detections, image.size, self.threshold)

  def close(self) -> None:
    self.model = None


class Florence2Detector(DetectorModel):
  model_name = "florence"

  def __init__(self, options: Mapping[str, Any] | None = None, threshold: float = 0.0):
    super().__init__(options, threshold)
    self.model: Any = None
    self.processor: Any = None
    self.torch: Any = None
    self.device = "cpu"
    self.dtype: Any = None

  def load_model(self) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_path = str(self.options.get("model_path", "/Models/Florence-2-large"))
    trust_remote_code = bool(self.options.get("trust_remote_code", True))
    self.device, self.dtype = _torch_device_and_dtype(torch, self.options)
    self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    self.model = AutoModelForCausalLM.from_pretrained(
      model_path, torch_dtype=self.dtype, trust_remote_code=trust_remote_code
    ).to(self.device)
    self.model.eval()
    self.torch = torch

  def predict(self, image: Image, prompt: str) -> list[Detection]:
    if self.model is None or self.processor is None or self.torch is None:
      raise RuntimeError("Florence2Detector.load_model() must be called before predict()")

    task = str(self.options.get("task_prompt", "<CAPTION_TO_PHRASE_GROUNDING>"))
    inputs = self.processor(text=task + prompt, images=image, return_tensors="pt")
    input_ids = inputs["input_ids"].to(self.device)
    pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)

    with self.torch.inference_mode():
      generated_ids = self.model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=int(self.options.get("max_new_tokens", 256)),
        do_sample=False,
        num_beams=int(self.options.get("num_beams", 3)),
      )

    generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = self.processor.post_process_generation(generated_text, task=task, image_size=image.size)
    result = parsed.get(task, {})
    boxes = result.get("bboxes", [])
    labels = result.get("labels", [])
    scores = result.get("scores", [])
    detections: list[Detection] = []

    for index, box in enumerate(boxes):
      score = float(scores[index]) if index < len(scores) else None
      label = str(labels[index]) if index < len(labels) else prompt
      detections.append(Detection(tuple(float(value) for value in box), score, label))

    return sanitize_detections(detections, image.size, self.threshold)

  def synchronize(self) -> None:
    if self.torch is not None and self.device.startswith("cuda"):
      self.torch.cuda.synchronize()

  def close(self) -> None:
    self.model = None
    self.processor = None

    if self.torch is not None and self.device.startswith("cuda"):
      self.torch.cuda.empty_cache()

    self.torch = None


class LLMDetDetector(DetectorModel):
  model_name = "llmdet"

  def __init__(self, options: Mapping[str, Any] | None = None, threshold: float = 0.0):
    super().__init__(options, threshold)
    self.model: Any = None
    self.processor: Any = None
    self.torch: Any = None
    self.device = "cpu"
    self.dtype: Any = None

  def load_model(self) -> None:
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_path = str(self.options.get("model_path", "/Models/llmdet-large"))
    self.device, self.dtype = _torch_device_and_dtype(torch, self.options)
    self.processor = AutoProcessor.from_pretrained(model_path)
    self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
      model_path, torch_dtype=self.dtype
    ).to(self.device)
    self.model.eval()
    self.torch = torch

  def predict(self, image: Image, prompt: str) -> list[Detection]:
    if self.model is None or self.processor is None or self.torch is None:
      raise RuntimeError("LLMDetDetector.load_model() must be called before predict()")

    inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

    with self.torch.inference_mode():
      with self.torch.amp.autocast(device_type="cuda", dtype=self.dtype):
        outputs = self.model(**inputs)

    result = self.processor.post_process_grounded_object_detection(
      outputs, threshold=self.threshold, target_sizes=[(image.height, image.width)]
    )[0]
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])
    labels = result.get("labels", [])
    detections: list[Detection] = []

    for index, box in enumerate(boxes):
      box_values = box.tolist() if hasattr(box, "tolist") else box
      score_value = scores[index] if index < len(scores) else None

      if hasattr(score_value, "item"):
        score_value = score_value.item()

      score = float(score_value) if score_value is not None else None
      label = str(labels[index]) if index < len(labels) else prompt
      detections.append(Detection(tuple(float(value) for value in box_values), score, label))

    return sanitize_detections(detections, image.size, self.threshold)

  def synchronize(self) -> None:
    if self.torch is not None and self.device.startswith("cuda"):
      self.torch.cuda.synchronize()

  def close(self) -> None:
    self.model = None
    self.processor = None

    if self.torch is not None and self.device.startswith("cuda"):
      self.torch.cuda.empty_cache()

    self.torch = None


MODEL_REGISTRY: dict[str, type[DetectorModel]] = {
  "moondream": MoondreamDetector,
  "florence": Florence2Detector,
  "florence2": Florence2Detector,
  "llmdet": LLMDetDetector,
}


def create_detector(
  model_name: str, options: Mapping[str, Any] | None = None, threshold: float = 0.0
) -> DetectorModel:
  """Construct one configured detector without loading it."""

  normalized_name = model_name.strip().lower()

  if normalized_name not in MODEL_REGISTRY:
    available = ", ".join(sorted(MODEL_REGISTRY))
    raise ValueError(f"Unknown detector {model_name!r}. Available detectors: {available}")

  return MODEL_REGISTRY[normalized_name](options=options, threshold=threshold)
