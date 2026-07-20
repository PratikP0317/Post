"""Configurable, model-agnostic RefDrone grounding benchmark.

Edit ``CONFIG`` below, then run:

    python benchmark_refdrone.py

The benchmark evaluates one detector per process and leaves ``file.py``
independent from dataset evaluation.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from benchmark_models import Detection, create_detector, sanitize_detections


# ---------------------------------------------------------------------------
# Configuration -- edit this object to run a benchmark
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
  annotations_path: Path = Path("/data/RefDrone_train_mdetr.json")
  images_dir: Path = Path("/data/VisDrone2019-DET-train/images")
  output_dir: Path = Path("benchmark_results")
  split_name: str = "train"
  sample_limit: int | None = None
  random_seed: int = 17

  model_name: str = "moondream"
  model_threshold: float = 0.3
  model_path: str | None = None
  device: str = "auto"
  dtype: str = "auto"
  model_options: dict[str, dict[str, Any]] = field(default_factory=lambda: {
    "moondream": {
      "api_key_env": "MOONDREAM_API_KEY",
    },
    "florence": {
      "model_path": "/Models/Florence-2-large",
      "task_prompt": "<CAPTION_TO_PHRASE_GROUNDING>",
      "max_new_tokens": 256,
      "num_beams": 3,
      "trust_remote_code": True,
    },
    "llmdet": {
      "model_path": "iSEE-Laboratory/llmdet_tiny",
    },
  })

  iou_threshold: float = 0.5
  warmup_samples: int = 3
  continue_on_error: bool = True
  save_failure_visuals: bool = True
  max_failure_visuals: int = 100
  run_name: str | None = None
  resume_run_dir: Path | None = None


CONFIG = BenchmarkConfig()


# ---------------------------------------------------------------------------
# Dataset records
# ---------------------------------------------------------------------------

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class RefDroneSample:
  sample_id: str
  image_id: str
  image_path: Path
  prompt: str
  gt_boxes: tuple[Box, ...]


# ---------------------------------------------------------------------------
# RefDrone MDETR loading
# ---------------------------------------------------------------------------

def _xywh_to_xyxy(bbox: Iterable[Any]) -> Box:
  values = list(bbox)

  if len(values) != 4:
    raise ValueError(f"Expected a four-value xywh box, received {values!r}")

  x, y, width, height = (float(value) for value in values)

  if not all(math.isfinite(value) for value in (x, y, width, height)):
    raise ValueError(f"Bounding box contains a non-finite value: {values!r}")

  if width <= 0 or height <= 0:
    raise ValueError(f"Bounding box has non-positive dimensions: {values!r}")

  return x, y, x + width, y + height


def _prompt_from_image_record(record: dict[str, Any]) -> str:
  for key in ("caption", "sentence", "expression", "text"):
    if key not in record:
      continue

    value = record[key]

    if isinstance(value, str):
      return value.strip()

  raise ValueError(f"Image record {record.get('id')!r} has no referring expression")


def _resolve_image_path(images_dir: Path, file_name: str) -> Path:
  direct_path = images_dir / file_name

  if direct_path.exists():
    return direct_path

  basename_path = images_dir / Path(file_name).name
  return basename_path if basename_path.exists() else direct_path


def load_refdrone_samples(config: BenchmarkConfig) -> list[RefDroneSample]:
  if not config.annotations_path.is_file():
    raise FileNotFoundError(f"RefDrone annotations not found: {config.annotations_path}")

  with config.annotations_path.open("r", encoding="utf-8") as handle:
    dataset = json.load(handle)

  image_records = dataset.get("images")
  annotation_records = dataset.get("annotations", dataset.get("annotaions"))

  if not isinstance(image_records, list) or not isinstance(annotation_records, list):
    raise ValueError("Expected MDETR JSON with 'images' and 'annotations' lists")

  annotations_by_image: dict[str, list[Box]] = {}

  for annotation in annotation_records:
    image_id = str(annotation.get("image_id"))

    if annotation.get("empty") is True:
      annotations_by_image.setdefault(image_id, [])
      continue

    annotations_by_image.setdefault(image_id, []).append(_xywh_to_xyxy(annotation["bbox"]))

  samples: list[RefDroneSample] = []
  seen_sample_ids: set[str] = set()

  for index, image_record in enumerate(image_records):
    image_id = str(image_record.get("id"))
    sample_id = image_id

    if sample_id in seen_sample_ids:
      sample_id = f"{image_id}:{index}"

    seen_sample_ids.add(sample_id)
    file_name = image_record.get("file_name", image_record.get("filename"))

    if not isinstance(file_name, str) or not file_name:
      raise ValueError(f"Image record {image_id!r} has no file_name")

    prompt = _prompt_from_image_record(image_record)
    gt_boxes = tuple(annotations_by_image.get(image_id, []))

    if not prompt and gt_boxes:
      raise ValueError(f"Positive image record {image_id!r} has an empty referring expression")

    samples.append(
      RefDroneSample(
        sample_id=sample_id,
        image_id=image_id,
        image_path=_resolve_image_path(config.images_dir, file_name),
        prompt=prompt,
        gt_boxes=gt_boxes,
      )
    )

  if config.sample_limit is not None:
    if config.sample_limit <= 0:
      raise ValueError("sample_limit must be positive or None")

    if config.sample_limit < len(samples):
      randomizer = random.Random(config.random_seed)
      selected_indices = sorted(randomizer.sample(range(len(samples)), config.sample_limit))
      samples = [samples[index] for index in selected_indices]

  missing_paths = sorted({sample.image_path for sample in samples if not sample.image_path.is_file()})

  if missing_paths:
    examples = "\n".join(f"  - {path}" for path in missing_paths[:10])
    remainder = len(missing_paths) - 10
    suffix = f"\n  ... and {remainder} more" if remainder > 0 else ""
    raise FileNotFoundError(
      f"{len(missing_paths)} RefDrone image files are missing:\n{examples}{suffix}"
    )

  if not samples:
    raise ValueError("No RefDrone samples were found in the annotation file")

  return samples


# ---------------------------------------------------------------------------
# IoU matching and official metrics
# ---------------------------------------------------------------------------

def box_iou(first: Box, second: Box) -> float:
  x1 = max(first[0], second[0])
  y1 = max(first[1], second[1])
  x2 = min(first[2], second[2])
  y2 = min(first[3], second[3])
  intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
  first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
  second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
  union = first_area + second_area - intersection
  return intersection / union if union > 0 else 0.0


def match_boxes(
  predicted_boxes: list[Box], gt_boxes: list[Box], iou_threshold: float
) -> list[tuple[int, int, float]]:
  """Reproduce RefDrone's greedy highest-IoU one-to-one matching."""

  candidates: list[tuple[float, int, int]] = []

  for prediction_index, predicted_box in enumerate(predicted_boxes):
    for gt_index, gt_box in enumerate(gt_boxes):
      iou = box_iou(predicted_box, gt_box)

      if iou >= iou_threshold:
        candidates.append((iou, prediction_index, gt_index))

  used_predictions: set[int] = set()
  used_gt: set[int] = set()
  matches: list[tuple[int, int, float]] = []

  for iou, prediction_index, gt_index in sorted(candidates, reverse=True):
    if prediction_index in used_predictions or gt_index in used_gt:
      continue

    used_predictions.add(prediction_index)
    used_gt.add(gt_index)
    matches.append((prediction_index, gt_index, iou))

  return sorted(matches, key=lambda item: item[0])


def _safe_divide(numerator: float, denominator: float) -> float:
  return numerator / denominator if denominator else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
  return _safe_divide(2 * tp, 2 * tp + fp + fn)


def _percentile(values: list[float], percentile: float) -> float:
  if not values:
    return 0.0

  ordered = sorted(values)
  position = (len(ordered) - 1) * percentile
  lower = math.floor(position)
  upper = math.ceil(position)

  if lower == upper:
    return ordered[lower]

  return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def score_records(records: list[dict[str, Any]], iou_threshold: float) -> dict[str, Any]:
  instance_tp = instance_tn = instance_fp = instance_fn = 0
  image_tp = image_tn = image_fp = image_fn = 0
  no_target_total = no_target_failures = 0
  slice_totals = {"single": 0, "multi": 0}
  slice_exact = {"single": 0, "multi": 0}
  size_totals = {"small": 0, "medium": 0, "large": 0}
  size_matches = {"small": 0, "medium": 0, "large": 0}
  count_errors: list[float] = []
  detector_latencies: list[float] = []
  total_latencies: list[float] = []
  model_errors = 0

  for record in records:
    gt_boxes = [tuple(float(value) for value in box) for box in record["gt_boxes"]]
    predictions = record.get("predictions", [])
    predicted_boxes = [
      tuple(float(value) for value in prediction["bbox_xyxy"])
      for prediction in predictions
    ]
    has_error = bool(record.get("error"))
    matches = [] if has_error else match_boxes(predicted_boxes, gt_boxes, iou_threshold)
    matched_gt = {match[1] for match in matches}

    detector_latencies.append(float(record.get("detector_latency_ms", 0.0)))
    total_latencies.append(float(record.get("total_latency_ms", 0.0)))
    count_errors.append(abs(len(predicted_boxes) - len(gt_boxes)))

    if has_error:
      model_errors += 1

      if gt_boxes:
        instance_fn += len(gt_boxes)
        slice_name = "single" if len(gt_boxes) == 1 else "multi"
        slice_totals[slice_name] += 1
        image_fn += 1
      else:
        instance_fp += 1
        image_fp += 1
    elif not gt_boxes:
      no_target_total += 1

      if predicted_boxes:
        # The official RefDrone evaluator counts a no-target failure once per image.
        instance_fp += 1
        image_fp += 1
        no_target_failures += 1
      else:
        instance_tn += 1
        image_tn += 1
    else:
      true_positives = len(matches)
      instance_tp += true_positives
      instance_fp += len(predicted_boxes) - true_positives
      instance_fn += len(gt_boxes) - true_positives
      exact = true_positives == len(gt_boxes) == len(predicted_boxes)

      if exact:
        image_tp += 1
      else:
        image_fn += 1

      slice_name = "single" if len(gt_boxes) == 1 else "multi"
      slice_totals[slice_name] += 1
      slice_exact[slice_name] += int(exact)

    if has_error and not gt_boxes:
      no_target_total += 1
      no_target_failures += 1

    for gt_index, gt_box in enumerate(gt_boxes):
      area = max(0.0, gt_box[2] - gt_box[0]) * max(0.0, gt_box[3] - gt_box[1])

      if area < 32 ** 2:
        size_name = "small"
      elif area <= 96 ** 2:
        size_name = "medium"
      else:
        size_name = "large"

      size_totals[size_name] += 1
      size_matches[size_name] += int(gt_index in matched_gt)

  instance_total = instance_tp + instance_tn + instance_fp + instance_fn
  image_total = image_tp + image_tn + image_fp + image_fn
  total_inference_seconds = sum(total_latencies) / 1000.0

  return {
    "iou_threshold": iou_threshold,
    "samples": len(records),
    "instance": {
      "tp": instance_tp,
      "tn": instance_tn,
      "fp": instance_fp,
      "fn": instance_fn,
      "f1": _f1(instance_tp, instance_fp, instance_fn),
      "accuracy": _safe_divide(instance_tp + instance_tn, instance_total),
    },
    "image": {
      "tp": image_tp,
      "tn": image_tn,
      "fp": image_fp,
      "fn": image_fn,
      "f1": _f1(image_tp, image_fp, image_fn),
      "accuracy": _safe_divide(image_tp + image_tn, image_total),
    },
    "slices": {
      "no_target": {
        "samples": no_target_total,
        "false_positive_rate": _safe_divide(no_target_failures, no_target_total),
      },
      "single_target": {
        "samples": slice_totals["single"],
        "exact_accuracy": _safe_divide(slice_exact["single"], slice_totals["single"]),
      },
      "multi_target": {
        "samples": slice_totals["multi"],
        "exact_accuracy": _safe_divide(slice_exact["multi"], slice_totals["multi"]),
      },
      "object_size_recall": {
        name: _safe_divide(size_matches[name], size_totals[name])
        for name in size_totals
      },
      "object_size_counts": size_totals,
    },
    "count_mae": _safe_divide(sum(count_errors), len(count_errors)),
    "model_errors": model_errors,
    "model_error_rate": _safe_divide(model_errors, len(records)),
    "latency_ms": {
      "detector_mean": _safe_divide(sum(detector_latencies), len(detector_latencies)),
      "total_mean": _safe_divide(sum(total_latencies), len(total_latencies)),
      "total_p50": _percentile(total_latencies, 0.50),
      "total_p95": _percentile(total_latencies, 0.95),
    },
    "inference_throughput_samples_per_second": _safe_divide(
      len(records), total_inference_seconds
    ),
  }


# ---------------------------------------------------------------------------
# Result persistence and visualization
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
  if isinstance(value, Path):
    return str(value)

  if isinstance(value, dict):
    return {str(key): _jsonable(item) for key, item in value.items()}

  if isinstance(value, (list, tuple)):
    return [_jsonable(item) for item in value]

  return value


def _redacted_config(config: BenchmarkConfig) -> dict[str, Any]:
  serialized = _jsonable(asdict(config))

  def redact(value: Any) -> Any:
    if isinstance(value, dict):
      return {
        key: "<redacted>" if any(token in key.lower() for token in ("api_key", "token")) else redact(item)
        for key, item in value.items()
      }

    if isinstance(value, list):
      return [redact(item) for item in value]

    return value

  return redact(serialized)


def _safe_name(value: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.+-]+", "-", value).strip("-") or "run"


def _write_json(path: Path, value: Any) -> None:
  temporary_path = path.with_suffix(path.suffix + ".tmp")
  temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  temporary_path.replace(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
  if not path.exists():
    return []

  records = []

  with path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
      if not line.strip():
        continue

      try:
        records.append(json.loads(line))
      except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error

  sample_ids = [str(record["sample_id"]) for record in records]

  if len(sample_ids) != len(set(sample_ids)):
    raise ValueError(f"Duplicate sample IDs found in {path}")

  return records


def _record_is_failure(record: dict[str, Any], iou_threshold: float) -> bool:
  if record.get("error"):
    return True

  gt_boxes = [tuple(box) for box in record["gt_boxes"]]
  predicted_boxes = [tuple(item["bbox_xyxy"]) for item in record.get("predictions", [])]

  if not gt_boxes:
    return bool(predicted_boxes)

  matches = match_boxes(predicted_boxes, gt_boxes, iou_threshold)
  return len(matches) != len(gt_boxes) or len(matches) != len(predicted_boxes)


def _render_failure(
  sample: RefDroneSample,
  record: dict[str, Any],
  output_path: Path,
  model_label: str,
  iou_threshold: float,
) -> None:
  from PIL import Image, ImageDraw

  with Image.open(sample.image_path) as source_image:
    image = source_image.convert("RGB")

  draw = ImageDraw.Draw(image)
  gt_boxes = [tuple(box) for box in record["gt_boxes"]]
  predicted_boxes = [tuple(item["bbox_xyxy"]) for item in record.get("predictions", [])]
  matches = match_boxes(predicted_boxes, gt_boxes, iou_threshold)

  for index, box in enumerate(gt_boxes):
    draw.rectangle(box, outline=(0, 255, 0), width=3)
    draw.text((box[0] + 2, box[1] + 2), f"GT {index}", fill=(0, 255, 0))

  for index, prediction in enumerate(record.get("predictions", [])):
    box = prediction["bbox_xyxy"]
    score = prediction.get("score")
    suffix = f" {score:.2f}" if score is not None else ""
    draw.rectangle(box, outline=(255, 0, 0), width=3)
    draw.text((box[0] + 2, max(0, box[1] - 12)), f"P {index}{suffix}", fill=(255, 0, 0))

  match_text = ", ".join(f"P{pred}-GT{gt}:{iou:.2f}" for pred, gt, iou in matches) or "none"
  error_text = f" | error: {record['error']}" if record.get("error") else ""
  header = (
    f"{model_label} | {record.get('total_latency_ms', 0.0):.1f} ms | matches {match_text}{error_text}\n"
    f"Prompt: {sample.prompt}"
  )
  draw.rectangle((0, 0, image.width, 34), fill=(0, 0, 0))
  draw.multiline_text((4, 3), header[:500], fill=(255, 255, 255), spacing=2)
  image.save(output_path, format="JPEG", quality=90)


def _append_summary_csv(path: Path, summary: dict[str, Any]) -> None:
  row = {
    "run_id": summary["run_id"],
    "model": summary["model"],
    "split": summary["split"],
    "samples": summary["metrics"]["samples"],
    "instance_f1": summary["metrics"]["instance"]["f1"],
    "instance_accuracy": summary["metrics"]["instance"]["accuracy"],
    "image_f1": summary["metrics"]["image"]["f1"],
    "image_accuracy": summary["metrics"]["image"]["accuracy"],
    "no_target_false_positive_rate": summary["metrics"]["slices"]["no_target"]["false_positive_rate"],
    "count_mae": summary["metrics"]["count_mae"],
    "model_error_rate": summary["metrics"]["model_error_rate"],
    "detector_latency_mean_ms": summary["metrics"]["latency_ms"]["detector_mean"],
    "total_latency_p50_ms": summary["metrics"]["latency_ms"]["total_p50"],
    "total_latency_p95_ms": summary["metrics"]["latency_ms"]["total_p95"],
    "throughput_samples_per_second": summary["metrics"]["inference_throughput_samples_per_second"],
  }
  existing_rows: list[dict[str, Any]] = []

  if path.exists() and path.stat().st_size > 0:
    with path.open("r", newline="", encoding="utf-8") as handle:
      existing_rows = [
        existing for existing in csv.DictReader(handle)
        if existing.get("run_id") != row["run_id"]
      ]

  temporary_path = path.with_suffix(path.suffix + ".tmp")

  with temporary_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(row))
    writer.writeheader()
    writer.writerows(
      {field: existing.get(field, "") for field in row}
      for existing in existing_rows
    )
    writer.writerow(row)

  temporary_path.replace(path)


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

def _selected_model_options(config: BenchmarkConfig) -> dict[str, Any]:
  normalized_name = config.model_name.strip().lower()
  options = dict(config.model_options.get(normalized_name, {}))
  options["device"] = config.device
  options["dtype"] = config.dtype

  if config.model_path is not None:
    options["model_path"] = config.model_path

  return options


def _run_label(config: BenchmarkConfig) -> str:
  return config.model_name.strip().lower()


def _prepare_run_directory(config: BenchmarkConfig) -> tuple[str, Path]:
  config.output_dir.mkdir(parents=True, exist_ok=True)

  if config.resume_run_dir is not None:
    run_dir = config.resume_run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir.name, run_dir

  timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  run_id = _safe_name(config.run_name or f"{_run_label(config)}-{config.split_name}-{timestamp}")
  run_dir = config.output_dir / run_id

  if run_dir.exists():
    raise FileExistsError(f"Run directory already exists: {run_dir}")

  run_dir.mkdir(parents=True)
  return run_id, run_dir


def _open_rgb_image(path: Path) -> Any:
  from PIL import Image

  with Image.open(path) as source_image:
    return source_image.convert("RGB")


def _serialize_detection(detection: Detection) -> dict[str, Any]:
  return {
    "bbox_xyxy": list(detection.bbox_xyxy),
    "score": detection.score,
    "label": detection.label,
  }


def run_benchmark(config: BenchmarkConfig = CONFIG) -> dict[str, Any]:
  if not 0.0 <= config.model_threshold <= 1.0:
    raise ValueError("model_threshold must be between 0 and 1")

  if not 0.0 < config.iou_threshold <= 1.0:
    raise ValueError("iou_threshold must be in (0, 1]")

  samples = load_refdrone_samples(config)
  run_id, run_dir = _prepare_run_directory(config)
  predictions_path = run_dir / "predictions.jsonl"
  config_path = run_dir / "config.json"
  existing_records = _load_jsonl(predictions_path)
  existing_ids = {str(record["sample_id"]) for record in existing_records}
  requested_ids = {sample.sample_id for sample in samples}

  if not existing_ids.issubset(requested_ids):
    unexpected = sorted(existing_ids - requested_ids)[:10]
    raise ValueError(f"Resume file contains samples outside this dataset selection: {unexpected}")

  if config.resume_run_dir is not None and config_path.exists():
    previous_config = json.loads(config_path.read_text(encoding="utf-8"))

    for key in ("annotations_path", "images_dir", "model_name", "model_threshold", "iou_threshold"):
      if previous_config.get(key) != _redacted_config(config).get(key):
        raise ValueError(f"Cannot resume because configuration field {key!r} changed")
  else:
    _write_json(config_path, _redacted_config(config))

  pending_samples = [sample for sample in samples if sample.sample_id not in existing_ids]
  model_label = _run_label(config)
  failures_dir = run_dir / "failures"

  if config.save_failure_visuals:
    failures_dir.mkdir(exist_ok=True)

  detector = create_detector(
    config.model_name,
    options=_selected_model_options(config),
    threshold=config.model_threshold,
  )
  load_start = time.perf_counter()
  detector.load_model()
  detector.synchronize()
  model_load_seconds = time.perf_counter() - load_start
  run_start = time.perf_counter()
  visual_count = len(list(failures_dir.glob("*.jpg"))) if failures_dir.exists() else 0

  try:
    for sample in pending_samples[:max(0, config.warmup_samples)]:
      try:
        image = _open_rgb_image(sample.image_path)
        detector.predict(image, sample.prompt)
        detector.synchronize()
      except Exception as error:
        print(f"Warmup failed for sample {sample.sample_id}: {error}")

    with predictions_path.open("a", encoding="utf-8") as predictions_file:
      for completed, sample in enumerate(pending_samples, start=1):
        detector_latency_ms = 0.0
        error_message = None
        raw_detections: list[Detection] = []
        image = None

        try:
          image = _open_rgb_image(sample.image_path)
          detector.synchronize()
          detector_start = time.perf_counter()

          try:
            raw_detections = detector.predict(image, sample.prompt)
            detector.synchronize()
          finally:
            detector_latency_ms = (time.perf_counter() - detector_start) * 1000.0

          raw_detections = sanitize_detections(
            raw_detections, image.size, config.model_threshold
          )

        except Exception as error:
          error_message = f"{type(error).__name__}: {error}"

          if not config.continue_on_error:
            raise

        total_latency_ms = detector_latency_ms
        record = {
          "sample_id": sample.sample_id,
          "image_id": sample.image_id,
          "image_path": str(sample.image_path),
          "prompt": sample.prompt,
          "gt_boxes": [list(box) for box in sample.gt_boxes],
          "raw_predictions": [_serialize_detection(item) for item in raw_detections],
          "predictions": [_serialize_detection(item) for item in raw_detections],
          "detector_latency_ms": detector_latency_ms,
          "total_latency_ms": total_latency_ms,
          "error": error_message,
        }
        predictions_file.write(json.dumps(record, sort_keys=True) + "\n")
        predictions_file.flush()

        if (
          config.save_failure_visuals
          and visual_count < config.max_failure_visuals
          and _record_is_failure(record, config.iou_threshold)
        ):
          visual_path = failures_dir / f"{visual_count:04d}-{_safe_name(sample.sample_id)}.jpg"

          try:
            _render_failure(
              sample, record, visual_path, model_label, config.iou_threshold
            )
            visual_count += 1
          except Exception as error:
            print(f"Could not render failure {sample.sample_id}: {error}")

        print(
          f"[{completed}/{len(pending_samples)}] {sample.sample_id} "
          f"GT={len(sample.gt_boxes)} predicted={len(raw_detections)} "
          f"latency={total_latency_ms:.1f} ms"
        )

  finally:
    detector.close()

  wall_seconds = time.perf_counter() - run_start
  records = _load_jsonl(predictions_path)
  final_ids = {str(record["sample_id"]) for record in records}

  if final_ids != requested_ids:
    missing = sorted(requested_ids - final_ids)[:10]
    raise RuntimeError(f"Benchmark did not produce exactly one record per sample; missing: {missing}")

  metrics = score_records(records, config.iou_threshold)
  summary = {
    "run_id": run_id,
    "model": model_label,
    "split": config.split_name,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "annotations_path": str(config.annotations_path),
    "images_dir": str(config.images_dir),
    "model_load_seconds": model_load_seconds,
    "benchmark_wall_seconds": wall_seconds,
    "failure_visuals": visual_count,
    "metrics": metrics,
  }
  _write_json(run_dir / "summary.json", summary)
  _append_summary_csv(config.output_dir / "summary.csv", summary)

  print(json.dumps(summary, indent=2))
  return summary


if __name__ == "__main__":
  run_benchmark()
