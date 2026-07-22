# RefDrone Grounding Benchmark

This repository contains a configurable benchmark for measuring how well an object-detection or visual-grounding model can locate an object described by natural language in a drone image.

For each RefDrone sample, the benchmark gives one image and one referring expression to a selected detector. An expression might be something like `person in a green shirt` or `the white vehicle near the intersection`. The detector returns zero or more bounding boxes. Those predictions are compared with the annotated boxes using intersection over union (IoU), then the benchmark reports accuracy, F1, object-size recall, counting error, latency, throughput, model errors, and saved failure examples.

The benchmark is implemented in two standalone files:

- [`benchmark_refdrone.py`](benchmark_refdrone.py) loads RefDrone data, runs inference, matches predictions to ground truth, computes metrics, saves results, and supports resumable runs.
- [`benchmark_models.py`](benchmark_models.py) defines the common detector interface and adapters for Moondream, Florence-2, and LLMDet.

This benchmark evaluates **static image grounding**. It does not process video sequences, preserve track IDs, or measure temporal tracking quality.

## Table of contents

- [What the benchmark tests](#what-the-benchmark-tests)
- [What it does not test](#what-it-does-not-test)
- [End-to-end benchmark flow](#end-to-end-benchmark-flow)
- [RefDrone input JSON format](#refdrone-input-json-format)
- [Bounding-box formats](#bounding-box-formats)
- [Configuration](#configuration)
- [Supported detector adapters](#supported-detector-adapters)
- [Detection cleanup](#detection-cleanup)
- [How predictions are matched](#how-predictions-are-matched)
- [How every metric is calculated](#how-every-metric-is-calculated)
- [Timing and throughput](#timing-and-throughput)
- [Running the benchmark](#running-the-benchmark)
- [Output directory and file formats](#output-directory-and-file-formats)
- [Failure visualizations](#failure-visualizations)
- [Warmup behavior](#warmup-behavior)
- [Error handling](#error-handling)
- [Resuming a run](#resuming-a-run)
- [How to interpret common outcomes](#how-to-interpret-common-outcomes)
- [Adding another model](#adding-another-model)
- [Testing the benchmark implementation](#testing-the-benchmark-implementation)
- [Validation checklist](#validation-checklist)

## What the benchmark tests

The central question is:

> Given a drone image and a natural-language expression, does the model return the correct bounding box or boxes for the described target?

The benchmark tests the following behaviors:

1. **Referring-expression understanding** — whether the model connects the words in the prompt to the correct visible object.
2. **Spatial localization** — whether the predicted box overlaps the annotated target closely enough.
3. **Multiple-target grounding** — whether the model finds all annotated instances when an expression refers to more than one object.
4. **No-target rejection** — whether the model returns no boxes when the described target is absent.
5. **Object counting** — whether the number of predicted boxes matches the number of annotated boxes.
6. **Object-size performance** — how recall changes for small, medium, and large targets.
7. **Inference reliability** — how many samples produce model or adapter errors.
8. **Inference speed** — detector latency, latency percentiles, and inference-only throughput.

The default localization requirement is IoU `>= 0.5`. This is controlled by `iou_threshold` and is independent of the model confidence threshold.

## What it does not test

This distinction is important when reading the results.

The benchmark does **not** test:

- video tracking across frames;
- track-ID consistency or ID switches;
- reacquisition after occlusion;
- tracker drift;
- end-to-end streaming frame rate;
- network latency separately from model latency for remote detectors;
- preprocessing, result-file writing, or failure-image rendering as part of detector latency;
- model quality on arbitrary data outside the selected RefDrone split.

A strong result means the model grounds RefDrone expressions well on individual images. It does not by itself prove that the model will track those targets well in video.

## End-to-end benchmark flow

```text
RefDrone MDETR JSON + image directory
                  |
                  v
       Load image/prompt samples
       Convert GT xywh -> xyxy
                  |
                  v
       Optional deterministic subset
                  |
                  v
       Load one selected detector
                  |
                  v
       Run untimed warmup samples
                  |
                  v
       For every pending sample:
         1. Open image as RGB
         2. Run detector with prompt
         3. Synchronize accelerator
         4. Sanitize predicted boxes
         5. Write one JSONL record
         6. Render the failure if needed
                  |
                  v
       Greedy IoU matching and scoring
                  |
                  v
 config.json + predictions.jsonl + summary.json
             + summary.csv + failure JPEGs
```

Only one detector is evaluated per process. To compare several detectors fairly, run the script once per model with the same dataset paths, sample selection, IoU threshold, and comparable hardware conditions.

## RefDrone input JSON format

The loader expects an MDETR/COCO-like JSON object containing an `images` list and an `annotations` list. A `categories` list may also be present, but this benchmark does not use it.

### Complete example

```json
{
  "images": [
    {
      "id": 101,
      "file_name": "000001.jpg",
      "caption": "the white car near the intersection"
    },
    {
      "id": 102,
      "file_name": "000002.jpg",
      "caption": "person wearing a yellow shirt"
    },
    {
      "id": 103,
      "file_name": "000003.jpg",
      "caption": ""
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 101,
      "bbox": [598, 387, 180, 179],
      "category_id": 1
    },
    {
      "id": 2,
      "image_id": 102,
      "bbox": [120, 80, 24, 55],
      "category_id": 1
    },
    {
      "id": 3,
      "image_id": 103,
      "bbox": [0, 0, 0, 0],
      "empty": true,
      "category_id": 1
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "object"
    }
  ]
}
```

### `images` records

Each image record becomes one benchmark sample.

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | Expected | Image/expression identifier. It is converted to a string internally. |
| `file_name` | Yes | Image path relative to `images_dir`. The alias `filename` is also accepted. |
| `caption` | Yes in normal RefDrone data | Natural-language referring expression sent to the detector. |

The prompt loader checks `caption`, `sentence`, `expression`, and `text`, in that order, and uses the first string-valued field it finds.

If duplicate image IDs appear, the first sample uses the ID as its `sample_id`; later duplicates receive `:<image-list-index>` so every output record still has a unique ID.

### `annotations` records

Annotations are grouped by `image_id`. An image may therefore have zero, one, or multiple ground-truth boxes.

| Field | Required | Meaning |
| --- | --- | --- |
| `image_id` | Yes | Connects the annotation to an image record. |
| `bbox` | Yes for a normal target | Ground-truth box in `[x, y, width, height]` format. |
| `empty` | Optional | When exactly `true`, marks a no-target sentinel. Its box is ignored. |

For compatibility with a known misspelling, the loader accepts a top-level `annotaions` list if `annotations` is absent. New datasets should use the correct `annotations` spelling.

### No-target samples

A no-target expression has no real ground-truth boxes. RefDrone may represent it with an annotation such as:

```json
{
  "image_id": 103,
  "bbox": [0, 0, 0, 0],
  "empty": true
}
```

When `empty` is `true`, the loader preserves the image as a zero-target sample but does not treat `[0, 0, 0, 0]` as a box. An empty prompt is accepted only for a sample with zero ground-truth boxes. A positive sample with boxes and an empty prompt is rejected as malformed.

### Image-path resolution

For `file_name: "subdirectory/000001.jpg"`, the loader tries:

1. `<images_dir>/subdirectory/000001.jpg`
2. `<images_dir>/000001.jpg`

Before model loading begins, the benchmark checks every selected path. If images are missing, it reports the first ten missing paths and the number of additional missing files.

## Bounding-box formats

The input annotations and benchmark predictions use different box conventions.

### Annotation input: `xywh`

RefDrone annotations enter as:

```text
[x, y, width, height]
```

Here, `x` and `y` are the top-left corner. The loader validates that all four values are finite and that width and height are positive.

### Internal and output format: `xyxy`

The benchmark immediately converts ground truth to:

```text
[x1, y1, x2, y2]
```

where:

```text
x2 = x + width
y2 = y + height
```

For example:

```text
Input annotation: [598, 387, 180, 179]
Internal box:      [598, 387, 778, 566]
```

Every detector adapter must also return absolute-pixel `xyxy` boxes on the original image. Output JSON therefore stores both `gt_boxes` and predicted `bbox_xyxy` values in `xyxy`, not `xywh`.

## Configuration

The benchmark is configured by editing the `BenchmarkConfig` object near the top of [`benchmark_refdrone.py`](benchmark_refdrone.py).

```python
CONFIG = BenchmarkConfig()
```

### Configuration fields

| Field | Default | Purpose |
| --- | --- | --- |
| `annotations_path` | `/data/RefDrone_train_mdetr.json` | RefDrone MDETR annotation JSON. |
| `images_dir` | `/data/VisDrone2019-DET-train/images` | Directory containing the referenced images. |
| `output_dir` | `benchmark_results` | Parent directory for run results and `summary.csv`. |
| `split_name` | `train` | Human-readable split name stored in result summaries and run IDs. |
| `sample_limit` | `None` | Run all samples, or choose a deterministic subset of this size. |
| `random_seed` | `17` | Seed used only for deterministic subset selection. |
| `model_name` | `moondream` | Detector adapter to run. |
| `model_threshold` | `0.3` | Minimum confidence for predictions that provide scores. |
| `model_path` | `None` | Optional top-level model-path override. |
| `device` | `auto` | `auto`, `cpu`, `cuda:0`, or another Torch device string. |
| `dtype` | `auto` | `auto` or a Torch dtype name such as `float16` or `float32`. |
| `model_options` | Model-specific mapping | Per-adapter paths and inference options. |
| `iou_threshold` | `0.5` | Minimum IoU for a prediction/ground-truth match. |
| `warmup_samples` | `3` | Number of pending samples used for untimed warmup calls. |
| `continue_on_error` | `True` | Save failed samples and continue instead of stopping immediately. |
| `save_failure_visuals` | `True` | Save annotated JPEGs for failed samples. |
| `max_failure_visuals` | `100` | Maximum failure images retained for one run. |
| `run_name` | `None` | Optional explicit run ID before filename sanitization. |
| `resume_run_dir` | `None` | Existing run directory to resume. |

### Example: small Florence-2 validation run

```python
CONFIG = BenchmarkConfig(
  annotations_path=Path("/data/RefDrone_val_mdetr.json"),
  images_dir=Path("/data/VisDrone2019-DET-val/images"),
  output_dir=Path("benchmark_results"),
  split_name="val",
  sample_limit=100,
  random_seed=17,
  model_name="florence",
  model_threshold=0.3,
  model_path="/Models/Florence-2-large",
  device="cuda:0",
  dtype="float16",
  iou_threshold=0.5,
  warmup_samples=3,
  continue_on_error=True,
  save_failure_visuals=True,
  max_failure_visuals=100,
  run_name="florence-val-100",
)
```

### Deterministic sample limiting

When `sample_limit` is smaller than the dataset:

1. Python's local random generator is initialized with `random_seed`.
2. That generator samples image-list indices without replacement.
3. The selected indices are sorted so processing retains original dataset order.

The same annotation file, limit, and seed produce the same subset. Change the seed to choose a different subset. Use `None` to benchmark the full split.

### Model-path precedence

Each model may define a path in `model_options`. If top-level `model_path` is not `None`, it overrides the selected adapter's `model_options[model_name]["model_path"]` value.

## Supported detector adapters

All adapters implement the same lifecycle:

```python
detector = create_detector(model_name, options, threshold)
detector.load_model()
detections = detector.predict(image, prompt)
detector.synchronize()
detector.close()
```

Every `predict` call returns a list of immutable `Detection` objects:

```python
Detection(
  bbox_xyxy=(x1, y1, x2, y2),
  score=0.87,
  label="the white car",
)
```

The list may contain zero, one, or many detections. `score` and `label` may be `None`.

### Moondream

Registry name: `moondream`

- Initializes the Moondream API client.
- Reads its API key from `MOONDREAM_API_KEY` by default.
- Calls `detect(image=image, object=prompt)`.
- Converts normalized coordinates in the API response to absolute image pixels.
- Uses the prompt as the fallback label.

Example configuration:

```python
model_name="moondream"
model_options={
  "moondream": {
    "api_key_env": "MOONDREAM_API_KEY",
  },
}
```

Set the key before running:

```bash
export MOONDREAM_API_KEY="your-key"
```

Moondream's recorded detector latency includes the remote API request because it occurs inside `predict`.

### Florence-2

Registry names: `florence` and `florence2`

- Loads `AutoProcessor` and `AutoModelForCausalLM` lazily.
- Uses `<CAPTION_TO_PHRASE_GROUNDING>` by default.
- Concatenates the task token and referring expression.
- Generates a model response, then calls Florence post-processing with the original image size.
- Returns parsed boxes, labels, and scores when scores are available.

Important options are `model_path`, `task_prompt`, `max_new_tokens`, `num_beams`, and `trust_remote_code`.

### LLMDet

Registry name: `llmdet`

- Loads `AutoProcessor` and `AutoModelForZeroShotObjectDetection` lazily.
- Sends one image and the plain prompt string to the processor.
- Runs inference under Torch inference mode and CUDA autocast.
- Calls `post_process_grounded_object_detection` with the original image dimensions.
- Applies `model_threshold` during model post-processing.

The adapter's built-in fallback path is `/Models/llmdet-large`. The current benchmark configuration provides `iSEE-Laboratory/llmdet_tiny` unless top-level `model_path` overrides it.

### Lazy dependencies

Heavy model packages are imported inside each adapter's `load_model` method. This allows the benchmark modules to be imported for dataset or metric work without loading every model backend. Running a particular adapter still requires that adapter's packages, model files, credentials, and compatible hardware environment.

Common requirements include Python 3, Pillow, and the dependencies for the selected model. Local Florence-2 and LLMDet runs additionally require compatible PyTorch and Transformers installations. This repository does not currently pin those packages in a requirements file, so the runtime environment must supply them.

## Detection cleanup

Predictions pass through `sanitize_detections` before scoring. This protects the evaluator from malformed adapter output.

For each detection, the sanitizer:

1. Converts all four box coordinates to floats.
2. Rejects boxes containing nonnumeric or non-finite values.
3. Converts a provided score to a float.
4. Rejects nonnumeric or non-finite scores.
5. Rejects scored predictions below `model_threshold`.
6. Clamps `x1` and `x2` into `[0, image_width]`.
7. Clamps `y1` and `y2` into `[0, image_height]`.
8. Rejects a box if `x2 <= x1` or `y2 <= y1` after clamping.

If a prediction has `score: null`, confidence thresholding is skipped for that prediction. It can still be accepted if its box is valid.

Some adapters sanitize their output internally, and the benchmark runner sanitizes the returned list again. The second pass establishes the runner-level contract regardless of the selected adapter.

## How predictions are matched

The benchmark uses greedy, highest-IoU, one-to-one matching.

### IoU

For a predicted box `P` and ground-truth box `G`:

```text
IoU(P, G) = intersection_area(P, G) / union_area(P, G)
```

IoU ranges from `0.0` for no overlap to `1.0` for identical boxes.

### Matching algorithm

For one sample:

1. Calculate IoU for every prediction/ground-truth pair.
2. Discard pairs below `iou_threshold`.
3. Sort the remaining pairs from highest to lowest IoU.
4. Accept the highest pair whose prediction and ground truth are both unused.
5. Mark both boxes as used and continue.
6. Stop after all candidates have been considered.

One prediction can match at most one ground-truth box, and one ground-truth box can match at most one prediction.

### Example

Suppose a sample has two ground-truth boxes and three predictions. The candidate IoUs are:

| Pair | IoU |
| --- | ---: |
| Prediction 0 → GT 0 | 0.82 |
| Prediction 0 → GT 1 | 0.10 |
| Prediction 1 → GT 0 | 0.61 |
| Prediction 1 → GT 1 | 0.73 |
| Prediction 2 → GT 1 | 0.42 |

With a threshold of `0.5`, the benchmark selects Prediction 0 → GT 0 at `0.82`, then Prediction 1 → GT 1 at `0.73`. Prediction 2 is unmatched. The result is two true positives and one false positive.

## How every metric is calculated

The benchmark reports both **instance-level** and **image-level** metrics. These answer different questions.

### Instance-level counts

For a positive sample:

- `TP` is the number of matched prediction/ground-truth pairs.
- `FP` is the number of unmatched predictions.
- `FN` is the number of unmatched ground-truth boxes.

For a no-target sample:

- no predictions adds one `TN`;
- one or more predictions adds one `FP`, regardless of how many boxes were returned.

This special no-target rule measures whether the image-level absence decision was correct without allowing many boxes on one no-target image to dominate the entire instance metric.

### Image-level counts

A positive image counts as an image-level `TP` only when it is an **exact detection**:

```text
number of matches == number of GT boxes == number of predictions
```

Therefore, all targets must be found and there must be no extra predictions. Any mismatch on a positive sample counts as one image-level `FN`.

For a no-target image:

- no predictions adds one image-level `TN`;
- one or more predictions adds one image-level `FP`.

### F1

Both instance and image F1 use:

```text
F1 = 2 * TP / (2 * TP + FP + FN)
```

If the denominator is zero, the benchmark returns `0.0`.

### Accuracy

Both levels use:

```text
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### No-target false-positive rate

```text
no_target_false_positive_rate =
  no-target samples with predictions or errors / all no-target samples
```

If there are no no-target samples, the value is `0.0`.

### Single-target and multi-target exact accuracy

Positive samples are split by ground-truth count:

- `single_target`: exactly one ground-truth box;
- `multi_target`: more than one ground-truth box.

For either slice:

```text
exact_accuracy = samples with all GT matched and no extra predictions / slice samples
```

### Object-size recall

Each ground-truth box is categorized by absolute pixel area:

| Slice | Ground-truth area |
| --- | ---: |
| Small | `< 32²` pixels |
| Medium | `>= 32²` and `<= 96²` pixels |
| Large | `> 96²` pixels |

For each size:

```text
recall = matched GT boxes in the size / all GT boxes in the size
```

These thresholds use the source image's pixels. The benchmark does not normalize object area by image area.

### Count MAE

For every sample:

```text
count error = abs(number of predictions - number of GT boxes)
```

`count_mae` is the mean of those absolute errors over all records.

### Model error rate

```text
model_error_rate = records with a nonempty error field / all records
```

Errors are deliberately penalized. They are not treated as correct empty detections:

- an error on a positive sample adds one FN per ground-truth box and one image-level FN;
- an error on a no-target sample adds one instance FP and one image-level FP.

## Timing and throughput

The benchmark records several different notions of time.

### Model load time

`model_load_seconds` starts immediately before `detector.load_model()` and ends after the first `detector.synchronize()`.

For a local model, it includes loading weights and moving the model to its selected device. For a remote model, it covers client initialization rather than necessarily loading the remote service's weights.

### Detector latency

For each measured sample, `detector_latency_ms` includes:

- the adapter's `predict` call;
- any remote API request performed inside `predict`;
- post-processing performed inside the adapter;
- a post-prediction accelerator synchronization.

It excludes:

- opening and converting the source image;
- the runner's final sanitization pass;
- JSONL serialization and disk writes;
- metric aggregation;
- failure-image rendering.

The runner synchronizes once before starting the timer so unrelated queued accelerator work is less likely to contaminate the measurement.

### Total latency

There is currently no second-stage benchmark verifier or other timed stage, so:

```text
total_latency_ms == detector_latency_ms
```

Both fields remain in the output schema.

### Latency summary

The summary reports:

- mean detector latency;
- mean total latency;
- total-latency p50;
- total-latency p95.

Percentiles use linear interpolation between neighboring sorted samples.

### Inference throughput

```text
throughput = number of records / sum(total_latency_ms converted to seconds)
```

This is inference-only throughput. It is not the same as end-to-end samples per second because it excludes image loading, warmups, output writing, scoring, and visual rendering.

### Benchmark wall time

`benchmark_wall_seconds` starts after model loading and ends after sample processing. Unlike summed detector latency, it includes warmups, image I/O, output writes, and failure rendering. It does not include model loading or final metric/summary writing.

## Running the benchmark

### 1. Prepare data

Place the annotation JSON and matching image files where the configured paths can find them. The defaults are:

```text
/data/RefDrone_train_mdetr.json
/data/VisDrone2019-DET-train/images/
```

### 2. Configure one model

Edit `CONFIG` in [`benchmark_refdrone.py`](benchmark_refdrone.py). For an initial smoke test, use a small `sample_limit` such as `5` or `10`.

### 3. Supply credentials or model files

- Moondream requires its configured API-key environment variable.
- Florence-2 requires a compatible local or downloadable Hugging Face model path.
- LLMDet requires a compatible model snapshot and Transformers environment.

An offline local model directory must contain the complete snapshot needed by `from_pretrained`, not only the weight file.

### 4. Run

```bash
python benchmark_refdrone.py
```

The console prints one progress line per measured sample:

```text
[17/100] 1017 GT=2 predicted=2 latency=83.4 ms
```

This means:

- `17/100`: seventeenth pending sample out of 100;
- `1017`: sample ID;
- `GT=2`: two annotated target boxes;
- `predicted=2`: two sanitized predictions;
- `latency=83.4 ms`: measured detector/total latency.

The progress line does not say whether those two predictions matched the two targets. Use `predictions.jsonl`, the failure visualization, or the final metrics for correctness.

### 5. Compare models

Change `model_name` and run again. Keep these fields constant for a controlled comparison:

- annotation file and images;
- split name;
- sample limit and seed;
- IoU threshold;
- intended confidence-threshold policy;
- device and dtype where applicable;
- hardware load and network conditions.

Confidence scores are not necessarily calibrated the same way across models. A threshold of `0.3` may not impose equivalent strictness on every adapter.

## Output directory and file formats

A new run normally creates:

```text
benchmark_results/
├── summary.csv
└── moondream-train-20260720T180000Z/
    ├── config.json
    ├── predictions.jsonl
    ├── summary.json
    └── failures/
        ├── 0000-1017.jpg
        └── 0001-1042.jpg
```

The default run ID is:

```text
<normalized-model-name>-<split-name>-<UTC timestamp>
```

If `run_name` is set, its sanitized value becomes the run ID. Characters outside letters, digits, `_`, `.`, `+`, and `-` become hyphens.

### `config.json`

This is the serialized configuration used to start the run. Paths become strings. Keys containing `api_key` or `token` are replaced with `<redacted>` recursively.

The broad key-based redaction also affects nonsecret option names containing `token`, such as `max_new_tokens`, in the current implementation. The live in-memory configuration is still used for inference; only the saved representation is redacted.

Example shape:

```json
{
  "annotations_path": "/data/RefDrone_train_mdetr.json",
  "continue_on_error": true,
  "device": "auto",
  "dtype": "auto",
  "images_dir": "/data/VisDrone2019-DET-train/images",
  "iou_threshold": 0.5,
  "max_failure_visuals": 100,
  "model_name": "moondream",
  "model_options": {
    "moondream": {
      "api_key_env": "<redacted>"
    }
  },
  "model_path": null,
  "model_threshold": 0.3,
  "output_dir": "benchmark_results",
  "random_seed": 17,
  "resume_run_dir": null,
  "run_name": null,
  "sample_limit": null,
  "save_failure_visuals": true,
  "split_name": "train",
  "warmup_samples": 3
}
```

### `predictions.jsonl`

JSONL means **JSON Lines**: each nonempty line is a complete JSON object for one sample. It is not one large JSON array.

Example line, formatted here across multiple lines for readability:

```json
{
  "sample_id": "101",
  "image_id": "101",
  "image_path": "/data/VisDrone2019-DET-train/images/000001.jpg",
  "prompt": "the white car near the intersection",
  "gt_boxes": [
    [598.0, 387.0, 778.0, 566.0]
  ],
  "raw_predictions": [
    {
      "bbox_xyxy": [603.0, 390.0, 775.0, 560.0],
      "score": 0.91,
      "label": "the white car near the intersection"
    }
  ],
  "predictions": [
    {
      "bbox_xyxy": [603.0, 390.0, 775.0, 560.0],
      "score": 0.91,
      "label": "the white car near the intersection"
    }
  ],
  "detector_latency_ms": 83.4,
  "total_latency_ms": 83.4,
  "error": null
}
```

Field reference:

| Field | Meaning |
| --- | --- |
| `sample_id` | Unique benchmark-record ID. |
| `image_id` | Original image ID converted to a string. |
| `image_path` | Resolved source-image path. |
| `prompt` | Referring expression sent to the detector. |
| `gt_boxes` | Zero or more ground-truth boxes in absolute `xyxy`. |
| `raw_predictions` | Serialized detector results after the runner's current sanitization path. |
| `predictions` | Predictions used for scoring. |
| `detector_latency_ms` | Timed detector inference in milliseconds. |
| `total_latency_ms` | Total timed inference stages; currently equal to detector latency. |
| `error` | `null` on success or `"ExceptionType: message"` on failure. |

In the current benchmark, `raw_predictions` and `predictions` contain the same sanitized list. Both fields are retained in the schema, but there is no second-stage benchmark filter separating them.

JSONL is useful for long or interrupted runs because each completed sample is flushed immediately. A partially completed run can retain valid earlier lines and later be resumed.

### `summary.json`

This contains run metadata and all aggregate metrics.

Example structure:

```json
{
  "run_id": "moondream-train-20260720T180000Z",
  "model": "moondream",
  "split": "train",
  "created_at": "2026-07-20T18:10:42.123456+00:00",
  "annotations_path": "/data/RefDrone_train_mdetr.json",
  "images_dir": "/data/VisDrone2019-DET-train/images",
  "model_load_seconds": 0.18,
  "benchmark_wall_seconds": 118.7,
  "failure_visuals": 36,
  "metrics": {
    "iou_threshold": 0.5,
    "samples": 100,
    "instance": {
      "tp": 72,
      "tn": 3,
      "fp": 18,
      "fn": 21,
      "f1": 0.7868852459,
      "accuracy": 0.6578947368
    },
    "image": {
      "tp": 61,
      "tn": 3,
      "fp": 2,
      "fn": 34,
      "f1": 0.7721518987,
      "accuracy": 0.64
    },
    "slices": {
      "no_target": {
        "samples": 5,
        "false_positive_rate": 0.4
      },
      "single_target": {
        "samples": 70,
        "exact_accuracy": 0.7
      },
      "multi_target": {
        "samples": 25,
        "exact_accuracy": 0.48
      },
      "object_size_recall": {
        "small": 0.51,
        "medium": 0.77,
        "large": 0.89
      },
      "object_size_counts": {
        "small": 35,
        "medium": 42,
        "large": 16
      }
    },
    "count_mae": 0.39,
    "model_errors": 0,
    "model_error_rate": 0.0,
    "latency_ms": {
      "detector_mean": 83.1,
      "total_mean": 83.1,
      "total_p50": 80.4,
      "total_p95": 101.6
    },
    "inference_throughput_samples_per_second": 12.03
  }
}
```

The numbers above illustrate the schema; they are not claimed benchmark results.

### `summary.csv`

The CSV stores one compact row per run so results can be sorted or plotted without opening every `summary.json`.

Columns are:

```text
run_id
model
split
samples
instance_f1
instance_accuracy
image_f1
image_accuracy
no_target_false_positive_rate
count_mae
model_error_rate
detector_latency_mean_ms
total_latency_p50_ms
total_latency_p95_ms
throughput_samples_per_second
```

Writing a summary with an existing `run_id` replaces that run's previous CSV row instead of duplicating it.

## Failure visualizations

A sample is considered a failure when:

- inference produced an error;
- a no-target sample received one or more predictions;
- a positive sample has an unmatched ground-truth box;
- a positive sample has an unmatched extra prediction.

When enabled, each failure JPEG contains:

- **green boxes** labeled `GT 0`, `GT 1`, and so on for ground truth;
- **red boxes** labeled `P 0`, `P 1`, and so on for predictions;
- a prediction score when one is available;
- a header with the model, latency, matched pairs and IoUs, error text, and prompt.

Only failures are rendered. Correct samples do not receive images. Rendering stops after `max_failure_visuals`, but every sample is still scored and written to JSONL.

## Warmup behavior

Warmup uses the first `warmup_samples` entries among the **pending** samples after resume filtering.

For each warmup sample, the benchmark opens the image, calls the same detector `predict` path, and synchronizes. Warmup calls:

- use the same already-loaded detector as measured inference;
- are not written to `predictions.jsonl`;
- are not included in per-sample detector latency;
- are included in `benchmark_wall_seconds`;
- do not require or load a second model.

A warmup failure prints a message and does not stop the run, even if `continue_on_error` is `False`. Because measured inference uses the same prediction path, a repeated warmup error usually indicates a real model, processor, input, device, or environment problem—not merely a warmup problem.

Set `warmup_samples=0` to skip warmup calls. This can shorten a smoke test, but it does not fix an inference-path error.

## Error handling

### `continue_on_error=True`

This is the default. A measured inference exception is converted to:

```json
{
  "predictions": [],
  "error": "RuntimeError: explanation from the adapter"
}
```

The record is flushed, the error is penalized during scoring, and the benchmark continues to the next sample.

### `continue_on_error=False`

The first measured inference exception is re-raised. The detector is still closed through the runner's `finally` block, and already-flushed JSONL lines remain on disk.

### Why the `error` field matters

The console may show `predicted=0` for both a legitimate empty detection and an inference failure. Check the JSONL record:

- `error: null` and no predictions means the detector completed and returned no accepted boxes;
- a non-null `error` means inference failed and the zero predictions should not be interpreted as model rejection behavior;
- nonempty predictions with poor metrics means the model ran but localized the wrong objects, returned extra objects, or missed the IoU threshold.

## Resuming a run

Set `resume_run_dir` to an existing run directory:

```python
CONFIG = BenchmarkConfig(
  # Keep the original dataset and model configuration.
  resume_run_dir=Path("benchmark_results/moondream-train-20260720T180000Z"),
)
```

Resume behavior is:

1. Load existing lines from `predictions.jsonl`.
2. Reject duplicate `sample_id` values.
3. Confirm existing IDs belong to the currently requested sample set.
4. Compare selected saved configuration fields.
5. Skip IDs already present.
6. Warm up on the first remaining pending samples.
7. Append new records.
8. Require the final ID set to exactly match the requested sample set.
9. Recompute metrics from all JSONL records.

The automatic configuration comparison checks:

- `annotations_path`;
- `images_dir`;
- `model_name`;
- `model_threshold`;
- `iou_threshold`.

It does not compare every behavior-affecting field, including all model options, device, dtype, sample seed, or model files behind an unchanged path. Therefore, resume only with the same effective configuration and environment. If model behavior or adapter code changed after a failed run, starting a fresh run is safer than mixing old and new predictions.

The JSONL loader rejects malformed JSON lines and duplicate sample IDs rather than silently scoring ambiguous data.

## How to interpret common outcomes

### High instance F1 but lower image F1

The model often finds individual objects, but many images contain at least one missed target or extra prediction. Image scoring requires exact per-image output.

### Good single-target accuracy but weak multi-target accuracy

The model can identify one described object but struggles to enumerate all objects matching a plural or shared expression.

### High recall with poor count MAE

The model finds many true targets but also returns extra boxes or duplicate detections.

### Low no-target false-positive rate

The model usually stays silent when the target is absent. Lower is better for this metric.

### Weak small-object recall

The model has difficulty localizing targets below `32²` pixels in the original image, a common challenge in aerial imagery.

### `predicted=0` on every sample

Inspect several JSONL lines before concluding the model simply found nothing:

1. Check `error` for processor, model, CUDA, path, or API failures.
2. Check that the prompt is populated.
3. Confirm model files and credentials are available.
4. Try a very small run and, for diagnosis, a lower confidence threshold.
5. Verify that the model adapter's Transformers version matches the installed model implementation.

### High throughput but long wall time

The throughput metric uses only summed detector latency. Image loading, warmup, file writing, and failure rendering can make wall time significantly longer.

## Adding another model

To benchmark another detector:

1. Create a subclass of `DetectorModel` in [`benchmark_models.py`](benchmark_models.py).
2. Implement `load_model()` and `predict(image, prompt)`.
3. Return `list[Detection]` using absolute original-image `xyxy` coordinates.
4. Implement `synchronize()` if inference can be asynchronous on an accelerator.
5. Implement `close()` if resources need cleanup.
6. Add the class to `MODEL_REGISTRY`.
7. Add default settings under `BenchmarkConfig.model_options`.
8. Run a small sample with failure visuals enabled before a full benchmark.

Minimal adapter shape:

```python
class MyDetector(DetectorModel):
  model_name = "my_detector"

  def load_model(self) -> None:
    self.model = load_my_model(self.options)

  def predict(self, image: Image, prompt: str) -> list[Detection]:
    result = self.model.predict(image, prompt)
    return [
      Detection(
        bbox_xyxy=(result.x1, result.y1, result.x2, result.y2),
        score=result.score,
        label=result.label,
      )
    ]


MODEL_REGISTRY["my_detector"] = MyDetector
```

Do not return normalized coordinates unless the adapter converts them to absolute pixels first. Do not return `xywh`. The runner assumes every adapter follows the `Detection` contract.

## Testing the benchmark implementation

There is currently no committed automated test suite in this repository. Model benchmarking and benchmark-code validation are separate tasks: a model run measures detector quality, while the checks below help establish that the loader, scorer, and output files behave as expected.

### Syntax check

This catches Python syntax errors without loading model weights:

```bash
python -m py_compile benchmark_models.py benchmark_refdrone.py
```

### Dataset-loader smoke test

Use the real annotation and image paths but stop before model loading:

```bash
python - <<'PY'
from pathlib import Path

from benchmark_refdrone import BenchmarkConfig, load_refdrone_samples

config = BenchmarkConfig(
  annotations_path=Path("/data/RefDrone_val_mdetr.json"),
  images_dir=Path("/data/VisDrone2019-DET-val/images"),
  sample_limit=5,
)
samples = load_refdrone_samples(config)

for sample in samples:
  print(sample.sample_id, sample.prompt, sample.gt_boxes, sample.image_path)
PY
```

This checks JSON structure, prompt extraction, `xywh` conversion, deterministic sampling, and image-path resolution. It does not call a detector.

### Synthetic scoring sanity check

The following creates one exact positive sample and one correct no-target sample:

```bash
python - <<'PY'
from pprint import pprint

from benchmark_refdrone import score_records

records = [
  {
    "sample_id": "positive",
    "gt_boxes": [[0, 0, 10, 10]],
    "predictions": [
      {"bbox_xyxy": [0, 0, 10, 10], "score": 1.0, "label": "target"}
    ],
    "detector_latency_ms": 10.0,
    "total_latency_ms": 10.0,
    "error": None,
  },
  {
    "sample_id": "no-target",
    "gt_boxes": [],
    "predictions": [],
    "detector_latency_ms": 10.0,
    "total_latency_ms": 10.0,
    "error": None,
  },
]

pprint(score_records(records, iou_threshold=0.5))
PY
```

Expected core results are one instance/image `TP`, one instance/image `TN`, zero `FP`, zero `FN`, F1 `1.0`, accuracy `1.0`, single-target exact accuracy `1.0`, small-object recall `1.0`, and count MAE `0.0`.

### Small end-to-end model test

Before a full split, configure:

```python
sample_limit=5
warmup_samples=1
continue_on_error=True
save_failure_visuals=True
```

Then run the normal command and inspect all five JSONL records. Confirm that:

- every record has the expected prompt and ground truth;
- `error` is `null`;
- predicted boxes use plausible image-pixel coordinates;
- latency is nonnegative;
- any saved failure JPEG agrees with the JSONL record;
- `summary.json` reports exactly five samples.

### Output integrity checks

Useful lightweight checks after a run are:

```bash
python -m json.tool benchmark_results/<run-id>/config.json >/dev/null
python -m json.tool benchmark_results/<run-id>/summary.json >/dev/null
```

Because `predictions.jsonl` contains one JSON value per line rather than one JSON document, validate it line by line:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("benchmark_results/<run-id>/predictions.jsonl")

with path.open(encoding="utf-8") as handle:
  records = [json.loads(line) for line in handle if line.strip()]

sample_ids = [str(record["sample_id"]) for record in records]
assert len(sample_ids) == len(set(sample_ids)), "duplicate sample IDs"
print(f"valid JSONL: {len(records)} unique records")
PY
```

These checks validate serialization and unique IDs. They do not establish detector accuracy; that comes from inspecting the annotations, predictions, matches, and aggregate metrics together.

## Validation checklist

Before trusting a large comparison, verify the following:

- [ ] Annotation JSON contains `images` and `annotations` lists.
- [ ] Image IDs in annotations correspond to image records.
- [ ] Annotation boxes are `xywh`, not already `xyxy`.
- [ ] `empty: true` is used for no-target sentinel annotations.
- [ ] Every selected image exists under `images_dir`.
- [ ] The model receives the intended free-form prompt.
- [ ] Model paths, credentials, PyTorch, Transformers, and device support are available.
- [ ] A small run completes with `model_error_rate == 0`.
- [ ] Failure JPEGs show green ground truth and red predictions in plausible locations.
- [ ] Compared runs use the same sample IDs and IoU threshold.
- [ ] Confidence-threshold differences are intentional and documented.
- [ ] Latency comparisons use comparable hardware and network conditions.
- [ ] JSONL `error` fields were inspected before treating empty predictions as valid model behavior.
- [ ] A changed adapter or model starts a fresh run instead of resuming incompatible predictions.

The most useful files for analysis are `summary.csv` for cross-run comparison, `summary.json` for full metrics, `predictions.jsonl` for sample-level diagnosis, and `failures/` for quickly seeing what went wrong.
