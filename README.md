# Open-vocabulary tracking on Jetson AGX

This project grounds a free-form description in video frames with one of three vision-language models, converts the boxes to BoxMOT's standard detection format, and assigns persistent track IDs with ByteTrack, OC-SORT, or BoT-SORT.

Implemented detector backends:

| Backend | Default checkpoint | Grounding operation |
|---|---|---|
| Moondream | `vikhyatk/moondream2` revision `2025-06-21` | native `detect(image, prompt)` |
| LocateAnything | `nvidia/LocateAnything-3B` | multi-instance phrase grounding, hybrid decoding |
| Florence-2 | `microsoft/Florence-2-base-ft` | caption-to-phrase grounding |

The default prompt is `person wearing a green shirt`. For high-altitude footage, simpler prompts such as `person wearing green clothing` often work better because shirt details may occupy only a few pixels.

## Jetson AGX Orin quick start

The container targets JetPack 6.2-or-newer AGX Orin systems. The PyTorch image must match the JetPack/L4T release installed on the host; override `JETSON_PYTORCH_IMAGE` if NVIDIA's compatibility table specifies another tag.

```bash
./scripts/download_drone_sample.sh data

docker compose -f compose.jetson.yaml build
docker compose -f compose.jetson.yaml run --rm ovtrack track \
  --detector moondream \
  --tracker botsort \
  --source /data/drone_people_sample.mp4 \
  --output /outputs/moondream-botsort.mp4 \
  --metadata /outputs/moondream-botsort.jsonl \
  --summary /outputs/moondream-botsort-summary.json \
  --prompt "person wearing green clothing"
```

Run all three detectors against the same clip:

```bash
docker compose -f compose.jetson.yaml run --rm --entrypoint bash ovtrack \
  /workspace/scripts/run_model_matrix.sh /data/drone_people_sample.mp4 \
  "person wearing green clothing" botsort
```

The model matrix is intentionally sequential. Loading all three checkpoints together needlessly increases unified-memory pressure. Model downloads persist in the mounted Hugging Face cache.

BoxMOT is pinned to 15.0.10. Newer releases changed their dependency floor to NumPy 2.2 and Hugging Face Hub 1.x, which conflicts with LocateAnything's required Transformers 4.57.1 environment. The pinned release supplies the same `Nx6` detection and `Mx8` track contract used here.

## Local development and smoke tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Install the model extras only on the inference machine:

```bash
pip install -e '.[models,dev]'
```

## Detector and tracker experiments

Any detector can be paired with any included tracker:

```bash
ovtrack track \
  --detector locate-anything \
  --tracker ocsort \
  --source data/drone_people_sample.mp4 \
  --output outputs/locate-ocsort.mp4 \
  --prompt "person wearing a green shirt" \
  --max-inference-side 960 \
  --detect-every 1
```

Useful tradeoffs:

- `--max-inference-side 960` reduces LocateAnything memory and latency. Increase it for tiny aerial subjects if memory permits.
- `--detect-every N` invokes the expensive VLM every Nth frame. BoxMOT receives empty detections on intervening frames and advances its motion model.
- BoT-SORT enables camera-motion compensation but disables its separate ReID model. This keeps the comparison focused on the three requested models and works better within Jetson memory constraints.
- ByteTrack is the lightest baseline. OC-SORT is a useful motion-only comparison for moving-camera footage.

Generative VLM grounding APIs do not expose calibrated per-box probabilities. The adapters therefore assign `--default-score 0.9` to every returned box. Use this score only to satisfy BoxMOT's association contract; do not interpret it as detector confidence.

## RefDrone grounding benchmark

RefDrone contains referring expressions and boxes for static VisDrone images. It can measure whether a model locates phrases such as “the person in green,” but it cannot measure temporal ID switches because it is not a video tracking annotation set.

1. Download RefDrone annotations with `./scripts/download_refdrone_annotations.sh`, or from the [official Hugging Face dataset](https://huggingface.co/datasets/sunzc-sunny/RefDrone).
2. Download the corresponding images from the [official VisDrone dataset](https://github.com/VisDrone/VisDrone-Dataset).
3. Run each detector (start with `--limit 25` on Jetson):

```bash
ovtrack refdrone \
  --detector florence2 \
  --annotations data/RefDrone/RefDrone_val_mdetr.json \
  --images data/RefDrone/all_image \
  --limit 25 \
  --output outputs/refdrone-florence2.json
```

The evaluator reports one-to-one box precision, recall, and F1 at IoU 0.50 and 0.75, plus per-image detector latency. For temporal tracking metrics (HOTA, IDF1, MOTA), use VisDrone-MOT sequences and their official annotations. A color-specific prompt cannot be scored directly against ordinary VisDrone-MOT labels because those labels identify object classes and track IDs, not clothing descriptions.

## Outputs

- Annotated MP4 with persistent IDs.
- Optional JSONL, one record per frame with BoxMOT rows in `[x1, y1, x2, y2, id, confidence, class, detection_index]` order.
- Optional summary JSON with detector time, wall time, counts, and processed FPS.

## Hardware and licensing notes

- LocateAnything's official implementation uses Transformers 4.57.1 and standard PyTorch SDPA on Jetson. Its optional MagiAttention path is for Hopper/Blackwell, not AGX Orin. NVIDIA reports substantial SDPA memory use at 4K input, so start at 960 or 1280 pixels.
- LocateAnything-3B is released for research/non-commercial use under NVIDIA's model terms. Review those terms before deployment.
- BoxMOT is AGPL-3.0. Florence-2 and Moondream have their own model licenses. This repository does not redistribute weights.
- The sample downloader fetches a CC BY 3.0 Wikimedia Commons drone clip and writes its attribution beside the video.

## Primary references

- [BoxMOT repository and Python detection contract](https://github.com/mikel-brostrom/boxmot)
- [Moondream 2 model card](https://huggingface.co/vikhyatk/moondream2)
- [LocateAnything official implementation](https://github.com/NVlabs/Eagle/tree/main/Embodied)
- [Florence-2 model card](https://huggingface.co/microsoft/Florence-2-base-ft)
- [RefDrone repository](https://github.com/sunzc-sunny/refdrone)
- [NVIDIA PyTorch for Jetson compatibility table](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)
