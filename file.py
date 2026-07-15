import os
import cv2
import time
import torch
import moondream as md

from flask import Flask, Response, jsonify, render_template_string
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


app = Flask(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Choose either:
# "moondream"
# "florence"
DETECTOR = "moondream"

DETECTION_PROMPT = "person"
DETECTION_THRESHOLD = 0.3
TRACKER_SCORE_THRESHOLD = 0.1
TRACK_ASSOCIATION_IOU_THRESHOLD = 0.1
VIDEO_PATH = "/MAVIK_dataset/DJI_0042.MP4"
INTERVAL_N = 30

TRACKER_MODEL_PATH = "/app/src/moondream/object_tracking_vittrack_2023sep.onnx"


# ---------------------------------------------------------------------------
# Moondream setup
# ---------------------------------------------------------------------------

moondream_model = None

if DETECTOR == "moondream":
  api_key = os.environ.get("MOONDREAM_API_KEY")

  if not api_key:
    raise RuntimeError("MOONDREAM_API_KEY is not set")

  moondream_model = md.vl(api_key=api_key)


# ---------------------------------------------------------------------------
# Florence-2 setup
# ---------------------------------------------------------------------------

florence_model = None
florence_processor = None

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
FLORENCE_MODEL_NAME = "/Models/Florence-2-large"

if DETECTOR == "florence":
  florence_processor = AutoProcessor.from_pretrained(FLORENCE_MODEL_NAME, trust_remote_code=True)

  florence_model = AutoModelForCausalLM.from_pretrained(
    FLORENCE_MODEL_NAME, torch_dtype=TORCH_DTYPE, trust_remote_code=True
  ).to(DEVICE)

  florence_model.eval()


if DETECTOR not in ["moondream", "florence"]:
  raise ValueError(f"Unsupported detector: {DETECTOR}")


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

analytics = {
  "fps": 0.0, "inference_time_ms": 0, "detected_items": 0, "frame_count": 0, "detector": DETECTOR,
  "detector_confidence_available": None, "detector_confidence": None,
  "detector_threshold": DETECTION_THRESHOLD, "tracker_confidence": None,
  "tracker_threshold": TRACKER_SCORE_THRESHOLD, "tracker_status": "Waiting"
}


# ---------------------------------------------------------------------------
# Video setup
# ---------------------------------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
  raise RuntimeError(f"Could not open video: {VIDEO_PATH}")


# ---------------------------------------------------------------------------
# TrackerVit setup
# ---------------------------------------------------------------------------

def create_tracker():
  params = cv2.TrackerVit_Params()

  params.net = TRACKER_MODEL_PATH
  params.tracking_score_threshold = TRACKER_SCORE_THRESHOLD

  return cv2.TrackerVit.create(params)


# ---------------------------------------------------------------------------
# Moondream detection
# ---------------------------------------------------------------------------

def moondream_detect(frame):
  height, width = frame.shape[:2]

  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  pil_image = Image.fromarray(rgb_frame)

  result = moondream_model.detect(image=pil_image, object=DETECTION_PROMPT)

  detections = []

  for obj in result.get("objects", []):
    x1 = int(obj["x_min"] * width)
    y1 = int(obj["y_min"] * height)
    x2 = int(obj["x_max"] * width)
    y2 = int(obj["y_max"] * height)
    score = obj.get("score", obj.get("confidence"))
    score = float(score) if score is not None else None

    if score is None or score >= DETECTION_THRESHOLD:
      detections.append((x1, y1, x2, y2, DETECTION_PROMPT, score))

  return detections


# ---------------------------------------------------------------------------
# Florence-2 detection
# ---------------------------------------------------------------------------

def florence_detect(frame):
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  pil_image = Image.fromarray(rgb_frame)

  task_prompt = "<OPEN_VOCABULARY_DETECTION>"
  prompt = task_prompt + DETECTION_PROMPT

  inputs = florence_processor(text=prompt, images=pil_image, return_tensors="pt")

  input_ids = inputs["input_ids"].to(DEVICE)
  pixel_values = inputs["pixel_values"].to(DEVICE, dtype=TORCH_DTYPE)

  with torch.inference_mode():
    generated_ids = florence_model.generate(
      input_ids=input_ids,
      pixel_values=pixel_values,
      max_new_tokens=256,
      do_sample=False,
      num_beams=3
    )

  generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

  parsed_result = florence_processor.post_process_generation(
    generated_text, task=task_prompt, image_size=pil_image.size
  )

  result = parsed_result.get(task_prompt, {})

  boxes = result.get("bboxes", [])
  labels = result.get("labels", [])
  scores = result.get("scores", [])

  detections = []

  for index, box in enumerate(boxes):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    label = DETECTION_PROMPT
    score = float(scores[index]) if index < len(scores) else None

    if index < len(labels):
      label = labels[index]

    if score is None or score >= DETECTION_THRESHOLD:
      detections.append((x1, y1, x2, y2, label, score))

  return detections


# ---------------------------------------------------------------------------
# Selected detector
# ---------------------------------------------------------------------------

def run_detection(frame):
  if DETECTOR == "moondream":
    return moondream_detect(frame)

  if DETECTOR == "florence":
    return florence_detect(frame)

  return []


# ---------------------------------------------------------------------------
# Coordinate validation
# ---------------------------------------------------------------------------

def clamp_bbox(bbox, frame_width, frame_height):
  x1, y1, x2, y2 = bbox

  x1 = max(0, min(x1, frame_width - 1))
  y1 = max(0, min(y1, frame_height - 1))
  x2 = max(0, min(x2, frame_width - 1))
  y2 = max(0, min(y2, frame_height - 1))

  return x1, y1, x2, y2


def bbox_iou(first, second):
  x1 = max(first[0], second[0])
  y1 = max(first[1], second[1])
  x2 = min(first[2], second[2])
  y2 = min(first[3], second[3])
  intersection = max(0, x2 - x1) * max(0, y2 - y1)

  first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
  second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
  union = first_area + second_area - intersection

  return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

def generate():
  global analytics

  local_frame_count = 0
  tracker_initialized = False
  object_tracker = None
  last_tracker_bbox = None
  secondary_detections = []
  item_count = 0

  while True:
    start_time = time.time()

    success, frame = cap.read()

    if not success:
      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

      local_frame_count = 0
      tracker_initialized = False
      object_tracker = None
      last_tracker_bbox = None
      secondary_detections = []
      item_count = 0

      continue

    frame = cv2.resize(frame, (854, 480), interpolation=cv2.INTER_AREA)
    frame_height, frame_width = frame.shape[:2]

    annotated = frame.copy()

    # Only trigger detector every N frames
    run_vlm_detection = local_frame_count % INTERVAL_N == 0

    if run_vlm_detection:
      inference_start = time.time()

      try:
        detections = run_detection(frame)

        valid_detections = []

        for x1, y1, x2, y2, label, score in detections:
          x1, y1, x2, y2 = clamp_bbox((x1, y1, x2, y2), frame_width, frame_height)

          if x2 > x1 and y2 > y1:
            valid_detections.append((x1, y1, x2, y2, label, score))

        if valid_detections:
          analytics["detector_confidence_available"] = any(detection[5] is not None for detection in valid_detections)
          scored_detections = [(index, detection) for index, detection in enumerate(valid_detections) if detection[5] is not None]
          primary_index = None

          if tracker_initialized and last_tracker_bbox is not None:
            overlaps = [bbox_iou(last_tracker_bbox, detection[:4]) for detection in valid_detections]
            best_overlap = max(overlaps)

            if best_overlap >= TRACK_ASSOCIATION_IOU_THRESHOLD:
              primary_index = overlaps.index(best_overlap)
          else:
            primary_index = max(scored_detections, key=lambda item: item[1][5])[0] if scored_detections else 0

          secondary_detections = [detection for index, detection in enumerate(valid_detections) if index != primary_index]

          # Draw every non-primary detection in gray.
          for x1, y1, x2, y2, label, score in secondary_detections:
            score_text = f" {score:.2f}" if score is not None else ""
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (140, 140, 140), 2)
            cv2.putText(annotated, f"{label}{score_text}", (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

          item_count = len(valid_detections)

          if primary_index is not None:
            # Match the current target by overlap. Use confidence/first result only for initial acquisition.
            x1, y1, x2, y2, label, score = valid_detections[primary_index]
            tracker_bbox = (x1, y1, x2 - x1, y2 - y1)
            object_tracker = create_tracker()
            object_tracker.init(frame, tracker_bbox)
            tracker_initialized = True
            last_tracker_bbox = (x1, y1, x2, y2)
            analytics["detector_confidence"] = score
            analytics["tracker_status"] = "Tracking (score not exposed)"

            score_text = f" {score:.2f}" if score is not None else ""
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(annotated, f"{DETECTOR}: {label}{score_text}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
          else:
            analytics["detector_confidence"] = None
            analytics["tracker_status"] = "No matching detection; keeping tracker"

        else:
          item_count = 0
          secondary_detections = []
          analytics["detector_confidence"] = None

          # Keep the existing tracker active if
          # the detector temporarily misses.

        analytics["inference_time_ms"] = int((time.time() - inference_start) * 1000)

      except Exception as error:
        print(f"{DETECTOR} inference error: {error}")

        analytics["inference_time_ms"] = 0

    # -----------------------------------------------------------------------
    # TrackerVit step
    # -----------------------------------------------------------------------

    if not run_vlm_detection:
      # These are the most recent detector boxes; they are not independently tracked.
      for x1, y1, x2, y2, label, score in secondary_detections:
        score_text = f" {score:.2f}" if score is not None else ""
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (100, 100, 100), 1)
        cv2.putText(annotated, f"last: {label}{score_text}", (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    if tracker_initialized and object_tracker is not None and not run_vlm_detection:
      tracking_success, bbox = object_tracker.update(frame)

      if tracking_success:
        x, y, width, height = [int(value) for value in bbox]

        last_tracker_bbox = (x, y, x + width, y + height)
        item_count = 1 + len(secondary_detections)
        analytics["tracker_status"] = "Tracking (score not exposed)"

        # Blue box for TrackerVit frames
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (255, 0, 0), 2)

        cv2.putText(annotated, f"TrackerVit (Frame +{local_frame_count % INTERVAL_N})", (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

      else:
        cv2.putText(annotated, "Tracker Lost Target", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        tracker_initialized = False
        object_tracker = None
        last_tracker_bbox = None
        item_count = 0
        analytics["tracker_status"] = "Lost"

    # Increment counters
    local_frame_count += 1
    analytics["frame_count"] += 1

    total_frame_time = time.time() - start_time

    if total_frame_time > 0:
      analytics["fps"] = round(1.0 / total_frame_time, 1)
    else:
      analytics["fps"] = 0.0

    analytics["detected_items"] = item_count

    encode_success, buffer = cv2.imencode(".jpg", annotated)

    if not encode_success:
      continue

    frame_bytes = buffer.tobytes()

    yield (
      b"--frame\r\n"
      b"Content-Type: image/jpeg\r\n\r\n"
      + frame_bytes
      + b"\r\n"
    )


# ---------------------------------------------------------------------------
# Flask webpage
# ---------------------------------------------------------------------------

@app.route("/")
def index():
  html_page = """
  <!DOCTYPE html>
  <html lang="en">

  <head>
    <title>Open Vocabulary Detection + TrackerVit</title>

    <style>
      body {
        font-family:
          -apple-system,
          BlinkMacSystemFont,
          "Segoe UI",
          Roboto,
          sans-serif;

        background: #0f172a;
        color: #f8fafc;
        margin: 0;
        padding: 20px;
      }

      .container {
        max-width: 1200px;
        margin: auto;
        display: grid;
        grid-template-columns: 2fr 1fr;
        gap: 20px;
      }

      header {
        grid-column: span 2;
        border-bottom: 1px solid #334155;
        padding-bottom: 10px;
        margin-bottom: 10px;
      }

      .video-box {
        background: #020617;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #334155;
        display: flex;
        justify-content: center;
      }

      .video-box img {
        width: 100%;
        height: auto;
        max-height: 70vh;
        object-fit: contain;
      }

      .analytics-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #334155;
      }

      .metric {
        margin-bottom: 25px;
        padding-bottom: 15px;
        border-bottom: 1px solid #334155;
      }

      .metric:last-child {
        border: none;
      }

      .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.05em;
        margin-bottom: 5px;
      }

      .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #38bdf8;
      }
    </style>
  </head>

  <body>
    <div class="container">
      <header>
        <h1>Open Vocabulary Detection + TrackerVit</h1>

        <p>
          Detector: <span style="color:#38bdf8;">{{ detector }}</span> |
          Prompt: <span style="color:#fbbf24;">{{ detection_prompt }}</span> |
          Status: <span style="color:#4ade80;">Active</span>
        </p>
      </header>

      <div class="video-box">
        <img src="/video" alt="Processed Video Feed">
      </div>

      <div class="analytics-card">
        <div class="metric">
          <div class="metric-title">Performance Speed</div>

          <div class="metric-value">
            <span id="fps">0.0</span>
            <span style="font-size:1rem; font-weight:normal; color:#64748b;">FPS</span>
          </div>
        </div>

        <div class="metric">
          <div class="metric-title">Detector Latency</div>

          <div class="metric-value">
            <span id="latency">0</span>
            <span style="font-size:1rem; font-weight:normal; color:#64748b;">ms</span>
          </div>
        </div>

        <div class="metric">
          <div class="metric-title">Detector Confidence</div>
          <div class="metric-value" id="detector-confidence" style="font-size:1.3rem;">Waiting</div>
          <div id="detector-confidence-note" style="font-size:0.75rem; color:#94a3b8; margin-top:5px;"></div>
        </div>

        <div class="metric">
          <div class="metric-title">Tracker Confidence</div>
          <div class="metric-value" id="tracker-confidence" style="font-size:1.3rem;">N/A</div>
          <div id="tracker-confidence-note" style="font-size:0.75rem; color:#94a3b8; margin-top:5px;">Waiting</div>
        </div>

        <div class="metric">
          <div class="metric-title">Active Targets Present</div>
          <div class="metric-value" id="targets" style="color:#fbbf24;">0</div>
        </div>

        <div class="metric">
          <div class="metric-title">Frames Processed</div>
          <div class="metric-value" id="frames">0</div>
        </div>
      </div>
    </div>

    <script>
      setInterval(async () => {
        try {
          const response = await fetch("/stats");

          const data = await response.json();

          document.getElementById("fps").innerText = data.fps;
          document.getElementById("latency").innerText = data.inference_time_ms;
          document.getElementById("targets").innerText = data.detected_items;
          document.getElementById("frames").innerText = data.frame_count;

          const detectorConfidence = document.getElementById("detector-confidence");
          const detectorNote = document.getElementById("detector-confidence-note");

          if (data.detector_confidence_available === null) {
            detectorConfidence.innerText = "Waiting";
            detectorNote.innerText = `Configured threshold: ${data.detector_threshold.toFixed(2)}`;
          } else if (!data.detector_confidence_available) {
            detectorConfidence.innerText = "Not provided";
            detectorNote.innerText = "This model output has no confidence scores";
          } else {
            detectorConfidence.innerText = data.detector_confidence === null ? "Available" : data.detector_confidence.toFixed(2);
            detectorNote.innerText = `Configured threshold: ${data.detector_threshold.toFixed(2)}`;
          }

          document.getElementById("tracker-confidence").innerText = data.tracker_confidence === null ? "Not exposed" : data.tracker_confidence.toFixed(2);
          document.getElementById("tracker-confidence-note").innerText = `${data.tracker_status}; internal threshold: ${data.tracker_threshold.toFixed(2)}`;

        } catch (error) {
          console.error("Telemetry error:", error);
        }
      }, 150);
    </script>
  </body>

  </html>
  """

  return render_template_string(html_page, detector=DETECTOR, detection_prompt=DETECTION_PROMPT)


@app.route("/stats")
def stats():
  return jsonify(analytics)


@app.route("/video")
def video():
  return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=1516, debug=False)
