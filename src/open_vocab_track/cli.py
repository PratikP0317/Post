from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_vocab_track.detectors import create_detector


def _common_detector(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--detector", required=True, choices=["moondream", "locate-anything", "florence2"])
    parser.add_argument("--model-id", help="Override the detector's Hugging Face model ID")
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    parser.add_argument("--default-score", type=float, default=0.9, help="Score assigned to generative boxes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ovtrack", description="Open-vocabulary VLM detection + BoxMOT")
    commands = parser.add_subparsers(dest="command", required=True)
    track = commands.add_parser("track", help="Detect and track a referring expression in video")
    _common_detector(track)
    track.add_argument("--tracker", default="botsort", choices=["bytetrack", "ocsort", "botsort"])
    track.add_argument("--source", required=True)
    track.add_argument("--output", required=True)
    track.add_argument("--prompt", default="person wearing a green shirt")
    track.add_argument("--detect-every", type=int, default=1)
    track.add_argument("--max-inference-side", type=int, default=1280)
    track.add_argument("--max-frames", type=int, default=0)
    track.add_argument("--metadata")
    track.add_argument("--summary")

    benchmark = commands.add_parser("refdrone", help="Evaluate phrase grounding on RefDrone")
    _common_detector(benchmark)
    benchmark.add_argument("--annotations", required=True)
    benchmark.add_argument("--images", required=True)
    benchmark.add_argument("--limit", type=int, default=0)
    benchmark.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    detector_kwargs = {"device": args.device, "default_score": args.default_score}
    if args.model_id:
        detector_kwargs["model_id"] = args.model_id
    detector = create_detector(args.detector, **detector_kwargs)
    if args.command == "track":
        import cv2
        from open_vocab_track.pipeline import run_video, write_summary
        from open_vocab_track.tracking import create_tracker

        capture = cv2.VideoCapture(args.source)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        capture.release()
        tracker = create_tracker(args.tracker, fps)
        summary = run_video(
            args.source,
            args.output,
            detector,
            tracker,
            args.prompt,
            args.detect_every,
            args.max_inference_side,
            args.max_frames,
            args.metadata,
        )
        if args.summary:
            write_summary(summary, args.summary)
        print(json.dumps({**summary.__dict__, "processed_fps": summary.processed_fps}, indent=2))
    else:
        from open_vocab_track.benchmark import run_refdrone

        metrics = run_refdrone(args.annotations, args.images, detector, args.limit)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
