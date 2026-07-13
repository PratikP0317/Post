#!/usr/bin/env python3
"""Small OpenCV fallback used when ffmpeg is unavailable on the host."""

from __future__ import annotations

import sys

import cv2


def main() -> int:
    source, output, start_arg, duration_arg, width_arg = sys.argv[1:]
    start, duration, output_width = float(start_arg), float(duration_arg), int(width_arg)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_height = round(source_height * output_width / source_width / 2) * 2
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height))
    capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    frames = round(duration * fps)
    try:
        for _ in range(frames):
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
        writer.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
