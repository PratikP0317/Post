#!/usr/bin/env bash
set -euo pipefail

source_video="${1:-/data/drone_people_sample.mp4}"
prompt="${2:-person wearing green clothing}"
tracker="${3:-botsort}"
mkdir -p /outputs

for detector in moondream locate-anything florence2; do
  ovtrack track \
    --detector "${detector}" \
    --tracker "${tracker}" \
    --source "${source_video}" \
    --output "/outputs/${detector}-${tracker}.mp4" \
    --metadata "/outputs/${detector}-${tracker}.jsonl" \
    --summary "/outputs/${detector}-${tracker}-summary.json" \
    --prompt "${prompt}"
done

