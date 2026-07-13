#!/usr/bin/env bash
set -euo pipefail

root="${1:-data}"
mkdir -p "${root}"
url='https://commons.wikimedia.org/wiki/Special:Redirect/file/Drone_video_of_people_in_water_-_body_surfing_(East_Sydney_at_Tamarama).webm'
raw="${root}/drone_people_raw.webm"
output="${root}/drone_people_sample.mp4"

curl --fail --location --retry 3 --output "${raw}" "${url}"
if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -y -ss 00:00:05 -i "${raw}" -t 00:00:15 -vf 'scale=960:-2' -an \
    -c:v libx264 -preset fast -crf 23 "${output}"
else
  python_bin="${PYTHON:-python3}"
  if ! "${python_bin}" "$(dirname "$0")/trim_video.py" "${raw}" "${output}" 5 15 960; then
    echo "ffmpeg or Python with opencv-python is required to trim the sample" >&2
    exit 1
  fi
fi
rm -f "${raw}"
printf '%s\n' \
  'Source: Poseidon’s Reach, “East Sydney body surf drone video at Tamarama”' \
  'License: CC BY 3.0' \
  'https://commons.wikimedia.org/wiki/File:Drone_video_of_people_in_water_-_body_surfing_(East_Sydney_at_Tamarama).webm' \
  > "${root}/drone_people_sample.ATTRIBUTION.txt"
echo "Created ${output}"

