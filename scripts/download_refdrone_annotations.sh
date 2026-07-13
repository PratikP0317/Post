#!/usr/bin/env bash
set -euo pipefail

root="${1:-data/RefDrone}"
base="https://huggingface.co/datasets/sunzc-sunny/RefDrone/resolve/main"
mkdir -p "${root}"
for split in train val test; do
  file="RefDrone_${split}_mdetr.json"
  curl --fail --location --retry 3 --output "${root}/${file}" "${base}/${file}"
done
echo "Downloaded RefDrone annotations to ${root}; add the matching VisDrone images under ${root}/all_image"
