#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: ./workflow.sh <video_path>"
    echo "Example: ./workflow.sh input/DJI_20260228092820_0008_D.MP4"
    exit 1
fi

VIDEO="$1"
if [ ! -f "$VIDEO" ]; then
    echo "Error: $VIDEO not found"
    exit 1
fi

echo "Processing: $VIDEO"

rm -rf data/scene

ns-process-data video --data "$VIDEO" --output-dir data/scene --matching-method sequential

python3 fix_colmap_model.py data/scene --regenerate-transforms

ns-train splatfacto --data data/scene --output-dir output/ --max-num-iterations 30000 --viewer.quit-on-train-completion True

LATEST=$(ls -t output/scene/splatfacto/ | head -1)

ns-export gaussian-splat --load-config output/scene/splatfacto/$LATEST/config.yml --output-dir export/

python3 cleanup_ply.py export/splat.ply export/splat_final.ply --distance 3.0 --min-opacity 0.0 --rotate-x 90

python3 -m http.server 8080
