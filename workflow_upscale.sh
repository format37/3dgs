#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: ./workflow_upscale.sh <video_path> [scale] [num_frames]"
    echo "  scale: 2 or 4 (default: 4)"
    echo "  num_frames: approximate number of frames to extract (default: 150)"
    echo ""
    echo "Example: ./workflow_upscale.sh input/old_phone_video.mp4 4 150"
    exit 1
fi

VIDEO="$1"
SCALE="${2:-4}"
NUM_FRAMES="${3:-150}"

if [ ! -f "$VIDEO" ]; then
    echo "Error: $VIDEO not found"
    exit 1
fi

echo "Processing: $VIDEO (upscale ${SCALE}x, ~${NUM_FRAMES} frames)"

rm -rf data/scene

# Get total frames to calculate spacing
TOTAL_FRAMES=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$VIDEO")
SPACING=$((TOTAL_FRAMES / NUM_FRAMES))
if [ "$SPACING" -lt 1 ]; then
    SPACING=1
fi
echo "Total frames: $TOTAL_FRAMES, spacing: $SPACING (selecting ~1 every $SPACING)"

# Step 1: Extract frames with ffmpeg
echo ""
echo "=== Step 1/7: Extracting frames ==="
mkdir -p data/scene/_frames_raw
ffmpeg -i "$VIDEO" -vsync vfr -filter_complex "thumbnail=$SPACING,setpts=N/TB" \
    "data/scene/_frames_raw/frame_%05d.png"
echo "Extracted $(ls data/scene/_frames_raw/*.png | wc -l) frames"

# Step 2: Upscale frames
echo ""
echo "=== Step 2/7: Upscaling frames (${SCALE}x) ==="
python3 upscale_frames.py data/scene/_frames_raw/ data/scene/_frames_upscaled/ --scale "$SCALE"

# Step 3: COLMAP on upscaled images
echo ""
echo "=== Step 3/7: Running COLMAP (ns-process-data images) ==="
ns-process-data images --data data/scene/_frames_upscaled --output-dir data/scene --matching-method sequential

# Step 4: Fix COLMAP model
echo ""
echo "=== Step 4/7: Fixing COLMAP model ==="
python3 fix_colmap_model.py data/scene --regenerate-transforms

# Step 5: Train splatfacto
echo ""
echo "=== Step 5/7: Training splatfacto ==="
ns-train splatfacto --data data/scene --output-dir output/ --max-num-iterations 30000 --viewer.quit-on-train-completion True

# Step 6: Export gaussian splat
echo ""
echo "=== Step 6/7: Exporting gaussian splat ==="
LATEST=$(ls -t output/scene/splatfacto/ | head -1)
ns-export gaussian-splat --load-config output/scene/splatfacto/$LATEST/config.yml --output-dir export/

# Step 7: Cleanup PLY
echo ""
echo "=== Step 7/7: Cleaning up PLY ==="
python3 cleanup_ply.py export/splat.ply export/splat_final.ply \
    --distance "${DISTANCE:-3.0}" --min-opacity "${MIN_OPACITY:--2.0}" --rotate-x 90

echo ""
echo "=== Done! ==="
python3 -m http.server 8080
