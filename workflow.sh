#!/bin/bash
set -e

# nerfstudio's resume path calls torch.load without weights_only=False, which
# PyTorch >= 2.6 rejects; our checkpoints are local and trusted.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

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

SCENE=data/scene
FPFILE=$SCENE/.source_video
FP="$(basename "$VIDEO") $(stat -c%s "$VIDEO")"
RUNS_DIR=output/scene/splatfacto
MAX_ITERS=30000

# A checkpoint interrupted mid-write is a truncated zip; never resume from one.
ckpt_valid() {
    python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).testzip()" "$1" 2>/dev/null
}

echo "Processing: $VIDEO"

# --- Stage 1: video -> COLMAP dataset ---
# The fingerprint is written only after processing fully succeeds, so its
# presence means the dataset is complete and identifies the source video.
if [ -f "$FPFILE" ]; then
    if [ "$(cat "$FPFILE")" = "$FP" ]; then
        echo "[skip] $SCENE already built from: $FP"
    else
        echo "Error: $SCENE was built from a different video: $(cat "$FPFILE")"
        echo "Rename or remove $SCENE before processing a new video, e.g.:"
        echo "  mv $SCENE data/scene_old   # or: rm -rf $SCENE"
        exit 1
    fi
else
    rm -rf "$SCENE"
    ns-process-data video --data "$VIDEO" --output-dir "$SCENE" --matching-method sequential
    python3 fix_colmap_model.py "$SCENE" --regenerate-transforms
    echo "$FP" > "$FPFILE"
fi

# --- Stage 2: train, resuming from the newest valid checkpoint of this dataset ---
# Only run dirs created after the dataset was built belong to this video.
LOAD_ARGS=()
LATEST=""
RESUME_ITERS=$MAX_ITERS
if [ -d "$RUNS_DIR" ]; then
    for run in $(ls -t "$RUNS_DIR"); do
        [ "$RUNS_DIR/$run/config.yml" -nt "$FPFILE" ] || continue
        ckpt=$(ls -t "$RUNS_DIR/$run/nerfstudio_models/"*.ckpt 2>/dev/null | head -1)
        [ -n "$ckpt" ] || continue
        if ckpt_valid "$ckpt"; then
            step=$(basename "$ckpt" | grep -oE '[0-9]+' | sed 's/^0*//')
            step=${step:-0}
            if [ "$step" -ge $((MAX_ITERS - 1)) ]; then
                echo "[skip] training already complete at step $step: $run"
                LATEST=$run
            else
                # nerfstudio resumes from the loaded step and then runs
                # max_num_iterations MORE steps, so pass only the remainder —
                # passing $MAX_ITERS again makes the run never finish.
                RESUME_ITERS=$((MAX_ITERS - step))
                # gsplat's densification accumulators are not checkpointed, so
                # re-entering refinement after a resume crashes with a CUDA
                # index assert. Disable grow/prune from the resume step onward.
                echo "[resume] from $ckpt ($RESUME_ITERS iterations remain; densification disabled)"
                LOAD_ARGS=(--load-dir "$RUNS_DIR/$run/nerfstudio_models"
                           --pipeline.model.stop-split-at "$step")
            fi
            break
        else
            echo "[warn] ignoring corrupt checkpoint: $ckpt"
        fi
    done
fi

if [ -z "$LATEST" ]; then
    # NVMe dies under sustained write load; log its temperature to a synced
    # file every 10s so a crash leaves evidence of the last-seen temps.
    (
        while :; do
            for d in /sys/class/nvme/nvme*; do
                t=$(cat "$d"/hwmon*/temp1_input 2>/dev/null | head -1)
                [ -n "$t" ] && echo "$(date '+%F %T') $(basename "$d") $((t / 1000))C" >> nvme_temp.log
            done
            sync nvme_temp.log 2>/dev/null
            sleep 10
        done
    ) &
    TEMP_PID=$!
    trap 'kill $TEMP_PID 2>/dev/null' EXIT

    # Keep every checkpoint while training: a crash mid-save must not be able
    # to destroy the only copy. Intermediates are pruned after export succeeds.
    ns-train splatfacto --data "$SCENE" --output-dir output/ \
        --max-num-iterations $RESUME_ITERS \
        --save-only-latest-checkpoint False \
        --viewer.quit-on-train-completion True \
        "${LOAD_ARGS[@]}"

    kill $TEMP_PID 2>/dev/null || true
    LATEST=$(ls -t "$RUNS_DIR" | head -1)
fi

# --- Stage 3: export + cleanup ---
# The final checkpoint's step number varies on resumed runs, so "final" is
# simply the newest checkpoint of the completed run.
FINAL_CKPT=$(ls -t "$RUNS_DIR/$LATEST/nerfstudio_models/"*.ckpt | head -1)
if [ -f export/splat_final.ply ] && [ export/splat_final.ply -nt "$FINAL_CKPT" ]; then
    echo "[skip] export/splat_final.ply is up to date"
else
    ns-export gaussian-splat --load-config "$RUNS_DIR/$LATEST/config.yml" --output-dir export/
    # DISTANCE crops gaussians farther than this from the scene center
    # (smaller = tighter crop), e.g.: DISTANCE=2.0 ./workflow.sh <video>
    python3 cleanup_ply.py export/splat.ply export/splat_final.ply \
        --distance "${DISTANCE:-3.0}" --min-opacity "${MIN_OPACITY:--2.0}" --rotate-x 90
    find "$RUNS_DIR/$LATEST/nerfstudio_models" -name 'step-*.ckpt' ! -name "$(basename "$FINAL_CKPT")" -delete
fi

# --- Stage 4: serve viewer (8080 is taken by the weaviate container) ---
echo "Open http://localhost:8081/viewer.html"
python3 -m http.server 8081
