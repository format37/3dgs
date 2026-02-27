# Video to 3D Gaussian Splat Pipeline

Convert a phone video into a 3D Gaussian Splat (`.ply`) and upload to [live.arrival.space](https://live.arrival.space).

```
Video → Frames → COLMAP SfM → Splatfacto Train → Export PLY → Clean & Rotate → Upload
         (auto via nerfstudio)
```

## Prerequisites

- NVIDIA GPU with 8+ GB VRAM (tested on RTX 4090)
- CUDA toolkit (tested with CUDA 12.5)
- CUDA-compatible PyTorch
- FFmpeg
- Node.js 20+ (only if you need SPZ conversion)

## Setup

### 1. Install COLMAP (with CUDA)

The default `sudo apt install colmap` installs a CPU-only build. For GPU-accelerated
feature extraction and matching (significantly faster), build from source.

**Important:** Use tag `3.11.1` — the latest `main` branch adds an OpenImageIO
dependency that causes linking conflicts with anaconda. Also, if you have anaconda
installed, strip it from `PATH` during the build to avoid library mismatches.

```bash
# Install dependencies (Ubuntu 22.04)
sudo apt install -y \
  build-essential cmake git ninja-build \
  libcgal-dev libeigen3-dev libsuitesparse-dev libfreeimage-dev \
  libgoogle-glog-dev libgflags-dev libsqlite3-dev libglew-dev \
  libflann-dev liblz4-dev libboost-all-dev libceres-dev \
  libopencv-dev qtbase5-dev libqt5opengl5-dev

# Clone and checkout 3.11.1
git clone https://github.com/colmap/colmap.git
cd colmap
git checkout 3.11.1
mkdir build && cd build

# Configure — strip anaconda from PATH, point to correct nvcc
# Replace 89 with your GPU's compute capability (89 = RTX 4090/Ada Lovelace)
# Replace /usr/local/cuda-12.5 with your CUDA toolkit path
PATH=/usr/local/cuda-12.5/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin CMAKE_PREFIX_PATH="" cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DCUDA_ENABLED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.5/bin/nvcc

# Build (strip anaconda from PATH here too)
PATH=/usr/local/cuda-12.5/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin ninja -j$(nproc)

sudo ninja install
```

Verify GPU support:
```bash
colmap -h  # Should say "with CUDA", NOT "without CUDA"
```

**Common build issues:**
- `Unsupported gpu architecture 'compute_'` — cmake too old for `native`; use explicit arch (e.g. `89`)
- `/usr/bin/nvcc` picked up instead of CUDA toolkit — set `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-XX/bin/nvcc`
- TIFF/GDAL linking errors — anaconda libs leaking in; strip anaconda from PATH
- OpenImageIO errors — use tag `3.11.1` instead of `main`

### 2. Install Nerfstudio

```bash
pip install nerfstudio
```

Verify: `ns-train --help`

### 3. PyTorch compatibility fix

Nerfstudio 1.1.5 has a compatibility issue with PyTorch 2.6+ (`weights_only` default changed to `True`). Patch the checkpoint loader:

```bash
# Find and patch eval_utils.py
EVAL_UTILS=$(python -c "import nerfstudio; print(nerfstudio.__path__[0])")/utils/eval_utils.py
sed -i 's/torch.load(load_path, map_location="cpu")/torch.load(load_path, map_location="cpu", weights_only=False)/' "$EVAL_UTILS"
```

## Usage

### 1. Process video (extract frames + COLMAP reconstruction)

```bash
ns-process-data video \
  --data input/YOUR_VIDEO.mp4 \
  --output-dir data/scene \
  --num-frames-target 150
```

Verify: `data/scene/` should contain `images/`, `colmap/sparse/`, and `transforms.json`.

### 2. Train splatfacto

```bash
ns-train splatfacto \
  --data data/scene \
  --output-dir output/ \
  --max-num-iterations 30000 \
  --viewer.quit-on-train-completion True
```

- First run includes a ~5-15 min torch.compile warmup (CPU-only, GPU idle — this is normal)
- Actual training: ~15-30 min on RTX 4090
- Live viewer at http://localhost:7007 during training
- If you hit OOM, add: `--pipeline.datamanager.camera-res-scale-factor 0.5`

### 3. Export to PLY

```bash
ns-export gaussian-splat \
  --load-config output/scene/splatfacto/<TIMESTAMP>/config.yml \
  --output-dir export/
```

Replace `<TIMESTAMP>` with the actual run directory (e.g., `2026-02-27_160905`).

### 4. Preview exported PLY

Before uploading, verify the export looks correct.

**Option A: Nerfstudio viewer** (shows trained model from checkpoint)

```bash
ns-viewer --load-config output/scene/splatfacto/<TIMESTAMP>/config.yml
```

Open http://localhost:7007 in your browser. Drag to rotate, scroll to zoom.

**Option B: Local web viewer** (shows the actual PLY file)

```bash
python3 -m http.server 8080
```

Open http://localhost:8080/viewer.html in your browser. Buttons in the top-right switch between PLY files in `export/`. Uses [@mkkellogg/gaussian-splats-3d](https://github.com/mkkellogg/GaussianSplats3D) via CDN.

### 5. Clean up and fix orientation

Exported PLY files typically need two fixes before uploading:

1. **Outlier removal** — a large cloud of floating artifact gaussians surrounds the scene
2. **Rotation** — phone video produces a scene where the subject is "lying down" (rotated 90°)

```bash
# Recommended: clean + fix phone video orientation (+90° around X axis)
python3 cleanup_ply.py export/splat.ply export/splat_final.ply \
  --distance 3.0 --min-opacity 0.0 --rotate-x 90

# Just cleanup, no rotation
python3 cleanup_ply.py export/splat.ply export/splat_clean.ply \
  --distance 3.0 --min-opacity 0.0

# Aggressive cleanup — also cap scale to reduce spiky artifacts
python3 cleanup_ply.py export/splat.ply export/splat_final.ply \
  --distance 3.0 --min-opacity 0.0 --max-scale 3.0 --rotate-x 90
```

The script supports:
- **`--distance`** — max distance from median center (removes outlier cloud)
- **`--min-opacity`** — min opacity in logit space (`0.0` = 50% visible; `-2.0` = 12%)
- **`--max-scale`** — max scale in log space (caps gaussian size, reduces spikes)
- **`--rotate-x/y/z`** — rotation in degrees around each axis
- **`--recenter`** — move scene center to origin

Preview the result in the local viewer (Step 4B) and adjust thresholds until
the scene looks clean. You can also use [SuperSplat](https://playcanvas.com/supersplat/editor)
for visual bounding-box cropping.

Alternatively, crop during export using nerfstudio's OBB options:

```bash
ns-export gaussian-splat \
  --load-config output/scene/splatfacto/<TIMESTAMP>/config.yml \
  --output-dir export/ \
  --obb-center 0.5 1.6 -0.5 \
  --obb-scale 6.0 6.0 6.0
```

### 6. Upload

Go to [live.arrival.space](https://live.arrival.space) and upload the cleaned PLY file (e.g., `export/splat_final.ply`).

arrival.space accepts both `.ply` and `.spz` formats.

### 7. (Optional) Convert PLY to SPZ

SPZ is ~10x smaller than PLY. Only needed if file size matters.

```bash
npm install              # first time only — installs spz-js
node convert_to_spz.mjs  # defaults: export/splat.ply → export/scene.spz

# Convert a specific file
node convert_to_spz.mjs export/splat_final.ply export/scene_final.spz
```

Note: `spz-js` requires PLY with SH coefficients (the default nerfstudio export). It will fail on PLY exported with `--ply-color-mode rgb`.

## Video Capture Tips

Quality of the input video is the single biggest factor for good results. Poor video = spiky artifacts.

### Do

- **Orbit the subject** — walk around it in a circle or arc, keeping it centered
- **Move slowly and smoothly** — avoid sudden movements, jerks, or shaking
- **Overlap views** — each frame should share ~70% of the scene with adjacent frames
- **Capture from multiple heights** — eye level + slightly above + slightly below
- **Good lighting** — diffuse, even lighting works best; avoid harsh shadows
- **Textured surfaces** — COLMAP needs visual features to match; plain white walls will fail
- **15-60 seconds** of footage (longer is better for coverage)

### Don't

- **Linear pan only** — a single straight-line walk gives almost no 3D information
- **Move too fast** — causes motion blur, which kills feature matching
- **Point at the sky** — featureless backgrounds confuse COLMAP
- **Have moving objects in the scene** — people, cars, flags will create ghosting artifacts
- **Shoot through glass or reflections** — confuses feature matching

### Ideal capture pattern

```
        ___
       /   \      Walk in a circle or arc around the subject
      |  *  |     keeping camera pointed at the center (*)
       \___/      at 2-3 different heights
```

## Project Structure

```
08-3dgs/
  input/              # Source videos
  data/scene/         # Processed frames + COLMAP output
    images/           # Extracted frames
    colmap/sparse/    # Camera poses
    transforms.json   # Nerfstudio dataset config
  output/             # Training runs
    scene/splatfacto/<timestamp>/
      config.yml
      nerfstudio_models/step-*.ckpt
  export/
    splat.ply         # Gaussian splat (raw export)
    splat_final.ply   # Cleaned + rotated PLY (ready to upload)
  cleanup_ply.py      # Clean outliers + rotate PLY
  convert_to_spz.mjs  # PLY→SPZ converter (optional)
  viewer.html         # Local 3DGS web viewer
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| COLMAP says "without CUDA" | Build from source with `-DCUDA_ENABLED=ON` (see Setup step 1) |
| COLMAP fails / few matches | More frames, better overlap, textured surfaces |
| OOM during training | Add `--pipeline.datamanager.camera-res-scale-factor 0.5` |
| `weights_only` error on export | Apply the PyTorch patch in Setup step 3 |
| Spiky / blobby result | Better video (see capture tips above) |
| Huge artifact cloud around scene | Run `cleanup_ply.py` with `--distance` filter (Step 5) |
| Scene tiny on arrival.space | Outlier cloud pushes camera too far; clean up first |
| Subject lying on its side | Add `--rotate-x 90` to cleanup_ply.py (Step 5) |
| `spz-js` error "Missing f_dc_0" | PLY was exported with `--ply-color-mode rgb`; use default `sh_coeffs` |
| torch.compile takes forever | Normal on first run (~5-15 min); subsequent runs are cached |
