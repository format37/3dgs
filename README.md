# Video to 3D Gaussian Splat Pipeline

Convert a phone video into a 3D Gaussian Splat (`.spz`) and upload to [live.arrival.space](https://live.arrival.space).

```
Video → Frames → COLMAP SfM → Splatfacto Train → Export PLY → Convert SPZ → Upload
         (auto via nerfstudio)
```

## Prerequisites

- NVIDIA GPU with 8+ GB VRAM (tested on RTX 4090)
- CUDA-compatible PyTorch
- FFmpeg
- Node.js 20+

## Setup

### 1. Install COLMAP

```bash
sudo apt install colmap
```

### 2. Install Nerfstudio

```bash
pip install nerfstudio
```

Verify: `ns-train --help`

### 3. Install SPZ converter

```bash
npm install   # installs spz-js from package.json
```

### 4. PyTorch compatibility fix

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

### 4. Convert PLY to SPZ

```bash
node convert_to_spz.mjs [input.ply] [output.spz]
# defaults: export/splat.ply → export/scene.spz
```

SPZ is ~10x smaller than PLY.

### 5. Upload

Go to [live.arrival.space](https://live.arrival.space) and upload `export/scene.spz`.

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
    splat.ply         # Gaussian splat (full)
    scene.spz         # Compressed splat (for upload)
  convert_to_spz.mjs  # PLY→SPZ converter
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| COLMAP fails / few matches | More frames, better overlap, textured surfaces |
| OOM during training | Add `--pipeline.datamanager.camera-res-scale-factor 0.5` |
| `weights_only` error on export | Apply the PyTorch patch in Setup step 4 |
| Spiky / blobby result | Better video (see capture tips above) |
| torch.compile takes forever | Normal on first run (~5-15 min); subsequent runs are cached |
