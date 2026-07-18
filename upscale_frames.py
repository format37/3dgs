#!/usr/bin/env python3
"""Upscale all images in a directory using Real-ESRGAN via spandrel.

Auto-downloads model weights (~64MB) on first run. Uses GPU with fp16 by default.

Usage:
    python upscale_frames.py input_dir/ output_dir/ --scale 4
    python upscale_frames.py input_dir/ output_dir/ --scale 2 --no-fp16
"""

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

import torch
from PIL import Image
import numpy as np
import spandrel


MODEL_URLS = {
    4: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    2: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
}

CACHE_DIR = Path.home() / ".cache" / "realesrgan"


def get_model_path(scale):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"RealESRGAN_x{scale}plus.pth"
    path = CACHE_DIR / filename
    if not path.exists():
        url = MODEL_URLS[scale]
        print(f"Downloading {filename}...")
        urlretrieve(url, path)
        print(f"Saved to {path}")
    return path


def upscale_frames(input_dir, output_dir, scale=4, fp16=True, tile_size=0):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not images:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images in {input_dir}")

    model_path = get_model_path(scale)
    model = spandrel.ModelLoader().load_from_file(model_path).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if fp16 and device.type == "cuda":
        model = model.half()
    model = model.to(device)
    dtype = torch.float16 if (fp16 and device.type == "cuda") else torch.float32

    print(f"Using device: {device}, dtype: {dtype}")

    for i, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB")
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=device, dtype=dtype) / 255.0

        with torch.no_grad():
            output = model(tensor)

        output = output.squeeze(0).clamp(0, 1).cpu().float()
        output = (output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        out_path = output_dir / img_path.name
        Image.fromarray(output).save(out_path)
        print(f"  [{i+1}/{len(images)}] {img_path.name} "
              f"({img.width}x{img.height} -> {output.shape[1]}x{output.shape[0]})")

    print(f"\nDone. Upscaled {len(images)} images to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upscale images with Real-ESRGAN via spandrel")
    parser.add_argument("input_dir", help="Directory with input images")
    parser.add_argument("output_dir", help="Directory for upscaled output")
    parser.add_argument("--scale", type=int, choices=[2, 4], default=4,
                        help="Upscale factor (default: 4)")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True,
                        help="Use fp16 inference (default: True)")
    parser.add_argument("--tile-size", type=int, default=0,
                        help="Tile size for processing (0 = no tiling)")
    args = parser.parse_args()
    upscale_frames(args.input_dir, args.output_dir, args.scale, args.fp16, args.tile_size)
