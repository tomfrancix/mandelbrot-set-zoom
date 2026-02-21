# main.py

import sys
import time
import subprocess
import math
from pathlib import Path

import argparse
import numpy as np
from tqdm import tqdm
from mpmath import mpf, log10
from PIL import Image

# ─── GPU‐accelerated renderer ────────────────────────────────────────────────
from gpu_renderer import render_gpu_frame, zoom_fits_double

# ─── CPU perturbation renderer for very deep zooms ───────────────────────────
from perturbation_renderer import render_perturbation_frame

# ─── CPU/mpmath arbitrary‐precision renderer ─────────────────────────────────
from renderer import mandelbrot_ap

# ─── Configuration ────────────────────────────────────────────────────────────
try:
    from config import (
        OUTPUT_DIR,
        WIDTH,
        HEIGHT,
        MAX_ITER,
        ZOOM_START,
        ZOOM_END,
        DURATION_SEC,
        START_CENTER,
        END_CENTER,
    )
except ImportError as e:
    print("ERROR: Could not import config.py. Make sure it defines all of:")
    print("  OUTPUT_DIR, WIDTH, HEIGHT, MAX_ITER, ZOOM_START, ZOOM_END, DURATION_SEC, START_CENTER, END_CENTER")
    print("Details:", e)
    sys.exit(1)

# ─── Optional “--force” flag ───────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--force", action="store_true",
    help="If present, overwrite existing frames even if they already exist."
)
args = parser.parse_args()
FORCE_OVERWRITE = args.force

# ─── Constants for a 60 FPS video ──────────────────────────────────────────────
FRAMERATE    = 60
TOTAL_FRAMES = DURATION_SEC * FRAMERATE  # = 36000 for 10 min × 60 fps

# ─── Simple ASCII‐only logger ─────────────────────────────────────────────────
def log(msg: str) -> None:
    with open("render.log", "a", encoding="utf-8", errors="ignore") as f:
        f.write(msg + "\n")

# ────────────────────────────────────────────────────────────────────────────────
# render_frame(i, center, zoom)
#
# Renders exactly one frame #i at (center, zoom). Saves to:
#    OUTPUT_DIR/frame_{i:04d}.png
# If zoom_fits_double(float(zoom)) → GPU double precision; else → CPU/mpmath.
# ────────────────────────────────────────────────────────────────────────────────
def render_frame(i: int, center: tuple, zoom) -> None:
    """
    i:      integer frame index ∈ [0 .. TOTAL_FRAMES-1]
    center: (cx, cy) as Python floats or mpf (float casting OK for GPU)
    zoom:   either a float (GPU path) or an mpf (CPU path)

    If frame_{i:04d}.png already exists and --force is False, skip rendering.
    """
    output_dir = Path(OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename    = f"frame_{i:04d}.png"
    output_path = output_dir / filename

    if output_path.exists() and not FORCE_OVERWRITE:
        log(f"[Frame {i:04d}] SKIP: already exists.")
        return

    log(f"[Frame {i:04d}] START: center={center}, zoom={zoom}, path={output_path}")
    t0 = time.time()
    try:
        # 1) GPU path when zoom is within a safe float64 regime.
        #    Note: float(mpf("1e480")) becomes inf, which correctly triggers non-GPU paths.
        z_float = float(zoom)
        if math.isfinite(z_float) and zoom_fits_double(z_float):
            img = render_gpu_frame(center, z_float, WIDTH, HEIGHT, MAX_ITER)
        else:
            # 2) Perturbation path for deep zooms.
            #    This is dramatically faster than per-pixel mpmath and avoids early stalls.
            try:
                img = render_perturbation_frame(
                    center=center,
                    zoom=zoom,
                    width=WIDTH,
                    height=HEIGHT,
                    max_iter=MAX_ITER,
                    frame_id=i,
                )
            except Exception as perturb_err:
                # 3) Last-resort: pure mpmath renderer (slow, but can salvage correctness).
                log(f"[Frame {i:04d}] Perturbation failed ({perturb_err}); falling back to mpmath CPU renderer")
                img = mandelbrot_ap(
                    center=center,
                    zoom=mpf(zoom),
                    width=WIDTH,
                    height=HEIGHT,
                    max_iter=MAX_ITER,
                    frame_id=i
                )
    except Exception as e:
        log(f"[Frame {i:04d}] ERROR during render: {e}")
        return

    try:
        img.save(str(output_path), format="PNG")
    except Exception as e:
        log(f"[Frame {i:04d}] ERROR saving file: {e}")
        return

    if not output_path.exists():
        log(f"[Frame {i:04d}] ERROR: file not found after save.")
    else:
        elapsed = time.time() - t0
        try:
            # Print zoom as “1eN” if possible
            z_msg = f"1e{int(log10(mpf(zoom)))}"
        except Exception:
            z_msg = str(zoom)
        log(f"[Frame {i:04d}] DONE: zoom~{z_msg}, time={elapsed:.2f}s")

# ────────────────────────────────────────────────────────────────────────────────
# crop_and_resize(prev_i, curr_i, zoom_factors)
#
# For an intermediate frame curr_i (where prev_i = floor(curr_i/10)*10 is the
# last anchor), open frame_{prev_i:04d}.png, crop its center by factor
#   crop_scale = zoom_factors[prev_i] / zoom_factors[curr_i],
# then resize back to (WIDTH, HEIGHT), and save as frame_{curr_i:04d}.png.
# ────────────────────────────────────────────────────────────────────────────────
def crop_and_resize(prev_i: int, curr_i: int, r, zoom_prev, zoom_curr) -> None:
    base_path   = Path(OUTPUT_DIR) / f"frame_{prev_i:04d}.png"
    target_path = Path(OUTPUT_DIR) / f"frame_{curr_i:04d}.png"

    # Skip if exists and no --force
    if target_path.exists() and not FORCE_OVERWRITE:
        log(f"[Frame {curr_i:04d}] SKIP crop: already exists.")
        return

    if not base_path.exists():
        log(f"[Frame {curr_i:04d}] ERROR: base frame {prev_i:04d} not found for cropping.")
        return

    # 1) Open the anchor image
    img = Image.open(str(base_path))
    w, h = img.size  # should match (WIDTH, HEIGHT)

    # 2) Compute crop_scale = zoom_prev / zoom_curr.
    #    Use mpf ratio for robustness at extreme zooms.
    crop_scale = float(mpf(zoom_prev) / mpf(zoom_curr))  # 0 < crop_scale ≤ 1

    # 3) Determine pixel dims of the crop
    crop_w = max(1, int(round(w * crop_scale)))
    crop_h = max(1, int(round(h * crop_scale)))

    left = (w - crop_w) // 2
    top  = (h - crop_h) // 2
    right  = left + crop_w
    bottom = top + crop_h

    # 4) Crop and resize back to full resolution
    cropped = img.crop((left, top, right, bottom))
    zoomed  = cropped.resize((w, h), resample=Image.LANCZOS)

    # 5) Save result
    zoomed.save(str(target_path), format="PNG")
    log(f"[Frame {curr_i:04d}] CROPPED from {prev_i:04d} (scale={crop_scale:.6e}), saved.")

# ────────────────────────────────────────────────────────────────────────────────
# deep_zoom_pipeline()
#
# 1) Compute per-frame multiplier r = (ZOOM_END/ZOOM_START)^(1/(TOTAL_FRAMES-1)).
# 2) Build zoom_factors list of length TOTAL_FRAMES: zoom_factors[i] = ZOOM_START * r^i (mpf).
# 3) For each frame i in [0..TOTAL_FRAMES-1]:
#      if i % 10 == 0 → call render_frame(i, ..., zoom_factors[i])
#      else           → call crop_and_resize(prev_anchor, i, zoom_factors)
# ────────────────────────────────────────────────────────────────────────────────
def deep_zoom_pipeline() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    log("===== BEGIN 10-MINUTE DEEP ZOOM (36 000 FRAMES) =====")

    if TOTAL_FRAMES <= 0:
        log(f"ERROR: TOTAL_FRAMES = {TOTAL_FRAMES} (must be > 0). Exiting.")
        print(f"→ TOTAL_FRAMES = {TOTAL_FRAMES}. Nothing to do.")
        return

    # 1) Compute per-frame multiplier r (mpf)
    from mpmath import mpf
    r = (ZOOM_END / ZOOM_START) ** ( mpf(1) / mpf(TOTAL_FRAMES - 1) )

    # 2) Stream zoom forward (avoid allocating 36k mpf objects and keeping them alive)
    zoom = mpf(ZOOM_START)
    last_anchor_i = 0
    last_anchor_zoom = zoom

    # 3) Loop through all frames
    for i in tqdm(range(TOTAL_FRAMES), desc="Rendering all frames"):
        center = START_CENTER  # no pan, pure zoom-in-place

        if i % 10 == 0:
            # Full Mandelbrot render every 10th frame
            render_frame(i, center, zoom)
            last_anchor_i = i
            last_anchor_zoom = zoom
        else:
            # Crop/resample from the last anchor
            crop_and_resize(last_anchor_i, i, r, last_anchor_zoom, zoom)

        # Advance zoom (except after final frame)
        if i != TOTAL_FRAMES - 1:
            zoom = zoom * r

    log("===== END DEEP ZOOM PIPELINE =====")

# ────────────────────────────────────────────────────────────────────────────────
# generate_video()
#
# Uses ffmpeg to stitch frames into mandelbrot_zoom_60fps.mp4 at 60 fps.
# ────────────────────────────────────────────────────────────────────────────────
def generate_video() -> None:
    if TOTAL_FRAMES <= 0:
        log(f"ERROR: Cannot encode video; TOTAL_FRAMES = {TOTAL_FRAMES}")
        print(f"→ TOTAL_FRAMES = {TOTAL_FRAMES}. No video to encode.")
        return

    output_video = "mandelbrot_zoom_60fps.mp4"
    log("===== BEGIN VIDEO ENCODING (60 FPS) =====")

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(FRAMERATE),
        "-i", f"{OUTPUT_DIR}/frame_%04d.png",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_video
    ]
    try:
        subprocess.run(cmd, check=True)
        log(f"===== VIDEO ENCODED: {output_video} =====")
    except subprocess.CalledProcessError as e:
        log(f"ERROR: ffmpeg failed: {e}")
        print(f"ERROR: ffmpeg failed: {e}")

# ────────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("MAIN: Starting 10-minute (36 000-frame) deep-zoom pipeline…")
    log("MAIN: Starting 10-minute deep-zoom pipeline…")
    start_time = time.time()

    try:
        deep_zoom_pipeline()
        generate_video()
    except Exception as e:
        log(f"MAIN: FATAL ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        print("MAIN: Fatal error—see render.log for details.")
        sys.exit(1)

    total_elapsed = time.time() - start_time
    log(f"MAIN: All done. Total elapsed time: {total_elapsed/60:.2f} minutes.")
    print(f"MAIN: All done. Elapsed ~ {total_elapsed/60:.2f} minutes.")
