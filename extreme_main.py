# main.py

import sys
import time
import subprocess
from pathlib import Path

import argparse
import numpy as np
from tqdm import tqdm
from mpmath import mpf, log10
from PIL import Image

# ─── GPU‐accelerated renderer (with deep Zoom support) ─────────────────────────
from extreme_gpu_renderer import render_gpu_frame, zoom_fits_double

# ─── CPU/mpmath arbitrary‐precision fallback (used only if absolutely needed) ──
from renderer import mandelbrot_ap

# ─── Configuration (from config.py) ────────────────────────────────────────────
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

# ─── Optional “--force” flag ────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--force", action="store_true",
    help="If present, overwrite existing frames even if they already exist."
)
args = parser.parse_args()
FORCE_OVERWRITE = args.force

# ─── Constants for a 60 FPS video ──────────────────────────────────────────────
FRAMERATE    = 60
TOTAL_FRAMES = DURATION_SEC * FRAMERATE  # e.g. 600 s × 60 fps = 36 000 frames

# ─── Simple ASCII‐only logger ─────────────────────────────────────────────────
def log(msg: str) -> None:
    with open("render.log", "a", encoding="utf-8", errors="ignore") as f:
        f.write(msg + "\n")

# ────────────────────────────────────────────────────────────────────────────────
# render_frame(i, center, zoom)
#
# Renders exactly one frame #i using either:
#   - double‐precision GPU (if zoom < 1e12), or
#   - perturbation+double‐double GPU (if zoom ≥ 1e12),
#   - fallback CPU/mpmath if GPU fails for some reason.
# Saves the result to OUTPUT_DIR/frame_{i:04d}.png.
# ────────────────────────────────────────────────────────────────────────────────
def render_frame(i: int, center: tuple, zoom) -> None:
    """
    i:      integer frame index ∈ [0 .. TOTAL_FRAMES-1]
    center: (cx, cy) as Python floats or mpf (float-castable)
    zoom:   either a float (for GPU) or mpf (for CPU fallback)

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
        # Convert zoom to float for the GPU decision
        zf = float(zoom)
        if zoom_fits_double(zf):
            # GPU path (shallow or deep) handles double‐precision or perturbation internally
            img = render_gpu_frame(center, zf, WIDTH, HEIGHT, MAX_ITER)
        else:
            # Extremely deep beyond GPU double-double range → CPU/mpmath fallback
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
            z_msg = f"1e{int(log10(mpf(zoom)))}"
        except Exception:
            z_msg = str(zoom)
        log(f"[Frame {i:04d}] DONE: zoom~{z_msg}, time={elapsed:.2f}s")

# ────────────────────────────────────────────────────────────────────────────────
# deep_zoom_pipeline()
#
# 1) Build per-frame zoom_factors using a geometric sequence (mpf).
# 2) For i in [0..TOTAL_FRAMES-1]:
#      u = i/(TOTAL_FRAMES-1)
#      center = interpolate(START_CENTER → END_CENTER) with smoothstep (mpf).
#      zoom   = zoom_factors[i]
#      render_frame(i, center, zoom)
# ────────────────────────────────────────────────────────────────────────────────
def deep_zoom_pipeline() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    log("===== BEGIN DEEP ZOOM (ALL FRAMES VIA GPU OR FALLBACK) =====")

    if TOTAL_FRAMES <= 0:
        log(f"ERROR: TOTAL_FRAMES = {TOTAL_FRAMES} (must be > 0). Exiting.")
        print(f"→ TOTAL_FRAMES = {TOTAL_FRAMES}. Nothing to do.")
        return

    # Build geometry‐spaced zoom array of length TOTAL_FRAMES:
    from mpmath import mpf
    zoom_factors = np.array([mpf(1)] * TOTAL_FRAMES, dtype=object)
    r = (ZOOM_END / ZOOM_START) ** ( mpf(1) / mpf(TOTAL_FRAMES - 1) )
    for i in range(TOTAL_FRAMES):
        zoom_factors[i] = ZOOM_START * (r ** mpf(i))

    # Helper functions for center interpolation and smoothstep
    def smoothstep(t):
        return t * t * (3 - 2 * t)

    def interpolate(start, end, t):
        return (
            float((1 - t) * mpf(start[0]) + t * mpf(end[0])),
            float((1 - t) * mpf(start[1]) + t * mpf(end[1]))
        )

    for i in tqdm(range(TOTAL_FRAMES), desc="Rendering all frames"):
        u = mpf(i) / mpf(TOTAL_FRAMES - 1)
        t = smoothstep(float(u))
        center = interpolate(START_CENTER, END_CENTER, t)
        zoom = zoom_factors[i]

        render_frame(i, center, zoom)

    log("===== END DEEP ZOOM PIPELINE =====")

# ────────────────────────────────────────────────────────────────────────────────
# generate_video()
#
#  Uses ffmpeg to stitch frames into mandelbrot_zoom_60fps.mp4 at 60 fps.
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
    print("MAIN: Starting deep‐zoom pipeline…")
    log("MAIN: Starting deep‐zoom pipeline…")
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
