# main.py

import numpy as np
from tqdm import tqdm
from pathlib import Path
import subprocess
from config import *
from renderer import mandelbrot_ap
from mpmath import mpf, log10
import time
import os

START_CENTER = (-0.5, 0.0)
END_CENTER = (-0.743643887037151, 0.13182590420533)

def log(msg):
    with open("render.log", "a") as f:
        f.write(msg + "\n")

def interpolate(start, end, t):
    return tuple(float((1 - t) * mpf(s) + t * mpf(e)) for s, e in zip(start, end))

def smoothstep(t):
    return t * t * (3 - 2 * t)

def render_frame(i, zoom, center):
    output_dir = Path(OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"frame_{i:04d}.png"
    log(f"[Frame {i}] START - Zoom={zoom}, Center={center}, Path={output_path}")

    if output_path.exists():
        log(f"[Frame {i}] Skipped: Already exists at {output_path}")
        return

    zoom = mpf(zoom)
    start = time.time()

    try:
        img = mandelbrot_ap(center=center, zoom=zoom, width=WIDTH, height=HEIGHT, max_iter=MAX_ITER, frame_id=i)
    except Exception as e:
        log(f"[Frame {i}] ERROR generating image: {str(e)}")
        return

    try:
        img.save(str(output_path), format="PNG")
    except Exception as e:
        log(f"[Frame {i}] ERROR saving to {output_path}: {str(e)}")
        return

    if not output_path.exists():
        log(f"[Frame {i}] ERROR: File not created at {output_path}")
    else:
        elapsed = time.time() - start
        log(f"[Frame {i}] SUCCESS: Zoom=1e{int(log10(zoom))}, Saved to {output_path}, Time={elapsed:.2f}s")

def deep_zoom():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    zooms = np.geomspace(ZOOM_START, ZOOM_END, num=FRAMES)

    log("===== BEGIN DEEP ZOOM =====")
    for i in tqdm(range(FRAMES)):
        t = smoothstep(i / (FRAMES - 1))
        center = interpolate(START_CENTER, END_CENTER, t)
        render_frame(i, zooms[i], center)
    log("===== END DEEP ZOOM =====")

def generate_video():
    log("===== BEGIN VIDEO ENCODING =====")
    output_video = "mandelbrot_zoom.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-framerate", "30",
        "-i", f"{OUTPUT_DIR}/frame_%04d.png",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_video
    ]
    subprocess.run(command)
    log(f"===== VIDEO ENCODED: Saved to {output_video} =====")

if __name__ == "__main__":
    deep_zoom()
    generate_video()
