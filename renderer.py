# renderer.py
from mpmath import mp, mpf, mpc, log10
from PIL import Image
from color import get_smooth_color
from utils import set_precision
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def render_row(args):
    y, width, height, re_min, re_max, im_min, im_max, max_iter, frame_id = args
    im = im_min + (im_max - im_min) * y / height
    row = np.zeros((width, 3), dtype=np.uint8)
    if y%10 == 0:
        with open("render.log", "a") as f:
            f.write(f"\nY: {y}")

    for x in range(width):
        re = re_min + (re_max - re_min) * x / width
        c = mpc(re, im)
        z = mpc(0, 0)
        n = 0
        try:
            max_internal_steps = 50000
            internal_steps = 0
            while (z.real ** 2 + z.imag ** 2) <= 4 and n < max_iter:
                z = z * z + c
                n += 1
                internal_steps += 1
                if internal_steps > max_internal_steps:
                    with open("render.log", "a") as f:
                        f.write(f"[Frame {frame_id}] STALL DETECTED @ ({x},{y}) z={z}, n={n}, breaking loop\n")
                    break
        except Exception as e:
            with open("render.log", "a") as f:
                f.write(f"[Frame {frame_id}] ERROR in pixel ({x},{y}) — z={z}, c={c}, n={n}: {str(e)}\n")
            row[x] = (255, 0, 0)
            continue

        try:
            r, g, b = get_smooth_color(z, n, max_iter)
            row[x] = (r, g, b)
        except Exception as e:
            with open("render.log", "a") as f:
                f.write(f"[Frame {frame_id}] COLOR FAIL @ ({x},{y}) z={z}, n={n} — {str(e)}\n")
            row[x] = (255, 0, 0)

    return y, row


def mandelbrot_ap(center, zoom, width, height, max_iter, frame_id=None):
    import time
    start = time.time()
    set_precision(zoom)

    re_center, im_center = map(mpf, center)
    scale = mpf(4) / zoom
    aspect_ratio = width / height

    re_min = re_center - scale * aspect_ratio / 2
    re_max = re_center + scale * aspect_ratio / 2
    im_min = im_center - scale / 2
    im_max = im_center + scale / 2

    buf = np.zeros((height, width, 3), dtype=np.uint8)

    with open("render.log", "a") as f:
        f.write(f"[Frame {frame_id}] Mandelbrot Start — Re:({re_min},{re_max}), Im:({im_min},{im_max})\n")

    try:
        with ProcessPoolExecutor() as pool:
            args = [
                (y, width, height, re_min, re_max, im_min, im_max, max_iter, frame_id)
                for y in range(height)
            ]
            for y, row in pool.map(render_row, args):
                buf[y] = row

    except Exception as e:
        with open("render.log", "a") as f:
            f.write(f"[Frame {frame_id}] FATAL LOOP ERROR: {str(e)}\n")
        raise

    try:
        img = Image.fromarray(buf, mode="RGB")
    except Exception as e:
        with open("render.log", "a") as f:
            f.write(f"[Frame {frame_id}] ERROR constructing Image object: {str(e)}\n")
        raise

    with open("render.log", "a") as f:
        f.write(f"[Frame {frame_id}] Mandelbrot Done — Time={time.time() - start:.2f}s\n")

    return img
