from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List

import numpy as np
from PIL import Image
from mpmath import mpf, mpc

from color import get_smooth_color
from utils import set_precision
from mandelzoom.util.logging_setup import get_logger, logging_initialiser

_G = {}

def _init_worker(re_min, re_max, im_min, im_max, width, height, max_iter, frame_id, log_queue, log_level):
    _G["re_min"] = re_min
    _G["re_max"] = re_max
    _G["im_min"] = im_min
    _G["im_max"] = im_max
    _G["width"] = width
    _G["height"] = height
    _G["max_iter"] = max_iter
    _G["frame_id"] = frame_id
    logging_initialiser(log_queue, log_level)

def _render_band(y0_y1: Tuple[int, int]):
    y0, y1 = y0_y1
    width = _G["width"]
    height = _G["height"]
    re_min = _G["re_min"]
    re_max = _G["re_max"]
    im_min = _G["im_min"]
    im_max = _G["im_max"]
    max_iter = _G["max_iter"]
    frame_id = _G["frame_id"]

    logger = get_logger()
    band = np.zeros((y1 - y0, width, 3), dtype=np.uint8)

    for yi, y in enumerate(range(y0, y1)):
        im = im_min + (im_max - im_min) * mpf(y) / mpf(height)
        for x in range(width):
            re = re_min + (re_max - re_min) * mpf(x) / mpf(width)
            c = mpc(re, im)
            z = mpc(0, 0)
            n = 0
            try:
                max_internal_steps = 50000
                internal_steps = 0
                while (z.real * z.real + z.imag * z.imag) <= 4 and n < max_iter:
                    z = z * z + c
                    n += 1
                    internal_steps += 1
                    if internal_steps > max_internal_steps:
                        logger.warning("[Frame %s] Stall at pixel (%s,%s) - breaking loop", frame_id, x, y)
                        break
            except Exception:
                logger.exception("[Frame %s] ERROR in pixel (%s,%s) - z=%s c=%s", frame_id, x, y, z, c)
                band[yi, x] = (255, 0, 0)
                continue

            try:
                r, g, b = get_smooth_color(z, n, max_iter)
                band[yi, x] = (r, g, b)
            except Exception:
                logger.exception("[Frame %s] COLOR FAIL @ (%s,%s) z=%s n=%s", frame_id, x, y, z, n)
                band[yi, x] = (255, 0, 0)

        if y % 50 == 0:
            logger.info("[Frame %s] Rendered row %s/%s", frame_id, y, height)

    return y0, band

def render_frame_cpu(
    *,
    center: Tuple[float, float],
    zoom: float,
    width: int,
    height: int,
    max_iter: int,
    frame_id: str,
    log_queue,
    log_level: int,
    band_height: int = 32,
) -> Image.Image:
    logger = get_logger()
    set_precision(zoom)

    re_center, im_center = map(mpf, center)
    scale = mpf(4) / mpf(zoom)
    aspect = mpf(width) / mpf(height)

    re_min = re_center - scale * aspect / 2
    re_max = re_center + scale * aspect / 2
    im_min = im_center - scale / 2
    im_max = im_center + scale / 2

    logger.info("[Frame %s] CPU render start zoom=%s iter=%s", frame_id, zoom, max_iter)

    buf = np.zeros((height, width, 3), dtype=np.uint8)

    bands: List[Tuple[int, int]] = []
    y = 0
    while y < height:
        y1 = min(height, y + band_height)
        bands.append((y, y1))
        y = y1

    with ProcessPoolExecutor(
        initializer=_init_worker,
        initargs=(re_min, re_max, im_min, im_max, width, height, max_iter, frame_id, log_queue, log_level),
    ) as pool:
        for y0, band in pool.map(_render_band, bands):
            buf[y0:y0 + band.shape[0]] = band

    img = Image.fromarray(buf, mode="RGB")
    logger.info("[Frame %s] CPU render done", frame_id)
    return img
