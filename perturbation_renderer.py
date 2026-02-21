# perturbation_renderer.py

"""High-depth Mandelbrot rendering via perturbation (NumPy implementation).

This renderer exists to prevent the "grinds to a halt" behaviour once the pipeline
falls back from GPU float64 to per-pixel mpmath.

Perturbation works by computing a **high-precision reference orbit** for the centre
point c0, then iterating small **deltas** (dz) for each pixel using fast floating
arithmetic.

Key property
------------

We never form ``c = c0 + dc`` in float arithmetic. We compute ``dc`` directly from
pixel offsets and keep it as the pixel's coordinate. That avoids catastrophic
cancellation at high magnifications.

Implementation notes
--------------------

* This file is deliberately NumPy-only (no Numba) to avoid environment/tooling
  compatibility issues.
* To control RAM, rendering is done in **row tiles**.
* Uses ``numpy.longdouble`` for deltas where available. On Linux x86_64 this is
  typically 80-bit extended precision and supports extremely deep zoom scales
  (e.g. 1e-480 spans). On Windows it is often just float64, so ultra-deep zooms
  will underflow earlier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from mpmath import mp, mpf, mpc, log10
from PIL import Image


@dataclass(frozen=True)
class PerturbationSettings:
    # mp.dps heuristic
    extra_digits: int = 80
    max_dps: int = 5000

    # Row-tiling to cap memory usage.
    tile_rows: int = 64

    # Enable smooth colouring.
    smooth_colours: bool = True


def _set_mp_precision_for_reference(zoom: mpf, max_iter: int, extra_digits: int, max_dps: int) -> int:
    try:
        zdigits = int(log10(zoom))
    except Exception:
        zdigits = 500

    iter_term = int(max(0.0, math.log10(max(10, int(max_iter)))) * 25)
    dps = max(100, zdigits + extra_digits + iter_term)
    dps = min(dps, max_dps)
    mp.dps = dps
    return dps


def _compute_reference_orbit(center: Tuple[mpf, mpf], max_iter: int) -> Tuple[np.ndarray, np.ndarray]:
    cx, cy = center
    c0 = mpc(cx, cy)
    z = mpc(0, 0)

    xr = np.empty(max_iter + 1, dtype=np.longdouble)
    yi = np.empty(max_iter + 1, dtype=np.longdouble)

    for n in range(max_iter + 1):
        xr[n] = np.longdouble(z.real)
        yi[n] = np.longdouble(z.imag)
        z = z * z + c0

    return xr, yi


def _palette_from_mu(mu: np.ndarray, max_iter: int) -> np.ndarray:
    """Vectorised version of colour.py's palette. Returns uint8 RGB array."""
    t = mu / float(max_iter)
    t = np.power(t, 0.7, dtype=np.float64)

    # Use float64 trig even if mu was longdouble; output is uint8 anyway.
    a = 6.0 * math.pi * t
    r = 255.0 * (0.5 + 0.5 * np.sin(a))
    g = 255.0 * (0.5 + 0.5 * np.sin(a + 2.0 * math.pi / 3.0))
    b = 255.0 * (0.5 + 0.5 * np.sin(a + 4.0 * math.pi / 3.0))

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    return rgb


def render_perturbation_frame(
    *,
    center: Tuple[object, object],
    zoom: object,
    width: int,
    height: int,
    max_iter: int,
    frame_id: int | None = None,
    settings: PerturbationSettings | None = None,
) -> Image.Image:
    settings = settings or PerturbationSettings()

    cx, cy = center
    cx = mpf(cx)
    cy = mpf(cy)
    zoom_mpf = mpf(zoom)

    dps = _set_mp_precision_for_reference(zoom_mpf, max_iter, settings.extra_digits, settings.max_dps)

    try:
        zexp = int(log10(zoom_mpf))
        zmsg = f"1e{zexp}"
    except Exception:
        zmsg = str(zoom_mpf)

    with open("render.log", "a", encoding="utf-8", errors="ignore") as f:
        f.write(
            f"[Frame {frame_id}] Perturbation START zoom~{zmsg} mp.dps={dps} (w={width}, h={height}, it={max_iter})\n"
        )

    # Reference orbit for c0.
    x_ref, y_ref = _compute_reference_orbit((cx, cy), max_iter)

    # Compute pixel delta scales.
    aspect = np.longdouble(width) / np.longdouble(height)
    scale = mpf(4) / zoom_mpf
    scale_ld = np.longdouble(scale)
    if scale_ld == 0.0:
        msg = (
            f"[Frame {frame_id}] Perturbation FAIL: scale underflow at zoom={zoom_mpf}. "
            "Your platform's numpy.longdouble cannot represent this depth." 
        )
        with open("render.log", "a", encoding="utf-8", errors="ignore") as f:
            f.write(msg + "\n")
        raise RuntimeError(msg)

    vert_span = scale_ld
    horiz_span = scale_ld * aspect

    x_offsets = (np.arange(width, dtype=np.longdouble) - (np.longdouble(width) * 0.5)) * (horiz_span / np.longdouble(width))

    out = np.zeros((height, width, 3), dtype=np.uint8)

    tile = max(8, int(settings.tile_rows))
    for y0 in range(0, height, tile):
        y1 = min(height, y0 + tile)
        tile_h = y1 - y0

        y_offsets = (np.arange(y0, y1, dtype=np.longdouble) - (np.longdouble(height) * 0.5)) * (vert_span / np.longdouble(height))

        dc_re = np.broadcast_to(x_offsets, (tile_h, width)).copy()
        dc_im = np.broadcast_to(y_offsets.reshape(tile_h, 1), (tile_h, width)).copy()

        dz_re = np.zeros((tile_h, width), dtype=np.longdouble)
        dz_im = np.zeros((tile_h, width), dtype=np.longdouble)

        escaped = np.zeros((tile_h, width), dtype=bool)
        iters = np.zeros((tile_h, width), dtype=np.int32)

        # For smooth colouring we need z at escape.
        esc_zre = np.zeros((tile_h, width), dtype=np.longdouble)
        esc_zim = np.zeros((tile_h, width), dtype=np.longdouble)

        active = ~escaped

        for n in range(max_iter):
            if not active.any():
                break

            zr = x_ref[n]
            zi = y_ref[n]

            # Compute next dz for active pixels.
            ar = dz_re[active]
            ai = dz_im[active]
            dcr = dc_re[active]
            dci = dc_im[active]

            # dz^2
            ar2 = ar * ar
            ai2 = ai * ai
            arai = ar * ai

            # 2*z_ref*dz
            t_re = (zr * ar - zi * ai) * 2.0
            t_im = (zr * ai + zi * ar) * 2.0

            # dz_{n+1}
            nr = t_re + (ar2 - ai2) + dcr
            ni = t_im + (2.0 * arai) + dci

            dz_re[active] = nr
            dz_im[active] = ni

            # z = z_ref + dz
            zre = zr + dz_re
            zim = zi + dz_im

            # Escape test
            mag2 = zre * zre + zim * zim
            newly_escaped = (~escaped) & (mag2 > 4.0)
            if newly_escaped.any():
                escaped[newly_escaped] = True
                iters[newly_escaped] = n
                esc_zre[newly_escaped] = zre[newly_escaped]
                esc_zim[newly_escaped] = zim[newly_escaped]

            active = ~escaped

        # Colour the tile.
        inside = ~escaped
        out[y0:y1, :, :] = 0

        if escaped.any():
            if settings.smooth_colours:
                # mu = n + 1 - log(log(|z|))/log(2)
                zabs = np.sqrt((esc_zre[escaped].astype(np.float64) ** 2) + (esc_zim[escaped].astype(np.float64) ** 2))
                # guard against numerical oddities
                zabs = np.maximum(zabs, 1e-300)
                mu = iters[escaped].astype(np.float64) + 1.0 - (np.log(np.log(zabs)) / math.log(2.0))
            else:
                mu = iters[escaped].astype(np.float64)

            rgb = _palette_from_mu(mu, max_iter)
            tile_rgb = out[y0:y1, :, :]
            tile_rgb[escaped] = rgb

    img = Image.fromarray(out, mode="RGB")

    with open("render.log", "a", encoding="utf-8", errors="ignore") as f:
        f.write(f"[Frame {frame_id}] Perturbation DONE\n")

    return img
