# gpu_renderer.py

import numpy as np
from numba import cuda
from PIL import Image
import math
from mpmath import mp, mpf
import sys

# Increase mpmath precision for building the reference orbit (e.g. ~200–300 digits)
mp.dps = 300

# ────────────────────────────────────────────────────────────────────────────────
# 1) Original CUDA KERNEL: double‐precision Mandelbrot iteration (shallow zoom)
# ────────────────────────────────────────────────────────────────────────────────
# Each thread computes one pixel. We stop iterating once |z| > 2.0 or max_iter.
# The iteration count is stored in out_iters; later we map that to an RGB palette.
# ────────────────────────────────────────────────────────────────────────────────
@cuda.jit
def mandelbrot_kernel_double(
    center_x,   # float64
    center_y,   # float64
    zoom,       # float64
    width,      # int32
    height,     # int32
    max_iter,   # int32
    out_iters   # int32[:, :]
):
    """
    For each pixel (px,py), compute the corresponding complex coordinate (x0,y0)
    with correct aspect ratio, then run a double‐precision Mandelbrot iteration.
    """
    # Thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    # Pixel coordinates in [0 .. width-1], [0 .. height-1]
    px = bx * bw + tx
    py = by * bh + ty

    # If pixel is out of image bounds, exit
    if px >= width or py >= height:
        return

    # 1) Compute the vertical span in the complex plane (always 4.0/zoom):
    vert_span = 4.0 / zoom

    # 2) Compute the horizontal span so that (horiz_span / width) == (vert_span / height):
    horiz_span = vert_span * (width / height)

    # 3) Map pixel (px,py) to complex plane (x0,y0):
    x0 = center_x + (px - 0.5 * width)  * (horiz_span / width)
    y0 = center_y + (py - 0.5 * height) * (vert_span  / height)

    # 4) Iterate Mandelbrot in double‐precision
    x = 0.0
    y = 0.0
    iter_count = 0

    while (x * x + y * y <= 4.0) and (iter_count < max_iter):
        xt = x * x - y * y + x0
        yt = 2.0 * x * y + y0
        x = xt
        y = yt
        iter_count += 1

    # 5) Store the iteration count in out_iters (int32).
    out_iters[py, px] = iter_count


# ────────────────────────────────────────────────────────────────────────────────
# 2) Device‐side Double‐Double arithmetic (TwoSum / TwoProd: ~106‐bit mantissa)
# ────────────────────────────────────────────────────────────────────────────────
@cuda.jit(device=True)
def dd_add(a_hi, a_lo, b_hi, b_lo):
    """
    Double‐Double addition: (a_hi + a_lo) + (b_hi + b_lo) → (r_hi, r_lo)
    where r_hi retains the top 53 bits and r_lo holds the rounding error.
    """
    # TwoSum on high parts
    s = a_hi + b_hi
    v = s - a_hi
    t = (a_hi - (s - v)) + (b_hi - v)

    # Add low parts + the error from high‐part addition
    t += a_lo + b_lo

    # Renormalize
    s2 = s + t
    err = t + (s - s2)
    return s2, err


@cuda.jit(device=True)
def dd_mul(a_hi, a_lo, b_hi, b_lo):
    """
    Double‐Double multiplication: (a_hi + a_lo)*(b_hi + b_lo) → (r_hi, r_lo).
    Uses Dekker’s TwoProd algorithm to recover full 106‐bit result.
    """
    p = a_hi * b_hi

    # Dekker’s splitting for a_hi
    SPLIT = 2**27 + 1.0  # for IEEE 754 double
    c = SPLIT * a_hi
    ahi = c - (c - a_hi)
    alo = a_hi - ahi

    # Dekker’s splitting for b_hi
    d = SPLIT * b_hi
    bhi = d - (d - b_hi)
    blo = b_hi - bhi

    # Error term from high‐part multiply
    err_hi = ((ahi * bhi - p) + ahi * blo + alo * bhi) + alo * blo

    # Cross‐terms and low‐part multiply
    s = err_hi + (a_hi * b_lo + a_lo * b_hi + a_lo * b_lo)

    # Renormalize
    prod_hi = p + s
    v = prod_hi - p
    prod_lo = (p - (prod_hi - v)) + (s - v)
    return prod_hi, prod_lo


# ────────────────────────────────────────────────────────────────────────────────
# 3) CUDA KERNEL: perturbation + double‐double Mandelbrot (deep zoom)
# ────────────────────────────────────────────────────────────────────────────────
@cuda.jit
def mandelbrot_kernel_perturb(
    cx_hi, cx_lo,         # center_x in double‐double
    cy_hi, cy_lo,         # center_y in double‐double
    zoom_hi, zoom_lo,     # zoom in double‐double
    ref_x_hi, ref_x_lo,   # reference‐orbit real parts (double‐double arrays)
    ref_y_hi, ref_y_lo,   # reference‐orbit imag parts
    width, height,        # image dimensions (int)
    max_iter,             # iteration limit
    out_iters             # int32[:, :] output buffer
):
    """
    Each thread computes one pixel by iterating the 'delta' sequence:
      δ_{n+1} = 2*z_ref[n]*δ_n + (δ_n)^2 + Δc,
    where z_ref[n] is the high‐precision reference orbit (double‐double), and Δc is
    the small offset from the reference center to this pixel (in plain float).

    Inputs:
      • (cx_hi, cx_lo), (cy_hi, cy_lo): reference center in double‐double
      • (zoom_hi, zoom_lo): zoom in double‐double
      • ref_x_hi/lo[n], ref_y_hi/lo[n]: the reference orbit z_ref[n]
      • width, height, max_iter
    Output:
      • out_iters[py, px] = iteration count (int32)
    """
    # Thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    # Pixel coordinates
    px = bx * bw + tx
    py = by * bh + ty

    if px >= width or py >= height:
        return

    # 1) Reconstruct float zoom from double‐double for span calculation
    zf = float(zoom_hi + zoom_lo)
    vert_span = 4.0 / zf
    horiz_span = vert_span * (width / height)

    # 2) Compute Δc = (dx, dy) in plain floats
    dx = (px - 0.5 * width)  * (horiz_span / width)
    dy = (py - 0.5 * height) * (vert_span  / height)

    # 3) Form pixel’s c as double‐double: c_hi = cx_hi + dx, c_lo = cx_lo
    #    i.e. dd_add(cx_hi, cx_lo, dx, 0.0)
    cpx_hi, cpx_lo = dd_add(cx_hi, cx_lo, dx, 0.0)
    cpy_hi, cpy_lo = dd_add(cy_hi, cy_lo, dy, 0.0)

    # 4) Initialize δ₀ = 0 + i·0 (as plain double)
    delta_x = 0.0
    delta_y = 0.0

    # 5) Iterate
    for n in range(max_iter):
        # Load z_ref[n] from global memory (double‐double)
        zrx_hi = ref_x_hi[n]
        zrx_lo = ref_x_lo[n]
        zry_hi = ref_y_hi[n]
        zry_lo = ref_y_lo[n]

        # 5a) Compute (δ_n)^2 in plain double:
        dx2 = delta_x * delta_x - delta_y * delta_y
        dy2 = 2.0 * delta_x * delta_y

        # 5b) Compute 2 * z_ref_x * δ_x  (DD multiply)
        two_zrx_hi = 2.0 * zrx_hi
        two_zrx_lo = 2.0 * zrx_lo
        prod1_hi, prod1_lo = dd_mul(two_zrx_hi, two_zrx_lo, delta_x, 0.0)

        # 5c) Compute 2 * z_ref_y * δ_y  (DD multiply)
        two_zry_hi = 2.0 * zry_hi
        two_zry_lo = 2.0 * zry_lo
        prod2_hi, prod2_lo = dd_mul(two_zry_hi, two_zry_lo, delta_y, 0.0)

        # 5d) Form new δ_x as (prod1 - prod2 + dx2) in DD
        diff_hi, diff_lo = dd_add(prod1_hi, prod1_lo, -prod2_hi, -prod2_lo)
        ndx_hi, ndx_lo = dd_add(diff_hi, diff_lo, dx2, 0.0)
        delta_x = float(ndx_hi + ndx_lo)

        # 5e) Form new δ_y as (2*z_ref_x*δ_y + 2*z_ref_y*δ_x + dy2) in DD
        prod3_hi, prod3_lo = dd_mul(two_zrx_hi, two_zrx_lo, delta_y, 0.0)
        prod4_hi, prod4_lo = dd_mul(two_zry_hi, two_zry_lo, delta_x,  0.0)
        sum_hi, sum_lo = dd_add(prod3_hi, prod3_lo, prod4_hi, prod4_lo)
        ndy_hi, ndy_lo = dd_add(sum_hi, sum_lo, dy2, 0.0)
        delta_y = float(ndy_hi + ndy_lo)

        # 5f) Check escape: |(z_ref + δ)|^2 > 4?
        #    Compute z_ref + δ in DD, then convert to double for norm
        zrpx_hi, zrpx_lo = dd_add(zrx_hi, zrx_lo, delta_x, 0.0)
        zrpy_hi, zrpy_lo = dd_add(zry_hi, zry_lo, delta_y, 0.0)
        zx = float(zrpx_hi + zrpx_lo)
        zy = float(zrpy_hi + zrpy_lo)
        if zx*zx + zy*zy > 4.0:
            out_iters[py, px] = n
            return

    # 6) If we never escaped, mark as inside
    out_iters[py, px] = max_iter


# ────────────────────────────────────────────────────────────────────────────────
# 4) Helper: convert an mpf to a Double‐Double (two float64)
# ────────────────────────────────────────────────────────────────────────────────
def mpf_to_doubledouble(a: mpf) -> (float, float):
    """
    Convert a high-precision mpf 'a' to a double-double representation (hi, lo).
    hi = nearest float(a), lo = float(a - hi). Together they store ~2×53 bits.
    """
    f_hi = float(a)
    diff = a - mpf(f_hi)
    f_lo = float(diff)
    return f_hi, f_lo


# ────────────────────────────────────────────────────────────────────────────────
# 5) CPU‐side: build a high‐precision reference orbit for c_ref = (cx_mpf, cy_mpf)
# ────────────────────────────────────────────────────────────────────────────────
def build_reference_orbit(cx_mpf: mpf, cy_mpf: mpf, max_iter: int):
    """
    Iterate z_{n+1} = z_n^2 + (cx_mpf + i cy_mpf) in mpf, store each z_n as double-double.
    Returns four NumPy float64 arrays of length max_iter:
      ref_x_hi, ref_x_lo, ref_y_hi, ref_y_lo
    representing the real+imag parts of z_ref[n] in double-double form.
    """
    z = mpf(0) + mpf(0)*1j
    orbit = [z]
    for n in range(1, max_iter):
        z = z*z + (cx_mpf + cy_mpf*1j)
        orbit.append(z)
        if abs(z) > 2:
            break

    # If we escaped early, pad the rest with the last value
    last = orbit[-1]
    while len(orbit) < max_iter:
        orbit.append(last)

    # Convert to double-double arrays
    ref_x_hi = np.zeros(max_iter, dtype=np.float64)
    ref_x_lo = np.zeros(max_iter, dtype=np.float64)
    ref_y_hi = np.zeros(max_iter, dtype=np.float64)
    ref_y_lo = np.zeros(max_iter, dtype=np.float64)

    for n, z_n in enumerate(orbit):
        x_dd_hi, x_dd_lo = mpf_to_doubledouble(z_n.real)
        y_dd_hi, y_dd_lo = mpf_to_doubledouble(z_n.imag)
        ref_x_hi[n], ref_x_lo[n] = x_dd_hi, x_dd_lo
        ref_y_hi[n], ref_y_lo[n] = y_dd_hi, y_dd_lo

    return ref_x_hi, ref_x_lo, ref_y_hi, ref_y_lo


# ────────────────────────────────────────────────────────────────────────────────
# 6) Python wrapper: allocate buffers, decide kernel, launch, colorize → PIL Image
# ────────────────────────────────────────────────────────────────────────────────
def render_gpu_frame(center, zoom, width, height, max_iter):
    """
    center: (center_x, center_y) as Python floats (double precision)
    zoom:   Python float → determines spans
    width, height, max_iter: ints

    If zoom < 1e12, we use mandelbrot_kernel_double (fast, pure FP64).
    Otherwise, we run:
      1) Build the reference orbit on CPU (mpf→DD).
      2) Upload reference‐orbit arrays and the center/zoom as DD to GPU.
      3) Launch mandelbrot_kernel_perturb (all in FP64 & DD).
    Finally, colorize via the same sinusoidal palette and return a PIL Image.
    """
    center_x, center_y = center

    # 1) Allocate host‐side array for iteration counts
    iters_host = np.zeros((height, width), dtype=np.int32)

    # 2) Decide which path to take:
    zf = float(zoom)
    if zf < 1e12:
        # ── FAST PATH: double‐precision GPU
        iters_device = cuda.device_array_like(iters_host)

        threads_per_block = (16, 16)
        blocks_x = math.ceil(width  / threads_per_block[0])
        blocks_y = math.ceil(height / threads_per_block[1])
        blocks_per_grid = (blocks_x, blocks_y)

        mandelbrot_kernel_double[blocks_per_grid, threads_per_block](
            float(center_x),
            float(center_y),
            float(zoom),
            int(width),
            int(height),
            int(max_iter),
            iters_device
        )

        iters_device.copy_to_host(iters_host)

    else:
        # ── DEEP PATH: perturbation + double‐double on GPU

        # 1a) Build the reference orbit (mpf) on the CPU
        cx_mpf = mpf(center_x)
        cy_mpf = mpf(center_y)
        ref_x_hi, ref_x_lo, ref_y_hi, ref_y_lo = build_reference_orbit(cx_mpf, cy_mpf, max_iter)

        # 1b) Convert center and zoom to double‐double
        cx_hi, cx_lo = mpf_to_doubledouble(cx_mpf)
        cy_hi, cy_lo = mpf_to_doubledouble(cy_mpf)
        zoom_mpf = mpf(zoom)
        zh_hi, zh_lo = mpf_to_doubledouble(zoom_mpf)

        # 2) Allocate & copy reference‐orbit arrays to device
        d_ref_x_hi = cuda.to_device(ref_x_hi)
        d_ref_x_lo = cuda.to_device(ref_x_lo)
        d_ref_y_hi = cuda.to_device(ref_y_hi)
        d_ref_y_lo = cuda.to_device(ref_y_lo)

        # 3) Prepare output buffer on device
        iters_device = cuda.device_array_like(iters_host)

        # 4) Launch perturbation kernel
        threads_per_block = (16, 16)
        blocks_x = math.ceil(width  / threads_per_block[0])
        blocks_y = math.ceil(height / threads_per_block[1])
        blocks_per_grid = (blocks_x, blocks_y)

        mandelbrot_kernel_perturb[blocks_per_grid, threads_per_block](
            float(cx_hi), float(cx_lo),    # center (double‐double)
            float(cy_hi), float(cy_lo),
            float(zh_hi), float(zh_lo),    # zoom (double‐double)
            d_ref_x_hi, d_ref_x_lo,        # ref orbit real
            d_ref_y_hi, d_ref_y_lo,        # ref orbit imag
            int(width), int(height),
            int(max_iter),
            iters_device
        )

        iters_device.copy_to_host(iters_host)

    # ─────────────────────────────────────────────────────────────────────────────
    # 3) Build RGB palette (sinusoidal) exactly as before
    # ─────────────────────────────────────────────────────────────────────────────
    escaped_mask = (iters_host < max_iter)
    n_vals = iters_host.astype(np.float32)
    raw_t = n_vals / float(max_iter)
    t = np.power(raw_t, 0.7, dtype=np.float32)
    img_arr = np.zeros((height, width, 3), dtype=np.uint8)

    sin_arg = 6.0 * math.pi * t
    two_pi_over_3  = 2.0 * math.pi / 3.0
    four_pi_over_3 = 4.0 * math.pi / 3.0

    r_float = (0.5 + 0.5 * np.sin(sin_arg)).astype(np.float32)
    g_float = (0.5 + 0.5 * np.sin(sin_arg + two_pi_over_3)).astype(np.float32)
    b_float = (0.5 + 0.5 * np.sin(sin_arg + four_pi_over_3)).astype(np.float32)

    r_8bit = (r_float * 255.0).astype(np.uint8)
    g_8bit = (g_float * 255.0).astype(np.uint8)
    b_8bit = (b_float * 255.0).astype(np.uint8)

    img_arr[..., 0][escaped_mask] = r_8bit[escaped_mask]
    img_arr[..., 1][escaped_mask] = g_8bit[escaped_mask]
    img_arr[..., 2][escaped_mask] = b_8bit[escaped_mask]

    # 4) Convert to PIL Image and return
    return Image.fromarray(img_arr, mode="RGB")


# ────────────────────────────────────────────────────────────────────────────────
# 7) Decide if we can safely use double‐precision GPU
# ────────────────────────────────────────────────────────────────────────────────
def zoom_fits_double(zoom):
    """
    Return True if `zoom < 1e12`, meaning double precision still has enough mantissa bits.
    Otherwise, return False → use perturbation path.
    """
    return zoom < 1e12
