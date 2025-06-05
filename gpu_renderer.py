# gpu_renderer.py

import numpy as np
from numba import cuda
from PIL import Image
import math

# ------------------------------------------------------------
# 1) CUDA KERNEL: double‐precision Mandelbrot iteration numbers
# ------------------------------------------------------------
# Each thread computes one pixel. We stop iterating once |z| > 2.0 or max_iter.
# The iteration count is stored in out_iters; later we map that to an RGB palette.
# ------------------------------------------------------------
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
    For each pixel (px,py), compute the corresponding complex coordinate (x0,y0).
    We fix the VERTICAL span = 4.0/zoom, then choose HORIZONTAL span = (4.0/zoom)*(width/height).
    That way, each pixel in the output image is 'square' in complex‐plane units,
    no matter what the aspect ratio (width/height) is.

    old method (bad, squeezes):
        span = 4.0/zoom
        x0 = center_x + (px - 0.5*width)  * (span/width)
        y0 = center_y + (py - 0.5*height) * (span/height)

    new method (correct, keeps 1:1 aspect):
        vert_span  = 4.0 / zoom
        horiz_span = vert_span * (width / height)
        x0 = center_x + (px - 0.5*width)  * (horiz_span / width)
        y0 = center_y + (py - 0.5*height) * (vert_span  / height)
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
    #    i.e. horiz_span = vert_span * (width / height)
    horiz_span = vert_span * (width / height)

    # 3) Map pixel (px,py) to complex plane (x0,y0):
    #    • x0 ranges from center_x - horiz_span/2  … center_x + horiz_span/2
    #    • y0 ranges from center_y - vert_span/2   … center_y + vert_span/2
    x0 = center_x + (px - 0.5 * width)  * (horiz_span / width)
    y0 = center_y + (py - 0.5 * height) * (vert_span  / height)

    # 4) Iterate Mandelbrot:
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


# ------------------------------------------------------------
# 2) Python wrapper: allocate buffers, launch kernel, colorize → PIL Image
# ------------------------------------------------------------
def render_gpu_frame(center, zoom, width, height, max_iter):
    """
    center: (center_x, center_y) as Python floats
    zoom:   Python float → determines spans = 4.0/zoom
    width, height, max_iter: ints

    Returns a PIL.Image in mode='RGB', colored with a sinusoidal palette:
    • Inside the set (n == max_iter) => (0,0,0) black
    • Escaped (n < max_iter) => a vibrant RGB color based on a smoothed iteration count.
    """

    center_x, center_y = center  # Python floats

    # 1) Allocate an int32 array on the host (height × width) to hold iteration counts
    iters_host = np.zeros((height, width), dtype=np.int32)

    # 2) Allocate a device array of the same shape & dtype
    iters_device = cuda.device_array_like(iters_host)

    # 3) Configure CUDA kernel launch (16×16 threads per block)
    threads_per_block = (16, 16)
    blocks_x = math.ceil(width  / threads_per_block[0])
    blocks_y = math.ceil(height / threads_per_block[1])
    blocks_per_grid = (blocks_x, blocks_y)

    # 4) Launch the kernel with corrected aspect‐ratio mapping
    mandelbrot_kernel_double[blocks_per_grid, threads_per_block](
        float(center_x),   # cast to float64
        float(center_y),
        float(zoom),
        int(width),
        int(height),
        int(max_iter),
        iters_device
    )

    # 5) Copy iteration counts back to host
    iters_device.copy_to_host(iters_host)

    # 6) Build an RGB “rainbow” palette exactly as before
    #    - Black for any pixel where iter_count == max_iter
    #    - Sinusoidal coloring for iter_count < max_iter

    # Create a boolean mask: True where the point “escaped”
    escaped_mask = (iters_host < max_iter)

    # Convert iteration counts to float32 for palette math
    n_vals = iters_host.astype(np.float32)

    # 6a) Normalize iteration counts to [0..1]
    raw_t = n_vals / float(max_iter)

    # 6b) Apply a contrast stretch: t = raw_t ^ 0.7
    t = np.power(raw_t, 0.7, dtype=np.float32)

    # 6c) Prepare an empty (height × width × 3) array for RGB output
    img_arr = np.zeros((height, width, 3), dtype=np.uint8)

    # 6d) Build sinusoids:  sin_arg = 6π * t
    sin_arg = 6.0 * math.pi * t  # float32 array

    # Precompute the phase shifts for G and B channels
    two_pi_over_3  = 2.0 * math.pi / 3.0
    four_pi_over_3 = 4.0 * math.pi / 3.0

    # 6e) Compute each channel in float form [0..1], then scale to [0..255]
    r_float = (0.5 + 0.5 * np.sin(sin_arg)).astype(np.float32)
    g_float = (0.5 + 0.5 * np.sin(sin_arg + two_pi_over_3)).astype(np.float32)
    b_float = (0.5 + 0.5 * np.sin(sin_arg + four_pi_over_3)).astype(np.float32)

    r_8bit = (r_float * 255.0).astype(np.uint8)
    g_8bit = (g_float * 255.0).astype(np.uint8)
    b_8bit = (b_float * 255.0).astype(np.uint8)

    # 6f) Fill only “escaped” pixels with color; “inside‐set” remain (0,0,0)
    img_arr[..., 0][escaped_mask] = r_8bit[escaped_mask]
    img_arr[..., 1][escaped_mask] = g_8bit[escaped_mask]
    img_arr[..., 2][escaped_mask] = b_8bit[escaped_mask]

    # 7) Convert NumPy array → PIL Image (RGB) and return
    img = Image.fromarray(img_arr, mode="RGB")
    return img


# ------------------------------------------------------------
# 3) Helper: decide if we can safely use double‐precision GPU
# ------------------------------------------------------------
def zoom_fits_double(zoom):
    """
    Return True if `zoom < 1e12`, meaning double precision still has enough mantissa bits
    so that “span = 4.0/zoom” is representable without catastrophic rounding. Otherwise,
    fall back to a CPU/mpmath renderer.
    """
    return zoom < 1e12
