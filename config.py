# config.py
#
# “10 minute deep‐zoom” edition: we compute a ZOOM_END ~ 10^480
# so that each of the 36 000 frames multiplies zoom by exactly the
# same factor as in your 10 s (600‐frame) version that went from 1→1e8.
#
# - Uses mpmath with high precision (200 digits) to avoid overflow.
# - START_CENTER == END_CENTER = [YOUR CLICKED COORDINATE: -1.156133259 + (-0.278616381)i].
# - DURATION_SEC = 600 → TOTAL_FRAMES = 600 s × 60 fps = 36 000 frames.

from mpmath import mp, mpf

# ─────────────────────────────────────────────────────────────────────────────────────────
# Increase mpmath precision so we can safely compute 10^480 etc.
mp.dps = 200

# -----------------------------------------------------------------------------
# OUTPUT_DIR: where to write PNG frames
# WIDTH, HEIGHT: pixel dimensions (e.g. 3840×2160 for 4K UHD)
# MAX_ITER: iteration limit for Mandelbrot
# ZOOM_START: initial zoom (span = 4/zoom)
# ZOOM_END: final zoom (computed below)
# DURATION_SEC: how many seconds long (at 60 fps)
# START_CENTER, END_CENTER: complex centers (mpf tuples or floats)
# -----------------------------------------------------------------------------

OUTPUT_DIR   = "frames"
WIDTH        = 3840
HEIGHT       = 2160
MAX_ITER     = 1500  # you may consider raising this (see notes below)

# The “old” 10 s version ended at 1e8 (10^8) over 600 frames (10 s × 60 fps = 600 frames).
ZOOM_START   = mpf(1)           # start magnification

# We will stretch to 600 s (10 minutes) → 36 000 frames (600 s × 60 fps).
DURATION_SEC = 600              # 10 minutes
TOTAL_FRAMES = DURATION_SEC * 60  # 36 000

# Compute the new ZOOM_END so that per‐frame multiplier r is identical:
#   - Old r = (10^8)^(1/(600−1))
#   - New ZOOM_END = r^(36 000−1)
old_zoom_end = mpf("1e8")           # what the old 10 s version ended at
old_frames   = mpf(600)             # 10 s × 60 fps = 600 frames
new_frames   = mpf(TOTAL_FRAMES)    # 10 min × 60 fps = 36 000 frames

# r = (old_zoom_end)^(1/(600−1))
r = old_zoom_end ** ( mpf(1) / (old_frames - mpf(1)) )

# Now raise r to (36 000−1) to get the new final zoom:
ZOOM_END = r ** (new_frames - mpf(1))
# Equivalently, ZOOM_END ≈ 10^(8 × 36 000/599) ≈ 10^480.78798…

# -----------------------------------------------------------------------------
# Panning / Center: No pan—pure zoom in place at the coordinates you clicked:
# (Both START_CENTER and END_CENTER must be identical for in‐place zoom.)
# -----------------------------------------------------------------------------
START_CENTER = (mpf("-1.156133259"), mpf("-0.278616381"))
END_CENTER   = START_CENTER        # no panning

# (TOTAL_FRAMES is defined here only for clarity; main.py will recompute it.)
#─────────────────────────────────────────────────────────────────────────────────────────
