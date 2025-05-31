# utils.py

import numpy as np
from mpmath import mp, log10

try:
    import gmpy2
    print("[INFO] gmpy2 is installed. mpmath will use it automatically.")
    with open("render.log", "a") as f:
        f.write("[INFO] gmpy2 is installed. Using optimized backend.\n")
except ImportError:
    print("[WARN] gmpy2 is NOT installed. Using slower pure Python backend.")
    with open("render.log", "a") as f:
        f.write("[WARN] gmpy2 is NOT installed. Using pure Python fallback.\n")

def set_precision(zoom_level):
    digits = int(log10(zoom_level)) + 50
    mp.dps = max(digits, 100)
    with open("render.log", "a") as f:
        f.write(f"[DEBUG] Precision set to {mp.dps} decimal places for zoom={zoom_level}\n")
