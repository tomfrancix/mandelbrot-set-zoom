# color.py

import numpy as np
from mpmath import log as mplog

def get_smooth_color(z, n, max_iter):
    if n >= max_iter:
        return (0, 0, 0)

    try:
        mu = n + 1 - float(mplog(mplog(abs(z))) / mplog(2))
        t = mu / max_iter
        t = t ** 0.7  # nonlinear stretch to improve contrast

        r = int(255 * (0.5 + 0.5 * np.sin(6 * np.pi * t)))
        g = int(255 * (0.5 + 0.5 * np.sin(6 * np.pi * t + 2 * np.pi / 3)))
        b = int(255 * (0.5 + 0.5 * np.sin(6 * np.pi * t + 4 * np.pi / 3)))

        return (r, g, b)
    except Exception as e:
        with open("render.log", "a") as f:
            f.write(f"[COLOR ERROR] z={z}, n={n}, Error={e}\n")
        return (255, 0, 0)  # fallback to red
