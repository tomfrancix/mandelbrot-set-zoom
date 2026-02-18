# Mandelbrot Set Zoom

![A snapshot](image-2.png)

This repo renders Mandelbrot deep-zoom frame sequences and encodes them into video.

---
# Mandelbrot Deep Zoom Renderer


A high-precision Mandelbrot set renderer written in Python that zooms smoothly from the classic view into Seahorse Valley — all the way to 1e12 magnification. The system supports parallelized frame rendering, smooth color gradients based on log-escape time, adaptive precision, and ffmpeg video encoding.

![A snapshot](image-1.png)

## Features

- Deep Zoom: Zooms from 1× to 1e12 into the Mandelbrot set
- Adaptive Precision: Uses `mpmath` to dynamically scale decimal precision for extreme zooms
- Smooth Coloring: Trigonometric gradient palette based on smoothed iteration counts
- Multiprocessing: Renders each frame in parallel by row for full CPU utilization
- Video Output: Uses `ffmpeg` to compile thousands of PNG frames into a high-quality MP4
- Failsafe Logging: Logs stalls, precision issues, and pixel errors to `render.log`

## Requirements

Install dependencies with:

```
pip install -r requirements.txt
```

Python 3.9+ recommended. Optional: install `gmpy2` for much faster precision math via optimized backend.

## Usage

### 1. Configure the Zoom

Edit `config.py` to customize:

```
START_CENTER = (-0.5, 0.0)
END_CENTER = (-0.743643887037151, 0.13182590420533)
FRAMES = 1800
ZOOM_START = 1e0
ZOOM_END = 1e12
WIDTH = 1920
HEIGHT = 1080
MAX_ITER = 1000
OUTPUT_DIR = "output"
```

### 2. Run the Renderer

```
python main.py
```

This will:
- Generate frames in `output/`
- Log rendering details to `render.log`
- Create `mandelbrot_zoom.mp4` using `ffmpeg`

## Video Encoding Only

To regenerate the video from frames:

```
from main import generate_video
generate_video()
```

## Unified CLI (recommended)

```bash
pip install -r requirements-core.txt
# optional:
# pip install -r requirements-gpu.txt
# pip install -r requirements-video.txt

python -m mandelzoom render --config config.sample.json --renderer auto
python -m mandelzoom encode --config config.sample.json
```

`render_every_n > 1` enables **preview interpolation** (crop/resample between keyframes). It is lossy; use `1` for true rendering.

Each render writes `artifacts/run.json` capturing config + environment details.


## Troubleshooting

- If rendering is slow, check if `gmpy2` is installed.
- If red pixels appear, inspect `render.log` for precision or overflow issues.
- Existing frames are skipped by default when rerunning.

## License

MIT License
