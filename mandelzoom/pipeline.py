from __future__ import annotations

import os
import math
from typing import Dict, Any, Tuple, Optional

from PIL import Image

from mandelzoom.util.logging_setup import get_logger
from mandelzoom.renderers.cpu_mpmath import render_frame_cpu
from mandelzoom.renderers.gpu import render_frame_gpu_double, render_frame_gpu_extreme, probe_cuda

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def _exp_lerp(a: float, b: float, t: float) -> float:
    if a <= 0 or b <= 0:
        raise ValueError("Zoom values must be positive.")
    la = math.log(a)
    lb = math.log(b)
    return math.exp(_lerp(la, lb, t))

def _save_frame(img: Image.Image, frames_dir: str, frame_index: int) -> str:
    path = os.path.join(frames_dir, f"frame_{frame_index:06d}.png")
    img.save(path, format="PNG", optimize=True)
    return path

def _crop_resample(img: Image.Image, factor: float, width: int, height: int) -> Image.Image:
    if factor <= 0:
        raise ValueError("factor must be > 0")
    if factor >= 1.0:
        resized = img.resize((max(1, int(width / factor)), max(1, int(height / factor))), Image.LANCZOS)
        canvas = Image.new("RGB", (width, height))
        x0 = (width - resized.size[0]) // 2
        y0 = (height - resized.size[1]) // 2
        canvas.paste(resized, (x0, y0))
        return canvas
    crop_w = max(1, int(width * factor))
    crop_h = max(1, int(height * factor))
    x0 = (width - crop_w) // 2
    y0 = (height - crop_h) // 2
    cropped = img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    return cropped.resize((width, height), Image.LANCZOS)

def choose_renderer(*, renderer: str, zoom: float) -> str:
    if renderer in ("cpu", "gpu", "gpu-extreme"):
        return renderer
    if renderer != "auto":
        raise ValueError("renderer must be one of: auto, cpu, gpu, gpu-extreme")

    cuda_info = probe_cuda()
    if not cuda_info.get("available"):
        return "cpu"

    try:
        from gpu_renderer import zoom_fits_double as zfd  # type: ignore
        if zfd(zoom):
            return "gpu"
    except Exception:
        pass

    try:
        from extreme_gpu_renderer import zoom_fits_double as zfd2  # type: ignore
        if zfd2(zoom):
            return "gpu-extreme"
    except Exception:
        pass

    return "cpu"

def renderer_info(resolved: str, zoom: float) -> Dict[str, Any]:
    info = {"resolved": resolved, "zoom": zoom}
    info.update({"cuda": probe_cuda()})
    return info

def render_sequence(*, cfg: Dict[str, Any], renderer: str, log_queue, log_level: int) -> Dict[str, Any]:
    logger = get_logger()

    width = int(cfg["width"])
    height = int(cfg["height"])
    total_frames = int(cfg["total_frames"])
    start_zoom = float(cfg["start_zoom"])
    end_zoom = float(cfg["end_zoom"])
    center = (float(cfg["center"][0]), float(cfg["center"][1]))
    max_iter = int(cfg["max_iter"])
    frames_dir = str(cfg["frames_dir"])
    render_every_n = int(cfg.get("render_every_n", 1))

    _ensure_dir(frames_dir)

    logger.info("Render start total_frames=%s size=%sx%s zoom=%s..%s every_n=%s renderer=%s",
                total_frames, width, height, start_zoom, end_zoom, render_every_n, renderer)

    last_key_img: Optional[Image.Image] = None
    last_key_index: Optional[int] = None
    last_key_zoom: Optional[float] = None

    for i in range(total_frames):
        t = 0.0 if total_frames == 1 else i / (total_frames - 1)
        zoom = _exp_lerp(start_zoom, end_zoom, t)
        is_key = (render_every_n <= 1) or (i % render_every_n == 0) or (i == total_frames - 1)

        if is_key:
            resolved = choose_renderer(renderer=renderer, zoom=zoom)
            frame_id = f"{i:06d}"

            if resolved == "cpu":
                img = render_frame_cpu(
                    center=center, zoom=zoom, width=width, height=height, max_iter=max_iter,
                    frame_id=frame_id, log_queue=log_queue, log_level=log_level
                )
            elif resolved == "gpu":
                img = render_frame_gpu_double(center=center, zoom=zoom, width=width, height=height, max_iter=max_iter, frame_id=frame_id)
            else:
                img = render_frame_gpu_extreme(center=center, zoom=zoom, width=width, height=height, max_iter=max_iter, frame_id=frame_id)

            path = _save_frame(img, frames_dir, i)
            last_key_img, last_key_index, last_key_zoom = img, i, zoom
            logger.info("Saved key frame %s -> %s (renderer=%s zoom=%s)", i, path, resolved, zoom)
            continue

        if last_key_img is None or last_key_index is None or last_key_zoom is None:
            raise RuntimeError("Interpolation requested before first keyframe rendered.")

        ratio = zoom / last_key_zoom
        factor = 1.0 / ratio
        img = _crop_resample(last_key_img, factor=factor, width=width, height=height)
        path = _save_frame(img, frames_dir, i)
        logger.info("Saved preview frame %s -> %s (from key=%s ratio=%s)", i, path, last_key_index, ratio)

    logger.info("Render complete frames_dir=%s", frames_dir)
    return {"frames_dir": frames_dir, "total_frames": total_frames, "width": width, "height": height, "render_every_n": render_every_n}
