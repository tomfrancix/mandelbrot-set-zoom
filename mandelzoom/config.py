import json
from typing import Any, Dict, Optional

def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError("Config JSON must be an object.")
        return cfg

    try:
        import importlib
        legacy = importlib.import_module("config")
        cfg = {k: getattr(legacy, k) for k in dir(legacy) if k.isupper()}
        mapping = {
            "WIDTH": "width",
            "HEIGHT": "height",
            "FPS": "fps",
            "TOTAL_FRAMES": "total_frames",
            "START_ZOOM": "start_zoom",
            "END_ZOOM": "end_zoom",
            "CENTER": "center",
            "MAX_ITER": "max_iter",
            "FRAMES_DIR": "frames_dir",
            "OUTPUT_VIDEO": "output_video",
            "RENDER_EVERY_N": "render_every_n",
            "MIN_PRECISION_DIGITS": "min_precision_digits",
            "MAX_PRECISION_DIGITS": "max_precision_digits",
        }
        out: Dict[str, Any] = {}
        for k, v in cfg.items():
            if k in mapping:
                out[mapping[k]] = v
        return out
    except Exception:
        return {
            "width": 1920,
            "height": 1080,
            "fps": 60,
            "total_frames": 600,
            "start_zoom": 1.0,
            "end_zoom": 1e6,
            "center": [-0.75, 0.0],
            "max_iter": 1000,
            "frames_dir": "frames",
            "output_video": "mandelbrot_zoom.mp4",
            "render_every_n": 1,
            "min_precision_digits": 50,
            "max_precision_digits": 3000,
        }

def normalise_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    required = ["width", "height", "total_frames", "start_zoom", "end_zoom", "center", "max_iter", "frames_dir"]
    for r in required:
        if r not in cfg:
            raise ValueError(f"Missing config field: {r}")

    width = int(cfg["width"])
    height = int(cfg["height"])
    total_frames = int(cfg["total_frames"])
    if width <= 0 or height <= 0 or total_frames <= 0:
        raise ValueError("width/height/total_frames must be positive.")

    center = cfg["center"]
    if not (isinstance(center, (list, tuple)) and len(center) == 2):
        raise ValueError("center must be [re, im].")

    out = dict(cfg)
    out["width"] = width
    out["height"] = height
    out["total_frames"] = total_frames
    out["fps"] = int(cfg.get("fps", 60))
    out["frames_dir"] = str(cfg.get("frames_dir", "frames"))
    out["output_video"] = str(cfg.get("output_video", "output.mp4"))
    out["render_every_n"] = int(cfg.get("render_every_n", 1))
    out["min_precision_digits"] = int(cfg.get("min_precision_digits", 50))
    out["max_precision_digits"] = int(cfg.get("max_precision_digits", 3000))
    return out
