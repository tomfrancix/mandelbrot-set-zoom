from __future__ import annotations

import argparse
import logging
import os
import subprocess
from typing import Optional

from mandelzoom.config import load_config, normalise_config
from mandelzoom.pipeline import render_sequence, renderer_info, choose_renderer
from mandelzoom.util.logging_setup import configure_root_logging, create_log_queue, start_queue_listener, get_logger
from mandelzoom.util.manifest import build_manifest, write_manifest
from mandelzoom.video.opencv_writer import encode_with_opencv

def _git_commit() -> Optional[str]:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return None

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mandelzoom", description="Mandelbrot zoom renderer (CPU/GPU) with reproducible runs.")
    p.add_argument("--config", type=str, default=None, help="Path to config JSON. If omitted, uses legacy config.py.")
    p.add_argument("--renderer", type=str, default="auto", choices=["auto", "cpu", "gpu", "gpu-extreme"], help="Renderer selection.")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Log level.")
    p.add_argument("--log-file", type=str, default="render.log", help="Log file path (rotating). Set empty to disable file logging.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="Render frames to the frames directory.")
    r.add_argument("--frames-dir", type=str, default=None, help="Override frames_dir from config.")

    e = sub.add_parser("encode", help="Encode frames into an MP4 video using OpenCV.")
    e.add_argument("--input-dir", type=str, default=None, help="Frames directory (defaults to config.frames_dir).")
    e.add_argument("--output", type=str, default=None, help="Output MP4 file (defaults to config.output_video).")
    e.add_argument("--fps", type=int, default=None, help="Frames per second (defaults to config.fps).")

    return p

def main(argv: Optional[list] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_file = args.log_file if args.log_file and args.log_file.strip() else None
    listener_logger = configure_root_logging(level=log_level, console=True, log_file=log_file)

    queue = create_log_queue()
    listener = start_queue_listener(queue, listener_logger)

    logger = get_logger()

    try:
        cfg = normalise_config(load_config(args.config))

        if args.cmd == "render":
            if args.frames_dir:
                cfg["frames_dir"] = args.frames_dir

            start_zoom = float(cfg["start_zoom"])
            resolved_for_manifest = choose_renderer(renderer=args.renderer, zoom=start_zoom)
            rinfo = renderer_info(resolved_for_manifest, start_zoom)

            render_sequence(cfg=cfg, renderer=args.renderer, log_queue=queue, log_level=log_level)

            manifest = build_manifest(config=cfg, renderer_info=rinfo, git_commit=_git_commit())
            write_manifest(os.path.join("artifacts", "run.json"), manifest)
            logger.info("Run manifest written: artifacts/run.json")
            return 0

        if args.cmd == "encode":
            input_dir = args.input_dir or str(cfg.get("frames_dir", "frames"))
            output = args.output or str(cfg.get("output_video", "output.mp4"))
            fps = args.fps or int(cfg.get("fps", 60))

            encode_with_opencv(input_dir=input_dir, output_file=output, fps=fps)
            return 0

        raise RuntimeError("Unknown command.")
    finally:
        try:
            listener.stop()
        except Exception:
            pass
