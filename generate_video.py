#!/usr/bin/env python3
"""
make_video.py

This script collects all image files in a specified folder (default: "frames/"),
sorts them in lexicographical order, and stitches them into a single MP4 video
(using OpenCV's VideoWriter). By default, it writes "output.mp4" at 60 FPS.

Usage:
    python make_video.py \
        --input_dir frames \
        --output_file mandelbrot_zoom.mp4 \
        --fps 60

Requirements:
    pip install opencv-python
"""

import os
import sys
import argparse
import glob

import cv2  # OpenCV for video writing
from natsort import natsorted  # Natural sorting (e.g. "frame_2.png" before "frame_10.png")
from PIL import Image

def parse_args():
    p = argparse.ArgumentParser(
        description="Stitch a sequence of images (PNG, JPG, etc.) into an MP4 video."
    )
    p.add_argument(
        "-i", "--input_dir",
        type=str,
        default="frames",
        help="Path to the folder containing image frames (default: \"frames\")."
    )
    p.add_argument(
        "-o", "--output_file",
        type=str,
        default="output.mp4",
        help="Output video filename (e.g. \"output.mp4\")."
    )
    p.add_argument(
        "-f", "--fps",
        type=int,
        default=15,
        help="Frames per second for the output video (default: 60)."
    )
    p.add_argument(
        "--ext",
        type=str,
        default="png",
        help="Image file extension to look for (default: png). "
             "You can also use a comma‐separated list like \"png,jpg,jpeg\"."
    )
    return p.parse_args()

def collect_image_paths(input_dir, extensions):
    """
    Walk `input_dir` (non-recursively) and collect all files whose extension
    matches one of `extensions`. Return a naturally sorted list of file paths.
    """
    all_paths = []
    for ext in extensions:
        # e.g. ext = "png" → search for "*.png" in input_dir
        pattern = os.path.join(input_dir, f"*.{ext}")
        found = glob.glob(pattern)
        all_paths.extend(found)

    if not all_paths:
        print(f"ERROR: No image files with extensions {extensions} found in '{input_dir}'.", file=sys.stderr)
        sys.exit(1)

    # Use natural sorting (so that "frame_2.png" < "frame_10.png")
    all_paths = natsorted(all_paths)
    return all_paths

def main():
    args = parse_args()

    input_dir = args.input_dir
    output_file = args.output_file
    fps = args.fps
    exts = [e.strip().lower() for e in args.ext.split(",")]

    # 1) Verify that input_dir exists and is a directory
    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory '{input_dir}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    # 2) Collect all image paths
    img_paths = collect_image_paths(input_dir, exts)

    # 3) Read the first image to get frame size
    first_img_path = img_paths[0]
    try:
        img0 = Image.open(first_img_path)
    except Exception as e:
        print(f"ERROR: Could not open image '{first_img_path}': {e}", file=sys.stderr)
        sys.exit(1)

    width, height = img0.size
    img0.close()

    print(f"Found {len(img_paths)} images in '{input_dir}'.")
    print(f"Frame size: {width}×{height}. FPS: {fps}.")
    print(f"Writing video to '{output_file}'...")

    # 4) Prepare OpenCV VideoWriter
    #    FourCC code for 'mp4v' (most reliable for .mp4). If you have H264 support in your
    #    OpenCV build, you could also use 'H264' or 'avc1', but 'mp4v' usually works.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print("ERROR: Could not open video writer. Check if OpenCV has proper codecs installed.", file=sys.stderr)
        sys.exit(1)

    # 5) Iterate over all images, read with OpenCV, and write to video
    for idx, img_path in enumerate(img_paths):
        # Read as BGR (OpenCV default). If reading with PIL is preferred, convert to BGR afterward.
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"WARNING: Skipping '{img_path}', could not read as an image.", file=sys.stderr)
            continue

        # Verify dimensions match
        h, w = frame.shape[:2]
        if (w, h) != (width, height):
            print(f"ERROR: Image '{img_path}' has size {w}×{h}, expected {width}×{height}.", file=sys.stderr)
            video_writer.release()
            sys.exit(1)

        video_writer.write(frame)

        if (idx + 1) % 100 == 0 or idx == len(img_paths) - 1:
            print(f"  • Written frame {idx + 1}/{len(img_paths)}")

    # 6) Release resource
    video_writer.release()
    print("Video writing complete. You can now play:", output_file)

if __name__ == "__main__":
    main()
