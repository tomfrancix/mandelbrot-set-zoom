#!/usr/bin/env python3
"""
reverse_speedup.py

A script to load an input video, play it backwards at a specified speed factor,
and write out a new output video file.

Usage:
    python reverse_speedup.py \
        --input output.mp4 \
        --output output_reversed_5x.mp4 \
        --factor 5

Requirements:
    - Python 3.8+
    - moviepy (pip install moviepy)
    - ffmpeg installed and on your system PATH
"""

import os
import sys
import argparse
from moviepy.editor import VideoFileClip, vfx
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def process_video(input_path: str, output_path: str, speed_factor: float = 5.0) -> None:
    """
    Load a video, reverse it, speed it up, and write to a new file.

    Parameters:
    -----------
    input_path : str
        Path to the input video file (e.g. "output.mp4").
    output_path : str
        Path where the processed video will be written (e.g. "output_reversed_5x.mp4").
    speed_factor : float
        By how many times to accelerate playback (both video and audio).
        Must be > 1.0. Default is 5.0.
    """

    # 1) Validate inputs
    if not os.path.isfile(input_path):
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if speed_factor <= 0:
        print(f"[ERROR] Speed factor must be > 0. Received: {speed_factor}", file=sys.stderr)
        sys.exit(1)

    # 2) Load the clip
    print(f"[INFO] Loading video from: {input_path}")
    try:
        clip = VideoFileClip(input_path)
    except Exception as e:
        print(f"[ERROR] Failed to load video: {e}", file=sys.stderr)
        sys.exit(1)

    original_duration = clip.duration
    print(f"[INFO] Original duration: {original_duration:.3f} seconds")

    # 3) Reverse (time-mirror) both video and audio
    print("[INFO] Reversing video and audio (time_mirror)...")
    try:
        reversed_clip = clip.fx(vfx.time_mirror)
    except Exception as e:
        print(f"[ERROR] Failed to reverse clip: {e}", file=sys.stderr)
        clip.reader.close()
        if clip.audio:
            clip.audio.reader.close_proc()
        sys.exit(1)

    reversed_duration = reversed_clip.duration
    print(f"[INFO] Reversed duration (should match original): {reversed_duration:.3f} seconds")

    # 4) Speed up the reversed clip
    print(f"[INFO] Speeding up by factor {speed_factor} (speedx)...")
    try:
        fast_reversed_clip = reversed_clip.fx(vfx.speedx, speed_factor)
    except Exception as e:
        print(f"[ERROR] Failed to speed up reversed clip: {e}", file=sys.stderr)
        reversed_clip.reader.close()
        if reversed_clip.audio:
            reversed_clip.audio.reader.close_proc()
        sys.exit(1)

    final_duration = fast_reversed_clip.duration
    print(f"[INFO] Final duration: {final_duration:.3f} seconds "
          f"(should be ~{original_duration / speed_factor:.3f} seconds)")

    # 5) Write the final video to disk
    print(f"[INFO] Writing output video to: {output_path}")
    try:
        fast_reversed_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            fps=clip.fps,
            preset="medium",
            threads=4,
            bitrate="2000k",
            logger="bar"  # can be None, "bar" (progress bar) or "verbose"
        )
    except Exception as e:
        print(f"[ERROR] Failed to write output video: {e}", file=sys.stderr)
        fast_reversed_clip.reader.close()
        if fast_reversed_clip.audio:
            fast_reversed_clip.audio.reader.close_proc()
        sys.exit(1)

    # 6) Clean up readers to avoid resource leaks
    clip.reader.close()
    if clip.audio:
        clip.audio.reader.close_proc()
    reversed_clip.reader.close()
    if reversed_clip.audio:
        reversed_clip.audio.reader.close_proc()
    fast_reversed_clip.reader.close()
    if fast_reversed_clip.audio:
        fast_reversed_clip.audio.reader.close_proc()

    print("[INFO] Processing complete! 🎬")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reverse a video and speed it up by a given factor."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="output.mp4",
        help="Path to input video file (default: output.mp4)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output_reversed_5x.mp4",
        help="Path to output video file (default: output_reversed_5x.mp4)"
    )
    parser.add_argument(
        "--factor", "-f",
        type=float,
        default=5.0,
        help="Speed‐up factor (default: 5.0). Must be > 0"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(
        input_path=args.input,
        output_path=args.output,
        speed_factor=args.factor
    )
