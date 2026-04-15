#!/usr/bin/env python3
"""
Re-encode all videos under static/demos/**/edited_with_mask/ to browser-friendly
H.264 (yuv420p) + AAC in MP4 with faststart. Fixes files that are MPEG-4 Part 2,
HEVC, odd profiles, etc.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

DEMOS_ROOT = os.path.join("static", "demos")
EDITED_SUBDIR = "edited_with_mask"
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".m4v"}


def find_edited_with_mask_videos(root: str) -> list[str]:
    out: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) != EDITED_SUBDIR:
            continue
        for name in filenames:
            if os.path.splitext(name)[1].lower() in VIDEO_EXTS:
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def _has_audio(path: str) -> bool:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(r.stdout.strip())


def reencode(
    src: str,
    crf: int,
    preset: str,
    dry_run: bool,
) -> bool:
    if dry_run:
        print(f"[dry-run] would re-encode: {src}")
        return True

    fd, tmp_path = tempfile.mkstemp(suffix=".mp4", prefix="reencode_")
    os.close(fd)
    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            src,
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-movflags",
            "+faststart",
        ]
        if _has_audio(src):
            cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "128k"]
        else:
            cmd += ["-an"]
        cmd.append(tmp_path)

        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print(f"FAIL {src}\n{p.stderr}", file=sys.stderr)
            os.unlink(tmp_path)
            return False

        bak = src + ".bak"
        shutil.move(src, bak)
        try:
            shutil.move(tmp_path, src)
        except OSError:
            shutil.move(bak, src)
            raise
        os.unlink(bak)
        print(f"OK {src}")
        return True
    except Exception as e:
        if os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        print(f"ERROR {src}: {e}", file=sys.stderr)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default=DEMOS_ROOT,
        help=f"Demos root (default: {DEMOS_ROOT})",
    )
    ap.add_argument("--crf", type=int, default=20, help="libx264 CRF (default: 20)")
    ap.add_argument(
        "--preset",
        default="medium",
        help="libx264 preset (default: medium)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List files only, do not run ffmpeg",
    )
    args = ap.parse_args()

    for bin_name in ("ffmpeg", "ffprobe"):
        if shutil.which(bin_name) is None:
            print(f"Missing `{bin_name}` in PATH.", file=sys.stderr)
            return 1

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    files = find_edited_with_mask_videos(root)
    if not files:
        print(f"No videos under **/{EDITED_SUBDIR}/ in {root}")
        return 0

    ok = 0
    for path in files:
        if reencode(path, crf=args.crf, preset=args.preset, dry_run=args.dry_run):
            ok += 1

    if args.dry_run:
        print(f"{ok} file(s) would be processed.")
    else:
        print(f"Done: {ok}/{len(files)} succeeded.")
    return 0 if ok == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
