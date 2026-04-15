#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_INPUT = Path("static/demos/substitution/ref/blackswan/blackswan.glb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rotate a GLB model +90 degrees around X axis."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input GLB path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GLB path. If omitted, writes '<input_stem>_x90.glb'.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file directly.",
    )
    return parser.parse_args()


def rotate_glb_x90(input_path: Path, output_path: Path) -> None:
    try:
        import numpy as np
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency. Please install: pip install trimesh numpy"
        ) from exc

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    loaded = trimesh.load(input_path, force="scene")
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)

    rotation = trimesh.transformations.rotation_matrix(np.deg2rad(-90.0), [1.0, 0.0, 0.0])
    scene.apply_transform(rotation)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    glb_bytes = scene.export(file_type="glb")
    output_path.write_bytes(glb_bytes)


def main() -> int:
    args = parse_args()
    input_path = args.input

    if args.in_place:
        output_path = input_path
    elif args.output is not None:
        output_path = args.output
    else:
        output_path = input_path.with_name(f"{input_path.stem}_x90.glb")

    try:
        rotate_glb_x90(input_path=input_path, output_path=output_path)
    except Exception as exc:  # noqa: BLE001 - CLI script, return readable errors.
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Rotated GLB saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
