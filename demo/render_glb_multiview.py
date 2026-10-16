# -*- coding: utf-8 -*-
"""
Load a GLB and render 4 turntable views with a black background (default 832x480).
Camera is fixed; the mesh rotates about +Z by --azimuths.

Depends: trimesh, pyrender, numpy, Pillow

Headless (no monitor): this script defaults to EGL when DISPLAY is unset (GPU offscreen).
If EGL fails, try:  --backend osmesa  (needs OSMesa system libs, CPU-only)
Or:  xvfb-run -a python3 ...   with  --backend glx
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

import trimesh


def configure_pyopengl_platform(backend: str) -> None:
    """
    Set PYOPENGL_PLATFORM before pyrender/OpenGL import.
    On headless servers the default GLX path fails with: Cannot connect to "None".
    """
    b = (backend or "auto").lower()
    if b == "auto":
        if os.environ.get("PYOPENGL_PLATFORM"):
            return
        disp = (os.environ.get("DISPLAY") or "").strip()
        if disp:
            return
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        return
    if b == "glx":
        os.environ.pop("PYOPENGL_PLATFORM", None)
        return
    os.environ["PYOPENGL_PLATFORM"] = b


def load_glb_trimesh(path: str) -> trimesh.Trimesh:
    obj = trimesh.load(path, process=False, force=None)
    if isinstance(obj, trimesh.Scene):
        geoms = list(obj.geometry.values())
        if not geoms:
            raise ValueError(f"No geometry in GLB: {path}")
        if len(geoms) == 1:
            return geoms[0]
        try:
            return trimesh.util.concatenate(geoms)
        except Exception:
            return geoms[0]
    if not isinstance(obj, trimesh.Trimesh):
        raise TypeError(f"Unsupported load result: {type(obj)}")
    return obj


def center_mesh_for_view(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, float]:
    """
    Put mesh in the middle of the render: translate so the bounding-sphere center
    is at the origin and size the camera from the sphere radius (stable across turntable).
    Falls back to AABB center if the sphere is unavailable.
    """
    mesh = mesh.copy()
    try:
        bs = mesh.bounding_sphere
        c = np.asarray(bs.center, dtype=np.float64)
        mesh.apply_translation(-c)
        r = float(bs.primitive.radius)
        if not np.isfinite(r) or r < 1e-8:
            r = float((mesh.bounds[1] - mesh.bounds[0]).max()) * 0.5
        extent = max(2.0 * r, float((mesh.bounds[1] - mesh.bounds[0]).max()))
    except Exception:
        center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-center)
        extent = float((mesh.bounds[1] - mesh.bounds[0]).max())
    if extent < 1e-8:
        extent = 1.0
    return mesh, extent


def camera_pose_orbit(
    azimuth_deg: float,
    elevation_deg: float,
    radius: float,
    target: np.ndarray | None = None,
) -> np.ndarray:
    """Pyrender camera pose: transforms camera frame to world (OpenGL, looks down -Z)."""
    if target is None:
        target = np.zeros(3, dtype=np.float64)
    else:
        target = np.asarray(target, dtype=np.float64)
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    offset = np.array(
        [
            radius * np.cos(el) * np.cos(az),
            radius * np.sin(el),
            radius * np.cos(el) * np.sin(az),
        ],
        dtype=np.float64,
    )
    eye = target + offset
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    up = up / (np.linalg.norm(up) + 1e-12)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def mat4_rotation_z_deg(deg: float) -> np.ndarray:
    """4x4 homogeneous rotation about +Z, degrees (right-handed, OpenGL Y-up)."""
    t = np.radians(deg)
    co, si = np.cos(t), np.sin(t)
    M = np.eye(4, dtype=np.float64)
    M[0, 0], M[0, 1], M[1, 0], M[1, 1] = co, -si, si, co
    return M


def render_four_views(
    glb_path: str,
    out_dir: str,
    width: int,
    height: int,
    azimuths_deg: list[float],
    elevation_deg: float,
    yfov_deg: float,
    *,
    gl_backend: str = "auto",
) -> list[str]:
    configure_pyopengl_platform(gl_backend)
    try:
        import pyrender
    except ImportError as e:
        raise ImportError(
            "pyrender is required: pip install pyrender PyOpenGL PyOpenGL_accelerate"
        ) from e

    mesh_tm, extent = center_mesh_for_view(load_glb_trimesh(glb_path))
    pr_mesh = pyrender.Mesh.from_trimesh(mesh_tm, smooth=False)

    yfov = np.radians(yfov_deg)
    aspect = float(width) / float(height)
    znear = max(extent * 0.01, 1e-4)
    zfar = max(extent * 100.0, 10.0)
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect, znear=znear, zfar=zfar)

    # Fit sphere (diameter ~ extent) in vertical FOV; margin keeps object centered with padding
    fit_margin = 2.12
    radius = (0.5 * extent * fit_margin) / max(np.tan(yfov * 0.5), 1e-6)

    scene = pyrender.Scene(
        ambient_light=np.array([0.35, 0.35, 0.35, 1.0], dtype=np.float32),
        bg_color=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )

    light = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=4.0)
    light_pose = np.eye(4, dtype=np.float64)
    light_pose[:3, 3] = [2.0, 4.0, 2.0]
    scene.add(light, pose=light_pose)

    fixed_cam_pose = camera_pose_orbit(0.0, elevation_deg, radius)
    fill_cam = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=2.0)

    try:
        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    except Exception as e:
        plat = os.environ.get("PYOPENGL_PLATFORM", "(default glx)")
        raise RuntimeError(
            f"OffscreenRenderer failed (PYOPENGL_PLATFORM={plat!r}): {e}\n"
            "Headless: try  --backend egl  (default when no DISPLAY), or  --backend osmesa, "
            "or  xvfb-run -a  with  --backend glx"
        ) from e
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(glb_path))[0]
    paths: list[str] = []

    try:
        cam_node = scene.add(camera, pose=fixed_cam_pose)
        fill_node = scene.add(fill_cam, pose=fixed_cam_pose)
        for i, az in enumerate(azimuths_deg):
            mesh_node = scene.add(pr_mesh, pose=mat4_rotation_z_deg(az))
            color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            scene.remove_node(mesh_node)
            out_path = os.path.join(out_dir, f"{base}_view{i:02d}_az{int(az)}.png")
            # Rotate the image by 90 degrees.
            color = np.rot90(color, k=-1)
            Image.fromarray(color).save(out_path)
            paths.append(out_path)
        scene.remove_node(fill_node)
        scene.remove_node(cam_node)
    finally:
        renderer.delete()

    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Render 4 views of a GLB (black background).")
    parser.add_argument("--glb", default=None, help="Path to a single .glb file")
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default=None,
        help="Glob pattern for many .glb files",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: next to each GLB, subfolder multiview_render)",
    )
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument(
        "--azimuths",
        type=float,
        nargs=4,
        default=[0.0, 90.0, 180.0, 270.0],
        help="Mesh rotation about +Z in degrees, camera fixed (4 values)",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=0.0,
        help="Fixed camera elevation in degrees (orbit angle at azimuth 0)",
    )
    parser.add_argument("--yfov", type=float, default=60.0, help="Vertical field of view in degrees")
    parser.add_argument(
        "--backend",
        choices=("auto", "egl", "osmesa", "glx"),
        default="auto",
        help="OpenGL platform: auto=no DISPLAY -> egl; use osmesa for CPU if EGL fails; glx needs a display or Xvfb",
    )
    args = parser.parse_args()

    if bool(args.glb) == bool(args.glob_pattern):
        print("Specify exactly one of --glb or --glob", file=sys.stderr)
        return 1

    glbs = [args.glb] if args.glb else sorted(glob.glob(args.glob_pattern))
    if not glbs:
        print("No GLB files found", file=sys.stderr)
        return 1

    for path in glbs:
        if not os.path.isfile(path):
            print(f"Missing file: {path}", file=sys.stderr)
            return 1
        out_dir = args.output_dir
        if not out_dir:
            out_dir = os.path.join(os.path.dirname(path) or ".", "multiview_render")
        print(f"Rendering {path} -> {out_dir}")
        try:
            render_four_views(
                path,
                out_dir,
                width=args.width,
                height=args.height,
                azimuths_deg=list(args.azimuths),
                elevation_deg=args.elevation,
                yfov_deg=args.yfov,
                gl_backend=args.backend,
            )
        except Exception as e:
            print(f"Failed: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
