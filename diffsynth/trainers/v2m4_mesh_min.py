"""
Minimal V2M4 mesh rendering utilities used by mesh_util.py.

Vendored from V2M4 (v2m4_trellis) so DiffSynth trainers do not need the full
V2M4 repo on PYTHONPATH. Requires: torch, numpy, nvdiffrast, easydict.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

try:
    import nvdiffrast.torch as dr
except Exception:  # pragma: no cover
    dr = None
    print("nvdiffrast is not installed. Please install nvdiffrast to use mesh rendering.")


# ---------------------------------------------------------------------------
# Mesh container (from v2m4_trellis.representations.mesh.cube2mesh.MeshExtractResult)
# ---------------------------------------------------------------------------

class MeshExtractResult:
    def __init__(
        self,
        vertices,
        faces,
        vertex_attrs=None,
        res=64,
        texture=None,
        uv=None,
    ):
        self.vertices = vertices
        self.faces = faces.long()
        self.vertex_attrs = vertex_attrs
        self.face_normal = self.comput_face_normals(vertices, faces)
        self.res = res
        self.success = vertices.shape[0] != 0 and faces.shape[0] != 0

        # training only
        self.tsdf_v = None
        self.tsdf_s = None
        self.reg_loss = None

        self.texture = texture
        self.uv = uv

    def deepcopy(self):
        return MeshExtractResult(
            self.vertices.detach().clone(),
            self.faces.detach().clone(),
            self.vertex_attrs.detach().clone() if self.vertex_attrs is not None else None,
            self.res,
            self.texture.detach().clone() if self.texture is not None else None,
            self.uv.detach().clone() if self.uv is not None else None,
        )

    def comput_face_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        return face_normals[:, None, :].repeat(1, 3, 1)

    def comput_v_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        v_normals = torch.zeros_like(verts)
        v_normals = v_normals.scatter_add(0, i0[..., None].repeat(1, 3), face_normals)
        v_normals = v_normals.scatter_add(0, i1[..., None].repeat(1, 3), face_normals)
        v_normals = v_normals.scatter_add(0, i2[..., None].repeat(1, 3), face_normals)

        v_normals = torch.nn.functional.normalize(v_normals, dim=1)
        return v_normals


# ---------------------------------------------------------------------------
# Camera helpers (subset of utils3d + V2M4 render_utils)
# ---------------------------------------------------------------------------

def _as_batched_1d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 0:
        return x.reshape(1)
    return x


def intrinsics_from_focal_center(fx, fy, cx, cy) -> torch.Tensor:
    fx = _as_batched_1d(torch.as_tensor(fx, dtype=torch.float32))
    fy = _as_batched_1d(torch.as_tensor(fy, dtype=torch.float32, device=fx.device))
    cx = _as_batched_1d(torch.as_tensor(cx, dtype=torch.float32, device=fx.device))
    cy = _as_batched_1d(torch.as_tensor(cy, dtype=torch.float32, device=fx.device))
    zeros = torch.zeros_like(fx)
    ones = torch.ones_like(fx)
    return torch.stack([fx, zeros, cx, zeros, fy, cy, zeros, zeros, ones], dim=-1).unflatten(-1, (3, 3))


def intrinsics_from_fov_xy(fov_x, fov_y) -> torch.Tensor:
    fov_x = torch.as_tensor(fov_x, dtype=torch.float32)
    fov_y = torch.as_tensor(fov_y, dtype=torch.float32, device=fov_x.device)
    focal_x = 0.5 / torch.tan(fov_x / 2)
    focal_y = 0.5 / torch.tan(fov_y / 2)
    return intrinsics_from_focal_center(focal_x, focal_y, 0.5, 0.5)


def extrinsics_look_at(eye: torch.Tensor, look_at: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """OpenCV extrinsics looking at a point. eye/look_at/up: [..., 3] -> [..., 4, 4]."""
    if eye.ndim == 1:
        eye = eye.unsqueeze(0)
        look_at = look_at.unsqueeze(0) if look_at.ndim == 1 else look_at
        up = up.unsqueeze(0) if up.ndim == 1 else up
        squeeze = True
    else:
        squeeze = False

    z = look_at - eye
    x = torch.cross(-up, z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    z = z / z.norm(dim=-1, keepdim=True)
    R = torch.stack([x, y, z], dim=-2)
    t = -torch.matmul(R, eye[..., None])
    ret = torch.zeros((eye.shape[0], 4, 4), dtype=eye.dtype, device=eye.device)
    ret[:, :3, :3] = R
    ret[:, :3, 3] = t[:, :, 0]
    ret[:, 3, 3] = 1.0
    return ret[0] if squeeze else ret


def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)

    yaws = torch.tensor(yaws, dtype=torch.float32).cuda()
    pitchs = torch.tensor(pitchs, dtype=torch.float32).cuda()
    rs = torch.tensor(rs, dtype=torch.float32).cuda()
    fovs = torch.deg2rad(torch.tensor(fovs, dtype=torch.float32)).cuda()

    origs = torch.stack(
        [
            torch.sin(yaws) * torch.cos(pitchs),
            torch.cos(yaws) * torch.cos(pitchs),
            torch.sin(pitchs),
        ],
        dim=1,
    ) * rs[:, None]

    lookat = torch.tensor([0, 0, 0], dtype=torch.float32).cuda()
    up = torch.tensor([0, 0, 1], dtype=torch.float32).cuda()

    extrinsics = extrinsics_look_at(origs, lookat.expand(origs.shape[0], -1), up.expand(origs.shape[0], -1))
    intrinsics = intrinsics_from_fov_xy(fovs, fovs)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


# ---------------------------------------------------------------------------
# Mesh renderer + render_frames (from v2m4_trellis renderers / render_utils)
# ---------------------------------------------------------------------------

def intrinsics_to_projection(intrinsics: torch.Tensor, near: float, far: float) -> torch.Tensor:
    """OpenCV intrinsics to OpenGL perspective matrix."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = -2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.0
    return ret


class MeshRenderer:
    def __init__(self, rendering_options=None, device="cuda"):
        if rendering_options is None:
            rendering_options = {}
        if dr is None:
            raise ImportError("nvdiffrast is required for MeshRenderer")
        self.rendering_options = edict(
            {
                "resolution": None,
                "near": None,
                "far": None,
                "ssaa": 1,
            }
        )
        self.rendering_options.update(rendering_options)
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.device = device

    def render(
        self,
        mesh: MeshExtractResult,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        return_types=None,
    ) -> edict:
        if return_types is None:
            return_types = ["mask", "normal", "depth", "color", "texture"]

        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]

        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            default_img = torch.zeros((1, resolution, resolution, 3), dtype=torch.float32, device=self.device)
            return {
                k: default_img if k in ["normal", "normal_map", "color"] else default_img[..., :1]
                for k in return_types
            }

        perspective = intrinsics_to_projection(intrinsics, near, far)
        RT = extrinsics.unsqueeze(0)
        full_proj = (perspective @ extrinsics).unsqueeze(0)

        vertices = mesh.vertices.unsqueeze(0)
        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        vertices_camera = torch.bmm(vertices_homo, RT.transpose(-1, -2))
        vertices_clip = torch.bmm(vertices_homo, full_proj.transpose(-1, -2))
        faces_int = mesh.faces.int()
        rast, rast_db = dr.rasterize(
            self.glctx, vertices_clip, faces_int, (resolution * ssaa, resolution * ssaa), grad_db=True
        )

        out_dict = edict()
        for typ in return_types:
            img = None
            if typ == "mask":
                img = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
            elif typ == "depth":
                img = dr.interpolate(vertices_camera[..., 2:3].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
            elif typ == "normal":
                img = dr.interpolate(
                    mesh.face_normal.reshape(1, -1, 3),
                    rast,
                    torch.arange(mesh.faces.shape[0] * 3, device=self.device, dtype=torch.int).reshape(-1, 3),
                )[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
                img = (img + 1) / 2
                mask = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
                img = torch.where(mask > 0, img, torch.ones_like(img))
            elif typ == "normal_map":
                img = dr.interpolate(mesh.vertex_attrs[:, 3:].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
            elif typ == "color":
                img = dr.interpolate(mesh.vertex_attrs[:, :3].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
            elif typ == "texture":
                try:
                    uv_map, uv_map_dr = dr.interpolate(mesh.uv, rast, faces_int, rast_db, diff_attrs="all")
                    img = dr.texture(mesh.texture.unsqueeze(0), uv_map, uv_map_dr)
                    mask = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
                    img = torch.where(mask > 0, img, torch.ones_like(img))
                except Exception:
                    continue
            else:
                continue

            if ssaa > 1:
                img = F.interpolate(
                    img.permute(0, 3, 1, 2),
                    (resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                img = img.squeeze()
            else:
                img = img.permute(0, 3, 1, 2).squeeze()
            out_dict[typ] = img

        return out_dict


def render_frames(sample, extrinsics, intrinsics, options=None, colors_overwrite=None, verbose=True, **kwargs):
    """Mesh-only render_frames compatible with V2M4's API for MeshExtractResult."""
    if options is None:
        options = {}
    if not isinstance(sample, MeshExtractResult):
        raise ValueError(f"Unsupported sample type: {type(sample)} (only MeshExtractResult is vendored)")

    renderer = MeshRenderer()
    renderer.rendering_options.resolution = options.get("resolution", 512)
    renderer.rendering_options.near = options.get("near", 1)
    renderer.rendering_options.far = options.get("far", 100)
    renderer.rendering_options.ssaa = options.get("ssaa", 4)
    renderer.rendering_options.bg_color = options.get("bg_color", (0, 0, 0))

    if extrinsics.ndim == 2:
        extrinsics = extrinsics.unsqueeze(0)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)

    rets = {}
    for extr, intr in zip(extrinsics, intrinsics):
        res = renderer.render(sample, extr, intr)
        if "normal" not in rets:
            rets["normal"] = []
        rets["normal"].append(
            np.clip(res["normal"].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        )
        if "color" not in rets:
            rets["color"] = []
        rets["color"].append(
            np.clip(res["color"].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        )
        if "texture" not in rets:
            rets["texture"] = []
        try:
            rets["texture"].append(
                np.clip(res["texture"].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            )
        except Exception:
            pass
    return rets
