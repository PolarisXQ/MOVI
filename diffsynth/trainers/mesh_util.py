import glob
import os

import numpy as np
import trimesh
import torch
import math
import copy
import imageio
import torch.nn.functional as F
from PIL import Image
from typing import Optional, List, Tuple

from diffsynth.trainers.v2m4_mesh_min import (
    MeshExtractResult,
    extrinsics_look_at,
    render_frames,
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
)

class RenderConfig:
    def __init__(self, rotation_angle_deg=0, texture=True, align_to_x_axis=False,
                light_type='none', light_location=None, light_direction=None,
                ambient_color=None, diffuse_color=None, specular_color=None):
        """
        Args:
            rotation_angle_deg: The rotation angle in degrees, applied after aligning the mesh to the x axis
            texture: Whether to use texture
            align_to_x_axis: Whether to align the mesh to the x axis
            light_type: The type of light, one of 'point', 'directional', 'none'
            light_location: The location of the light, one of 'up', 'front', 'back', 'left', 'right'
            light_direction: The direction of the light, one of 'up', 'front', 'back', 'left', 'right'
            ambient_color: The ambient color of the light, a list of 3 floats
            diffuse_color: The diffuse color of the light, a list of 3 floats
            specular_color: The specular color of the light, a list of 3 floats
        """
        self.rotation_angle_deg = rotation_angle_deg
        self.texture = texture
        self.align_to_x_axis = align_to_x_axis
        self.light_type = light_type
        self.light_location = light_location
        self.light_direction = light_direction
        self.ambient_color = ambient_color
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color

class MeshLoader:
    def __init__(self, data_root, dataset="VIPSeg"):
        self.data_root = data_root
        self.dataset = dataset

    def load_mesh(self, major_class_name:str, frame_index:str):
        # check if major_class_name is in the data_root
        # A NOT GRACEFUL WAY TO HANDLE THIS, BUT IT'S OK FOR 
        if self.dataset == "VIPSeg":
            video_seq, major_class_name, instance_id = major_class_name.split("/")
            if not os.path.exists(os.path.join(self.data_root, video_seq, major_class_name, instance_id, f"{frame_index}_output_model.glb")):
                raise ValueError(f"{video_seq}/{major_class_name}/{instance_id}/{frame_index}_output_model.glb not found in {self.data_root}")
            mesh_path = os.path.join(self.data_root, video_seq, major_class_name, instance_id, f"{frame_index}_output_model.glb")
        elif self.dataset == "DAVIS17":
            major_class_name, instance_id = major_class_name.split("/")
            if not os.path.exists(os.path.join(self.data_root, major_class_name, instance_id, f"{frame_index}_output_model.glb")):
                raise ValueError(f"{major_class_name}/{instance_id}/{frame_index}_output_model.glb not found in {self.data_root}")
            mesh_path = os.path.join(self.data_root, major_class_name, instance_id, f"{frame_index}_output_model.glb")
        elif self.dataset == "ROSE":
            # ROSE default layout: <data_root>/<mesh_key>/<frame_index>_output_model.glb
            mesh_path = os.path.join(self.data_root, major_class_name, f"{frame_index}_output_model.glb")
            if not os.path.exists(mesh_path):
                raise ValueError(f"{major_class_name}/{frame_index}_output_model.glb not found in {self.data_root}")
        else:
            raise ValueError(f"Invalid dataset: {self.dataset}")
        tm_mesh = trimesh.load(mesh_path, process=False, force='mesh')

        if isinstance(tm_mesh, trimesh.Scene):
            if len(tm_mesh.geometry) == 0:
                raise ValueError(f"No geometry found in mesh file {mesh_path}")
            tm_mesh = tm_mesh.as_mesh()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        vertices_np = tm_mesh.vertices.astype(np.float32)
        faces_np = tm_mesh.faces.astype(np.int64)

        vertices = torch.from_numpy(vertices_np).to(device)
        faces = torch.from_numpy(faces_np).long().to(device)

        colors = None
        if hasattr(tm_mesh, "visual") and getattr(tm_mesh.visual, "vertex_colors", None) is not None and len(tm_mesh.visual.vertex_colors):
            colors_np = np.asarray(tm_mesh.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
            colors = torch.from_numpy(colors_np).to(device)
        else:
            colors = torch.full((vertices.shape[0], 3), 0.7, dtype=torch.float32, device=device)

        vertex_attrs = torch.cat([colors, torch.zeros_like(colors)], dim=1)

        mesh_result = MeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=vertex_attrs)
        normals = mesh_result.comput_v_normals(vertices, faces)
        if normals is not None:
            mesh_result.vertex_attrs[:, 3:6] = normals

        return mesh_result

    def load_glb_path(self, glb_path: str, normalize: bool = True):
        """
        Load any .glb into MeshExtractResult. If normalize, center and scale max bbox extent to ~1.8
        (matches typical camera distance in render()).
        """
        tm_mesh = trimesh.load(glb_path, process=False, force="mesh")

        if isinstance(tm_mesh, trimesh.Scene):
            if len(tm_mesh.geometry) == 0:
                raise ValueError(f"No geometry found in mesh file {glb_path}")
            tm_mesh = tm_mesh.as_mesh()

        if normalize:
            tm_mesh = tm_mesh.copy()
            center = tm_mesh.bounds.mean(axis=0)
            tm_mesh.apply_translation(-center)
            mx = float((tm_mesh.bounds[1] - tm_mesh.bounds[0]).max())
            if mx > 1e-8:
                tm_mesh.apply_scale(1.8 / mx)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vertices_np = tm_mesh.vertices.astype(np.float32)
        faces_np = tm_mesh.faces.astype(np.int64)

        vertices = torch.from_numpy(vertices_np).to(device)
        faces = torch.from_numpy(faces_np).long().to(device)

        if hasattr(tm_mesh, "visual") and getattr(tm_mesh.visual, "vertex_colors", None) is not None and len(tm_mesh.visual.vertex_colors):
            colors_np = np.asarray(tm_mesh.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
            colors = torch.from_numpy(colors_np).to(device)
        else:
            colors = torch.full((vertices.shape[0], 3), 0.7, dtype=torch.float32, device=device)

        vertex_attrs = torch.cat([colors, torch.zeros_like(colors)], dim=1)

        mesh_result = MeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=vertex_attrs)
        normals = mesh_result.comput_v_normals(vertices, faces)
        if normals is not None:
            mesh_result.vertex_attrs[:, 3:6] = normals

        return mesh_result

    # def extract_ptv3_features(
    #     self,
    #     mesh,
    #     use_colors: bool = True,
    #     use_normals: bool = True,
    # ) -> Optional[torch.Tensor]:
    #     """
    #     Extract PTv3 features from mesh vertices.
        
    #     Args:
    #         mesh: MeshExtractResult object
    #         use_colors: Whether to use vertex colors as input
    #         use_normals: Whether to use vertex normals as input
        
    #     Returns:
    #         [N, C] per-vertex features or None if PTv3 not available
    #     """
    #     if not self.enable_ptv3 or self.ptv3_extractor is None:
    #         return None
        
    #     return self.ptv3_extractor.extract_from_mesh(mesh, use_colors, use_normals)
    
    # def render_ptv3_features(
    #     self,
    #     mesh,
    #     params,
    #     rotation_angle_deg: float = 0,
    #     height: int = 480,
    #     width: int = 832,
    # ) -> Optional[Image.Image]:
    #     """
    #     Extract PTv3 features and render them to a 2D image.
        
    #     Args:
    #         mesh: MeshExtractResult object
    #         params: Camera parameters
    #         rotation_angle_deg: Rotation angle in degrees
    #         height: Output height
    #         width: Output width
        
    #     Returns:
    #         PIL.Image with rendered PTv3 features or None
    #     """
    #     if not self.enable_ptv3 or self.ptv3_extractor is None:
    #         return None
        
    #     # Extract features
    #     features = self.extract_ptv3_features(mesh)
    #     if features is None:
    #         return None
        
    #     # Apply rotation to vertices (same as render method)
    #     mesh_center = (mesh.vertices.min(dim=0).values + mesh.vertices.max(dim=0).values) / 2
    #     vertices_centered = mesh.vertices - mesh_center
        
    #     rotation_angle_rad = math.radians(rotation_angle_deg)
    #     cos_yaw = math.cos(rotation_angle_rad)
    #     sin_yaw = math.sin(rotation_angle_rad)
    #     rotation_matrix = torch.tensor([
    #         [cos_yaw, -sin_yaw, 0],
    #         [sin_yaw, cos_yaw, 0],
    #         [0, 0, 1]
    #     ], dtype=torch.float32, device=mesh.vertices.device)
        
    #     vertices_rotated = vertices_centered @ rotation_matrix.T + mesh_center
        
    #     # Update renderer dimensions
    #     self.ptv3_renderer.height = height
    #     self.ptv3_renderer.width = width
        
    #     # Render features
    #     return self.ptv3_renderer.render_features_to_image(
    #         vertices_rotated, features, params
    #     )
    
    # def get_mesh_vertices_and_features(
    #     self,
    #     mesh,
    # ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    #     """
    #     Get mesh vertices and PTv3 features as tensors.
        
    #     Args:
    #         mesh: MeshExtractResult object
        
    #     Returns:
    #         Tuple of (vertices [N, 3], features [N, C] or None)
    #     """
    #     vertices = mesh.vertices
    #     features = self.extract_ptv3_features(mesh) if self.enable_ptv3 else None
    #     return vertices, features

    def load_camera_parameters(self, major_class_name, frame_index):
        camera_parameters_path = os.path.join(self.data_root, major_class_name, f"{frame_index}_camera_parameters.npy")
        return np.load(camera_parameters_path)

    def get_mesh_without_texture(self, mesh):
        """Return a copy of the input mesh with texture and per-vertex colors removed."""
        mesh_no_tex = mesh.deepcopy() if hasattr(mesh, "deepcopy") else copy.deepcopy(mesh)

        if getattr(mesh_no_tex, "vertex_attrs", None) is not None and mesh_no_tex.vertex_attrs.shape[1] >= 3:
            mesh_no_tex.vertex_attrs = mesh_no_tex.vertex_attrs.clone()
            mesh_no_tex.vertex_attrs[:, :3] = 0.7

        if hasattr(mesh_no_tex, "texture"):
            mesh_no_tex.texture = None
        if hasattr(mesh_no_tex, "uv"):
            mesh_no_tex.uv = None

        return mesh_no_tex


    def align_mesh_bbox_to_x_axis(self, mesh):
        """
        Align the mesh so that the longer side of its horizontal (XY plane) bounding box
        is parallel to the global X axis. The mesh is rotated in-place around the Z axis.

        Args:
            mesh: MeshExtractResult-like object with `vertices` attribute of shape (N, 3).

        Returns:
            The input mesh after alignment (rotation applied in-place).
        """
        if mesh is None or not hasattr(mesh, "vertices") or mesh.vertices is None:
            return mesh

        vertices = mesh.vertices
        if vertices.numel() == 0 or vertices.shape[1] < 2:
            return mesh

        device = vertices.device
        dtype = vertices.dtype

        xy = vertices[:, :2]
        xy_center = xy.mean(dim=0, keepdim=True)
        centered_xy = xy - xy_center

        if torch.allclose(centered_xy, torch.zeros_like(centered_xy)):
            return mesh

        cov = centered_xy.t().matmul(centered_xy)
        num_points = centered_xy.shape[0]
        if num_points > 0:
            cov = cov / num_points

        eigvals, eigvecs = torch.linalg.eigh(cov)
        longest_axis_idx = torch.argmax(eigvals)
        principal_axis = eigvecs[:, longest_axis_idx]

        if torch.allclose(principal_axis.abs().sum(), torch.tensor(0.0, device=device, dtype=dtype)):
            return mesh

        angle = torch.atan2(principal_axis[1], principal_axis[0])
        rotation_angle = -angle  # rotate to align with +X

        cos_angle = torch.cos(rotation_angle)
        sin_angle = torch.sin(rotation_angle)

        rot2d = torch.stack(
            [
                torch.stack([cos_angle, -sin_angle]),
                torch.stack([sin_angle, cos_angle]),
            ]
        )
        rot = torch.eye(3, device=device, dtype=dtype)
        rot[:2, :2] = rot2d.to(device=device, dtype=dtype)

        center3d = vertices.mean(dim=0, keepdim=True)
        centered_vertices = vertices - center3d
        rotated_vertices = centered_vertices.matmul(rot.t()) + center3d
        mesh.vertices = rotated_vertices

        if getattr(mesh, "vertex_attrs", None) is not None and mesh.vertex_attrs.shape[1] >= 6:
            normals = mesh.vertex_attrs[:, 3:6]
            rotated_normals = normals.matmul(rot.t())
            mesh.vertex_attrs[:, 3:6] = F.normalize(rotated_normals, dim=1)

        if hasattr(mesh, "face_normal"):
            mesh.face_normal = mesh.comput_face_normals(mesh.vertices, mesh.faces)

        return mesh

    def apply_lighting_nvdiffrast(self, target_mesh, params, ambient_color=[0.2, 0.2, 0.2], diffuse_color=[1.0, 1.0, 1.0], specular_color=[0.2, 0.2, 0.2], light_type='point', light_location=None, light_direction=None):
        device = target_mesh.vertices.device

        lighted_mesh = copy.deepcopy(target_mesh)

        if lighted_mesh.vertex_attrs is not None and lighted_mesh.vertex_attrs.shape[1] >= 6:
            normals = lighted_mesh.vertex_attrs[:, 3:6]
        else:
            normals = lighted_mesh.comput_v_normals(lighted_mesh.vertices, lighted_mesh.faces)
            if lighted_mesh.vertex_attrs is None:
                lighted_mesh.vertex_attrs = torch.zeros((lighted_mesh.vertices.shape[0], normals.shape[1] + 3), device=device)
            lighted_mesh.vertex_attrs[:, 3:3 + normals.shape[1]] = normals

        normals = F.normalize(normals, dim=1)
        base_color = lighted_mesh.vertex_attrs[:, :3].clamp(0.0, 1.0).clone()

        ambient = torch.tensor(ambient_color, dtype=torch.float32, device=device)
        diffuse = torch.tensor(diffuse_color, dtype=torch.float32, device=device)
        specular = torch.tensor(specular_color, dtype=torch.float32, device=device)

        if light_type == 'directional':
            if light_direction is None or isinstance(light_direction, str):
                if light_direction == 'up' or light_direction == None:
                    light_dir = [0, 0, -1.0]
                elif light_direction == 'front':
                    light_dir = [0, 1.0, 0]
                elif light_direction == 'back':
                    light_dir = [0, -1.0, 0]
                elif light_direction == 'left':
                    light_dir = [1.0, 0, 0]
                elif light_direction == 'right':
                    light_dir = [-1.0, 0, 0]
                else:
                    print(f"Invalid light direction: {light_direction}, using default direction")
                    light_dir = [0, 0, -1.0]
            light_dir_tensor = torch.tensor(light_dir, dtype=torch.float32, device=device)
            light_dir_tensor = F.normalize(light_dir_tensor, dim=0)
            light_vec = -light_dir_tensor.unsqueeze(0).expand_as(normals)
            attenuation = torch.ones_like(light_vec[:, :1])
        else:
            if light_location is None or isinstance(light_location, str):
                mesh_center = (lighted_mesh.vertices.min(dim=0).values + lighted_mesh.vertices.max(dim=0).values) / 2
                if light_location == 'up' or light_location == None:
                    light_loc_tensor = mesh_center + torch.tensor([0, 0, 1.0], dtype=torch.float32, device=device)
                elif light_location == 'back':
                    light_loc_tensor = mesh_center + torch.tensor([0, 1.0, 0], dtype=torch.float32, device=device)
                elif light_location == 'front':
                    light_loc_tensor = mesh_center - torch.tensor([0, 1.0, 0], dtype=torch.float32, device=device)
                elif light_location == 'right':
                    light_loc_tensor = mesh_center + torch.tensor([1.0, 0, 0], dtype=torch.float32, device=device)
                elif light_location == 'left':
                    light_loc_tensor = mesh_center - torch.tensor([1.0, 0, 0], dtype=torch.float32, device=device)
                else:
                    print(f"Invalid light location: {light_location}, using default location")
                    light_loc_tensor = mesh_center + torch.tensor([0, 0, 1.0], dtype=torch.float32, device=device)
            to_light = light_loc_tensor.unsqueeze(0) - lighted_mesh.vertices
            distance = torch.norm(to_light, dim=1, keepdim=True).clamp_min(1e-6)
            light_vec = to_light / distance
            attenuation = 1.0 / (distance ** 2)

        diffuse_intensity = torch.clamp((normals * F.normalize(light_vec, dim=1)).sum(dim=1, keepdim=True), min=0.0) * attenuation

        yaw, pitch, r, lookat_x, lookat_y, lookat_z = params
        yaw_tensor = torch.tensor(yaw, dtype=torch.float32, device=device)
        pitch_tensor = torch.tensor(pitch, dtype=torch.float32, device=device)
        r_tensor = torch.tensor(r, dtype=torch.float32, device=device)

        camera_origin = torch.stack([
            torch.sin(yaw_tensor) * torch.cos(pitch_tensor),
            torch.cos(yaw_tensor) * torch.cos(pitch_tensor),
            torch.sin(pitch_tensor)
        ], dim=0) * r_tensor

        view_vec = F.normalize(camera_origin.unsqueeze(0) - lighted_mesh.vertices, dim=1)
        reflect_dir = F.normalize(2 * (normals * light_vec).sum(dim=1, keepdim=True) * normals - light_vec, dim=1)
        specular_intensity = torch.clamp((reflect_dir * view_vec).sum(dim=1, keepdim=True), min=0.0) ** 32 * attenuation

        ambient_term = ambient.view(1, 3) * base_color
        diffuse_term = diffuse_intensity * diffuse.view(1, 3) * base_color
        specular_term = specular_intensity * specular.view(1, 3)

        shaded = torch.clamp(ambient_term + diffuse_term + specular_term, 0.0, 1.0)
        lighted_mesh.vertex_attrs[:, :3] = shaded

        return lighted_mesh

    def rotate_render_and_save(self, mesh, base_name, output_path, params, rotation_angle_deg=30, suffix="", 
                            light_type='point', light_location=None, light_direction=None,
                            ambient_color=None, diffuse_color=None, specular_color=None):
        """
        Rotate the model along its own center for a specified angle in yaw and save the rendered image.
        
        Args:
            mesh: MeshExtractResult object containing the mesh to render
            base_name: Base name for the output file
            output_path: Path to save the rendered image
            params: Camera parameters (yaw, pitch, r, lookat_x, lookat_y, lookat_z)
            rotation_angle_deg: Rotation angle in degrees (default: 30)
            suffix: Suffix to add to the filename (e.g., "_textured" or "_no_texture")
            light_type: Type of light - 'point' or 'directional' (default: 'point')
            light_location: Location of point light [x, y, z] (default: [0.0, 0.0, -1.0])
            light_direction: Direction of directional light [x, y, z] (default: [0.0, 0.0, -1.0])
            ambient_color: Ambient light color [r, g, b] (default: [0.5, 0.5, 0.5])
            diffuse_color: Diffuse light color [r, g, b] (default: [1.0, 1.0, 1.0])
            specular_color: Specular light color [r, g, b] (default: [1.0, 1.0, 1.0])
        
        Returns:
            Rendered image array
        """
        # Extract camera parameters
        yaw, pitch, r, lookat_x, lookat_y, lookat_z = params
        
        # Get the mesh center (mean of min max of vertices)
        mesh_center = (mesh.vertices.min(dim=0).values + mesh.vertices.max(dim=0).values) / 2
        # mesh_center = mesh.vertices.mean(dim=0)
        
        # Translate vertices to center around origin
        vertices_centered = mesh.vertices - mesh_center
        
        # Convert rotation angle from degrees to radians
        rotation_angle_rad = math.radians(rotation_angle_deg)
        
        # Create rotation matrix for yaw rotation around Z-axis
        # Yaw rotation: rotation around the vertical axis (Z-axis)
        cos_yaw = math.cos(rotation_angle_rad)
        sin_yaw = math.sin(rotation_angle_rad)
        rotation_matrix = torch.tensor([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ], dtype=torch.float32).cuda()
        
        # Rotate vertices
        vertices_rotated = vertices_centered @ rotation_matrix.T
        
        # Translate back to original position
        vertices_rotated = vertices_rotated + mesh_center
        
        # Create a temporary mesh with rotated vertices
        rotated_mesh = copy.deepcopy(mesh)
        rotated_mesh.vertices = vertices_rotated
        
        # Update vertex normals if they exist
        if hasattr(rotated_mesh, 'face_normal'):
            rotated_mesh.face_normal = rotated_mesh.comput_face_normals(vertices_rotated, rotated_mesh.faces)
        if hasattr(rotated_mesh, 'vertex_attrs') and rotated_mesh.vertex_attrs is not None and rotated_mesh.vertex_attrs.shape[1] > 3:
            rotated_mesh.vertex_attrs[:, 3:] = rotated_mesh.comput_v_normals(vertices_rotated, rotated_mesh.faces)
        
        # Get extrinsics and intrinsics using the existing function
        # Keep the camera position the same (use original params)
        fov = 40
        extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics([yaw], [pitch], [r], [fov])

        if extr.dim() == 2:
            extr = extr.unsqueeze(0)

        if intr.dim() == 3:
            intr_batch = intr
            intr = intr[0]
        else:
            intr_batch = intr.unsqueeze(0)
        
        # Update lookat point to account for rotation around mesh center
        # lookat = torch.tensor([lookat_x, lookat_y, lookat_z], dtype=torch.float32).cuda()
        # lookat_centered = lookat - mesh_center
        # lookat_rotated = lookat_centered @ rotation_matrix.T
        # updated_lookat = lookat_rotated + mesh_center
        updated_lookat = mesh_center
        
        # Update extrinsics to use the rotated lookat point
        yaw_tensor = torch.tensor([yaw], dtype=torch.float32).cuda()
        pitch_tensor = torch.tensor([pitch], dtype=torch.float32).cuda()
        r_tensor = torch.tensor([r], dtype=torch.float32).cuda()
        
        orig = torch.stack([
            torch.sin(yaw_tensor) * torch.cos(pitch_tensor),
            torch.cos(yaw_tensor) * torch.cos(pitch_tensor),
            torch.sin(pitch_tensor),
        ], dim=1).squeeze() * r_tensor
        
        extr = extrinsics_look_at(orig.unsqueeze(0), updated_lookat.unsqueeze(0), torch.tensor([[0, 0, 1]], dtype=torch.float32).cuda())
        
        resolution = 512
        rend_img = None
        
        render_options = {'resolution': resolution, 'bg_color': (0, 0, 0)}

        
        result = render_frames(rotated_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
    
        # Save the rendered image
        suffix_str = f"_{suffix}" if suffix else ""
        output_filename = output_path + "/" + base_name + f"_rotated_{rotation_angle_deg}deg{suffix_str}_no_light.png"
        imageio.imsave(output_filename, rend_img)

        lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='point', light_location='up')
        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        imageio.imsave(output_filename.replace('_no_light.png', '_up_point_light.png'), rend_img)

        lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='point', light_location='front')
        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        imageio.imsave(output_filename.replace('_no_light.png', '_front_point_light.png'), rend_img)

        lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='point', light_location='back')
        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        imageio.imsave(output_filename.replace('_no_light.png', '_back_point_light.png'), rend_img)

        lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='point', light_location='left')
        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        imageio.imsave(output_filename.replace('_no_light.png', '_left_point_light.png'), rend_img)

        lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='point', light_location='right')
        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        imageio.imsave(output_filename.replace('_no_light.png', '_right_point_light.png'), rend_img)

        lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='directional', light_direction='up')
        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        imageio.imsave(output_filename.replace('_no_light.png', '_up_directional_light.png'), rend_img)

        lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='directional', light_direction='front')
        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        imageio.imsave(output_filename.replace('_no_light.png', '_front_directional_light.png'), rend_img)

        lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='directional', light_direction='back')
        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        imageio.imsave(output_filename.replace('_no_light.png', '_back_directional_light.png'), rend_img)

        lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='directional', light_direction='left')
        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        imageio.imsave(output_filename.replace('_no_light.png', '_left_directional_light.png'), rend_img)

        lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='directional', light_direction='right')
        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        imageio.imsave(output_filename.replace('_no_light.png', '_right_directional_light.png'), rend_img)

    def render(
        self,
        mesh,
        params,
        rotation_angle_deg,
        light_type,
        light_location,
        light_direction,
        ambient_color,
        diffuse_color,
        specular_color,
        verbose=False,
        height=480,
        width=832,
    ):
        """
        Render the model with the given rotation angle, light type, light location, light direction, ambient color, diffuse color, specular color.

        Args:
            mesh: The mesh to render
            params: The camera parameters
            rotation_angle_deg: The rotation angle in degrees
            light_type: The type of light
            light_location: The location of the light
            light_direction: The direction of the light
            ambient_color: The ambient color
            diffuse_color: The diffuse color
            specular_color: The specular color

        Returns:
            rend_img: rendered image
        """
        # Extract camera parameters
        yaw, pitch, r, lookat_x, lookat_y, lookat_z = params
        
        # Get the mesh center (mean of min max of vertices)
        mesh_center = (mesh.vertices.min(dim=0).values + mesh.vertices.max(dim=0).values) / 2
        # mesh_center = mesh.vertices.mean(dim=0)
        
        # Translate vertices to center around origin
        vertices_centered = mesh.vertices - mesh_center
    
        # Convert rotation angle from degrees to radians
        rotation_angle_rad = math.radians(rotation_angle_deg)
        
        # Create rotation matrix for yaw rotation around Z-axis
        # Yaw rotation: rotation around the vertical axis (Z-axis)
        cos_yaw = math.cos(rotation_angle_rad)
        sin_yaw = math.sin(rotation_angle_rad)
        rotation_matrix = torch.tensor([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ], dtype=torch.float32).cuda()
        
        # Rotate vertices
        vertices_rotated = vertices_centered @ rotation_matrix.T
        
        # Translate back to original position
        vertices_rotated = vertices_rotated + mesh_center
        
        # Create a temporary mesh with rotated vertices
        rotated_mesh = copy.deepcopy(mesh)
        rotated_mesh.vertices = vertices_rotated
        
        # Update vertex normals if they exist
        if hasattr(rotated_mesh, 'face_normal'):
            rotated_mesh.face_normal = rotated_mesh.comput_face_normals(vertices_rotated, rotated_mesh.faces)
        if hasattr(rotated_mesh, 'vertex_attrs') and rotated_mesh.vertex_attrs is not None and rotated_mesh.vertex_attrs.shape[1] > 3:
            rotated_mesh.vertex_attrs[:, 3:] = rotated_mesh.comput_v_normals(vertices_rotated, rotated_mesh.faces)
        
        # Get extrinsics and intrinsics using the existing function
        # Keep the camera position the same (use original params)
        fov = 20
        extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics([yaw], [pitch], [r], [fov])

        if extr.dim() == 2:
            extr = extr.unsqueeze(0)

        if intr.dim() == 3:
            intr_batch = intr
            intr = intr[0]
        else:
            intr_batch = intr.unsqueeze(0)
        
        # Update lookat point to account for rotation around mesh center
        # lookat = torch.tensor([lookat_x, lookat_y, lookat_z], dtype=torch.float32).cuda()
        # lookat_centered = lookat - mesh_center
        # lookat_rotated = lookat_centered @ rotation_matrix.T
        # updated_lookat = lookat_rotated + mesh_center
        updated_lookat = mesh_center
        
        # Update extrinsics to use the rotated lookat point
        yaw_tensor = torch.tensor([yaw], dtype=torch.float32).cuda()
        pitch_tensor = torch.tensor([pitch], dtype=torch.float32).cuda()
        r_tensor = torch.tensor([r], dtype=torch.float32).cuda()
        
        orig = torch.stack([
            torch.sin(yaw_tensor) * torch.cos(pitch_tensor),
            torch.cos(yaw_tensor) * torch.cos(pitch_tensor),
            torch.sin(pitch_tensor),
        ], dim=1).squeeze() * r_tensor
        
        extr = extrinsics_look_at(orig.unsqueeze(0), updated_lookat.unsqueeze(0), torch.tensor([[0, 0, 1]], dtype=torch.float32).cuda())
        
        resolution = 512
        rend_img = None

        render_options = {'resolution': resolution, 'bg_color': (0, 0, 0)}

        if light_type == 'point':
            lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='point', light_location=light_location)
        elif light_type == 'directional':
            lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='directional', light_direction=light_direction)
        elif light_type == 'none':
            lighted_mesh = rotated_mesh
        else:
            raise ValueError(f"Invalid light type: {light_type}")

        result = render_frames(
            lighted_mesh, extr, intr_batch, render_options, None, verbose
        )
        rend_img = result['color'][0]

        size = min(width, height)
        rend_img = Image.fromarray(rend_img)
        rend_img = rend_img.resize((size, size), Image.Resampling.LANCZOS)
        # pad with black pixels to the size of (width, height)
        pad_img = Image.new('RGB', (width, height), (0, 0, 0))
        # paste the rendered image to the center of the new image
        pad_img.paste(rend_img, ((width - size) // 2, (height - size) // 2))
        return pad_img

    def render_normal_map(
        self,
        mesh,
        params,
        rotation_angle_deg=0,
        use_abs_coor=True,
        verbose=False,
        height=512,
        width=512,
    ):
        """
        Render normal map from the mesh, similar to Hunyuan3D.
        
        Args:
            mesh: The mesh to render
            params: The camera parameters (yaw, pitch, r, lookat_x, lookat_y, lookat_z)
            rotation_angle_deg: The rotation angle in degrees
            use_abs_coor: If True, use absolute world coordinates for normals. 
                         If False, use camera-space coordinates (default: True)
            verbose: Whether to print verbose information
            height: Output image height (default: 512)
            width: Output image width (default: 512)
        
        Returns:
            PIL.Image: Normal map as RGB image where:
                - R channel: X normal component (normalized to [0, 1])
                - G channel: Y normal component (normalized to [0, 1])
                - B channel: Z normal component (normalized to [0, 1])
                - Background pixels: [1, 1, 1] (white)
        """
        # Extract camera parameters
        yaw, pitch, r, lookat_x, lookat_y, lookat_z = params
        
        # Get the mesh center
        mesh_center = (mesh.vertices.min(dim=0).values + mesh.vertices.max(dim=0).values) / 2
        
        # Translate vertices to center around origin
        vertices_centered = mesh.vertices - mesh_center
        
        # Convert rotation angle from degrees to radians
        rotation_angle_rad = math.radians(rotation_angle_deg)
        
        # Create rotation matrix for yaw rotation around Z-axis
        cos_yaw = math.cos(rotation_angle_rad)
        sin_yaw = math.sin(rotation_angle_rad)
        rotation_matrix = torch.tensor([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ], dtype=torch.float32).cuda()
        
        # Rotate vertices
        vertices_rotated = vertices_centered @ rotation_matrix.T
        vertices_rotated = vertices_rotated + mesh_center
        
        # Create a temporary mesh with rotated vertices
        normal_mesh = copy.deepcopy(mesh)
        normal_mesh.vertices = vertices_rotated
        
        # Compute normals
        if hasattr(normal_mesh, 'vertex_attrs') and normal_mesh.vertex_attrs is not None and normal_mesh.vertex_attrs.shape[1] >= 6:
            normals = normal_mesh.vertex_attrs[:, 3:6]
            # Rotate normals if mesh was rotated
            if rotation_angle_deg != 0:
                normals = normals.matmul(rotation_matrix.T)
        else:
            normals = normal_mesh.comput_v_normals(vertices_rotated, normal_mesh.faces)
            if normals is None:
                raise ValueError("Failed to compute vertex normals")
        
        # Normalize normals
        normals = F.normalize(normals, dim=1)
        
        # If not using absolute coordinates, transform to camera space
        if not use_abs_coor:
            # Get camera position
            yaw_tensor = torch.tensor([yaw], dtype=torch.float32).cuda()
            pitch_tensor = torch.tensor([pitch], dtype=torch.float32).cuda()
            r_tensor = torch.tensor([r], dtype=torch.float32).cuda()
            
            camera_origin = torch.stack([
                torch.sin(yaw_tensor) * torch.cos(pitch_tensor),
                torch.cos(yaw_tensor) * torch.cos(pitch_tensor),
                torch.sin(pitch_tensor)
            ], dim=0).squeeze() * r_tensor
            
            # Transform normals to camera space
            view_dir = F.normalize(camera_origin - mesh_center, dim=0)
            # Simple approximation: transform normals relative to view direction
            # This is a simplified version - for exact camera-space normals, 
            # you'd need the full view matrix transformation
            pass  # Keep world-space normals for now
        
        # Normalize normals to [0, 1] range for RGB encoding
        # Normal maps: (normal + 1) / 2 to convert from [-1, 1] to [0, 1]
        normal_rgb = (normals + 1.0) / 2.0
        normal_rgb = normal_rgb.clamp(0.0, 1.0)
        
        # Set vertex colors to normal values
        if normal_mesh.vertex_attrs is None:
            normal_mesh.vertex_attrs = torch.zeros((normal_mesh.vertices.shape[0], 6), device=normal_mesh.vertices.device)
        normal_mesh.vertex_attrs[:, :3] = normal_rgb
        
        # Get extrinsics and intrinsics
        fov = 20
        extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics([yaw], [pitch], [r], [fov])
        
        if extr.dim() == 2:
            extr = extr.unsqueeze(0)
        
        if intr.dim() == 3:
            intr_batch = intr
        else:
            intr_batch = intr.unsqueeze(0)
        
        # Update lookat point
        updated_lookat = mesh_center
        
        # Update extrinsics
        yaw_tensor = torch.tensor([yaw], dtype=torch.float32).cuda()
        pitch_tensor = torch.tensor([pitch], dtype=torch.float32).cuda()
        r_tensor = torch.tensor([r], dtype=torch.float32).cuda()
        
        orig = torch.stack([
            torch.sin(yaw_tensor) * torch.cos(pitch_tensor),
            torch.cos(yaw_tensor) * torch.cos(pitch_tensor),
            torch.sin(pitch_tensor),
        ], dim=1).squeeze() * r_tensor
        
        extr = extrinsics_look_at(orig.unsqueeze(0), updated_lookat.unsqueeze(0), torch.tensor([[0, 0, 1]], dtype=torch.float32).cuda())
        
        resolution = 512
        render_options = {'resolution': resolution, 'bg_color': (1, 1, 1)}  # White background for normal maps
        
        result = render_frames(
            normal_mesh, extr, intr_batch, render_options, None, verbose
        )
        normal_img = result['color'][0]
        
        # Resize and pad if needed
        size = min(width, height)
        normal_img = Image.fromarray(normal_img)
        normal_img = normal_img.resize((size, size), Image.Resampling.LANCZOS)
        pad_img = Image.new('RGB', (width, height), (0, 0, 0))  # Black background
        pad_img.paste(normal_img, ((width - size) // 2, (height - size) // 2))
        
        return pad_img

    def render_position_map(
        self,
        mesh,
        params,
        rotation_angle_deg=0,
        verbose=False,
        height=512,
        width=512,
    ):
        """
        Render position map from the mesh, similar to Hunyuan3D.
        
        Args:
            mesh: The mesh to render
            params: The camera parameters (yaw, pitch, r, lookat_x, lookat_y, lookat_z)
            rotation_angle_deg: The rotation angle in degrees
            verbose: Whether to print verbose information
            height: Output image height (default: 512)
            width: Output image width (default: 512)
        
        Returns:
            PIL.Image: Position map as RGB image where:
                - R channel: X coordinate (normalized to [0, 1])
                - G channel: Y coordinate (normalized to [0, 1])
                - B channel: Z coordinate (normalized to [0, 1])
                - Background pixels: [1, 1, 1] (white)
        """
        # Extract camera parameters
        yaw, pitch, r, lookat_x, lookat_y, lookat_z = params
        
        # Get the mesh center
        mesh_center = (mesh.vertices.min(dim=0).values + mesh.vertices.max(dim=0).values) / 2
        
        # Get mesh bounding box to normalize positions
        mesh_min = mesh.vertices.min(dim=0).values
        mesh_max = mesh.vertices.max(dim=0).values
        mesh_size = mesh_max - mesh_min
        mesh_size = torch.clamp(mesh_size, min=1e-6)  # Avoid division by zero
        
        # Translate vertices to center around origin
        vertices_centered = mesh.vertices - mesh_center
        
        # Convert rotation angle from degrees to radians
        rotation_angle_rad = math.radians(rotation_angle_deg)
        
        # Create rotation matrix for yaw rotation around Z-axis
        cos_yaw = math.cos(rotation_angle_rad)
        sin_yaw = math.sin(rotation_angle_rad)
        rotation_matrix = torch.tensor([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ], dtype=torch.float32).cuda()
        
        # Rotate vertices
        vertices_rotated = vertices_centered @ rotation_matrix.T
        vertices_rotated = vertices_rotated + mesh_center
        
        # Create a temporary mesh with rotated vertices
        position_mesh = copy.deepcopy(mesh)
        position_mesh.vertices = vertices_rotated
        
        # Normalize positions to [0, 1] range
        # Similar to Hunyuan3D: tex_position = 0.5 - vtx_pos / scale_factor
        # We'll use a simpler approach: normalize to bounding box
        vertices_normalized = (vertices_rotated - mesh_min) / mesh_size
        # Clamp to [0, 1] and ensure background is distinguishable
        vertices_normalized = vertices_normalized.clamp(0.0, 1.0)
        
        # Set vertex colors to position values
        if position_mesh.vertex_attrs is None:
            position_mesh.vertex_attrs = torch.zeros((position_mesh.vertices.shape[0], 6), device=position_mesh.vertices.device)
        position_mesh.vertex_attrs[:, :3] = vertices_normalized
        
        # Get extrinsics and intrinsics
        fov = 20
        extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics([yaw], [pitch], [r], [fov])
        
        if extr.dim() == 2:
            extr = extr.unsqueeze(0)
        
        if intr.dim() == 3:
            intr_batch = intr
        else:
            intr_batch = intr.unsqueeze(0)
        
        # Update lookat point
        updated_lookat = mesh_center
        
        # Update extrinsics
        yaw_tensor = torch.tensor([yaw], dtype=torch.float32).cuda()
        pitch_tensor = torch.tensor([pitch], dtype=torch.float32).cuda()
        r_tensor = torch.tensor([r], dtype=torch.float32).cuda()
        
        orig = torch.stack([
            torch.sin(yaw_tensor) * torch.cos(pitch_tensor),
            torch.cos(yaw_tensor) * torch.cos(pitch_tensor),
            torch.sin(pitch_tensor),
        ], dim=1).squeeze() * r_tensor
        
        extr = extrinsics_look_at(orig.unsqueeze(0), updated_lookat.unsqueeze(0), torch.tensor([[0, 0, 1]], dtype=torch.float32).cuda())
        
        resolution = 512
        render_options = {'resolution': resolution, 'bg_color': (1, 1, 1)}  # White background for position maps
        
        result = render_frames(
            position_mesh, extr, intr_batch, render_options, None, verbose
        )
        position_img = result['color'][0]
        
        # Resize and pad if needed
        size = min(width, height)
        position_img = Image.fromarray(position_img)
        position_img = position_img.resize((size, size), Image.Resampling.LANCZOS)
        pad_img = Image.new('RGB', (width, height), (0, 0, 0))  # Black background
        pad_img.paste(position_img, ((width - size) // 2, (height - size) // 2))
        
        return pad_img

    def render_normal_multiview(
        self,
        mesh,
        params,
        camera_elevs,
        camera_azims,
        use_abs_coor=True,
        height=480,
        width=832,
    ):
        """
        Render normal maps from multiple viewpoints, similar to Hunyuan3D's render_normal_multiview.
        
        Args:
            mesh: The mesh to render
            params: The camera parameters (yaw, pitch, r, lookat_x, lookat_y, lookat_z)
            camera_elevs: List of elevation angles in degrees
            camera_azims: List of azimuth angles in degrees
            use_abs_coor: If True, use absolute world coordinates for normals (default: True)
            height: Output image height (default: 480)
            width: Output image width (default: 832)
        
        Returns:
            List[PIL.Image]: List of normal maps, one for each viewpoint
        """
        normal_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            # Convert elevation/azimuth to yaw/pitch for the render function
            # Note: This is a simplified conversion - you may need to adjust based on your coordinate system
            yaw = azim
            pitch = elev
            
            # Create modified params with new yaw/pitch
            modified_params = (yaw, pitch, params[2], params[3], params[4], params[5])
            
            normal_map = self.render_normal_map(
                mesh,
                modified_params,
                rotation_angle_deg=0,
                use_abs_coor=use_abs_coor,
                height=height,
                width=width,
            )
            normal_maps.append(normal_map)
        
        return normal_maps

    def render_position_multiview(
        self,
        mesh,
        params,
        camera_elevs,
        camera_azims,
        height=480,
        width=832,
    ):
        """
        Render position maps from multiple viewpoints, similar to Hunyuan3D's render_position_multiview.
        
        Args:
            mesh: The mesh to render
            params: The camera parameters (yaw, pitch, r, lookat_x, lookat_y, lookat_z)
            camera_elevs: List of elevation angles in degrees
            camera_azims: List of azimuth angles in degrees
            height: Output image height (default: 480)
            width: Output image width (default: 832)
        
        Returns:
            List[PIL.Image]: List of position maps, one for each viewpoint
        """
        position_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            # Convert elevation/azimuth to yaw/pitch for the render function
            yaw = azim
            pitch = elev
            
            # Create modified params with new yaw/pitch
            modified_params = (yaw, pitch, params[2], params[3], params[4], params[5])
            
            position_map = self.render_position_map(
                mesh,
                modified_params,
                rotation_angle_deg=0,
                height=height,
                width=width,
            )
            position_maps.append(position_map)
        
        return position_maps

    def process_frame_index(self, major_class_name, frame_index, frame_selecting_strategy="farthest"):
        # cope with frame_index
        glb_files = [f for f in os.listdir(os.path.join(self.data_root, major_class_name)) if f.endswith("_output_model.glb")]
        avail_frame_indices = [f.split("_")[0] for f in glb_files]
        avail_frame_indices = [int(f) for f in avail_frame_indices]
        avail_frame_indices.sort()
        if frame_index in ["head", "tail", "middle"]:
            if frame_index == "head":
                target_frame_index = avail_frame_indices[0]
            elif frame_index == "tail":
                target_frame_index = avail_frame_indices[-1]
            else:
                target_frame_index = avail_frame_indices[1]
            # format to 5 digits
            if self.dataset == "VIPSeg":
                target_frame_index = str(target_frame_index)
            else:
                target_frame_index = str(target_frame_index).zfill(5)
        else:
            frame_index_int = int(frame_index)
            frame_indices_int = [int(f) for f in avail_frame_indices]
            frame_indices_arr = np.array(frame_indices_int)
            # print(f"frame_indices_arr: {frame_indices_arr}")
            # print(f"frame_index_int: {frame_index_int}")
            # print(f"frame_selecting_strategy: {frame_selecting_strategy}")
            if frame_selecting_strategy == "farthest":
                target_frame_index = frame_indices_int[np.argmax(np.abs(frame_indices_arr - frame_index_int))]
            elif frame_selecting_strategy == "nearest":
                target_frame_index = frame_indices_int[np.argmin(np.abs(frame_indices_arr - frame_index_int))]
            elif frame_selecting_strategy == "random":
                target_frame_index = np.random.choice(frame_indices_int)
            elif frame_selecting_strategy == "exact":
                # check if the frame is available
                if frame_index_int not in frame_indices_int:
                    print(f"Frame {frame_index} is not available, fallback to farthest strategy.")
                    target_frame_index = frame_indices_int[np.argmax(np.abs(frame_indices_arr - frame_index_int))]
                else:
                    target_frame_index = frame_index_int
            else:
                raise ValueError(f"Invalid frame selecting strategy: {frame_selecting_strategy}")
            if self.dataset == "VIPSeg":
                target_frame_index = str(target_frame_index)
            elif self.dataset == "DAVIS17":
                target_frame_index = str(target_frame_index).zfill(5)
            elif self.dataset == "ROSE":
                target_frame_index = str(target_frame_index).zfill(5)
            else:
                raise ValueError(f"Invalid dataset: {self.dataset}")
        # print(f"Selected frame index for {major_class_name}, frame index: {frame_index}, with {frame_selecting_strategy} strategy: {target_frame_index}")
        return target_frame_index

    def load_mesh_and_render(
        self,
        major_class_name,
        frame_index,
        render_config: list[RenderConfig],
        frame_selecting_strategy="farthest",
        verbose=False,
        height=480,
        width=832,
    ):
        """
        Load the mesh and render the model with the given render configuration.

        Args:
            major_class_name: The name of the major class
            frame_index: The index of the frame, selected from 'head', 'tail', 'middle' or a 5 digits with leading zeros string
            render_config: The configuration for the render, a list of RenderConfig objects
            frame_selecting_strategy: The strategy for selecting the frame, only valid when frame_index is a 5 digits with leading zeros string
                "exact": select the frame with the exact index, may lead to error if the frame is not available
                "farthest": select the frame with the farthest index from the current frame
                "nearest": select the frame with the nearest index from the current frame
                "random": select the frame randomly from the available frames
        Returns:
            len(render_config) rendered images, each image is a numpy array of shape (512, 512, 3)
        """
        frame_index = self.process_frame_index(major_class_name, frame_index, frame_selecting_strategy)
        if self.dataset == "DAVIS17":
            img_path = os.path.join('/inspurfs/group/mayuexin/xiaqi/DAVIS-2017/v2m4_results/masked_samples_new', major_class_name, f'{frame_index}_rotated_*deg_textured_no_light.png')
        elif self.dataset == "VIPSeg":
            img_path = os.path.join('/inspurfs/group/mayuexin/xiaqi/VIPSeg/v2m4_results', major_class_name, f'{frame_index}_rotated_*deg_textured_no_light.png')
        elif self.dataset == "ROSE":
            img_path = os.path.join('/inspurfs/group/mayuexin/xiaqi/ROSE-Dataset/v2m4_results', major_class_name, f'{frame_index}_rotated_*deg_textured_no_light.png')
        else:
            raise ValueError(f"Invalid dataset: {self.dataset}")
        img_paths = glob.glob(img_path)
        
        # sort img_paths by the rotation angle
        img_paths.sort(key=lambda x: int(x.split("_")[-4][:-3]))
        # print(f"img_paths: {img_paths}")
        if len(img_paths) == 0:
            raise ValueError(f'No image found for {major_class_name} {frame_index}')
        rendered_images = []
        for img_path in img_paths:
            img = Image.open(img_path)
            # create a target resolution black canvas and paste the img to the center
            target_canvas = Image.new('RGB', (width, height), (0, 0, 0))
            target_canvas.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
            rendered_images.append(target_canvas)
        # mesh = self.load_mesh(major_class_name, frame_index)
        
        # params = self.load_camera_parameters(major_class_name, frame_index)
        # rendered_images = []
        # for config in render_config:
        #     if config.align_to_x_axis:
        #         mesh = self.align_mesh_bbox_to_x_axis(mesh)
        #     if not config.texture:
        #         mesh = self.get_mesh_without_texture(mesh)
        #     rendered_image = self.render(
        #         mesh,
        #         params,
        #         config.rotation_angle_deg,
        #         config.light_type,
        #         config.light_location,
        #         config.light_direction,
        #         config.ambient_color,
        #         config.diffuse_color,
        #         config.specular_color,
        #         verbose=verbose,
        #         height=height,
        #         width=width,
        #     )
        #     rendered_images.append(rendered_image)
        return rendered_images

if __name__ == "__main__":
    import os
    assert os.getenv("DAVIS17_V2M4_RESULTS_PATH") is not None, "DAVIS17_V2M4_RESULTS_PATH is not set, please source path_setup.sh first!"
    mesh_loader = MeshLoader(data_root=os.getenv("DAVIS17_V2M4_RESULTS_PATH"), dataset="DAVIS17")
    render_configs = [RenderConfig(rotation_angle_deg=0, texture=True, align_to_x_axis=True,
                                  light_type='none'),
                                  RenderConfig(rotation_angle_deg=90, texture=True, align_to_x_axis=True,
                                  light_type='none'),
                                  RenderConfig(rotation_angle_deg=180, texture=True, align_to_x_axis=True,
                                  light_type='none'),
                                  RenderConfig(rotation_angle_deg=270, texture=True, align_to_x_axis=True,
                                  light_type='none'),
                                #   RenderConfig(rotation_angle_deg=60, texture=True, align_to_x_axis=False,
                                #   light_type='none') ,
                                #   RenderConfig(rotation_angle_deg=-60, texture=True, align_to_x_axis=False,
                                #   light_type='none') ,
                                #   RenderConfig(rotation_angle_deg=120, texture=False, align_to_x_axis=False,
                                #   light_type='point', light_location='front'),
                                #   RenderConfig(rotation_angle_deg=-120, texture=False, align_to_x_axis=False,
                                #   light_type='point', light_location='front'),
                                  ]
    rendered_images = mesh_loader.load_mesh_and_render("bear/1", "head", render_configs, frame_selecting_strategy="farthest")
    os.makedirs("bear_renders", exist_ok=True)
    for i, rendered_image in enumerate(rendered_images):
        texture_type = "textured" if render_configs[i].texture else "untextured"    
        align_type = "aligned" if render_configs[i].align_to_x_axis else "unaligned"
        light_type = render_configs[i].light_type
        light_location = render_configs[i].light_location
        light_direction = render_configs[i].light_direction
        image_name = os.path.join("bear_renders", f"bear_{render_configs[i].rotation_angle_deg}deg_{texture_type}_{align_type}_{light_type}_{light_location}_{light_direction}.png")
        imageio.imsave(image_name, rendered_image)
    
    # Test normal map and position map rendering from same viewpoints as render_configs
    print("\n" + "="*50)
    print("Testing Normal Map and Position Map Rendering")
    print("Rendering from same viewpoints and rotations as render_configs")
    print("="*50)
    
    frame_index = mesh_loader.process_frame_index("bear/1", "head", "farthest")
    # Load mesh and camera parameters
    mesh = mesh_loader.load_mesh("bear/1", frame_index)
    params = mesh_loader.load_camera_parameters("bear/1", frame_index)
    
    os.makedirs("bear_normal_position_maps", exist_ok=True)
    
    # Render normal and position maps for each render_config
    print("\nRendering normal and position maps for each render_config...")
    for i, config in enumerate(render_configs):
        print(f"\nProcessing render_config {i}: rotation={config.rotation_angle_deg}deg, align={config.align_to_x_axis}")
        
        # Prepare mesh (apply alignment if needed)
        test_mesh = copy.deepcopy(mesh)
        if config.align_to_x_axis:
            test_mesh = mesh_loader.align_mesh_bbox_to_x_axis(test_mesh)
        
        # Render normal map with absolute coordinates
        normal_map = mesh_loader.render_normal_map(
            test_mesh, 
            params, 
            rotation_angle_deg=config.rotation_angle_deg,
            use_abs_coor=True,
            height=480,
            width=832
        )
        
        # Render normal map with camera-space coordinates
        normal_map_cam = mesh_loader.render_normal_map(
            test_mesh, 
            params, 
            rotation_angle_deg=config.rotation_angle_deg,
            use_abs_coor=False,
            height=480,
            width=832
        )
        
        # Render position map
        position_map = mesh_loader.render_position_map(
            test_mesh, 
            params, 
            rotation_angle_deg=config.rotation_angle_deg,
            height=480,
            width=832
        )
        
        # Save with descriptive filenames matching render_config naming
        align_suffix = "aligned" if config.align_to_x_axis else "unaligned"
        base_name = f"bear_{config.rotation_angle_deg}deg_{align_suffix}"
        
        normal_filename = os.path.join("bear_normal_position_maps", f"{base_name}_normal_abs.png")
        normal_map.save(normal_filename)
        print(f"  Saved: {normal_filename}")
        
        normal_cam_filename = os.path.join("bear_normal_position_maps", f"{base_name}_normal_cam.png")
        normal_map_cam.save(normal_cam_filename)
        print(f"  Saved: {normal_cam_filename}")
        
        position_filename = os.path.join("bear_normal_position_maps", f"{base_name}_position.png")
        position_map.save(position_filename)
        print(f"  Saved: {position_filename}")
    
    print("\n" + "="*50)
    print("Normal Map and Position Map Testing Complete!")
    print("="*50)