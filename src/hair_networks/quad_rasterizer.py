from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer import MeshRasterizer,  RasterizationSettings, TexturesVertex
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from .soft_shader import SoftShader
import torch.nn.functional as F

from src.utils.build_quads import build_quads

class QuadRasterizer(torch.nn.Module):

    def __init__(
        self,
        render_size: int,
        feats_dim: int,
        antialiasing_factor: int = 1,
        head_mesh: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_gpu: bool = True,
        faces_per_pixel: int = 16,
        znear: Union[float, torch.Tensor] = 0.1,
        zfar: Union[float, torch.Tensor] = 10.0,
        use_silh=False,
        use_orients_cond=False,

    ):
        super().__init__()

        self.use_orients_cond = use_orients_cond
        self.use_silh = use_silh
        self.render_size = render_size
        self.feats_dim = feats_dim
        self.antialiasing_factor = antialiasing_factor
        self.use_gpu = use_gpu

        raster_settings = RasterizationSettings(
            image_size=self.render_size * antialiasing_factor,
            blur_radius=1e-4,
            faces_per_pixel=faces_per_pixel,
            bin_size=0,
            cull_backfaces=False,
            perspective_correct=False
        )
        
        self.head_mesh: Meshes = None

        if head_mesh is not None:
            self.set_head_mesh(head_mesh)
            self.num_head_faces = self.head_mesh.faces_list()[0].shape[0]
        else:
            self.num_head_faces = None

        self.rasterizer = MeshRasterizer(raster_settings=raster_settings)
        
        self.shader = SoftShader(
            image_size=render_size,
            feats_dim=feats_dim,
            sigma=1e-5,
            gamma=1e-5,
            zfar=zfar,
            znear=znear,
            num_head_faces=self.num_head_faces,
            use_orients_cond=use_orients_cond,
        )

        if self.use_gpu:
            self.rasterizer = self.rasterizer.cuda()
            self.shader = self.shader.cuda()

    def forward(self, hair: torch.Tensor, feats: torch.Tensor, cam_extr: torch.Tensor, cam_intr: torch.Tensor):

        cameras = cameras_from_opencv_projection(
            camera_matrix=cam_intr.unsqueeze(0),
            R=cam_extr[:3, :3].unsqueeze(0),
            tvec=cam_extr[:3, 3].unsqueeze(0),
            image_size=torch.tensor([self.render_size, self.render_size]).unsqueeze(0)
        ).cuda()

        hair_verts, hair_faces, indices = build_quads(hair @ cameras.R[0], w=0.0005)
        hair_verts = hair_verts @ cameras.R[0].T

        texture_tensor = \
            feats.unsqueeze(1) + \
            torch.zeros(len(hair_verts), hair_verts.shape[1], self.feats_dim, device=feats.device)
        
        if self.use_orients_cond:
            texture_tensor = texture_tensor.reshape(-1, self.feats_dim)
            texture_tensor = torch.cat((texture_tensor, indices), dim=-1)

        hair_texture = TexturesVertex([texture_tensor.reshape(-1, self.feats_dim + self.use_orients_cond)])

        hair_mesh = Meshes(
            verts=[hair_verts.reshape(-1, 3)],
            faces=[hair_faces.reshape(-1, 3)],
            textures=hair_texture
        )

        if self.use_gpu:
            cameras = cameras.cuda()
            hair_mesh = hair_mesh.cuda()

        self.rasterizer.cameras = cameras

        if self.head_mesh is not None:
            meshes = join_meshes_as_scene([hair_mesh, self.head_mesh])
        else:
            meshes = hair_mesh
            
        fragments = self.rasterizer(meshes)
        images, mask, colors = self.shader(fragments, meshes)


        if self.use_orients_cond:
            images_1 = colors[:, :, :, 0, :].permute(0, 3, 1, 2).to(images.dtype)
            images[:, -1:] = images_1[:, -1:]

        if self.antialiasing_factor > 1:
            images = torch.nn.functional.avg_pool2d(
                images,
                kernel_size=self.antialiasing_factor,
                stride=self.antialiasing_factor
            )
        
        return images


    def set_head_mesh(self, head_mesh: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        
        head_verts, head_faces = head_mesh
        head_texture = torch.zeros(len(head_verts), self.feats_dim + self.use_orients_cond)
        if self.use_gpu:
            head_texture = head_texture.cuda()
        
        self.head_mesh = Meshes(
            verts=[head_verts],
            faces=[head_faces],
            textures=TexturesVertex([head_texture])
        )
        
        if self.use_gpu:
            self.head_mesh = self.head_mesh.cuda()