import sys
import os

from npbgpp.npbgplusplus.modeling.refiner.unet import RefinerUNet

from pytorch3d.io import load_obj, load_ply, save_obj
import cv2
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import yaml

from .quad_rasterizer import QuadRasterizer

from NeuS.models.dataset import load_K_Rt_from_P
from src.utils.util import tensor2image
from src.utils.geometry import project_orient_to_camera, soft_interpolate, hard_interpolate


class Renderer(nn.Module):
    def __init__(
        self,
        config,
        device='cuda',
        save_dir='strands_rasterizer'
    ):
        super(Renderer, self).__init__()
            
        self.image_size = config.get('image_size', 512)
        self.out_channels = config.get('out_channels', -1)
        self.logging_freq = config.get('logging_freq', 5000)
        self.feat_size = config.get('feat_size', -1)
        self.num_strands = config.get('num_strands', -1)
        self.use_orients_cond = config.get('use_orients_cond', False)
        self.use_silh = config.get('use_silh', False)
        self.device = device

         # Load head mesh for occlusion
        if config.get('mesh_path', -1).split('.')[-1] == 'obj':
            verts, faces, _ = load_obj(config.get('mesh_path', -1))
            occlusion_faces = faces.verts_idx
        else:
            verts, faces = load_ply(config.get('mesh_path', -1))
            occlusion_faces = faces

        verts = verts.to(self.device)
        occlusion_faces = occlusion_faces.to(self.device)
        
        # Init rasterizers and renderers
        self.rasterizer = QuadRasterizer(
                                    render_size=self.image_size,
                                    feats_dim=self.feat_size + self.use_silh,
                                    head_mesh=(verts, occlusion_faces),
                                    use_silh=self.use_silh,
                                    use_orients_cond=self.use_orients_cond,
                                ).to(self.device)

        self.refiner_unet = RefinerUNet(
            conv_block='gated',
            num_input_channels=self.feat_size + 4 * self.use_orients_cond ,
            feature_scale=4,
            num_output_channels=self.out_channels
        ).to(self.device)

        self.save_dir = os.path.join(save_dir, 'strands_rasterizers')
        os.makedirs(self.save_dir, exist_ok=True)


    def forward(self, strands_origins, z_app, raster_dict, iter):

        rasterized = self.rasterizer(
            strands_origins,
            torch.cat((z_app, torch.ones(self.num_strands, 1).cuda()), dim=1),
            raster_dict['cam_extr'],
            raster_dict['cam_intr']
        )

        rasterized_features = rasterized[:, : self.feat_size, :, :]
        raster_dict['rasterized_img'] = rasterized_features[0]
        
        if self.use_orients_cond:
            raster_idxs = rasterized[0, self.feat_size + 1:, :, :]
            rasterized = rasterized[:, :self.feat_size + 1, :, :]

            orients = torch.zeros_like(strands_origins)
            orients[:, :orients.shape[1] - 1] = (strands_origins[:, 1:] - strands_origins[:, :-1])
            orients[:, orients.shape[1] - 1: ] = orients[:, orients.shape[1] - 2: orients.shape[1] - 1]
            orients = orients.reshape(-1, 3)
            
            r = raster_idxs
            r[r == 0] = -1
            valid_pixels = r[r != -1] 

            strands_origins = soft_interpolate(valid_pixels.cuda(), strands_origins.view(-1, 3))
            
            # Hard rasterize orientations
            hard_orients = hard_interpolate(valid_pixels.cuda(), orients) 

            # Project orients and points from 3d to 2d with camera
            projected_orients = project_orient_to_camera(hard_orients.unsqueeze(1), strands_origins, cam_intr=raster_dict['cam_intr'],  cam_extr=raster_dict['cam_extr']) 

            plane_orients = torch.zeros(self.image_size, self.image_size, 1, device=hard_orients.device)
            plane_orients[r[0]!=-1, :] = projected_orients
            raster_dict['pred_orients'] = plane_orients.permute(2, 0, 1)

            orient_cos_fine = plane_orients.cos()
            orient_fine_cam = torch.cat(
                [
                    orient_cos_fine * (orient_cos_fine >= 0),
                    plane_orients.sin(),
                    orient_cos_fine.abs() * (orient_cos_fine < 0)
                ],
                dim=-1
            )
            raster_dict['visual_pred_orients'] = (orient_fine_cam.detach().cpu().numpy() * 255).clip(0, 255)
        
        raster_dict['pred_silh'] = rasterized[0, self.feat_size:, :, :]
        # Inpaint holes with unet
        inp = [rasterized_features]
        if self.use_orients_cond:
            inp.append(plane_orients.permute(2, 0, 1)[None])
            inp.append(orient_fine_cam.permute(2, 0, 1)[None])
        inp = torch.cat(inp, dim=1)

        rgb_hair_image = self.refiner_unet([inp]) #[1, 3, 512, 512]
        raster_dict['pred_rgb'] = rgb_hair_image[0, :3]
        return raster_dict

    def calculate_losses(self, raster_dict, iter):
        losses = {}

        color_strands_error = (raster_dict['pred_rgb'] - raster_dict['gt_rgb']) * raster_dict['gt_silh']
        
        losses['l1'] = F.l1_loss(
            color_strands_error,
            torch.zeros_like(color_strands_error), reduction='sum'
        ) / (raster_dict['gt_silh'].sum() + 1e-5)            
        
        losses['silh'] = ((raster_dict['pred_silh'] - raster_dict['gt_silh']).abs()).mean()

        if iter % self.logging_freq == 0:
            self.visualize(raster_dict, iter)
            
        return losses
    
    
    def visualize(self, visuals, iter):
        cv2.imwrite(os.path.join(self.save_dir, f'rasterize_{iter}_resol_512.png'), tensor2image(visuals['rasterized_img'][:3, :, :])) 

        cv2.imwrite(os.path.join(self.save_dir, f'pred_silh_{iter}_resol_512.png'), tensor2image(visuals['pred_silh'][:3, :, :]))
        cv2.imwrite(os.path.join(self.save_dir, f'gt_silh_{iter}_resol_512.png'), tensor2image(visuals['gt_silh'][:3, :, :]))

        cv2.imwrite(
            os.path.join(self.save_dir, f'pred_hair_strands_{iter}.png'),
            tensor2image(visuals['pred_rgb'][:3, :, :] * visuals['gt_silh'] + 1 - visuals['gt_silh'])
        )
        cv2.imwrite(
            os.path.join(self.save_dir, f'gt_hair_strands_{iter}.png'),
            tensor2image(visuals['gt_rgb'][:3, :, :] * visuals['gt_silh'] + 1 - visuals['gt_silh'])
        )

        if self.use_orients_cond or self.use_gabor_loss:
            cv2.imwrite(os.path.join(self.save_dir, f'pred_hair_or_{iter}.png'), visuals['visual_pred_orients'])
            cv2.imwrite(os.path.join(self.save_dir, f'gt_hair_or_{iter}.png'), visuals['visual_gt_orients'])
            
