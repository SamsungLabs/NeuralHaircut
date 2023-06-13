import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
import json
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import imageio
from pathlib import Path
import toml
import math
from .cameras import OptimizableCameras
from pytorch3d.io import load_obj
import pickle

import sys
sys.path.append(os.path.join(sys.path[0], '../..'))
from NeuS.models.dataset import load_K_Rt_from_P
from src.utils.util import glob_imgs, tensor2image


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = Path(conf.get('data_dir'))
        self.render_cameras_name = conf.get('render_cameras_name')
        self.object_cameras_name = conf.get('object_cameras_name')


        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'image')))


        self.orientations_np, self.hair_masks_np, self.hair_masks_np_white_gray = None, None, None
        self.num_bins = conf.get('orient_num_bins')  

        fitted_camera_path = conf.get('fitted_camera_path', '')

        self.masks_lis  = sorted(glob_imgs(os.path.join(self.data_dir, 'mask')))

        self.hair_masks_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'hair_mask')))
                
        self.orientations_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'orientation_maps')))
        self.variance_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'confidence_maps')))
        
        if self.conf.get('mask_based_ray_sampling', False):
            self.binary_hair_masks = self.hair_masks[..., 0] > self.conf.get('mask_binarization_threshold', 0.5)

        self.n_images = len(self.images_lis)

        # H3DS dataset
        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.filter_views()
        print("Number of views:", self.n_images)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 255.0

        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 255.0

        self.hair_masks_np = np.stack([cv.imread(im_name) for im_name in self.hair_masks_lis]) / 255.0     
        
        self.orientations_np = np.stack([cv.imread(im_name) for im_name in self.orientations_lis]) / float(self.num_bins) * math.pi
        self.variance_np = np.stack([np.load(im_name) for im_name in self.variance_lis])
        self.hair_masks = torch.from_numpy(self.hair_masks_np.astype(np.float32)).cpu() 
        self.orientation_maps = torch.from_numpy(self.orientations_np.astype(np.float32)).cpu()
        self.variance_maps = torch.from_numpy(self.variance_np.astype(np.float32)).cpu()[..., None]
        # Convert variances to confidences
            
        self.confidence_maps = 1 / self.variance_maps ** 2

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
        

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]

        print(fitted_camera_path)
        if fitted_camera_path:
            camera_model = OptimizableCameras(len(self.images_lis), pretrain_path=fitted_camera_path).to(self.device)
            with torch.no_grad():
                print('camera model create')
                self.intrinsics_all, self.pose_all = camera_model(torch.arange(len(self.images_lis)), self.intrinsics_all, self.pose_all)

        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all_inv = torch.inverse(self.pose_all)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        if self.conf.get('mask_based_ray_sampling', False):
            self.binary_masks = self.masks[..., 0] > self.conf.get('mask_binarization_threshold', 0.5)

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        if 'scale_mat_0' in camera_dict.keys():
            # Object scale mat: region of interest to **extract mesh**
            object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
            object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
            object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
            object_bbox_min = object_bbox_min[:, 0]
            object_bbox_max = object_bbox_max[:, 0]
        self.object_bbox_min = object_bbox_min[:3]
        self.object_bbox_max = object_bbox_max[:3]

        # Calculate radii for all pixels in all images
        self.radii = torch.empty(self.n_images, self.W, self.H)
        for img_idx in range(self.n_images):
            tx = torch.linspace(0, self.W - 1, self.W)  #
            ty = torch.linspace(0, self.H - 1, self.H)  # TODO: this code seems to be incorrect, because meshgrid reverses the order
            pixels_x, pixels_y = torch.meshgrid(tx, ty) #       of values: pixels_x change across i, pixels_y -- across j 
            p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
            v = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None].cuda()).squeeze()  # W, H, 3
            
            ###
            ### Code below is borrowed from https://github.com/hjxwhy/mipnerf_pl/blob/master/datasets/datasets.py
            ###
            # Distance from each unit-norm direction vector to its x-axis neighbor.
            dx = torch.sqrt(torch.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1))
            dx = torch.cat([dx, dx[-2:-1, :]], 0)
            # Cut the distance in half, and then round it out so that it's
            # halfway between inscribed by / circumscribed about the pixel.
            self.radii[img_idx] = dx * 2 / math.sqrt(12)
        self.radii = self.radii.transpose(1, 2)[..., None] # [n_images, H, W, 1]

        print('Load data: End')
    
    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        radii = self.radii[img_idx][(pixels_y.long(), pixels_x.long())]
        orientation = self.orientation_maps[img_idx].cuda() # W, H, 3

        return p, radii, orientation, self.intrinsics_all[img_idx], self.pose_all[img_idx]
        
    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        if self.conf.get('mask_based_ray_sampling', False):
            # Sample 50% hair rays, 45% foreground rays and 5% background rays
            num_pixels_bg = round(batch_size * 0.05)
            num_pixels_fg = round(batch_size * 0.45)
            num_pixels_hair = batch_size - num_pixels_bg - num_pixels_fg

            pixels_bg = torch.nonzero(~self.binary_masks[img_idx])
            pixels_bg = pixels_bg[torch.randperm(pixels_bg.shape[0])][:num_pixels_bg]

            pixels_fg = torch.nonzero(self.binary_masks[img_idx])
            pixels_fg = pixels_fg[torch.randperm(pixels_fg.shape[0])][:num_pixels_fg]

            pixels_hair = torch.nonzero(self.binary_hair_masks[img_idx])
            pixels_hair = pixels_hair[torch.randperm(pixels_hair.shape[0])][:num_pixels_hair]

            pixels_x, pixels_y = torch.cat([pixels_bg, pixels_fg, pixels_hair], dim=0).cuda().split(1, dim=1)
            pixels_x = pixels_x[:, 0]
            pixels_y = pixels_y[:, 0]
        else:
            pixels_x = torch.randint(low=0, high=self.W, size=[batch_size]).cpu()
            pixels_y = torch.randint(low=0, high=self.H, size=[batch_size]).cpu()

        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        radii = self.radii[img_idx][(pixels_y, pixels_x)]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
            
        hair_mask = self.hair_masks[img_idx][(pixels_y, pixels_x)]
        orientation = self.orientation_maps[img_idx][(pixels_y, pixels_x)]
        confidence = self.confidence_maps[img_idx][(pixels_y, pixels_x)]
        
        orientation = self.orientation_maps[img_idx][(pixels_y, pixels_x)]
        confidence = self.confidence_maps[img_idx][(pixels_y, pixels_x)]      

        concated_tensor = torch.cat([p.cpu(),
                            radii.cpu(), 
                            color, 
                            mask[:, :1],
                            hair_mask[:, :1]], dim=-1)
        

        return (torch.cat([concated_tensor,
                           orientation[:, :1], 
                           confidence[:, :1]], dim=-1).cuda(), 
                self.intrinsics_all[img_idx].cuda(),
                self.pose_all[img_idx].cuda())

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3

        ###
        ### Code below is borrowed from https://github.com/hjxwhy/mipnerf_pl/blob/master/datasets/datasets.py
        ###
        v = p
        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = torch.sqrt(torch.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1))
        dx = torch.cat([dx, dx[-2:-1, :]], 0)
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = dx[..., None] * 2 / math.sqrt(12) # W, H, 1

        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose_inv = np.linalg.inv(pose)
        rot = torch.from_numpy(pose_inv[:3, :3]).cuda()
        trans = torch.from_numpy(pose_inv[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3

        # TODO: return intrinsics and pose

        return rays_o.transpose(0, 1), rays_v.transpose(0, 1), radii.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level), 
                          interpolation=cv.INTER_CUBIC)
               ).clip(0, 255)

    def orient_at(self, idx, resolution_level):
        orient = cv.imread(self.orientations_lis[idx])[..., :1] / float(self.num_bins) * math.pi
        cos = np.cos(orient)
        orient_img = np.concatenate([cos * (cos >= 0), np.sin(orient), np.abs(cos) * (cos < 0)], axis=-1)
        orient_img = (orient_img * 255).round().astype('uint8')
        return (cv.resize(orient_img, (self.W // resolution_level, self.H // resolution_level),
                          interpolation=cv.INTER_CUBIC)
               ).clip(0, 255)

    def orient_confs_at(self, idx, resolution_level):
        confs = 1 - np.load(self.variance_lis[idx]) # for better vis
        confs = np.round(confs.clip(0, 1) * 255.0).astype('uint8')
        return cv.resize(confs, (self.W // resolution_level, self.H // resolution_level),
                         interpolation=cv.INTER_CUBIC)[..., None] / 255.0

    def filter_views(self, view_config_id='32'):
        view_configs = toml.load(self.data_dir.parent / 'config.toml')['scenes'][self.data_dir.name]['default_views_configs']
        views = view_configs[view_config_id]
        self.hair_masks_lis = [self.hair_masks_lis[i] for i in views]
        self.orientations_lis = [self.orientations_lis[i] for i in views]
        self.variance_lis = [self.variance_lis[i] for i in views]
        self.images_lis = [self.images_lis[i] for i in views]
        self.masks_lis = [self.masks_lis[i] for i in views]
        self.n_images = len(self.images_lis)
        self.scale_mats_np = [self.scale_mats_np[i] for i in views]
        self.world_mats_np = [self.world_mats_np[i] for i in views]
        

class MonocularDataset(Dataset):
    def __init__(self, conf):
        self.device = torch.device('cuda')
        self.conf = conf
        self.data_dir = Path(conf.get('data_dir'))
        self.render_cameras_name = conf.get('render_cameras_name')
        self.object_cameras_name = conf.get('object_cameras_name')

        self.views_idx = conf.get('views_idx', '')

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        
        fitted_camera_path = conf.get('fitted_camera_path')

        # Define scale into unit sphere
        self.scale_mat = np.eye(4, dtype=np.float32)
        if conf.get('path_to_scale', None) is not None:
            with open(conf['path_to_scale'], 'rb') as f:
                transform = pickle.load(f)
                print('upload transform', transform, conf['path_to_scale'])
                self.scale_mat[:3, :3] *= transform['scale']
                self.scale_mat[:3, 3] = np.array(transform['translation'])
    
        self.num_bins = conf.get('orient_num_bins')
        self.intrinsics_all = []
        self.pose_all = []

        self.images_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'image')))

        self.masks_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'mask')))
#        load hair mask
        self.hair_masks_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'hair_mask')))
#             load orientations
        self.orientations_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'orientation_maps')))
#            load variance
        self.variance_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'confidence_maps')))
#         Load camera
        self.cameras = camera_dict['arr_0']

        if self.views_idx:
            self.filter_views()

        self.n_images  = len(self.images_lis)
        print("Number of views:", self.n_images) 
        
        for i in range(self.n_images):
            world_mat = self.cameras[i]  
            P = world_mat @ self.scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)                       
            self.pose_all.append(torch.from_numpy(pose).float())
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())     
        
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 255.0
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 255.0
        
        self.hair_masks_np = np.stack([cv.imread(im_name) for im_name in self.hair_masks_lis]) / 255.0
        self.orientations_np = np.stack([cv.imread(im_name) for im_name in self.orientations_lis]) / float(self.num_bins) * math.pi
        self.variance_np = np.stack([np.load(im_name) for im_name in self.variance_lis])

        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
            
        self.hair_masks = torch.from_numpy(self.hair_masks_np.astype(np.float32)).cpu() 
        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]

        self.orientation_maps = torch.from_numpy(self.orientations_np.astype(np.float32)).cpu()
        self.variance_maps = torch.from_numpy(self.variance_np.astype(np.float32)).cpu()[..., None].float()
#         Convert variances to confidences
        self.confidence_maps = 1 / self.variance_maps ** 2
        self.confidence_maps[torch.isinf(self.confidence_maps)] = 100000

        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        
        if fitted_camera_path:
            camera_model = OptimizableCameras(len(self.images_lis), pretrain_path=fitted_camera_path).to(self.device)
            with torch.no_grad():
                print('camera model create')
                self.intrinsics_all, self.pose_all = camera_model(torch.arange(len(self.images_lis)), self.intrinsics_all, self.pose_all)
        
#         self.intrinsics_all = self.intrinsics_all.to(self.device)
#         self.pose_all = self.pose_all.to(self.device)
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all_inv = torch.inverse(self.pose_all)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        if self.conf.get('mask_based_ray_sampling', False):
            self.binary_masks = self.masks[..., 0] > self.conf.get('mask_binarization_threshold', 0.5)

        self.object_bbox_min = np.array([ -1.01,  -1.01,  -1.01])
        self.object_bbox_max = np.array([ 1.01,  1.01,  1.01])

        # Calculate radii for all pixels in all images
        self.radii = torch.empty(self.n_images, self.W, self.H)
        for img_idx in range(self.n_images):
            tx = torch.linspace(0, self.W - 1, self.W)  #
            ty = torch.linspace(0, self.H - 1, self.H)  # TODO: this code seems to be incorrect, because meshgrid reverses the order
            pixels_x, pixels_y = torch.meshgrid(tx, ty) #       of values: pixels_x change across i, pixels_y -- across j 
            p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
            v = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None].cuda()).squeeze()  # W, H, 3
            # Distance from each unit-norm direction vector to its x-axis neighbor.
            dx = torch.sqrt(torch.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1))
            dx = torch.cat([dx, dx[-2:-1, :]], 0)
            # Cut the distance in half, and then round it out so that it's
            # halfway between inscribed by / circumscribed about the pixel.
            self.radii[img_idx] = dx * 2 / math.sqrt(12)
        self.radii = self.radii.transpose(1, 2)[..., None] # [n_images, H, W, 1]
        print('Load data: End')
    
    def filter_views(self):
        print('Filter scene!')
        with open(self.views_idx, 'rb') as f:
            filter_idx = pickle.load(f)
            print(filter_idx)
        self.cameras = [self.cameras[i] for i in filter_idx]
        self.hair_masks_lis = [self.hair_masks_lis[i] for i in filter_idx]
        self.orientations_lis = [self.orientations_lis[i] for i in filter_idx]
        self.variance_lis = [self.variance_lis[i] for i in filter_idx]
        self.images_lis = [self.images_lis[i] for i in filter_idx]
        self.masks_lis = [self.masks_lis[i] for i in filter_idx]
