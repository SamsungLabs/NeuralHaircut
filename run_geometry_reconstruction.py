from collections import namedtuple
import errno
import os
import sys
sys.path.append('..')
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from src.models.dataset import Dataset, MonocularDataset
from src.models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from src.models.renderer import extract_orientation
from src.hair_networks.sdf import HairSDFNetwork
from src.models.head_prior import HeadPriorMesh
from src.models.renderer import NeusHairRenderer
from pathlib import Path
from datetime import datetime
import math
import random
import yaml

from src.models.cameras import OptimizableCameras

from src.utils.util import set_seed
from src.utils.geometry import project_orient_to_camera

import warnings
warnings.filterwarnings("ignore")


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False, checkpoint_name=None,  exp_name=None,  train_cameras=False):
        self.device = torch.device('cuda')

        # Configuration of geometry      
        self.conf_path = conf_path 
        with open(conf_path, 'r') as f:
            replaced_conf = str(yaml.load(f, Loader=yaml.Loader)).replace('CASE_NAME', case)
            self.conf = yaml.load(replaced_conf, Loader=yaml.Loader)

        if exp_name is not None:
            date, time = str(datetime.today()).split('.')[0].split(' ')
            exps_dir = Path('./exps_first_stage') / exp_name / case / Path(conf_path).stem
            if is_continue:
                prev_exps = sorted(exps_dir.iterdir())
                if len(prev_exps) > 0:
                    cur_dir = prev_exps[-1].name
                else:
                    raise FileNotFoundError(errno.ENOENT, "No previous experiment in directory", exps_dir)
            else:
                cur_dir = date + '_' + time   
            self.base_exp_dir =  exps_dir / cur_dir        
        else:
            self.base_exp_dir = self.conf['general']['base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        

        if Path(self.conf['dataset']['data_dir']).parent.name == 'h3ds':
            self.dataset = Dataset(self.conf['dataset'])
        else:
            self.dataset = MonocularDataset(self.conf['dataset'])

        self.iter_step = 0
        

        # Training parameters
        train_conf = self.conf['train']
        
        self.end_iter = train_conf['end_iter']
        self.save_freq = train_conf['save_freq']
        self.report_freq = train_conf['report_freq']
        self.val_freq = train_conf['val_freq']
        self.val_mesh_freq = train_conf['val_mesh_freq']
        self.val_orients_freq = train_conf['val_orients_freq']
        self.batch_size = train_conf['batch_size']
        self.val_render_resolution_level = train_conf['val_render_resolution_level']
        self.val_mesh_resolution = train_conf['val_mesh_resolution']        
        self.val_orients_resolution = train_conf['val_orients_resolution']
        self.learning_rate = train_conf['learning_rate']
        self.learning_rate_alpha = train_conf['learning_rate_alpha']
        self.use_white_bkgd = train_conf['use_white_bkgd']
        self.warm_up_end = train_conf['warm_up_end']
        self.anneal_end = train_conf['anneal_end']


        # Sampling parameters
        self.n_images_sampling = train_conf['n_images_sampling']
        self.bs_sampling = train_conf['bs_sampling']

        # Weights
        self.igr_weight = train_conf['igr_weight']
        self.mask_weight = train_conf['mask_weight']
        self.hair_mask_weight = train_conf['hair_mask_weight']
        self.orient_weight = train_conf['orient_weight']
        
        # Weights for scalp regularization
        self.head_prior_reg_weight = train_conf['head_prior_reg_weight']
        self.head_prior_off_sdf_weight = train_conf['head_prior_off_sdf_weight']
        self.head_prior_normal_weight = train_conf['head_prior_normal_weight']
        self.head_prior_sdf_weight = train_conf['head_prior_sdf_weight']
        
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        set_seed(42)

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model']['nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model']['sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model']['variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model']['rendering_network']).to(self.device)
        
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        
        # Networks for hair geometry and orientation field
        self.hair_network = HairSDFNetwork(**self.conf['model']['hair_sdf_network']).to(self.device)
        self.hair_deviation_network = SingleVarianceNetwork(**self.conf['model']['hair_variance_network']).to(self.device)
        params_to_train += list(self.hair_network.parameters())
        params_to_train += list(self.hair_deviation_network.parameters())

        # Scalp regularization
        self.head_prior_mesh = None
        if self.head_prior_reg_weight:
            self.head_prior_mesh = HeadPriorMesh(self.conf['dataset']['path_to_mesh_prior'],  device=self.device)

        self.renderer = NeusHairRenderer(
            self.nerf_outside,
            self.sdf_network,
            self.deviation_network,
            self.color_network,
            self.hair_network,
            self.hair_deviation_network,
            self.head_prior_mesh,
            **self.conf['model']['neus_renderer'])
    
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
        
        # Tune cameras
        self.tune_cameras_start = train_conf['tune_cameras_start']
        self.tune_cameras_end = train_conf['tune_cameras_end']

        params_to_train_cam = []
        if train_cameras:
            self.camera_model = OptimizableCameras(self.dataset.n_images)
            params_to_train_cam += list(self.camera_model.parameters())
            self.optimizer_camera = torch.optim.Adam(params_to_train_cam, lr=train_conf['lr_cameras'])
        else:
            self.camera_model = None
    
        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            if checkpoint_name is not None and checkpoint_name + ".pth" in model_list:
                latest_model_name = checkpoint_name + ".pth"
            else:
                latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    
    def sample_one_image(self, batch=None):
        image_perm = self.get_image_perm()
        bs = self.batch_size if batch is None else batch

        data, cam_intr, cam_pose = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], batch_size=bs)
      
        return data, cam_intr, cam_pose, image_perm[self.iter_step % len(image_perm)].repeat(bs)
    
    def sample_rays(self): 

        if self.n_images_sampling == 1:
            # at each iteration sample pixels from one image
            data, cam_intr, cam_pose, image_perm = self.sample_one_image()
        else:
            # at each iteration sample pixels from n_images_sampling images
            cam_pose = []
            cam_intr = []
            sampled_pixels = []
            image_perm = []
            for i in range(self.n_images_sampling):            
                dat, cintr, cpose, image_perms =  self.sample_one_image(batch=self.bs_sampling)             
                cam_pose.append(cpose[None].repeat(self.bs_sampling, 1, 1))
                cam_intr.append(cintr[None].repeat(self.bs_sampling, 1, 1))
                sampled_pixels.append(dat)
                image_perm.append(image_perms)
                
            data = torch.cat(sampled_pixels, dim=0)
            cam_intr = torch.cat(cam_intr, dim=0)
            cam_pose = torch.cat(cam_pose, dim=0)
            image_perm = torch.cat(image_perm, dim=0)

        return data, cam_intr, cam_pose, image_perm

        
    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step

        for iter_i in tqdm(range(res_step)):
            self.hair_network.embed_fn_fine.current_iter = self.iter_step
            self.sdf_network.embed_fn_fine.current_iter = self.iter_step
            self.color_network.embedview_fn.current_iter = self.iter_step
            self.nerf_outside.embed_fn.current_iter = self.iter_step
            self.nerf_outside.embed_fn_view.current_iter = self.iter_step

            data, cam_intr_0, cam_pose_0, image_index = self.sample_rays()

            pixels, radii, true_rgb, mask, hair_mask, orient_angle, orient_conf = data.split([3, 1, 3, 1, 1, 1, 1], dim=1)
            
            if self.camera_model is not None and self.iter_step > self.tune_cameras_start:
                cam_intr_, cam_pose_ = self.camera_model(image_index, cam_intr_0, cam_pose_0)
            else:
                cam_intr_ = cam_intr_0
                cam_pose_ = cam_pose_0

            p = torch.matmul(torch.linalg.inv(cam_intr_)[None, :3, :3], pixels[:, :, None]).squeeze() if len(cam_intr_.shape) < 3 else torch.matmul(torch.linalg.inv(cam_intr_)[:, :3, :3], pixels[:, :, None]).squeeze() # batch_size, 3
            rays_d = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
            rays_d = torch.matmul(cam_pose_[None, :3, :3], rays_d[:, :, None]).squeeze() if len(cam_pose_.shape) < 3 else torch.matmul(cam_pose_[:, :3, :3], rays_d[:, :, None]).squeeze()# batch_size, 3
            if len(cam_pose_.shape) < 3: 
                rays_o = cam_pose_[None, :3, 3].expand(rays_d.shape) # batch_size, 3
            elif cam_pose_.shape[0] == 1:
                rays_o = cam_pose_[:, :3, 3].expand(rays_d.shape) 
            else:
                rays_o = cam_pose_[:, :3, 3]

            cam_intr = cam_intr_.detach()
            cam_extr = torch.linalg.inv(cam_pose_.detach())


            near, far = self.dataset.near_far_from_sphere(rays_o.detach(), rays_d.detach())

            
            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                if self.conf['train']['binarize_gt_masks']:
                    mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)
            
            if  self.hair_mask_weight > 0.0:
                if self.conf['train']['binarize_gt_masks']:
                    hair_mask = (hair_mask > 0.5).float()
            else:
                hair_mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            hair_mask_sum = hair_mask.sum() + 1e-5

            render_out = self.renderer.render(rays_o, rays_d, radii, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            s_val_head = render_out['s_val_head']
            cdf_head = render_out['cdf_head']
            gradient_error_head = render_out['gradient_error_head']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']
        
            s_val_hair = render_out['s_val_hair']
            sampled_orient_fine = render_out['sampled_orient_fine']
            pts_fine = render_out['pts_fine']

            weights_hair = render_out['weights_hair']
            weight_hair_max = render_out['weight_hair_max']
            weight_hair_sum = render_out['weight_hair_sum']

            cdf_hair = gradient_error_hair = None

            cdf_hair = render_out['cdf_hair']
            gradient_error_hair = render_out['gradient_error_hair']

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            
            color_error_hair = (color_fine - true_rgb) * hair_mask
            color_hair_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / hair_mask_sum
            
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss_head = gradient_error_head


            mask_fine_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss +\
                   eikonal_loss_head * self.igr_weight +\
                   mask_fine_loss * self.mask_weight

            
            orient_fine_loss = orient_coarse_loss = 0.0

            if self.orient_weight:

                orient_angle_fine = project_orient_to_camera(orient_3d=sampled_orient_fine, org_3d=pts_fine, cam_intr=cam_intr, cam_extr=cam_extr)

                # Get masks for orientation loss
                orient_mask = hair_mask
                if self.conf['train']['orient_use_conf']:
                    orient_mask = orient_mask * orient_conf                    

                orient_mask_sum = orient_mask.sum() + 1e-5
                orient_fine_error = torch.min((orient_angle_fine - orient_angle).abs(),
                                              torch.min((orient_angle_fine - orient_angle - np.pi).abs(), 
                                                        (orient_angle_fine - orient_angle + np.pi).abs()))
                orient_fine_error = orient_fine_error * orient_mask 
                orient_fine_loss = orient_fine_error.mean() * (orient_fine_error.shape[0] / orient_mask_sum)

            hair_mask_fine_loss = F.binary_cross_entropy(weight_hair_sum.clip(1e-3, 1.0 - 1e-3), hair_mask)


            loss += hair_mask_fine_loss * self.mask_weight

            if self.orient_weight:
                loss += orient_fine_loss * self.orient_weight

            if self.head_prior_reg_weight:
                loss += self.head_prior_reg_weight * render_out['head_prior_inside_points'].mean()
            if self.head_prior_off_sdf_weight:
                loss += self.head_prior_off_sdf_weight * render_out['head_prior_off_sdf']
            if self.head_prior_normal_weight:
                loss += self.head_prior_normal_weight * render_out['head_prior_normal']
            if self.head_prior_sdf_weight:
                loss += self.head_prior_sdf_weight * render_out['head_prior_sdf']

            eikonal_loss_hair = gradient_error_hair
            loss += eikonal_loss_hair * self.igr_weight

            
            self.optimizer.zero_grad()
            if self.camera_model is not None and self.iter_step > self.tune_cameras_start and self.iter_step < self.tune_cameras_end:
                self.optimizer_camera.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.camera_model is not None and self.iter_step > self.tune_cameras_start and self.iter_step < self.tune_cameras_end:
                self.optimizer_camera.step()
            
            losses = {}

            self.iter_step += 1
            
            self.writer.add_scalar('Loss/hair_mask_fine_loss', hair_mask_fine_loss, self.iter_step)
            if self.orient_weight:
                self.writer.add_scalar('Loss/orient_loss', orient_fine_loss, self.iter_step)
            self.writer.add_scalar('Statistics/hair_weight_max', (weight_hair_max * hair_mask).sum() / hair_mask_sum, self.iter_step)

            if self.head_prior_reg_weight:
                self.writer.add_scalar('Loss/head_prior_reg_loss', render_out['head_prior_inside_points'].mean(), self.iter_step)
                self.writer.add_scalar('Loss/num_inside_points', render_out['num_inside_points'], self.iter_step)
            if self.head_prior_off_sdf_weight:
                self.writer.add_scalar('Loss/head_prior_off_sdf', render_out['head_prior_off_sdf'], self.iter_step)
            if self.head_prior_normal_weight:
                self.writer.add_scalar('Loss/head_prior_normal', render_out['head_prior_normal'], self.iter_step)
            if self.head_prior_sdf_weight:
                self.writer.add_scalar('Loss/head_prior_sdf', render_out['head_prior_sdf'], self.iter_step)

            self.writer.add_scalar('Statistics/s_val_hair', s_val_hair.mean(), self.iter_step)

            self.writer.add_scalar('Loss/eikonal_loss_hair', eikonal_loss_hair, self.iter_step)
            self.writer.add_scalar('Statistics/cdf_hair', (cdf_hair[:, :1] * hair_mask).sum() / hair_mask_sum, self.iter_step)

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_fine_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/color_fine_hair_loss', color_hair_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/mask_fine_loss', mask_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss_head', eikonal_loss_head, self.iter_step)
            self.writer.add_scalar('Statistics/s_val_head', s_val_head.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf_head', (cdf_head[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
            if losses is not None:
                for k, v in losses.items():
                    self.writer.add_scalar(f'Loss/{k}', v, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh(resolution=self.val_mesh_resolution, filter=False)
                
            if self.orient_weight:
                if self.iter_step % self.val_orients_freq == 0:   
                    self.validate_orientation(resolution=self.val_orients_resolution)

            self.update_learning_rate()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general']['recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            if Path(dir_name).is_dir():
                cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
                os.makedirs(cur_dir, exist_ok=True)
                files = os.listdir(dir_name)
                for f_name in files:
                    if f_name[-3:] == '.py':
                        copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))
            else:
                copyfile(dir_name, os.path.join(self.base_exp_dir, 'recording', Path(dir_name).name))
                
        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.yaml'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.hair_network.load_state_dict(checkpoint['hair_network'])
        self.hair_deviation_network.load_state_dict(checkpoint['hair_variance_network_fine'])
        
        if self.camera_model is not None:
            self.camera_model.load_state_dict(checkpoint['camera_model'])

        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
                'nerf': self.nerf_outside.state_dict(),
                'sdf_network_fine': self.sdf_network.state_dict(),
                'variance_network_fine': self.deviation_network.state_dict(),
                'color_network_fine': self.color_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iter_step': self.iter_step,
            }

        checkpoint['hair_network'] = self.hair_network.state_dict()
        checkpoint['hair_variance_network_fine'] = self.hair_deviation_network.state_dict()
        
        if self.camera_model is not None:
            checkpoint['camera_model'] = self.camera_model.state_dict()
            
            os.makedirs(os.path.join(self.base_exp_dir, 'cameras'), exist_ok=True)
            self.camera_model.save_weights(os.path.join(self.base_exp_dir, 'cameras', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))


    @torch.no_grad()
    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.val_render_resolution_level        
        pixels, radii, orient_angle, cam_intr_0, cam_pose_0 = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        # print('before cam model', cam_intr_0.shape, cam_pose_0.shape)
        if self.camera_model is not None:
            cam_intr_, cam_pose_ = self.camera_model(torch.tensor([idx]), cam_intr_0, cam_pose_0)
            cam_intr_ = cam_intr_[0]
            cam_pose_ = cam_pose_[0]
        else:
            cam_intr_ = cam_intr_0
            cam_pose_ = cam_pose_0

        cam_intr = cam_intr_.detach()
        cam_extr = torch.linalg.inv(cam_pose_.detach())

        p = torch.matmul(torch.linalg.inv(cam_intr_)[None, None, :3, :3], pixels[:, :, :, None]).squeeze()  # W, H, 3
        rays_d = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_d = torch.matmul(cam_pose_[None, None, :3, :3], rays_d[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = cam_pose_[None, None, :3, 3].expand(rays_d.shape)  # W, H, 3

        H, W, _ = rays_o.shape

        rays_o = rays_o.transpose(0, 1).reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.transpose(0, 1).reshape(-1, 3).split(self.batch_size)
        
        radii = radii.reshape(-1, 1).split(self.batch_size)
        orient_angle = orient_angle.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_orient_fine = []
        out_normal_fine = []
        out_normal_hair_fine = []
        
        for rays_o_batch, rays_d_batch, radii_batch, orient_angle_batch in zip(rays_o, rays_d, radii, orient_angle):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              radii_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            color_fine = render_out['color_fine'].detach().cpu().numpy()


            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(color_fine)

            if self.orient_weight:
                if feasible('sampled_orient_fine'):
                    # Project orientations into camera
                    orient_angle_fine = project_orient_to_camera(orient_3d=render_out['sampled_orient_fine'], org_3d=render_out['pts_fine'], cam_intr=cam_intr, cam_extr=cam_extr)
                    
                    orient_cos_fine = orient_angle_fine.cos()
                    orient_fine_cam = torch.cat(
                        [
                            orient_cos_fine * (orient_cos_fine >= 0),
                            orient_angle_fine.sin(),
                            orient_cos_fine.abs() * (orient_cos_fine < 0)
                        ],
                        dim=1
                    )
                    out_orient_fine.append(orient_fine_cam.cpu().numpy())                        
        
            if feasible('gradients_head') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients_head'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]

                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)

            if feasible('gradients_hair') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals_hair = render_out['gradients_hair'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals_hair = normals_hair * render_out['inside_sphere'][..., None]

                normals_hair = normals_hair.sum(dim=1).detach().cpu().numpy()
                out_normal_hair_fine.append(normals_hair)  
                
            del render_out

        img_fine = None
        orient_fine = None
        
        if out_rgb_fine:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 255).clip(0, 255)
        if self.orient_weight:     
            if out_orient_fine:
                orient_fine = (np.concatenate(out_orient_fine, axis=0).reshape([H, W, 3, -1]) * 255).clip(0, 255)
        
        normal_img = None
        if out_normal_fine:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)
        normal_hair_img = None
        if out_normal_hair_fine:
            normal_hair_img = np.concatenate(out_normal_hair_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_hair_img = (np.matmul(rot[None, :, :], normal_hair_img[:, :, None])
                               .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)
            
        os.makedirs(os.path.join(self.base_exp_dir, 'renders_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)


        if self.orient_weight: 
            os.makedirs(os.path.join(self.base_exp_dir, 'orients_fine'), exist_ok=True)
            os.makedirs(os.path.join(self.base_exp_dir, 'orients_masked_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals_hair'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if out_rgb_fine:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'renders_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if self.orient_weight: 
                if out_orient_fine:
                    confs = self.dataset.orient_confs_at(idx, resolution_level=resolution_level)
                    orients_gt = self.dataset.orient_at(idx, resolution_level=resolution_level)
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            'orients_fine',
                                            '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                               np.concatenate([orient_fine[..., i], orients_gt]))

                    # Masked orients
                    orient_fine[..., i] = (orient_fine[..., i] * confs).round().astype('uint8')
                    orients_gt = (orients_gt * confs).round().astype('uint8')
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            'orients_masked_fine',
                                            '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                               np.concatenate([orient_fine[..., i], orients_gt]))
            if out_normal_fine:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

            if out_normal_hair_fine:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals_hair',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_hair_img[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        off_sdfpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 255).clip(0, 255).astype(np.uint8)
        return img_fine
    
    @torch.no_grad()
    def validate_orientation(self, resolution=64):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
        out = extract_orientation(bound_min, bound_max, resolution, self.hair_network)
        # This code works only for SDF implicit hair!
        sdf = out[..., 0]
        filter_outside = sdf <= 0
        orientation = out[..., 1:] * filter_outside[..., None]

        os.makedirs(os.path.join(self.base_exp_dir, 'orients'), exist_ok=True)
        with open(os.path.join(self.base_exp_dir, 'orients', '{:0>8d}_orients.npy'.format(self.iter_step)), 'wb') as f:
            np.save(f, orientation)
        
    @torch.no_grad()
    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0, filter=True):
        ### Extract the sdf mesh
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
        
        vertices, triangles = self.renderer.extract_geometry(
            bound_min, bound_max, 
            resolution=resolution, 
            threshold=threshold
        )
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        if filter:
            meshes = mesh.split(only_watertight=False)
            mesh = meshes[np.argmax([len(m.faces) for m in meshes])]
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_head.ply'.format(self.iter_step)))

        ### Extract the hair mesh
        try: 
            hair_vertices, hair_triangles =\
                self.renderer.extract_hair_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)

            if world_space:
                hair_vertices = hair_vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

            hair_mesh = trimesh.Trimesh(hair_vertices, hair_triangles)
            if filter:
                hair_meshes = hair_mesh.split(only_watertight=False)
                hair_mesh = hair_meshes[np.argmax([len(m.faces) for m in hair_meshes])]

            hair_mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_hair.ply'.format(self.iter_step)))
        except Exception as e:
            print("Failed to extract hair mesh", e)
        

        logging.info('End')

    def off_sdfpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()
    
    @torch.no_grad()
    def render_img(self, img_idx, resolution):
        pixels, radii, _, cam_intr_0, cam_pose_0 = self.dataset.gen_rays_at(img_idx, resolution_level=resolution)
        if self.camera_model is not None:
            cam_intr_, cam_pose_ = self.camera_model(img_idx, cam_intr_0, cam_pose_0)
        else:
            cam_intr_ = cam_intr_0
            cam_pose_ = cam_pose_0

        p = torch.matmul(torch.linalg.inv(cam_intr_)[None, None, :3, :3], pixels[:, :, :, None]).squeeze()  # W, H, 3
        rays_d = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_d = torch.matmul(cam_pose_[None, None, :3, :3], rays_d[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = cam_pose_[None, None, :3, 3].expand(rays_d.shape)  # W, H, 3
        
        H, W, _ = rays_o.shape
        rays_o = rays_o.transpose(0, 1).reshape(-1, 3).split(800)
        rays_d = rays_d.transpose(0, 1).reshape(-1, 3).split(800)

        radii = radii.reshape(-1, 1).split(800)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch, radii_batch in tqdm(zip(rays_o, rays_d, radii)):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              radii_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb, perturb_overwrite=0)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        img_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(img_dir, exist_ok=True)
        cv.imwrite(os.path.join(img_dir, f'{self.iter_step}_{img_idx}.png'), img_fine)


if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--no_filtering', default=False, action="store_true")
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to continue training')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--train_cameras', default=False, action="store_true")

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    print(args.train_cameras)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue, checkpoint_name=args.checkpoint, exp_name=args.exp_name,  train_cameras=args.train_cameras)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=False, resolution=512, threshold=args.mcube_threshold, filter=not args.no_filtering)
    elif args.mode.startswith('off_sdfpolate'):  # off_sdfpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.off_sdfpolate_view(img_idx_0, img_idx_1)
    elif args.mode.startswith('render'):
        # img_idx = int(args.mode.split('_')[1])
        for i in range(runner.dataset.n_images):
            runner.render_img(i, 1)
