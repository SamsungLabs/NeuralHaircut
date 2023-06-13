import errno
import os
import sys
from collections import namedtuple
import random
sys.path.append('..')
import argparse
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from shutil import copyfile

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from src.models.dataset import Dataset, MonocularDataset
from pyhocon import ConfigFactory
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.hair_networks.sdf import HairSDFNetwork
from src.strands_trainer import StrandsTrainer
import yaml
from copy import deepcopy

from src.utils.util import set_seed, scale_mat

import warnings
warnings.filterwarnings("ignore")
    
class Runner:
    def __init__(self, conf_path, case='CASE_NAME', scene_type='DATASET_TYPE', checkpoint_name=None, hair_conf_path=None, exp_name=None):
        
        self.device = torch.device('cuda')
        
        # Configuration of geometry      
        self.conf_path = conf_path 
        with open(conf_path, 'r') as f:
            replaced_conf = str(yaml.load(f, Loader=yaml.Loader)).replace('CASE_NAME', case)
            self.conf = yaml.load(replaced_conf, Loader=yaml.Loader)
        
        # Configuration of hair strands
        self.hair_conf_path = hair_conf_path
        with open(hair_conf_path, 'r') as f:
            replaced_conf = str(yaml.load(f, Loader=yaml.Loader)).replace('CASE_NAME', case)
            replaced_conf = replaced_conf.replace('DATASET_TYPE', scene_type)
            self.hair_conf = yaml.load(replaced_conf, Loader=yaml.Loader)
        

        train_conf = self.conf['train']
        
        self.end_iter = train_conf['end_iter']
        self.report_freq = train_conf['report_freq']
        self.batch_size = train_conf['batch_size']
        
        if exp_name is not None:
            date, time = str(datetime.today()).split('.')[0].split(' ')
            exps_dir = Path('./exps_second_stage') / exp_name / case / Path(conf_path).stem
            cur_dir = date + '_' + time   
            self.base_exp_dir =  exps_dir / cur_dir        
        else:
            self.base_exp_dir = self.conf['general']['base_exp_dir']
                 
        self.img_size = self.hair_conf['render']['image_size']
        
        os.makedirs(self.base_exp_dir, exist_ok=True)                    
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'hair_primitives'), exist_ok=True)
        
        if scene_type == 'h3ds':
            self.dataset = Dataset(self.conf['dataset'])
        else:
            self.dataset = MonocularDataset(self.conf['dataset'])
        
        self.iter_step = 0

        self.writer = None
        set_seed(42)

        self.hair_primitives_trainer = StrandsTrainer(self.hair_conf, run_model= lambda model, x: model(x, calc_orient=True), device=self.device, save_dir=self.base_exp_dir)
        
        self.hair_network = HairSDFNetwork(**self.conf['model']['hair_sdf_network']).to(self.device)
        
#         Upload volumetric geometry and surface orientation fields
        if train_conf['pretrain_path']:
            print('Upload sdf hair geometry and orientation field!')
            checkpoint = torch.load(train_conf['pretrain_path'], map_location=self.device)
            self.hair_network.load_state_dict(checkpoint['hair_network'])

#         Upload strand-based geometry         
        if train_conf['pretrain_strands_path']: 
            print('Upload strands!')
            self.hair_primitives_trainer.load_weights(train_conf['pretrain_hair_path'])
           
        # Backup codes and configs for debug
        self.file_backup()

    
    def train(self):
        res_step = self.end_iter - self.iter_step
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        image_perm = self.get_image_perm()
        losses = {}

        
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
        
        for iter_i in tqdm(range(res_step)):            
            cur_img = image_perm[self.iter_step % len(image_perm)]
            _, cam_intr, cam_pose = self.dataset.gen_random_rays_at(cur_img, self.batch_size)
            cam_extr = torch.linalg.inv(cam_pose.detach())

            orig_data_size = self.dataset.images[cur_img].shape[1] # H shape, suppose square images
            
            raster_dict = {}
            raster_dict['iter'] = self.iter_step
            raster_dict['cam_extr'] = cam_extr
            
            if orig_data_size is None:
                raster_dict['cam_intr'] = cam_intr
                raster_dict['gt_silh'] = self.dataset.hair_masks[cur_img].permute(2, 0, 1).cuda()
                raster_dict['gt_rgb'] = self.dataset.images[cur_img].permute(2, 0, 1).cuda()
            else:
                # need to change cameras intrinsic as we render in resolution 512x512
                scale_factor = orig_data_size / self.img_size
                raster_dict['cam_intr'] = scale_mat(deepcopy(cam_intr), scale_factor)
                raster_dict['gt_silh'] = F.interpolate(self.dataset.hair_masks[cur_img].permute(2, 0, 1)[None], size=self.img_size, mode='bilinear')[0].cuda()
                raster_dict['gt_rgb'] = F.interpolate(self.dataset.images[cur_img].permute(2, 0, 1)[None], size=self.img_size, mode='bilinear')[0].cuda()

            raster_dict['visual_gt_orients'] = self.dataset.orient_at(cur_img, resolution_level=1)
            
            losses.update({
                'hair_' + str(key): val for key, val in self.hair_primitives_trainer.train_step(model=self.hair_network, it=self.iter_step, raster_dict=raster_dict).items()
                })

            self.iter_step += 1

            if losses is not None:
                for k, v in losses.items():
                    self.writer.add_scalar(f'Loss/{k}', v, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                self.save_strands_pointcloud()
            
            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()
            
            if self.iter_step % self.report_freq == 0:
                self.hair_primitives_trainer.save_weights(os.path.join(self.base_exp_dir, 'hair_primitives', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))
                
    
    def save_strands_pointcloud(self):
        if self.hair_primitives_trainer:
            strands_origins = self.hair_primitives_trainer.strands_origins.reshape(-1, 100, 3)
            
            cols = torch.cat((torch.rand(strands_origins.shape[0], 3).unsqueeze(1).repeat(1, 100, 1), torch.ones(strands_origins.shape[0], 100, 1)), dim=-1).reshape(-1, 4).cpu()           
            trimesh.PointCloud(strands_origins.reshape(-1, 3).detach().cpu(), colors=cols).export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_strands_points.ply'.format(self.iter_step)))

    
    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def file_backup(self):
        dir_lis = self.conf['general']['recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.yaml'))
        copyfile(self.hair_conf_path, os.path.join(self.base_exp_dir, 'recording', 'hair_config.yaml'))

if __name__ == '__main__':
    print('Hello Wooden')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--scene_type', type=str, default='')
    parser.add_argument('--hair_conf', type=str, default=None, help='Use hair primitives config')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to continue training')
    parser.add_argument('--exp_name', type=str, default=None)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf,  args.case, args.scene_type, hair_conf_path=args.hair_conf, checkpoint_name=args.checkpoint, exp_name=args.exp_name)

    runner.train()
