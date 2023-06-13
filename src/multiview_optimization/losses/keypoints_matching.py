import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Union
from losses.op_kp_loss import cam_project


class KeypointsMatchingLoss(nn.Module):
    def __init__(self, device, use_3d = True):
        super(KeypointsMatchingLoss, self).__init__()
        self.weights = torch.ones(68, device=device)
        self.weights[5:7] = 1.0
        self.weights[10:12] = 1.0
        self.weights[27:36] = 5.5
        self.weights[30] = 7.0
        self.weights[31] = 7.0
        self.weights[35] = 3.0
        self.weights[60:68] = 3.5
        self.weights[48:60] = 3.5
        self.weights[48] = 3
        self.weights[54] = 3
        self.label = 'fa_kpts'
        self.use_3d = use_3d

    def compute(self,  prediction, data, img_size=512, weight=1.):
        pred_lmks_2d = cam_project(prediction['face_kpt'], data['intrinsics'])
        pred_lmks_2d = torch.cat([pred_lmks_2d[:,-17:], pred_lmks_2d[:,:-17]], dim=1) 
        gt_lmks = data['lmks3d'][:, :, :-1] if self.use_3d  else data['lmks']
        diff = (pred_lmks_2d - gt_lmks) / img_size
        loss = weight*(diff.abs().mean(-1) * self.weights[None] / self.weights.sum()).sum(-1).mean()
        return loss, gt_lmks, pred_lmks_2d