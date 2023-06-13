import os
import enum
import torch
from torch import nn
import numpy as np
import pickle
import json
import utils.index_mapping as im
from icecream import ic

def cam_project(points, K):
    """
    :param points: torch.Tensor of shape [b, n, 3]
    :param K: torch.Tensor intrinsics matrix of shape [b, 3, 3]
    :return: torch.Tensor points projected to 2d using K, shape: [b, n, 2]
    """
    b = points.shape[0]
    n = points.shape[1]

    points_K = torch.matmul(
        K.reshape(b, 1, 3, 3).repeat(1, n, 1, 1),
        points.reshape(b, n, 3, 1)
    )  # shape: [b, n, 3, 1]

    points_2d = points_K[:, :, :2, 0] / points_K[:, :, [2], 0]  # shape: [b, n, 2]
    return points_2d


class OPType(enum.Enum):
    body = 0
    left_hand = 1
    right_hand = 2
    face = 3
    face_no_mouth = 4
    mouth = 5


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return torch.mean(self.rho ** 2 * dist)


loss_type2loss = dict({
    'SmoothL1': nn.SmoothL1Loss(),
    'L2': nn.MSELoss(),
    'GMoF': GMoF(rho=1)
})


class DefaultLoss:
    def __init__(self, label='default', norm=1, weight=1, verbose=False):
        self.label = label
        self.weight = weight
        self.norm = norm

        if verbose:
            print(f'Init {self.label} loss with norm={norm:.1E} and weight={weight:.1E}')

    def handle_nan(self, loss):
        if loss != loss:
            print('NaN in {}'.format(self.label))
            exit()

    def weighted_handle_nan(self, loss):
        self.handle_nan(loss)
        loss *= self.weight
        self.handle_nan(loss)
        return loss

    def compute_1(self, x):
        loss = self.loss(x )
        return self.weighted_handle_nan(loss)

    def compute_2(self, pred, target):
        loss = self.loss(
            pred ,
            target
        )
        return self.weighted_handle_nan(loss)

    def compute(self, a, b):
        raise NotImplemented
        
class RegShapeLoss(DefaultLoss):
    def __init__(self, weight):
        super().__init__(label='reg_shape')
        print('init reg shape loss')
        self.info_betas = torch.tensor(np.eye(200), requires_grad=False, dtype=torch.float32)
        self.loss = lambda x: x
        self.weight =  weight
        self.label = 'reg_shape'

    def compute(self, model_output, data, img_size=512, weight=1.):
        betas = model_output['betas']

        if self.info_betas.device != betas.device:
            self.info_betas = self.info_betas.to(betas.device)
        return self.weight * betas.dot(torch.mv(self.info_betas, betas)), None, None

class OpenPoseLoss(DefaultLoss):
    def __init__(
        self,
        norm=1, weight=1, verbose=False,
        mode='face', kp_weights=None, device='cuda'
    ):

        super().__init__(label=f'openpose_{mode}', norm=norm, weight=weight, verbose=verbose)
        self.device = device
        self.kp_weights = kp_weights  # weights for every keypoint
        self.ignore_inds = []
        self.lm_inds = []
        if mode == 'body':
            self.jtr_inds = im.get_smplx_ids_op_body_25()
            self.data_key = 'op_pose'
            self.conf_key = 'op_conf_pose'
        if mode == 'left_hand':
            self.jtr_inds = im.get_smplx_ids_op_lh()
            self.data_key = 'op_hand_left'
            self.conf_key = 'op_conf_hand_left'
        if mode == 'right_hand':
            self.jtr_inds = im.get_smplx_ids_op_rh()
            self.data_key = 'op_hand_right'
            self.conf_key = 'op_conf_hand_right'
        if mode == 'face':
            self.jtr_inds = im.get_smplx_ids_op_face()
            self.data_key = 'op_face'
            self.conf_key = 'op_conf_face'
            self.ignore_inds = np.asarray([17, 26], dtype=int)
        if mode == 'face_no_mouth':
            self.jtr_inds = im.get_smplx_ids_op_face()
            self.lm_inds = np.concatenate([np.arange(0, 48), np.arange(68, 70)])
            self.data_key = 'op_face'
            self.conf_key = 'op_conf_face'
            self.ignore_inds = np.asarray([17, 26], dtype=int)
        if mode == 'mouth':
            self.jtr_inds = im.get_smplx_ids_op_face()
            self.lm_inds = np.arange(48, 68)
            self.data_key = 'op_face'
            self.conf_key = 'op_conf_face'            

    def compute(self, model_output, data, img_size=512, weight=1.):
        lms = data['openpose_lmks'][self.data_key].to(self.device).float()
        lms_mask = data['openpose_lmks'][self.conf_key].to(self.device).float()
        lm_3d = model_output['Jtr'][:, self.jtr_inds, :]
        if len(self.lm_inds) > 0:
            lms = lms[:, self.lm_inds, :]
            lms_mask = lms_mask[:, self.lm_inds, :]
            lm_3d = lm_3d[:, self.lm_inds, :]

        lms_mask[:, self.ignore_inds, :] = 0
        
        lm_pred = cam_project(lm_3d, data['intrinsics'])

        if self.kp_weights is not None:
            b = lms.shape[0]
            lms_mask *= self.kp_weights.reshape(1, -1, 1).repeat(b, 1, 2)
        diff = (((lm_pred - lms) * lms_mask) / img_size).abs() #normalize kp to be [0, 1]
        loss = weight*(diff.mean(-1)).sum(-1).mean()
        return loss, lms, lm_pred