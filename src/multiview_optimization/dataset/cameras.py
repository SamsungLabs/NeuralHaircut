'''
    Camera model reference: Self-Calibrating Neural Radiance Fields
    Code: https://github.com/POSTECH-CVLab/SCNeRF
'''
import torch
from torch import nn


def intrinsic_param_to_K(intrinsics):
    device = intrinsics.device
    intrinsic_mat = torch.eye(4, 4).repeat(intrinsics.shape[0], 1, 1).to(device)
    intrinsic_mat[:, [0, 1, 0, 1], [0, 1, 2, 2]] = intrinsics
    return intrinsic_mat


def ortho2rotation(poses):
    r"""
    poses: batch x 6
    From https://github.com/chrischoy/DeepGlobalRegistration/blob/master/core
    /registration.py#L16
    Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) 
    and Wei Dong (weidong@andrew.cmu.edu)
    """

    def normalize_vector(v):
        r"""
        Batch x 3
        """
        v_mag = torch.sqrt((v ** 2).sum(1, keepdim=True))
        v_mag = torch.clamp(v_mag, min=1e-8)
        v = v / (v_mag + 1e-10)
        return v

    def cross_product(u, v):
        r"""S
        u: batch x 3
        v: batch x 3
        """
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

        i = i[:, None]
        j = j[:, None]
        k = k[:, None]
        return torch.cat((i, j, k), 1)

    def proj_u2a(u, a):
        r"""
        u: batch x 3
        a: batch x 3
        """
        inner_prod = (u * a).sum(1, keepdim=True)
        norm2 = (u ** 2).sum(1, keepdim=True)
        norm2 = torch.clamp(norm2, min=1e-8)
        factor = inner_prod / (norm2 + 1e-10)
        return factor * u

    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]

    x = normalize_vector(x_raw)
    y = normalize_vector(y_raw - proj_u2a(x, y_raw))
    z = cross_product(x, y)

    x = x[:, :, None]
    y = y[:, :, None]
    z = z[:, :, None]

    return torch.cat((x, y, z), 2)


class OptimizableCameras(nn.Module):
    def __init__(self, n_cameras, pretrain_path=''):
        super().__init__()
        self.intrinsics_res = nn.Parameter(torch.zeros(n_cameras, 4))
        self.rotation_res = nn.Parameter(torch.zeros(n_cameras, 2, 3))
        self.translation_res = nn.Parameter(torch.zeros(n_cameras, 3))

        if pretrain_path:
            self.load_weights(pretrain_path)
    
    def load_weights(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
    
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
    
    def forward(self, cam_index, intrinsics_0, pose_0):
        if len(intrinsics_0.shape) < 3:
            intrinsics_0 = intrinsics_0[None]
            pose_0 = pose_0[None]
            cam_index = cam_index[0].unsqueeze(-1)

        bs = intrinsics_0.shape[0]
        
        intrinsics = intrinsic_param_to_K(intrinsics_0[:, [0, 1, 0, 1], [0, 1, 2, 2]] + self.intrinsics_res[cam_index].cuda())
        rotation_mat = ortho2rotation((self.rotation_res[cam_index].cuda() + pose_0[:, :3, :2].transpose(1, 2)).view(bs, -1))

        pose = rotation_mat.new_zeros((bs, 4, 4))
        pose[:, :3, :3] = rotation_mat
        pose[:, 3, 3] = 1
        pose[:, :3, -1] = self.translation_res[cam_index].cuda() + pose_0[:, :3, -1]
        return intrinsics, pose