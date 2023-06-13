import torch

from typing import Optional


def smooth(x: torch.Tensor, n: int = 3) -> torch.Tensor:
    ret = torch.cumsum(torch.concat((torch.repeat_interleave(x[:1], n - 1, dim=0), x)), 0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def xy_normal(strands: torch.Tensor, normalize: bool = True, smooth_degree: Optional[int] = None):
    d = torch.empty(strands.shape[:2] + (2, ), device=strands.device)

    if smooth_degree is not None:
        strands = smooth(strands, smooth_degree)

    d[:, :-1, :] = strands[:, 1:, [0, 1]] - strands[:, : -1, [0, 1]]
    d[:, -1, :] = d[:, -2, :]

    n = torch.cat((d[:, :, [1]], -d[:, :, [0]]), dim=2)
    
    if normalize:
        n = n / torch.linalg.norm(n, dim=2, keepdims=True)
    
    return n


def build_quads(
    strands: torch.Tensor,
    w: float = 0.0001,
    return_in_strands_shape: bool = True,
    calculate_faces: bool = True):

    n_strands, n_points = strands.shape[:2]

    n_xy = xy_normal(strands)
    n_xyz = torch.cat((n_xy, torch.zeros(n_strands, n_points, 1, device=strands.device)), axis=2)

    verts = torch.empty((n_strands, 2 * n_points, 3), device=strands.device)
    verts[:, 0::2, :] = strands + w * n_xyz
    verts[:, 1::2, :] = strands - w * n_xyz
    
    indices = torch.empty((n_strands, 2 * n_points, 1), device=strands.device, dtype=verts.dtype)
    values = torch.tensor(list(range(n_strands * n_points)), dtype=indices.dtype, device=indices.device)
    values = values.view((n_strands, n_points, 1))
    indices[:, 0::2, :] = values
    indices[:, 1::2, :] = values
    indices = indices.reshape(-1, 1)
    
    if calculate_faces:
        faces = torch.empty((2 * n_points - 2, 3), dtype=torch.long, device=strands.device)

        # Second edge in each face is boundary one, but the're inversed sinced triangles must have the same orientation

        faces[0::2, :] = \
            torch.stack((
                torch.arange(0, 2 * n_points - 3, step=2),
                torch.arange(1, 2 * n_points - 1, step=2),
                torch.arange(3, 2 * n_points + 1, step=2)
            )).T

        faces[1::2, :] = \
            torch.stack((
                torch.arange(3, 2 * n_points    , step=2),
                torch.arange(2, 2 * n_points    , step=2),
                torch.arange(0, 2 * n_points - 3, step=2)
            )).T

        full_faces_array = \
            torch.arange(0, verts.shape[1] * n_strands, verts.shape[1], device=strands.device).reshape(n_strands, 1, 1) + \
            faces.unsqueeze(0)
        
        if not return_in_strands_shape:
            full_faces_array = full_faces_array.reshape(-1, 3)

    if not return_in_strands_shape:
        verts = verts.reshape(-1, 3)

    if calculate_faces:
        return verts, full_faces_array, indices
    else:
        return verts, indices