import torch
from torch import nn
from torch.nn import functional as F

from typing import Union, Tuple



class GradScaler(nn.Module):
    def __init__(self, size: Union[int, Tuple[int]]):
        super(GradScaler, self).__init__()

    def forward(self, inputs):
        return inputs

    def backward(self, grad_output):
        if isinstance(self.size, tuple):
            for i in range(len(self.size)):
                grad_output[..., i] /= self.size[i]

        elif isinstance(self.size, int):
            grad_output /= self.size

        return grad_output


class GridSample(nn.Module):
    def __init__(self, size: Union[int, Tuple[int]]):
        super(GridSample, self).__init__()
        self.scaler = GradScaler(size)

    def forward(self, input, grid, padding_mode='reflection', align_corners=False):
        return F.grid_sample(input, self.scaler(grid), padding_mode=padding_mode, align_corners=align_corners)


def make_grid(h, w, device=torch.device('cpu'), dtype=torch.float32):
    grid_x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    grid_y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    v, u = torch.meshgrid(grid_y, grid_x)
    grid = torch.stack([u, v], dim=2).view(1, h, w, 2)

    return grid


def grid_sampler_backward(grad_out, grid, h=None, w=None, padding_mode='zeros', align_corners=False):
    with torch.no_grad():
        b, c = grad_out.shape[:2]
        if h is None or w is None:
            h, w = grad_out.shape[2:]
        size = torch.FloatTensor([w, h]).to(grad_out.device)
        grad_in = torch.zeros(b, c, h, w, device=grad_out.device)

        if align_corners:
            grid_ = (grid + 1) / 2 * (size - 1)
        else:
            grid_ = ((grid + 1) * size - 1) / 2

        if padding_mode == 'border':
            assert False, 'TODO'

        elif padding_mode == 'reflection':
            assert False, 'TODO'

        grid_nw = grid_.floor().long()
        
        grid_ne = grid_nw.clone()
        grid_ne[..., 0] += 1
        
        grid_sw = grid_nw.clone()
        grid_sw[..., 1] += 1
        
        grid_se = grid_nw.clone() + 1
        
        nw = (grid_se - grid_).prod(3)
        ne = (grid_ - grid_sw).abs().prod(3)
        sw = (grid_ne - grid_).abs().prod(3)
        se = (grid_ - grid_nw).prod(3)

        indices_ = torch.cat([
            (
                (
                    g[:, None, ..., 0] + g[:, None,..., 1] * w
                ).repeat_interleave(c, dim=1) 
                + torch.arange(c, device=g.device)[None, :, None, None] * (h*w) # add channel shifts
                + torch.arange(b, device=g.device)[:, None, None, None] * (c*h*w) # add batch size shifts
            ).view(-1) 
            for g in [grid_nw, grid_ne, grid_sw, grid_se]
        ])

        masks = torch.cat([
            (
                (g[..., 0] >= 0) & (g[..., 0] < w) & (g[..., 1] >= 0) & (g[..., 1] < h)
            )[:, None].repeat_interleave(c, dim=1).view(-1)
            for g in [grid_nw, grid_ne, grid_sw, grid_se]
        ])
    
    values_ = torch.cat([
        (m[:, None].repeat_interleave(c, dim=1) * grad_out).view(-1)
        for m in [nw, ne, sw, se]
    ])

    indices = indices_[masks]
    values = values_[masks]
    
    grad_in.put_(indices, values, accumulate=True)

    return grad_in