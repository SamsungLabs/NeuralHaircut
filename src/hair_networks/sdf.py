import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from src.models.embedder import get_embedder



# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class HairSDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 n_layers_orient,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 predict_color=False,
                 start=-1,
                 end=-1):
        super(HairSDFNetwork, self).__init__()
        self.n_layers_orient = n_layers_orient
        if self.n_layers_orient:
            d_out -= 3
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        
        self.embed_fn_fine = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in, start=start, end=end)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)
        
        if self.n_layers_orient:
            layers_orient = []
            dims_orient = [d_hidden + input_ch] + [d_hidden for _ in range(self.n_layers_orient)] + [3]
            num_layers_orient = len(dims_orient)

            for l in range(0, num_layers_orient - 1):
                lin = nn.Linear(dims_orient[l], dims_orient[l + 1])
                if weight_norm:
                    lin = nn.utils.weight_norm(lin)
                layers_orient.append(lin)

                if l < num_layers_orient - 2:
                    layers_orient.append(nn.ReLU())

            self.orient_head = nn.Sequential(*layers_orient)
        
        print(self)

    def forward(self, inputs, calc_orient=False):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        
        if calc_orient:
            if self.n_layers_orient:
                sdf = x[..., :1] / self.scale
                feats = x[..., 1:]
                inputs_orient = torch.cat([feats, inputs], -1)
                orients = torch.tanh(self.orient_head(inputs_orient))
            else:
                sdf = x[..., :1] / self.scale
                orients = torch.tanh(x[..., 1:4])
                feats = x[..., 4:]

            return torch.cat([sdf, feats, orients], dim=-1)
        
        else:
            sdf = x[..., :1] / self.scale
            feats = x[..., 1:]

            if not self.n_layers_orient:
                feats = feats[..., :-3]

            return torch.cat([sdf, feats], dim=-1)

    def sdf(self, x):
        return self.forward(x)[..., :1].clone()

    def sdf_hidden_appearance(self, x):
        return torch.cat(self.forward(x), dim=-1)

    @torch.enable_grad()
    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)
