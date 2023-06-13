import torch
import torch.nn as nn
import numpy as np

import sys
import os
sys.path.append(os.path.join(sys.path[0], '../..'))

from NeuS.models.embedder import Embedder

# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class SchedulerEmbedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.current_iter = 0
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)
        
        k = torch.linspace(0., max_freq, N_freqs)

        alpha = (self.current_iter - self.kwargs['start']) / (self.kwargs['end'] - self.kwargs['start']) * max_freq
        weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2
        
        for freq, w in zip(freq_bands, weight):
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq, w=w: w*p_fn(x * freq))
                out_dim += d
        
        
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        self.create_embedding_fn()
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    

def get_embedder(multires, input_dims=3, start=-1, end=-1):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
        'start': start,
        'end': end,
    }
    
    if start == -1 and end == -1:
        embedder_obj = Embedder(**embed_kwargs)
    else:
        embedder_obj = SchedulerEmbedder(**embed_kwargs)
        
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim
