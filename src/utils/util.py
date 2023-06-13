import torch
import numpy as np
import os
from glob import glob
import contextlib
import random


def scale_mat(mat, scale_factor):
    mat[0, 0] /= scale_factor
    mat[1, 1] /= scale_factor
    mat[0, 2] /= scale_factor
    mat[1, 2] /= scale_factor
    return mat

def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded. B x C x ...
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=1)


def param_to_buffer(module):
    """Turns all parameters of a module into buffers."""
    for submodule in module.children():
        param_to_buffer(submodule)
    for name, param in dict(module.named_parameters(recurse=False)).items():
        delattr(module, name) # Unregister parameter
        module.register_buffer(name, param, persistent=False)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image * 255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1,2,0)
    return image.astype(np.uint8).copy()

def glob_imgs(path):
    imgs = []
    for ext in ['*.jpg', '*.JPEG', '*.JPG', '*.png', '*.PNG', '*.npy', '*.NPY']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def fill_tensor(x, mask, c=0):
    if x is not None:
        out = x.new_ones((mask.shape[0], *x.shape[1:])) * c
        out[mask] = x
        return out 
    else:
        return x
    
@contextlib.contextmanager
def freeze_gradients(model):
    is_training = model.training
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    yield
    if is_training:
        model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    