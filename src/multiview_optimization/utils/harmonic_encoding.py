import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import math



def harmonic_encoding(tensor, num_encoding_functions=6, include_input=True, log_sampling=True) -> torch.Tensor:
    r"""Apply harmonic encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be harmonically encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a harmonic encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            harmonic encoding (default: True).
    Returns:
    (torch.Tensor): harmonic encoding of the input tensor.

    Source: https://github.com/krrish94/nerf-pytorch
    """
    # TESTED
    # Trivially, the input tensor is added to the harmonic encoding.
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

    # Special case, for no harmonic encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)