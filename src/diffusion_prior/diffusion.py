from functools import partial
import os

import sys
sys.path.append(os.path.join(sys.path[0], 'k-diffusion'))

from k_diffusion.layers import DenoiserWithVariance as Denoiser
from k_diffusion.utils import append_dims

def make_denoiser_wrapper(config):
    config = config['model']
    sigma_data = config.get('sigma_data', 1.)
    has_variance = config.get('has_variance', False)
    return partial(DenoiserWithVariance, sigma_data=sigma_data)


class DenoiserWithVariance(Denoiser):

    def loss_wo_logvar(self, input, noise, sigma, mask, unet_cond, **kwargs):
        c_skip, c_out, c_in = [append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * append_dims(sigma, input.ndim)
        model_output, logvar = self.inner_model(noised_input * c_in, sigma, unet_cond=unet_cond, return_variance=True, **kwargs)
        logvar = append_dims(logvar, model_output.ndim)
        target = (input - c_skip * noised_input) / c_out
        if mask is not None:
            losses = ((model_output - target) ** 2) * mask / 2
        else:
            losses = ((model_output - target) ** 2) / 2
        return losses.flatten(1).mean(1), model_output * c_out + c_skip * noised_input, noised_input