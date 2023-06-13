import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
import os
sys.path.append(os.path.join(sys.path[0], '../..'))

from NeuS.models.renderer import  extract_fields, NeuSRenderer
from src.utils.util import fill_tensor


def extract_orientation(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.full([resolution, resolution, resolution], np.nan, dtype=np.float32)
    u = np.repeat(np.expand_dims(u, axis=-1), 4, -1)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    in_spheres = pts.new_ones(pts.shape[0], dtype=torch.bool)
                    # This code works only for SDF query func
                    outputs = query_func(pts[in_spheres].cuda(), calc_orient=True).detach().cpu().numpy()
                    feats_size = outputs.shape[1] - 4
                    sdf, _, orients = np.split(outputs, [1, 1 + feats_size], axis=1)
                    val = np.concatenate([sdf, orients], axis=1)
                    in_spheres = in_spheres.reshape(len(xs), len(ys), len(zs)).cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs), :][in_spheres] = val
        return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    from skimage import measure, morphology
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    mask = ~np.isnan(u)
    u[~mask] = 0
    mask = morphology.binary_erosion(mask, np.ones((3, 3, 3)))
    vertices, triangles, _, _ = measure.marching_cubes(u, threshold, mask=mask)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False, detailed_output=False, valid_interval=None):
    # This implementation is from NeRF
    # Get pdf
    if valid_interval is None:
        weights = weights + 1e-5
    else:
        weights[valid_interval] += 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])
    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    if detailed_output:
        inds_bin = torch.where(t < 0.5, inds_g[..., 0], inds_g[..., 1])
        return samples, inds_bin
    else:
        return samples


def sphere_trace_surface_points(network, rays_o, rays_d, near, far, N_iters=20):
    device = rays_o.device
    d_preds = near.clone()
    mask = torch.ones_like(d_preds, dtype=torch.bool, device=device)
    for _ in range(N_iters):
        pts = rays_o + rays_d * d_preds
        surface_val = network.sdf(pts)
        d_preds[mask] += surface_val[mask]
        mask[d_preds > far] = False
        mask[d_preds < near] = False
    pts = rays_o + rays_d * d_preds
    return d_preds, pts, mask.squeeze(-1)


class NeusHairRenderer(NeuSRenderer):
    def __init__(self,
            nerf,
            sdf_network,
            deviation_network,
            color_network,
            hair_network,
            hair_deviation_network,
            head_prior_mesh,       
            n_samples,
            n_importance,
            n_outside,
            up_sample_steps,
            perturb,
            blended_upsample,
            head_prior_attraction
            ):
        
        super().__init__(nerf, sdf_network, deviation_network, color_network, n_samples, n_importance, n_outside, up_sample_steps, perturb)

        self.hair_network = hair_network
        self.hair_deviation_network = hair_deviation_network

        self.blended_upsample = blended_upsample #consider both geometry fields for upsample
        self.head_prior_mesh = head_prior_mesh
        self.head_prior_attraction = head_prior_attraction


    def eval_sdf(self, pts, dirs, dists, sdf_network, deviation_network, cos_anneal_ratio, valid_interval, output_orients=False):
        n_samples = pts.shape[0]

        if output_orients:
            sdf_nn_output = fill_tensor(sdf_network(pts[valid_interval], calc_orient=output_orients), valid_interval)
        else:
            sdf_nn_output = fill_tensor(sdf_network(pts[valid_interval]), valid_interval)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        orients = None
        if output_orients:
            orients = feature_vector[:, -3:]
            feature_vector = feature_vector[:, :-3]
        
        gradients = fill_tensor(sdf_network.gradient(pts[valid_interval]).squeeze(1), valid_interval, 1)

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6) # Single parameter
        inv_s = inv_s.expand(n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
        alpha *= valid_interval.view_as(alpha)

        return alpha, c, sdf, inv_s, gradients, feature_vector, orients

    def render_core(self,
                    rays_o,
                    rays_d,
                    radii,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    head_interval_indices, 
                    hair_interval_indices,
                    near, far,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, sample_dist.expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)
    
        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        head_valid_interval = (head_interval_indices[..., :-1] == head_interval_indices[..., 1:]) & (head_interval_indices[..., :-1] != -1)
        head_valid_interval = torch.cat([head_valid_interval, head_valid_interval.new_zeros(batch_size, 1)], dim=-1).view(-1)
        hair_valid_interval = (hair_interval_indices[..., :-1] == hair_interval_indices[..., 1:]) & (hair_interval_indices[..., :-1] != -1)
        hair_valid_interval = torch.cat([hair_valid_interval, hair_valid_interval.new_zeros(batch_size, 1)], dim=-1).view(-1)

        (
            alpha_head,
            cdf_head,
            sdf_head,
            inv_s_head,
            gradients_head,
            feature_vector_head,
            _
        ) = self.eval_sdf(pts, dirs, dists, sdf_network, deviation_network, cos_anneal_ratio, head_valid_interval)

        (
            alpha_hair,
            cdf_hair,
            sdf_hair,
            inv_s_hair,
            gradients_hair,
            feature_vector_hair,
            _
        ) = self.eval_sdf(pts, dirs, dists, 
                          self.hair_network, 
                          self.hair_deviation_network, 
                          cos_anneal_ratio, 
                          hair_valid_interval)

        gradients_hair_norm = gradients_hair / (gradients_hair.norm(dim=-1, keepdim=True) + 1e-5)
        gradients_head_norm = gradients_head / (gradients_head.norm(dim=-1, keepdim=True) + 1e-5)

#      soft blend alphas of hair and head
        mixing_weights = alpha_head / (alpha_head + alpha_hair + 1e-5)
        alpha = (alpha_head + alpha_hair).clamp(0, 1)

#      obtain blended features and normals
        gradients_norm = gradients_head_norm * mixing_weights + gradients_hair_norm * (1 - mixing_weights)
        feature_vector = feature_vector_head * mixing_weights + feature_vector_hair * (1 - mixing_weights)

        sampled_color = color_network(pts, gradients_norm, dirs, feature_vector)
            
        alpha = alpha.reshape(batch_size, n_samples)
        alpha_hair = alpha_hair.reshape(batch_size, n_samples)

        sampled_color = sampled_color.reshape(batch_size, n_samples, sampled_color.shape[-1])

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        weights_hair = alpha_hair * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
            
        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)
        
        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Do surface rendering at estimated surface points 
        with torch.no_grad():
            depth_z_vals, pts_depth, mask = sphere_trace_surface_points(self.hair_network, rays_o, rays_d, near, far)
        
        orient = self.eval_sdf(pts_depth[mask], rays_d[mask], dists.mean(-1)[mask], 
                                    self.hair_network, 
                                    self.hair_deviation_network, 
                                    cos_anneal_ratio,
                                    valid_interval=pts_depth.new_ones((batch_size,), dtype=torch.bool)[mask], 
                                    output_orients=True)[-1][:, None] # n_rays, 1, 3
        sampled_orient = fill_tensor(orient, mask)

        # Eikonal loss
        gradient_error_head = (torch.linalg.norm(gradients_head.reshape(batch_size, n_samples, 3), ord=2,
                                                 dim=-1) - 1.0) ** 2
        gradient_error_head = (relax_inside_sphere * gradient_error_head).sum() / (relax_inside_sphere.sum() + 1e-5)


        gradient_error_hair = (torch.linalg.norm(gradients_hair.reshape(batch_size, n_samples, 3), ord=2,
                                                 dim=-1) - 1.0) ** 2
        gradient_error_hair = (relax_inside_sphere * gradient_error_hair).sum() / (relax_inside_sphere.sum() + 1e-5)
            
        outputs = {
            'color': color,
            'sampled_orient': sampled_orient,
            'pts':  pts_depth,
            'sdf_head': sdf_head,
            'cdf_head': cdf_head.reshape(batch_size, n_samples),
            'alpha_head': alpha_head,
            'alpha_hair': alpha_hair,
            'dists': dists,
            'gradients_head': gradients_head.reshape(batch_size, n_samples, 3),
            'gradients_hair': gradients_hair.reshape(batch_size, n_samples, 3),
            's_val_head': 1.0 / inv_s_head,
            's_val_hair': 1.0 / inv_s_hair,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'weights_hair': weights_hair,
            'gradient_error_head': gradient_error_head,
            'inside_sphere': inside_sphere
        }


        outputs['head_prior_inside_points'] = None   
        outputs['num_inside_points'] = None
        outputs['head_prior_sdf'] = None   
        outputs['head_prior_normal'] = None
        outputs['head_prior_off_sdf'] = None

        if self.head_prior_mesh is not None and pts.numel() > 0:
            idx_inside_mesh = self.head_prior_mesh.compute_inside_points(pts)
            outputs['head_prior_inside_points'] = alpha_hair.reshape(-1)[idx_inside_mesh]
            outputs['num_inside_points'] = idx_inside_mesh.shape[0]
            
            if self.head_prior_attraction:          
                # sample the same number of points as in pts
                pts_mesh, normals_on_mesh = self.head_prior_mesh.sample_from_mesh(num_points=pts.shape[0])
                sdf_on_mesh = sdf_network(pts_mesh)[:, :1]  #[pts.shape[0], 3]
                gradients_sdf_on_mesh = sdf_network.gradient(pts_mesh).squeeze()
                # choose points not on mesh
                not_mesh_idx, on_mesh_idx = self.head_prior_mesh.points_on_mesh(pts) 
                off_sdf_constraint = torch.exp(-1e2 * torch.abs(sdf_head[not_mesh_idx])).mean()
                # F(x)=0 on mesh
                sdf_constraint = torch.abs(torch.cat((sdf_head[on_mesh_idx], sdf_on_mesh), dim=0)).mean()
                # Normals on mesh are the same
                normal_constraint = (1 - F.cosine_similarity(gradients_sdf_on_mesh, normals_on_mesh, dim=-1)).mean()
                
                outputs['head_prior_sdf'] = sdf_constraint   
                outputs['head_prior_normal'] = normal_constraint
                outputs['head_prior_off_sdf'] = off_sdf_constraint
                

        outputs['sdf_hair'] = sdf_hair
        outputs['cdf_hair'] = cdf_hair.reshape(batch_size, n_samples)
        outputs['gradient_error_hair'] = gradient_error_hair

        return outputs

    def render(self, rays_o, rays_d, radii, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):

        batch_size = len(rays_o)   
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]
        sample_dist = z_vals.new_ones((batch_size, 1)) * 2.0 / self.n_samples # Assuming the region of interest is a unit sphere
        head_interval_indices = torch.ones_like(z_vals, dtype=torch.int)
        hair_interval_indices = torch.ones_like(z_vals, dtype=torch.int)
        ray_has_intersection = z_vals.new_ones((batch_size,), dtype=torch.bool)

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * sample_dist

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        z_vals_coarse = z_vals.clone()

        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)
                geometry_hair = None
                if self.blended_upsample:
                    geometry_hair = self.hair_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)


                for i in range(self.up_sample_steps):
                    new_z_vals, _ = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                geometry_hair,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i,
                                                head_interval_indices,
                                                hair_interval_indices)
                    z_vals, sdf, geometry_hair, _ = self.cat_z_vals(
                                                rays_o,
                                                rays_d,
                                                z_vals,
                                                new_z_vals,
                                                sdf,
                                                geometry_hair,
                                                last=(i + 1 == self.up_sample_steps))
                    head_interval_indices = head_interval_indices.new_ones(z_vals.shape)
                    hair_interval_indices = hair_interval_indices.new_ones(z_vals.shape)
            n_samples = self.n_samples + self.n_importance

        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    radii,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    head_interval_indices, 
                                    hair_interval_indices,
                                    near,
                                    far,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)
        

        weights = fill_tensor(ret_fine['weights'], ray_has_intersection)
        weights_hair = fill_tensor(ret_fine['weights_hair'], ray_has_intersection)
        weights_sum = weights.sum(dim=-1, keepdim=True)
        weights_hair_sum = weights_hair.sum(dim=-1, keepdim=True)
        s_val_head = ret_fine['s_val_head'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)
        s_val_hair = ret_fine['s_val_hair'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        outputs = {
            'color_fine': fill_tensor(ret_fine['color'], ray_has_intersection, background_rgb if background_rgb is not None else 0),
            'sampled_orient_fine': fill_tensor(ret_fine['sampled_orient'], ray_has_intersection),
            'pts_fine': fill_tensor(ret_fine['pts'], ray_has_intersection),
            'alpha_head': ret_fine['alpha_head'],
            'alpha_hair': ret_fine['alpha_hair'],
            's_val_head': s_val_head,
            's_val_hair': s_val_hair,
            'cdf_head': fill_tensor(ret_fine['cdf_head'], ray_has_intersection), 
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'weight_hair_sum': weights_hair_sum,
            'weight_hair_max': torch.max(weights_hair, dim=-1, keepdim=True)[0],
            'weights': weights,
            'weights_hair': weights_hair,
            'gradients_head': fill_tensor(ret_fine['gradients_head'], ray_has_intersection, 1),
            'gradients_hair': fill_tensor(ret_fine['gradients_hair'], ray_has_intersection, 1),
            'head_prior_inside_points': ret_fine['head_prior_inside_points'],
            'num_inside_points': ret_fine['num_inside_points'],
            'head_prior_sdf': ret_fine['head_prior_sdf'],   
            'head_prior_normal': ret_fine['head_prior_normal'],
            'head_prior_off_sdf': ret_fine['head_prior_off_sdf'],
            'gradient_error_head': ret_fine['gradient_error_head'],
            'inside_sphere': fill_tensor(ret_fine['inside_sphere'], ray_has_intersection),
        }


        outputs['cdf_hair'] = fill_tensor(ret_fine['cdf_hair'], ray_has_intersection)
        outputs['gradient_error_hair'] = ret_fine['gradient_error_hair']
        
        return outputs 

    def up_sample(self, rays_o, rays_d, z_vals, sdf, geometry_hair, n_importance, inv_s, head_label, hair_label):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        
        alpha_sdf = self.compute_sdf_alpha_for_upsample(z_vals, sdf, inside_sphere, inv_s)
        valid_head_interval = (head_label[..., :-1] == head_label[..., 1:]) & (head_label[..., :-1] != -1)
        valid_hair_interval = (hair_label[..., :-1] == hair_label[..., 1:]) & (hair_label[..., :-1] != -1) 
        alpha_sdf *= valid_head_interval
        if self.blended_upsample:
            alpha_hair = self.compute_sdf_alpha_for_upsample(z_vals, geometry_hair, inside_sphere, inv_s)
            alpha_hair *= valid_hair_interval

            alpha = (alpha_sdf + alpha_hair).clamp(0, 1)
        else:
            alpha = alpha_sdf

        valid_interval = valid_head_interval | valid_hair_interval
        
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + valid_interval * 1e-7], -1), -1)[:, :-1]
        z_samples = sample_pdf(z_vals, weights, n_importance, det=True, valid_interval=valid_interval, detailed_output=True)
        return z_samples

    def compute_sdf_alpha_for_upsample(self, z_vals, sdf, inside_sphere, inv_s):
        batch_size, n_samples = z_vals.shape
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha_sdf = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)

        return alpha_sdf

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, geometry_hair, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            sdf = sdf[(xx, index.reshape(-1))].reshape(batch_size, n_samples + n_importance)
            if self.blended_upsample:
                new_geometry_hair = self.hair_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance) 
                geometry_hair = torch.cat([geometry_hair, new_geometry_hair], dim=-1)
                geometry_hair = geometry_hair[(xx, index.reshape(-1))].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf, geometry_hair, index
    
    def extract_hair_geometry(self, bound_min, bound_max, resolution, threshold=0.0, dilation_radius=50):
        vertices, triangles = extract_geometry(
                                bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: self.hair_network.sdf(pts))
        return vertices, triangles
