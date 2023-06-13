import torch

from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.renderer.mesh.rasterizer import Fragments


from typing import Union, Optional, Tuple


class SoftShader(ShaderBase):

    def __init__(
        self,
        image_size: int,
        feats_dim: int = 32,
        sigma: float = 1e-3,
        gamma: float = 1e-5,
        return_alpha: bool = False,
        znear: Union[float, torch.Tensor] = 1.0,
        zfar: Union[float, torch.Tensor] = 100,
        num_head_faces: Optional[int] = None,
        use_orients_cond=False,
        **kwargs
    ):
        
        super().__init__(**kwargs)
        
        self.use_orients_cond = use_orients_cond
        self.image_size = image_size
        self.feats_dim = feats_dim
        self.sigma = sigma
        self.gamma = gamma
        self.return_alpha = return_alpha
        self.background_color = torch.zeros(feats_dim + self.use_orients_cond, dtype=torch.float32)
        self.znear = znear
        self.zfar = zfar
        self.num_head_faces = num_head_faces

    def get_colors(self, fragments: Fragments, meshes: Meshes):
        pix_to_face, barycentric_coords = fragments.pix_to_face, fragments.bary_coords

        vertex_attributes = meshes.textures.verts_features_packed().unsqueeze(0)
        faces = meshes.faces_packed().unsqueeze(0)

        res = vertex_attributes[range(vertex_attributes.shape[0]), faces.flatten(start_dim=1).T]
        res = torch.transpose(res, 0, 1)
        attributes = res.reshape(faces.shape[0], faces.shape[1], 3, vertex_attributes.shape[-1])
        
        if self.return_alpha:
            alpha_mask = (pix_to_face != -1).float()[:, :, :, 0].unsqueeze(1)

        # Reshaping for torch.gather
        D = attributes.shape[-1]
        attributes = attributes.clone() # Needed for backprop
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])

        N, H, W, K, _ = barycentric_coords.shape

        # Needed for correct working of torch.gather
        mask = (pix_to_face == -1)
        pix_to_face = pix_to_face.clone() # Needed for backprop
        pix_to_face[mask] = 0

        # Building a tensor of sampled values
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)

        # Barycentric interpolation
        pixel_vals = (barycentric_coords.unsqueeze(-1) * pixel_face_vals).sum(-2)
        pixel_vals[mask] = 0
        
        if self.return_alpha:
            pixel_vals = torch.cat((pixel_vals, alpha_mask), 1)

        return pixel_vals
    
    def forward(self, fragments: Fragments, meshes: Meshes):

        N, H, W, K = fragments.pix_to_face.shape
        
        # Mask for padded pixels.
        if self.num_head_faces is None:
            mask = fragments.pix_to_face >= 0
        else:
            first_head_face = meshes.faces_list()[0].shape[0] - self.num_head_faces
            head_mask = fragments.pix_to_face >= first_head_face
            mask = head_mask
            
            for i in range(1, head_mask.shape[-1]):
                mask[..., i] += mask[..., i - 1]

            mask += fragments.pix_to_face < 0
            mask = ~mask.bool()

        colors = self.get_colors(fragments, meshes)
        pixel_colors = torch.ones((N, H, W, self.feats_dim), dtype=colors.dtype, device=colors.device)
        background_color = self.background_color.to(colors.device)

        '''
        # Fragments dists recalculation
        grid = torch.stack(torch.meshgrid((torch.linspace(1, -1, 512, device=meshes.device), ) * 2, indexing='xy'))

        s = torch.ones_like(fragments.dists[mask])
        s[fragments.dists[mask] < 0] = -1

        verts = meshes.verts_packed()

        face_to_edge = fragments.pix_to_face[mask] + 1 - 2 * (fragments.pix_to_face[mask] % 2)

        p0 = verts[face_to_edge][:, :2]
        p1 = verts[face_to_edge + 2][:, :2]

        t0 = verts[fragments.pix_to_face[mask]][:, :2]
        t1 = verts[fragments.pix_to_face[mask] + 2][:, :2]

        p_edges_norm = (p1 - p0).norm(dim=1, keepdim=True)
        t_edges_norm = (t1 - t0).norm(dim=1, keepdim=True)

        i = torch.argwhere(mask)
        x = grid[:, i[:, 1], i[:, 2]].permute(1, 0)

        d_p = ((x - p0)[:, 0] * (p1 - p0)[:, 1] - (x - p0)[:, 1] * (p1 - p0)[:, 0]).squeeze(0).abs() / (p_edges_norm.squeeze(1) + 1e-8)
        d_t = ((x - t0)[:, 0] * (t1 - t0)[:, 1] - (x - t0)[:, 1] * (t1 - t0)[:, 0]).squeeze(0).abs() / (t_edges_norm.squeeze(1) + 1e-8)

        fragments.dists[mask] = s * (torch.min(d_p, d_t) ** 2)
        fragments.pix_to_face[mask][d_t < d_p] = fragments.pix_to_face[mask][d_t < d_p] + s[d_t < d_p].long()
        '''

        # Weight for background color
        eps = 1e-10

        # Sigmoid probability map based on the distance of the pixel to the face.
        prob_map = torch.sigmoid(-fragments.dists / self.sigma) * mask

        # Weights for each face. Adjust the exponential by the max z to prevent
        # overflow. zbuf shape (N, H, W, K), find max over K.
        # TODO: there may still be some instability in the exponent calculation.

        alpha = torch.prod((1.0 - prob_map), dim=-1)

        # Reshape to be compatible with (N, H, W, K) values in fragments
        if torch.is_tensor(self.zfar):
            # pyre-fixme[16]
            self.zfar = self.zfar[:, None, None, None]
        if torch.is_tensor(self.znear):
            # pyre-fixme[16]: Item `float` of `Union[float, Tensor]` has no attribute
            #  `__getitem__`.
            self.znear = self.znear[:, None, None, None]

        z_inv = (self.zfar - fragments.zbuf) / (self.zfar - self.znear) * mask

        z_inv_max = torch.max(z_inv, dim=-1).values.unsqueeze(-1).clamp(min=eps)
        weights_num = prob_map * torch.exp((z_inv - z_inv_max) / self.gamma)

        # Also apply exp normalize trick for the background color weight.
        # Clamp to ensure delta is never 0.
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
        delta = torch.exp((eps - z_inv_max) / self.gamma).clamp(min=eps)

        # Normalize weights.
        # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
        denom = weights_num.sum(dim=-1).unsqueeze(-1) + delta

        # Sum: weights * textures + background color
        weighted_colors = (weights_num.unsqueeze(-1) * colors).sum(-2)
        weighted_background = delta * background_color
        pixel_colors = (weighted_colors + weighted_background) / denom
        
        # The cumulative product ensures that alpha will be 0.0 if at least 1
        # face fully covers the pixel as for that face, prob will be 1.0.
        # This results in a multiplication by 0.0 because of the (1.0 - prob)
        # term. Therefore 1.0 - alpha will be 1.0.
        # alpha = torch.prod((1.0 - prob_map), dim=-1)
        pixel_colors = pixel_colors * (1.0 - alpha.unsqueeze(-1))

        return pixel_colors.permute(0, 3, 1, 2), mask, colors