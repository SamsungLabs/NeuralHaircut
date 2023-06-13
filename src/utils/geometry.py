import torch

def project_orient_to_camera(orient_3d, org_3d, cam_intr, cam_extr):
        reshape_out = len(orient_3d.shape) == 3
        if reshape_out:
            b, n, _ = orient_3d.shape
        org_3d = org_3d.view(-1, 3)
        orient_3d = orient_3d.view(-1, 3)

        dst_3d = org_3d + orient_3d

        dummy_ones = torch.ones(dst_3d.shape[0], 1, device=dst_3d.device, dtype=dst_3d.dtype)
        org_3d = torch.cat([org_3d, dummy_ones], dim=1)[..., None] # M x 4 x 1
        dst_3d = torch.cat([dst_3d, dummy_ones], dim=1)[..., None]

        if cam_extr.dim() == 3:
            org_3d_cam = torch.matmul(cam_intr, torch.matmul(cam_extr, org_3d))[:, :3, 0] # M x 3
            dst_3d_cam = torch.matmul(cam_intr, torch.matmul(cam_extr, dst_3d))[:, :3, 0] # M x 3
        else:
            org_3d_cam = torch.matmul(cam_intr[None], torch.matmul(cam_extr[None], org_3d))[:, :3, 0] # M x 3
            dst_3d_cam = torch.matmul(cam_intr[None], torch.matmul(cam_extr[None], dst_3d))[:, :3, 0] # M x 3

        org_2d_cam = org_3d_cam[:, :2] / (org_3d_cam[:, [2]] + 1e-5)
        dst_2d_cam = dst_3d_cam[:, :2] / (dst_3d_cam[:, [2]] + 1e-5)

        orient_2d_cam = dst_2d_cam - org_2d_cam
        orient_2d_cam = orient_2d_cam / (orient_2d_cam.norm(dim=-1, keepdim=True) + 1e-5)
        if reshape_out:
            orient_2d_cam = orient_2d_cam.view(b, n, 2)


        orient_sin = orient_2d_cam[..., 0]
        to_mirror = torch.ones_like(orient_sin)
        to_mirror[orient_sin < 0] *= -1
        orient_cos = orient_2d_cam[..., 1] * to_mirror

        sampled_orient_angle = torch.acos(orient_cos.clamp(-1 + 1e-5, 1 - 1e-5))

        orient_angle = sampled_orient_angle[:, 0]
        orient_angle = orient_angle[:, None]
        return orient_angle
    

def soft_interpolate(valid_pixels, x):
    feat_size = x.shape[1]
    alpha_j = (1 - valid_pixels % 1).unsqueeze(-1).repeat(1, feat_size)
    vj = x[(valid_pixels // 1).long()]
    vj_ = x[
        torch.minimum((valid_pixels // 1 + 1).long(), torch.tensor(x.shape[0] - 1).long())
    ]
    interp_x = alpha_j * vj + (1 - alpha_j) * vj_
    return interp_x

def hard_interpolate(valid_pixels, x):
    feat_size = x.shape[1]
    vj = x[(valid_pixels // 1).long()]
    return vj

###
### https://github.com/krrish94/nerf-pytorch/blob/master/nerf/nerf_helpers.py
###
def barycentric_coordinates_of_projection(points, vertices):
    ''' https://github.com/MPI-IS/mesh/blob/master/mesh/geometry/barycentric_coordinates_of_projection.py
    '''
    """Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.
    See
        **Heidrich**, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf
    
    :param p: point to project. [B, 3]
    :param v0: first vertex of triangles. [B, 3]
    :returns: barycentric coordinates of ``p``'s projection in triangle defined by ``q``, ``u``, ``v``
            vectorized so ``p``, ``q``, ``u``, ``v`` can all be ``3xN``
    """
    #(p, q, u, v)
    v0, v1, v2 = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    p = points
    q = v0
    u = v1 - v0
    v = v2 - v0
    n = torch.cross(u, v)
    s = torch.sum(n * n, dim=1)
    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    s[s == 0] = 1e-6
    oneOver4ASquared = 1.0 / s
    w = p - q
    b2 = torch.sum(torch.cross(u, w) * n, dim=1) * oneOver4ASquared
    b1 = torch.sum(torch.cross(w, v) * n, dim=1) * oneOver4ASquared
    weights = torch.stack((1 - b1 - b2, b1, b2), dim=-1)
    # check barycenric weights
    # p_n = v0*weights[:,0:1] + v1*weights[:,1:2] + v2*weights[:,2:3]
    return weights


###
### https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
###
def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])

    nd = vertices.shape[2]
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, nd))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]

