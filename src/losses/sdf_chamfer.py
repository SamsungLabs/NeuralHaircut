from pytorch3d.io import IO
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import  Pointclouds
from .one_way_chamfer import chamfer_distance
from pytorch3d import _C


class SdfChamfer:
    def __init__(self, device='cuda', mesh_outer_hair_remeshed=None,  mesh_outer_hair=None, num_points=10000):

        self.num_points = num_points
        self.mesh_outer_hair = IO().load_mesh(mesh_outer_hair, device=device)
        # to ease the calculation of points2face distanse
        self.mesh_outer_hair_remeshed =IO().load_mesh(mesh_outer_hair_remeshed, device=device) 
        
    
    def points2face(self, mesh, points):
        pcl = Pointclouds(points=[points.float()])
        points = pcl.points_packed()
        points_first_idx = pcl.cloud_to_packed_first_idx()
        max_points = pcl.num_points_per_cloud().max().item()
        verts_packed = mesh.verts_packed()
        faces_packed = mesh.faces_packed()
        tris = verts_packed[faces_packed]
        tris_first_idx = mesh.mesh_to_faces_packed_first_idx()
        # Compute point to face distance
        dists, idxs = _C.point_face_dist_forward(points.float(), points_first_idx, tris.float(), tris_first_idx, max_points, 1e-10)
        pp = tris[idxs].mean(1)
        # Return idx of closest face, distance and center point of closest face
        return dists, idxs, pp
        
    def calc_chamfer(self, points):
        # sample points from visible outer hair surface
        sample_points  = sample_points_from_meshes(self.mesh_outer_hair, self.num_points)
        # calculate one-way chamfer
        loss_chamf, _ = chamfer_distance(sample_points, points)
        return loss_chamf
