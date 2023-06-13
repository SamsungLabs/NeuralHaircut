import numpy as np
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.ops import  knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d import _C
import torch
import pickle

class HeadPriorMesh:
    def __init__(self, path_to_mesh, device='cuda', tol=1e-8):
        verts, faces, aux = load_obj(path_to_mesh, device='cuda')
        self.mesh =  Meshes(verts=[(verts).to(device)], faces=[faces.verts_idx.to(device)])
            
        self.tol = tol
    
    def get_vertex_normals(self, idx):
        return self.mesh.verts_normals_packed()[idx].squeeze(1)
    
    def get_vertex(self, idx):
        return self.mesh.verts_packed()[idx].squeeze(1)
    
    def get_faces_normals(self, idx):
        return self.mesh.faces_normals_packed()[idx]
    
    def points_on_mesh(self, points):
        # Return idxes of points not on mesh
        dists, idxs, pp = self.points2face(points)
        out_idxes = torch.where(dists > self.tol)[0] 
        on_idxes = torch.where(dists <= self.tol)[0] 
        return out_idxes, on_idxes
    
    def sample_from_mesh(self, num_points=512*128):
        points_mesh, points_normals = sample_points_from_meshes(self.mesh, num_samples=num_points, return_normals=True)
        return points_mesh.squeeze(0), points_normals.squeeze(0) #[num_points, 3]
    
    def points2face(self, points):
        pcl = Pointclouds(points=[points.float()])
        points = pcl.points_packed()
        points_first_idx = pcl.cloud_to_packed_first_idx()
        max_points = pcl.num_points_per_cloud().max().item()
        verts_packed = self.mesh.verts_packed()
        faces_packed = self.mesh.faces_packed()
        tris = verts_packed[faces_packed]
        tris_first_idx = self.mesh.mesh_to_faces_packed_first_idx()
        # Compute point to face distance
        dists, idxs = _C.point_face_dist_forward(points.float(), points_first_idx, tris.float(), tris_first_idx, max_points, 5e-3)
        pp = tris[idxs].mean(1)
        # Return idx of closest face, distance and center point of closest face
        return dists, idxs, pp
        
    def compute_inside_points(self, points):
        _, idxs, pp = self.points2face(points)
        cosine = ((points - pp) * self.get_faces_normals(idxs)).sum(dim=1)
        return torch.where(cosine < 0)[0] 
