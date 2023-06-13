import os
import sys
from pyhocon import ConfigFactory
from pathlib import Path

import torch

from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer.cameras import  FoVPerspectiveCameras
from pytorch3d.renderer import TexturesVertex, look_at_view_transform
from pytorch3d.io import load_ply, save_ply

sys.path.append(os.path.join(sys.path[0], '..'))
from src.models.dataset import Dataset, MonocularDataset

import argparse
import yaml

from tqdm import tqdm

def create_visibility_map(camera, rasterizer, mesh):
    fragments = rasterizer(mesh, cameras=camera)
    pix_to_face = fragments.pix_to_face  
    packed_faces = mesh.faces_packed() 
    packed_verts = mesh.verts_packed() 
    vertex_visibility_map = torch.zeros(packed_verts.shape[0]) 
    faces_visibility_map = torch.zeros(packed_faces.shape[0]) 
    visible_faces = pix_to_face.unique()[1:] # not take -1
    visible_verts_idx = packed_faces[visible_faces] 
    unique_visible_verts_idx = torch.unique(visible_verts_idx)
    vertex_visibility_map[unique_visible_verts_idx] = 1.0
    faces_visibility_map[torch.unique(visible_faces)] = 1.0
    return vertex_visibility_map, faces_visibility_map


def check_visiblity_of_faces(cams, meshRasterizer, full_mesh, mesh_head, n_views=2):
    # collect visibility maps
    vis_maps = []
    for cam in tqdm(range(len(cams))):
        v, _ = create_visibility_map(cams[cam], meshRasterizer, full_mesh)
        vis_maps.append(v)

    # took faces that were visible at least from n_views to reduce noise
    vis_mask = (torch.stack(vis_maps).sum(0) > n_views).float()

    idx = torch.nonzero(vis_mask).squeeze(-1).tolist()
    idx = [i for i in idx if i > mesh_head.verts_packed().shape[0]]
    indices_mapping = {j: i for i, j in enumerate(idx)}

    face_faces = []
    face_idx = torch.tensor(idx).to('cuda')
    vertex = full_mesh.verts_packed()[face_idx]

    for fle, i in enumerate(full_mesh.faces_packed()):
        if i[0] in face_idx and i[1] in face_idx and i[2] in face_idx:
            face_faces += [[indices_mapping[i[0].item()], indices_mapping[i[1].item()], indices_mapping[i[2].item()]]]
    return vertex, torch.tensor(face_faces)

    
def main(args):
    
    scene_type = args.scene_type
    case = args.case
    save_path = f'./implicit-hair-data/data/{scene_type}/{case}'
                             
    # upload mesh hair and bust
    verts, faces = load_ply(os.path.join(save_path, 'final_hair.ply'))
    mesh_hair =  Meshes(verts=[(verts).float().to(args.device)], faces=[faces.to(args.device)])

    verts_, faces_ = load_ply(os.path.join(save_path, 'final_head.ply'))
    
    mesh_head =  Meshes(verts=[(verts_).float().to(args.device)], faces=[faces_.to(args.device)])

    raster_settings_mesh = RasterizationSettings(
                        image_size=args.img_size, 
                        blur_radius=0.000, 
                        faces_per_pixel=1, 
                    )

    # init camera
    R = torch.ones(1, 3, 3)
    t = torch.ones(1, 3)
    cam_intr = torch.ones(1, 4, 4)
    size = torch.tensor([args.img_size, args.img_size ]).to(args.device)

    cam = cameras_from_opencv_projection(
                                        camera_matrix=cam_intr.cuda(), 
                                        R=R.cuda(),
                                        tvec=t.cuda(),
                                        image_size=size[None].cuda()
                                          ).cuda()

    # init mesh rasterization
    meshRasterizer = MeshRasterizer(cam, raster_settings_mesh)

    mesh_hair.textures = TexturesVertex(verts_features=torch.ones_like(mesh_hair.verts_packed()).float().cuda()[None])
    mesh_head.textures = TexturesVertex(verts_features=torch.zeros_like(mesh_head.verts_packed()).float().cuda()[None])

    # join hair and bust mesh to handle occlusions
    full_mesh = join_meshes_as_scene([mesh_head, mesh_hair])


    # take from config dataset parameters
    with open(args.conf_path, 'r') as f:
        replaced_conf = str(yaml.load(f, Loader=yaml.Loader)).replace('CASE_NAME', args.case)
        conf = yaml.load(replaced_conf, Loader=yaml.Loader)
            

    if scene_type == 'h3ds':
        dataset = Dataset(conf['dataset'])
    else:
        dataset = MonocularDataset(conf['dataset'])


    # add upper cameras for h3ds data as they haven't got such views
    cams_up = []
    if scene_type == 'h3ds':
        elevs = [-100, -90, -80]
        for elev in elevs:
            R, T = look_at_view_transform(dist=2., elev=elev, azim=100)
            cam = FoVPerspectiveCameras(device=args.device, R=R, T=T)
            cams_up.append(cam)

    # add dataset cameras
    intrinsics_all = dataset.intrinsics_all #intrinsics
    pose_all_inv = torch.inverse(dataset.pose_all) #extrinsics

    cams_dataset = [cameras_from_opencv_projection(
                                    camera_matrix=intrinsics_all[idx][None].cuda(), 
                                    R=pose_all_inv[idx][:3, :3][None].cuda(),
                                    tvec=pose_all_inv[idx][:3, 3][None].cuda(),
                                    image_size=size[None].cuda()
                                     ).cuda() for idx in range(dataset.n_images)]

    cams = cams_dataset + cams_up
    vis_vertex, vis_face = check_visiblity_of_faces(cams, meshRasterizer, full_mesh, mesh_head, n_views=args.n_views)

    os.makedirs(save_path, exist_ok=True)
    save_ply(os.path.join(save_path, 'hair_outer.ply'), vis_vertex, vis_face)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--conf_path', default='./configs/monocular/neural_strands.yaml', type=str)
    
    parser.add_argument('--case', default='person_1', type=str)
    
    parser.add_argument('--scene_type', default='monocular', type=str) 
    
    parser.add_argument('--device', default='cuda', type=str)
    
    parser.add_argument('--img_size', default=2160, type=int)
    parser.add_argument('--n_views', default=2, type=int)
    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)