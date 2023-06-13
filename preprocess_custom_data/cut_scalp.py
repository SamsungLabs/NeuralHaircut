import matplotlib.pyplot as plt

from skimage.draw import polygon
import numpy as np

import pickle

from tqdm import tqdm
import yaml
import sys
import os

import torch
from pytorch3d.io import load_obj, load_objs_as_meshes, save_obj
from pytorch3d.structures import Meshes

sys.path.append(os.path.join(sys.path[0], '..'))
from src.hair_networks.sdf import HairSDFNetwork
from src.utils.geometry import face_vertices

import argparse

import cv2


def create_scalp_mask(scalp_mesh, scalp_uvs):
    img = np.zeros((256, 256, 1), 'uint8')
    
    for i in range(scalp_mesh.faces_packed().shape[0]):
        text = scalp_uvs[0][scalp_mesh.faces_packed()[i]].reshape(-1, 2).cpu().numpy()
        poly = 255/2 * (text + 1)
        rr, cc = polygon(poly[:,0], poly[:,1], img.shape)
        img[rr,cc, :] = (255)

    scalp_mask = np.flip(img.transpose(1, 0, 2), axis=0)
    return scalp_mask


def main(args):

    # indices of scalp vertices
    scalp_vert_idx = torch.load('./data/new_scalp_vertex_idx.pth').long().cuda()
    # faces that form a scalp
    scalp_faces = torch.load('./data/new_scalp_faces.pth')[None].cuda() 
    scalp_uvs = torch.load('./data/new_scalp_uvcoords.pth')[None].cuda()

    path_to_mesh = os.path.join(args.path_to_data, args.scene_type, args.case, 'head_prior.obj')
    path_to_ckpt = os.path.join(args.path_to_data, args.scene_type, args.case, 'ckpt_final.pth')
    
    save_path = os.path.join(args.path_to_data, args.scene_type, args.case)
    
    # Upload FLAME head geometry
    head_mesh = load_objs_as_meshes([path_to_mesh], device=args.device)
    verts, faces, _ = load_obj(path_to_mesh)
    head_mesh =  Meshes(verts=[(verts).float()], faces=[faces.verts_idx]).cuda()
    
    # Convert the head mesh into a scalp mesh
    scalp_verts = head_mesh.verts_packed()[None, scalp_vert_idx]
    scalp_face_verts = face_vertices(scalp_verts, scalp_faces)[0]
    scalp_mesh = Meshes(verts=scalp_verts, faces=scalp_faces).cuda()
    
    # Upload config  

    with open(args.conf_path, 'r') as f:
        replaced_conf = str(yaml.load(f, Loader=yaml.Loader)).replace('CASE_NAME', args.case)
        conf = yaml.load(replaced_conf, Loader=yaml.Loader)
            

    hair_network = HairSDFNetwork(**conf['model']['hair_sdf_network']).to(args.device)
    checkpoint = torch.load(path_to_ckpt, map_location=args.device)
    hair_network.load_state_dict(checkpoint['hair_network'])
    
    # Calculate sdf at scalp vertices
    sdf = hair_network(scalp_mesh.verts_packed())[..., 0]
    
    # Consider verices that close to sdf
    sorted_idx = torch.where(sdf < args.distance)[0]
    
    # Cut new scalp
    a = np.array(sorted(sorted_idx.cpu()))
    b = np.arange(a.shape[0])
    d = dict(zip(a,b))

    full_scalp_list = sorted(sorted_idx)

    faces_masked = []
    for face in scalp_mesh.faces_packed():
#         print(face[0] , full_scalp_list)
#         input()
        if face[0] in full_scalp_list and face[1] in full_scalp_list and  face[2] in full_scalp_list:
            faces_masked.append(torch.tensor([d[int(face[0])], d[int(face[1])], d[int(face[2])]]))
#         print(faces_masked, full_scalp_list)
        save_obj(os.path.join(save_path, 'scalp.obj'), scalp_mesh.verts_packed()[full_scalp_list], torch.stack(faces_masked))

    with open(os.path.join(save_path, 'cut_scalp_verts.pickle'), 'wb') as f:
        pickle.dump(list(torch.tensor(sorted_idx).detach().cpu().numpy()), f)
       
    # Create scalp mask for diffusion
    scalp_uvs = scalp_uvs[:, full_scalp_list]    
    scalp_mesh = load_objs_as_meshes([os.path.join(save_path, 'scalp.obj')], device=args.device)
    
    scalp_mask = create_scalp_mask(scalp_mesh, scalp_uvs)
    cv2.imwrite(os.path.join(save_path, 'dif_mask.png'), scalp_mask)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--conf_path', default='./configs/monocular/neural_strands.yaml', type=str)
        
    parser.add_argument('--case', default='person_1', type=str)
    
    parser.add_argument('--scene_type', default='monocular', type=str, choices=['h3ds', 'monocular']) 
    
    parser.add_argument('--path_to_data', default='./implicit-hair-data/data/', type=str)  

    parser.add_argument('--device', default='cuda', type=str)
    
    parser.add_argument('--distance', default=0.07, type=float)

    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)