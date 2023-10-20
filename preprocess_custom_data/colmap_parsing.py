import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
from tqdm import tqdm
import trimesh
import argparse

from scipy.spatial.transform import Rotation as R

    
def main(args):
    
    images_file = f'{args.path_to_scene}/colmap/sparse_txt/images.txt'
    points_file = f'{args.path_to_scene}/colmap/sparse_txt/points3D.txt'
    camera_file = f'{args.path_to_scene}/colmap/sparse_txt/cameras.txt'
    
    # Parse colmap cameras and used images
    with open(camera_file) as f:
        lines = f.readlines()

        u = float(lines[3].split()[4])
        h, w = [int(x) for x in lines[3].split()[5: 7]]

        intrinsic_matrix = np.array([
            [u, 0, h, 0],
            [0, u, w, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    with open(images_file) as f:
        images_file_lines = f.readlines()

    n_images = len(images_file_lines[4:]) // 2

    data = {}

    for i in range(n_images):

        line_split = images_file_lines[4 + i * 2].split()


        q = np.array([float(x) for x in line_split[1: 5]]) # w, x, y, z
        t = np.array([float(x) for x in line_split[5: 8]])

        image_name = line_split[-1]

        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = R.from_quat(np.roll(q, -1)).as_matrix()
        extrinsic_matrix[:3, 3] = t

        data[image_name] = intrinsic_matrix @ extrinsic_matrix    


    with open(points_file) as f:
        points_3d_lines = f.readlines()

    points = []
    colors = []

    for line in points_3d_lines[3:]:
        split_line = line.split()
        point = np.array([float(v) for v in split_line[1: 4]])
        color = np.array([int(v) for v in split_line[4: 7]])
        points.append(point)
        colors.append(color)

    points = np.stack(points)
    colors = np.stack(colors)
    
    cameras = []

    for k in sorted(data.keys()):
        T = data[k]
        cameras.append(T)

    np.savez(os.path.join(args.path_to_scene, 'cameras.npz'), np.stack(cameras))
    trimesh.points.PointCloud(points).export(os.path.join(args.path_to_scene, 'point_cloud.ply'));


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--path_to_scene', default='./implicit-hair-data/data/monocular/person_1', type=str)
    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)