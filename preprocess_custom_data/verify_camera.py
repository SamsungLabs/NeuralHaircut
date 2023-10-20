import argparse
import numpy as np
import trimesh
import os
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cameras_npz', type=str, required=True, help='Path to cameras.npz file')
    parser.add_argument('--point_cloud_ply', type=str, required=True, help='Path to point_cloud.ply file')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to directory containing images')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--point_indices', type=str, default=None, help='Comma-separated list of point indices to visualize')
    parser.add_argument('--point_radius', type=int, default=4, help='Radius of points in pixels [4]')
    parser.add_argument('--seed_offset', type=int, default=0, help='Seed offset for random number generator [0]')
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cameras = np.load(args.cameras_npz)['arr_0']

    points = np.array(trimesh.load(args.point_cloud_ply).vertices)

    if args.point_indices is not None:
        args.point_indices = [int(x) for x in args.point_indices.split(',')]

    for i, image_file in enumerate(sorted(os.listdir(args.image_dir))):
        image = cv2.imread(os.path.join(args.image_dir, image_file))

        width, height = image.shape[1], image.shape[0]
        assert(width == height)

        KR = cameras[i, 0:3, 0:3]
        Kt = cameras[i, 0:3, 3]

        for j, point in enumerate(points):
            if args.point_indices is not None and j not in args.point_indices:
                continue

            # Project point into image
            uv = KR @ point + Kt
            uv /= uv[2]
            uv = np.round(uv).astype(int)

            # Choose random color deterministically by index j
            np.random.seed(j + args.seed_offset)
            color = np.random.randint(0, 255, 3)

            # Rasterize point
            r = args.point_radius
            for du in range(-r, r+1):
                for dv in range(-r, r+1):
                    if (du * du + dv * dv <= r*r + 3):
                        if (uv[0] + du >= 0 and uv[0] + du < width and
                            uv[1] + dv >= 0 and uv[1] + dv < height):
                            image[uv[1] + dv, uv[0] + du] = color
            
        cv2.imwrite(os.path.join(args.output_dir, image_file), image)
