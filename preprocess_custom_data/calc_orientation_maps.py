from PIL import Image
import numpy as np
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from skimage.filters import difference_of_gaussians
import math
import os
import tqdm
import cv2
import argparse
from joblib import Parallel, delayed

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def generate_gabor_filters(sigma_x, sigma_y, freq, num_filters):
    thetas = np.linspace(0, math.pi * (num_filters - 1) / num_filters, num_filters)
    print(thetas / math.pi * 180)
    kernels = []
    for theta in thetas:
        kernel = np.real(gabor_kernel(freq, theta=math.pi - theta, sigma_x=sigma_x, sigma_y=sigma_y))
        kernels.append(kernel)
    return kernels


def calc_orients(img, kernels, n_jobs):
    gray_img = rgb2gray(img)
    filtered_image = difference_of_gaussians(gray_img, 0.4, 10)
    gabor_filtered_images = list(Parallel(n_jobs=n_jobs, prefer="threads")(delayed(ndi.convolve)(filtered_image, kernels[i], mode='wrap') for i in range(len(kernels))))
    F_orients = np.abs(np.stack(gabor_filtered_images)) # abs because we only measure angle in [0, pi]
    return F_orients


def calc_confidences(F_orients, orientation_map, num_filters, n_jobs):
    orients_bins = np.linspace(0, math.pi * (num_filters - 1) / num_filters, num_filters)
    
    def calc_dists(i):
        dists = np.abs(orientation_map - orients_bins[i])
        dists = np.minimum(np.abs(orientation_map - orients_bins[i] - math.pi), dists)
        dists = np.minimum(np.abs(orientation_map - orients_bins[i] + math.pi), dists)
        return dists
    dists = np.stack(Parallel(n_jobs=n_jobs, prefer="threads")(delayed(calc_dists)(i) for i in range(num_filters)))
        
    F_orients_norm = F_orients / F_orients.sum(axis=0, keepdims=True)
    
    V_F = (dists**2 * F_orients_norm).sum(0)
    
    return V_F

def main(args):

    os.makedirs(args.orient_dir, exist_ok=True)
    os.makedirs(args.conf_dir, exist_ok=True)
    
    kernels = generate_gabor_filters(args.sigma_x, args.sigma_y, args.freq, args.num_filters)
    
    img_list = sorted(os.listdir(args.img_path))
    for img_name in tqdm.tqdm(img_list):
        basename = img_name.split('.')[0]
        img = np.array(Image.open(os.path.join(args.img_path, img_name)))
        F_orients = calc_orients(img, kernels, args.n_jobs)
        orientation_map = F_orients.argmax(0)
        orientation_map_rad = orientation_map.astype('float16') / args.num_filters * math.pi
        confidence_map = calc_confidences(F_orients, orientation_map_rad, args.num_filters, args.n_jobs)

        cv2.imwrite(f'{args.orient_dir}/{basename}.png', orientation_map.astype('uint8'))
        np.save(f'{args.conf_dir}/{basename}.npy', confidence_map.astype('float16'))


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--img_path', default='./implicit-hair-data/data/h3ds/00141/image/', type=str)
    parser.add_argument('--orient_dir', default='./implicit-hair-data/data/h3ds/00141/orientation_maps/', type=str)
    parser.add_argument('--conf_dir', default='./implicit-hair-data/data/h3ds/00141/confidence_maps/', type=str)
    parser.add_argument('--sigma_x', default=1.8, type=float)
    parser.add_argument('--sigma_y', default=2.4, type=float)
    parser.add_argument('--freq', default=0.23, type=float)
    parser.add_argument('--num_filters', default=180, type=int)
    parser.add_argument('--n_jobs', default=8, type=int)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)