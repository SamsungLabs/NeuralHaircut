import argparse
import os
import pickle
import numpy
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--case', default='person_1', type=str)
    parser.add_argument('--scene_type', default='monocular', type=str)
    parser.add_argument('--path_to_data', default='../../implicit-hair-data/data/', type=str)
    
    args = parser.parse_args()

    data_dir = os.path.join(args.path_to_data, args.scene_type, args.case)

    data_param = []
    for img_file in os.listdir(os.path.join(data_dir, 'image')):
        img_file_stem = img_file.split('.')[0]
        # Read individual pixie parameter file
        with open(os.path.join(data_dir, 'pixie', img_file_stem,f'{img_file_stem}_param.pkl'), 'rb') as f:
            kv = pickle.load(f)
            # Convert numpy.ndarray to torch.tensor
            for key in kv.keys():
                if type(kv[key]) is numpy.ndarray:
                    kv[key] = torch.from_numpy(kv[key])
                    # Insert a new axis to the front
                    kv[key] = kv[key].unsqueeze(0)
            data_param.append(kv)

    # Write to initialization_pixie
    with open(os.path.join(data_dir, 'initialization_pixie'), 'wb') as f:
        for p in data_param:
            pickle.dump(p, f)
