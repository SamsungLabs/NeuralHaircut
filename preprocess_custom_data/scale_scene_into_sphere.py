import trimesh
import numpy as np
import pickle

import os



def main(args):
    
    path_to_scene = os.path.join(args.path_to_data, args.scene_type, args.case)

    pc = np.array(trimesh.load(os.path.join(path_to_scene, 'point_cloud_cropped.ply')).vertices)

    translation = (pc.min(0) + pc.max(0)) / 2
    scale = np.linalg.norm(pc - translation, axis=-1).max().item() / 1.1

    tr = (pc - translation) / scale
    assert tr.min() >= -1 and tr.max() <= 1

    print('Scaling into the sphere', tr.min(), tr.max())

    d = {'scale': scale,
        'translation': list(translation)}

    with open(os.path.join(path_to_scene, 'scale.pickle'), 'wb') as f:
        pickle.dump(d, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--case', default='person_1', type=str)
    parser.add_argument('--scene_type', default='monocular', type=str)
    
    parser.add_argument('--path_to_data', default='./implicit-hair-data/data/', type=str) 

    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)