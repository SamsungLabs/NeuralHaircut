from shutil import copyfile
import os
from pathlib import Path

def main(args):

    exp_name = args.exp_name
    case = args.case
    conf_path = args.conf_path

    exps_dir = Path('./exps_first_stage') / exp_name / case / Path(conf_path).stem
    prev_exps = sorted(exps_dir.iterdir())
    cur_dir = prev_exps[-1].name      

    path_to_mesh = os.path.join(exps_dir, cur_dir, 'meshes')
    path_to_ckpt = os.path.join(exps_dir, cur_dir, 'checkpoints')
    path_to_fitted_camera = os.path.join(exps_dir, cur_dir, 'cameras')

    meshes = sorted(os.listdir(path_to_mesh))

    last_ckpt = sorted(os.listdir(path_to_ckpt))[-1]
    last_hair = [i for i in head_string if i.split('_')[-1].split('.')[0]=='hair'][-1]
    last_head = [i for i in head_string if i.split('_')[-1].split('.')[0]=='head'][-1]

    print(f'Copy obtained from the first stage checkpoint, hair and head geometry to folder ./implicit-hair-data/data/{case}')
    
    copyfile(os.path.join(path_to_mesh, last_hair), f'./implicit-hair-data/data/{case}/final_hair.ply')
    copyfile(os.path.join(path_to_mesh, last_head), f'./implicit-hair-data/data/{case}/final_head.ply')
    copyfile(os.path.join(path_to_ckpt, last_ckpt), f'./implicit-hair-data/data/{case}/ckpt_final.ply')

    if os.path.exists(path_to_fitted_camera):
        print(f'Copy obtained from the first stage camera fitting checkpoint to folder ./implicit-hair-data/data/{case}')
        last_camera = sorted(os.listdir(path_to_fitted_camera))[-1]
        copyfile(os.path.join(path_to_fitted_camera, last_camera), f'./implicit-hair-data/data/{case}/fitted_cameras.pth')


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--conf_path', default='./configs/example_config/monocular/neural_strands-monocular.yaml', type=str)
    
    parser.add_argument('--case', default='person_1', type=str)
       
    parser.add_argument('--exp_name', default='first_stage_reconctruction_person_1', type=str)  

    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)