import warnings
import os
import torch
from dataset import Multiview_dataset
from losses import OpenPoseLoss, RegShapeLoss, KeypointsMatchingLoss
from runner import Runner
import argparse
from pyhocon import ConfigFactory
from models.SMPLX import SMPLX
from utils.config import cfg as pixie_cfg
warnings.filterwarnings("ignore")


def main(conf_path, batch_size, train_rotation, train_pose, train_shape, checkpoint_path, save_path):
    with open(conf_path) as f:
        conf_text = f.read()
    conf = ConfigFactory.parse_string(conf_text)
    device = torch.device(conf['general.device'])

    # Create SMPLX model from PIXIE
    smplx_model = SMPLX(pixie_cfg.model).to(device) 

    dataset = Multiview_dataset(**conf['dataset'], device=conf['general.device'], batch_size=batch_size)

#         Create train losses
    losses = []
    if conf.get_float('loss.fa_kpts_2d_weight'):
        losses += [KeypointsMatchingLoss(device=device, use_3d=False)]
    if conf.get_float('loss.fa_kpts_3d_weight'):
        losses += [KeypointsMatchingLoss(device=device, use_3d=True)]
    if conf.get_float('loss.openpose_face_weight'):
        losses += [OpenPoseLoss(mode='face', device=device)]
    if conf.get_float('loss.openpose_body_weight'):
        losses += [OpenPoseLoss(mode='body', device=device)]
    if conf.get_float('loss.reg_shape_weight'):
        losses += [RegShapeLoss(weight=float(conf['loss.reg_shape_weight']))]
    print(losses)
    loss_weights = {}
    loss_weights['reg_shape'] = conf.get_float('loss.reg_shape_weight')
    loss_weights['openpose_body'] = conf.get_float('loss.openpose_body_weight')
    loss_weights['openpose_face'] = conf.get_float('loss.openpose_face_weight')
    loss_weights['fa_kpts'] = conf.get_float('loss.fa_kpts_2d_weight')

    os.makedirs(save_path, exist_ok=True)

    runner = Runner(
            dataset,
            losses,
            smplx_model,
            device,
            save_path,
            cut_flame_head=conf.get_bool('general.cut_flame_head'),
            loss_weights=loss_weights,
            train_rotation=train_rotation, 
            train_shape=train_shape, 
            train_pose=train_pose, 
            checkpoint_path=checkpoint_path
        )
    runner.fit(
            epochs=conf.get_int('train.epochs'),
            lr=conf.get_float('train.learning_rate'),
            max_iter=conf.get_int('train.max_iter'),
        )
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--train_rotation', type=bool, default=False)
    parser.add_argument('--train_pose', type=bool, default=False)
    parser.add_argument('--train_shape', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    args = parser.parse_args()
    
    main(
        conf_path=args.conf,
        batch_size=args.batch_size,
        train_rotation=args.train_rotation, 
        train_pose=args.train_pose,
        train_shape=args.train_shape,
        checkpoint_path=args.checkpoint_path,
        save_path = args.save_path
     )