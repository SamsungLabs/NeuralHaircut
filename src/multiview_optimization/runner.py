import tqdm
from tensorboardX import SummaryWriter
import cv2
import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.lbs import batch_rodrigues
from utils import misc
from pytorch3d.io import save_obj
from opt_params import OptParams


def process_visuals(visuals):
    visual_list = []
    keys_visuals = list(visuals.keys())
    for visual_key in keys_visuals:
        visual_list += misc.prepare_visual(visuals, visual_key, preprocessing_op=None)
    visual_list = torch.cat(visual_list, 3) # cat w.r.t. width
    visual_list = visual_list.clamp(0, 1)    
    visual_list = torch.cat(visual_list.split(1, 0), 2)[0] # cat batch dim in lines w.r.t. height
    visual_list = visual_list.cpu()
    return visual_list

def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image * 255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1, 2, 0)
    return image.astype(np.uint8).copy()

    
class Runner:
    def __init__(
        self,
        dataset,
        losses,
        smplx_model,
        device,
        save_path,
        cut_flame_head,
        loss_weights,
        train_rotation, 
        train_pose, 
        train_shape,
        checkpoint_path
    ):

        self.dataset = dataset
        self.losses = losses
        self.smplx = smplx_model
        self.opt_params = OptParams(
                                    device,
                                    dataset,
                                    train_rotation, 
                                    train_pose, 
                                    train_shape,
                                    checkpoint_path
                                    )  
        self.device = device
        self.save_path = save_path

        self.eye_pose = torch.eye(3, requires_grad=False, device=self.device).unsqueeze(0).repeat(self.dataset.nimages, 2, 1, 1)
        self.left_hand_pose = torch.eye(3, requires_grad=False, device=self.device).unsqueeze(0).repeat(self.dataset.nimages, 15, 1, 1)
        self.right_hand_pose = torch.eye(3, requires_grad=False, device=self.device).unsqueeze(0).repeat(self.dataset.nimages, 15, 1, 1)

        self.cut_flame_head = cut_flame_head
        self.loss_weights = loss_weights
        self.img_size = dataset.images.shape[2] 
        
        os.makedirs(os.path.join(save_path, 'mesh'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'opt_params'), exist_ok=True)
        
    @torch.no_grad()
    def get_visuals(self, lm_pred, lm_gt, label, visuals, batch):
        name = 'target_stickman' + str(label)
        image_size = batch['img'].shape[1]
        if label == 'openpose_face':
            visuals[name] = misc.draw_stickman(lm_gt / (image_size / 2) - 1, image_size, images=batch['img']) #lmks[-1, 1]
            visuals['pred_' + name] = misc.draw_stickman(lm_pred / (image_size / 2) - 1, image_size, images=batch['img'])
        elif label =='openpose_body':
            visuals[name] = misc.draw_stickman_body(lm_gt / (image_size / 2) - 1, image_size, images=batch['img'])
            visuals['pred_' + name] = misc.draw_stickman_body(lm_pred / (image_size / 2) - 1, image_size, images=batch['img'])
        else:
            visuals[name] = misc.draw_stickman_fa(lm_gt/ (image_size / 2) - 1, image_size, images=batch['img'])
            visuals['pred_' + name] = misc.draw_stickman_fa(lm_pred / (image_size / 2) - 1, image_size, images=batch['img'])
        return visuals
    

    def fit(self, epochs, lr, max_iter, tol=1e-9):

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.dataset.nimages,
            shuffle=False,
            num_workers=0
        )

        for batch_idx, batch in enumerate(dataloader):
            if epochs > 0:
                param_lst = self.opt_params.get_train_params_list()
                # need to create optimizer for every batch so that data for prev batches is not changed
                optimizer = optim.LBFGS(
                    param_lst,
                    lr=lr, max_iter=max_iter,
                    line_search_fn='strong_wolfe',
                    tolerance_grad=tol, tolerance_change=tol)
            frame_id_0 = batch['frame_ids'][0]
            writer = SummaryWriter(log_dir=os.path.join(self.save_path, f'logs/{frame_id_0:06d}'))
            step = 0

            for epoch in range(0, epochs):
                tq = tqdm.tqdm(ncols=100)
                tq.set_description(f'Epoch {epoch}')
                tq.refresh()

                def closure():
                    nonlocal step, optimizer, tq, epoch, writer

                    def log(loss_str, label, value_gpu):
                        loss_value_cpu = value_gpu.detach().cpu().numpy()
                        loss_str += f'\n\t{label}: {loss_value_cpu:.1E} '
                        writer.add_scalar(os.path.join('train', label), loss_value_cpu, step)
                        return loss_str, loss_value_cpu

                    if torch.is_grad_enabled():
                        optimizer.zero_grad()

                    pred = self.forward(batch)
                    writer.add_scalar(os.path.join('train', 'epoch'), epoch, step)
                    loss_str = 'losses:'
                    step += 1
                    loss_values = []
                    
                    visuals = {}
                    for loss in self.losses:
                        loss_value, lm_gt, lm_pred = loss.compute(pred, batch, img_size=self.img_size, weight=self.loss_weights[loss.label])
                        loss_str, _ = log(loss_str, loss.label, loss_value)
                        loss_values.append(loss_value)
                        if loss.label ==  'openpose_face' or loss.label ==  'openpose_body' or loss.label == 'fa_kpts':
                            visuals = self.get_visuals(lm_pred, lm_gt, loss.label, visuals, batch)                            
                    
                    if step % 10 == 0:
                        self.dump_results(pred, step, epoch, batch_idx)
                        
                    visual_list = process_visuals(visuals)
                    writer.add_image(f'images', visual_list, step)

                    total_loss = torch.stack(loss_values).sum()
                    if step % 20 == 0:
                        print(total_loss)

                    loss_str, loss_value_cpu = log(loss_str, 'total_loss', total_loss)
                    total_loss.backward()

                    tq.update()
                    tq.set_postfix({'loss': f'{loss_value_cpu:.3E}'})
                    tq.refresh()

                    return total_loss

                optimizer.step(closure)

                tq.close()
    
    
    def dump_results(self, pred, step, epoch, batch_idx):
        save_obj(os.path.join(self.save_path, 'mesh', f'mesh{epoch}_{step}_{batch_idx}.obj'), pred['verts_world'][0], pred['faces_world']) 
        self.opt_params.dump_json_dict(os.path.join(self.save_path, 'opt_params', f'opt_params_{epoch}_{step}_{batch_idx}'))

    def obtain_global_matx(self, batch, ds_ids):
        b = len(ds_ids)
        scale = (torch.eye(3, device=self.device) * self.opt_params.global_scale.repeat(3)).repeat(b, 1, 1)
        global_rot = torch.bmm(batch_rodrigues(self.opt_params.global_rot.repeat(b, 1)), scale) #[b, 3, 3]
        global_trans = self.opt_params.global_trans.repeat(b, 1).reshape(b, 3, 1)  #[b, 3, 1]
        return global_rot.unsqueeze(1), global_trans.reshape(b, 3).unsqueeze(1)

    def forward(self, batch):
        ds_ids = batch['frame_ids']
        b = len(ds_ids)
        
        pose_jaw_rotmtx =  torch.index_select(self.opt_params.pose_jaw, 0, ds_ids)
        pose_body_rotmtx  =  torch.index_select(self.opt_params.pose_body, 0, ds_ids)
        face_expression = torch.index_select(self.opt_params.face_expression, 0, ds_ids)
        betas = self.opt_params.beta.reshape(1, -1).repeat(b, 1)
            
        global_rotmtx, global_trans = self.obtain_global_matx(batch, ds_ids)

        #  use default pose
        eye_pose = torch.index_select(self.eye_pose, 0, ds_ids)
        left_hand_pose = torch.index_select(self.left_hand_pose, 0, ds_ids)
        right_hand_pose = torch.index_select(self.right_hand_pose, 0, ds_ids)
        
        verts, landmarks, joints = self.smplx(shape_params=betas, expression_params=face_expression,  global_pose=global_rotmtx, body_pose=pose_body_rotmtx, jaw_pose=pose_jaw_rotmtx, eye_pose=eye_pose, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)
     
        # Add translation
        verts += global_trans
        joints += global_trans
        landmarks += global_trans
        
#        Change axis to be in NEUS space
        verts[:, :, :0] = -verts[:, :, :0] 
        joints[:, :, :0]  = -joints[:, :, :0] 
        landmarks[:, :, :0]  = -landmarks[:, :, :0] 
        
        result = {}
        
        if self.cut_flame_head:
            flame_verts, flame_faces = self.smplx.cut_flame_head(verts)
            result['verts_world'] = flame_verts
            result['faces_world'] = flame_faces
        else:
            result['verts_world'] = verts
            result['faces_world'] = self.smplx.faces_tensor
        
#       get extrinsics
        extrinsics_rot = batch['extrinsics_rvec'].unsqueeze(1)
        extrinsics_trans = batch['extrinsics_tvec'].unsqueeze(1)
        
#         world to camera transform
        joints = torch.matmul(extrinsics_rot.repeat(1, joints.shape[1], 1, 1), joints.unsqueeze(-1)).squeeze(-1)
        verts = torch.matmul(extrinsics_rot.repeat(1, verts.shape[1], 1, 1), verts.unsqueeze(-1)).squeeze(-1)
        landmarks = torch.matmul(extrinsics_rot.repeat(1, landmarks.shape[1], 1, 1), landmarks.unsqueeze(-1)).squeeze(-1)

        Jtr = joints + extrinsics_trans
        verts_trans = verts + extrinsics_trans
        landmarks += extrinsics_trans

        result['verts_extrinsics'] = verts #[bs, 10475, 3]
        result['verts'] = verts_trans 
        result['Jtr'] = Jtr #[166, 3]
        result['betas'] = self.opt_params.beta
        result['face_kpt'] = landmarks
        return result