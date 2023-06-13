import numpy as np
import torch
import pickle
import os
import glob

def rec_convert(d, f):
    result = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = rec_convert(v, f)
        else:
            result[k] = f(v)
    return result

class OptParams:
    def __init__(self, device, dataset, train_rotation, train_pose, train_shape, checkpoint_path):
        self.device = device
        self.dataset = dataset
        
        # sequence specific params
        self.ssp = ['global_rot', 'global_trans', 'global_scale']
        self.bsp = ['beta']

        self.beta = None
        self.global_rot = None
        self.global_trans = None
        self.global_scale = None
        
        # frame specific params
        self.fsp = ['pose_body', 'pose_jaw', 'face_expression']

        self.pose_jaw = None
        self.pose_body = None
        self.face_expression = None

        self.train_rotation = train_rotation
        self.train_pose = train_pose
        self.train_shape = train_shape

        self.initialize_parameters()

        if checkpoint_path:
            files = glob.glob(f'{checkpoint_path}/*')
            files.sort(key=os.path.getmtime)
            self.upload_parameters(files[-1])

    
    def tensor_kwargs(self, requires_grad=False):
        return dict(device=self.device, dtype=torch.float32, requires_grad=requires_grad)
    

    def upload_parameters(self, upload_path):
        print('upload parameters from', upload_path)

        with open(upload_path, 'rb') as f:
            data_ckpt = pickle.load(f)

        self.beta = torch.tensor(data_ckpt['beta'], **self.tensor_kwargs(requires_grad=self.train_shape))
        
        if 'pose_jaw' in data_ckpt and data_ckpt['pose_jaw'].shape[0]==self.dataset.nimages:
            self.pose_jaw = torch.tensor(data_ckpt['pose_jaw'], **self.tensor_kwargs(requires_grad=self.train_pose))
            self.face_expression = torch.tensor(data_ckpt['face_expression'], **self.tensor_kwargs(requires_grad=self.train_pose))
            self.pose_body = torch.tensor(data_ckpt['pose_body'], **self.tensor_kwargs(requires_grad=self.train_pose))

        self.global_rot = torch.tensor(data_ckpt['global_rot'], **self.tensor_kwargs(requires_grad=self.train_rotation))
        self.global_trans = torch.tensor(data_ckpt['global_trans'], **self.tensor_kwargs(requires_grad=self.train_rotation))
        self.global_scale = torch.tensor(data_ckpt['global_scale'], **self.tensor_kwargs(requires_grad=self.train_rotation))
    

    def initialize_parameters(self):
        
        datas = []
        with open(self.dataset.pixie_init_path, 'rb') as fp:
            N = len(os.listdir(self.dataset.image_path))
            for i in range(N):
                datas.append(pickle.load(fp))

        shapes = np.array([i['shape'].cpu().numpy()[0] for i in datas])
        exps = np.array([i['exp'].cpu().numpy()[0] for i in datas])
        jaw_pose = np.array([i['jaw_pose'].cpu().numpy()[0] for i in datas])  
        
        # filter views that have no keypoints
        filtered_views = self.dataset.get_filter_views().cpu().numpy()

        shapes = shapes[filtered_views]
        jaw_pose = jaw_pose[filtered_views]
        exps = exps[filtered_views]
        
        self.beta = torch.tensor(shapes.mean(axis=0),  **self.tensor_kwargs(requires_grad=self.train_shape))

        self.pose_jaw = torch.tensor(jaw_pose,  **self.tensor_kwargs(requires_grad=self.train_pose))
        self.face_expression = torch.tensor(exps,  **self.tensor_kwargs(requires_grad=self.train_pose))

        self.pose_body = torch.tensor(torch.eye(3).repeat(self.dataset.nimages, 21, 1, 1 ).numpy(),  **self.tensor_kwargs(requires_grad=self.train_pose))
 
        self.global_rot = torch.zeros((3,), **self.tensor_kwargs(requires_grad=self.train_rotation))
        self.global_trans = torch.zeros((3,), **self.tensor_kwargs(requires_grad=self.train_rotation))
        self.global_scale = torch.tensor(np.array([3.]), **self.tensor_kwargs(requires_grad=self.train_rotation))
      
    def get_params_dict(self, mode='train'):
        result = dict()

        if mode == 'train':
            if self.train_rotation:
                for k in self.ssp:
                    result[k] = getattr(self, k)

            if self.train_shape:
                for k in self.bsp:
                    result[k] = getattr(self, k)
            
            if self.train_pose:
                for k in self.fsp:
                    result[k] = getattr(self, k)
        else:
            # dump all tensors
            all_key_list = self.ssp + self.bsp + self.fsp
            for k in all_key_list:
                result[k] = getattr(self, k)
        return result

    def get_params_dict_cpu_copy(self, mode):
        result = rec_convert(
            self.get_params_dict(mode),
            lambda x: x.detach().cpu().numpy()
        )
        return result

    def get_train_params_list(self, **kwargs):
        result = []
        for k, v in self.get_params_dict(**kwargs).items():
            if isinstance(v, dict):
                result += list(v.values())
            else:
                result.append(v)
        return result
    
    def dump_json_dict(self, fp):
        print('dump data')
        result = self.get_params_dict_cpu_copy(mode='save')
        with open(fp, 'wb') as f:
            pickle.dump(result, f)
