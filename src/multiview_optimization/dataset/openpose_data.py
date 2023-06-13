import json
import os
import numpy as np
from copy import deepcopy


class OpenposeData:
    def __init__(self, path, views, device, filter_views_mapping=None):
        data = []
        self.sns = sorted(os.listdir(path))
        for sn in self.sns:
            with open(os.path.join(path, sn), 'r') as f:
                data.append(json.load(f))
        
        if filter_views_mapping is not None:
            views = [filter_views_mapping[i] for i in views]
        self.filtered_data = [data[i] for i in range(len(data)) if i in views]
        self.part2lmk_count = {
            'pose': 25,
            'face': 70,
            'hand_left': 21,
            'hand_right': 21
        }
        self.device = device
        self.postfix_data = '_keypoints_2d'
        self.dump_data = dict()
        for part, lmk_count in self.part2lmk_count.items():
            lmks = np.zeros((lmk_count, 2), dtype=np.float32)
            confs = np.zeros_like(lmks)

            self.dump_data[f'op_{part}'] = lmks
            self.dump_data[f'op_conf_{part}'] = confs

    def get_sample(self, index):
        result = dict()
        for part, lmk_count in self.part2lmk_count.items():
            part_data = np.array(self.filtered_data[index]['people'][0][part+self.postfix_data], dtype=np.float32).reshape(-1, 3)
            result[f'op_{part}'] = part_data[:, 0:2]
            result[f'op_conf_{part}'] = np.repeat(part_data[:, [2]], 2, axis=1)

        # ignore hand lmks if root hand joint has low confidence
        if result['op_conf_pose'][4, 0] < 0.5:
            result['op_conf_hand_right'][:] = 0
        if result['op_conf_pose'][7, 0] < 0.5:
            result['op_conf_hand_left'][:] = 0

        # ignore hand lmks if they intersect
        llu = np.min(result['op_hand_left'], axis=0)
        lrb = np.max(result['op_hand_left'], axis=0)
        rlu = np.min(result['op_hand_right'], axis=0)
        rrb = np.max(result['op_hand_right'], axis=0)
        int_lu = np.max(np.stack([llu, rlu]), axis=0)

        int_rb = np.min(np.stack([lrb, rrb]), axis=0)
        if int_rb[0] > int_lu[0] and int_rb[1] > int_lu[1]:
            int_area = (int_rb[0] - int_lu[0]) * (int_rb[1] - int_lu[1])

            l_area = (lrb[0] - llu[0]) * (lrb[1] - llu[1]) + 1e-5
            l_iou = int_area / l_area
            if l_iou > 0.5:
                result['op_conf_hand_left'][:] = 0

            r_area = (rrb[0] - rlu[0]) * (rrb[1] - rlu[1]) + 1e-5
            r_iou = int_area / r_area
            if r_iou > 0.5:
                result['op_conf_hand_right'][:] = 0

        return result