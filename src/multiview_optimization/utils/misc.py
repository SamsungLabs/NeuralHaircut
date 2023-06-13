import torch
import numpy as np
import cv2
from torchvision import transforms
from torch import nn
import torch.nn.functional as F

def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image * 255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1,2,0)
    return image.astype(np.uint8).copy()


def apply_imagenet_normalization(input):
    r"""Normalize using ImageNet mean and std.
    Args:
        input (4D tensor NxCxHxW): The input images, assuming to be [0, 1].
    Returns:
        Normalized inputs using the ImageNet normalization.
    """
    # normalize the input using the ImageNet mean and std
    mean = input.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = input.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    output = (input - mean) / std
    return output

@torch.no_grad()
def grid_sampler_backward(grad_out, grid, h=None, w=None, padding_mode='zeros', align_corners=False):
    b, c = grad_out.shape[:2]
    if h is None or w is None:
        h, w = grad_out.shape[2:]
    size = torch.FloatTensor([w, h]).to(grad_out.device)
    grad_in = torch.zeros(b, c, h, w, device=grad_out.device)

    if align_corners:
        grid_ = (grid + 1) / 2 * (size - 1)
    else:
        grid_ = ((grid + 1) * size - 1) / 2

    if padding_mode == 'border':
        assert False, 'TODO'

    elif padding_mode == 'reflection':
        assert False, 'TODO'

    grid_nw = grid_.floor().long()
    
    grid_ne = grid_nw.clone()
    grid_ne[..., 0] += 1
    
    grid_sw = grid_nw.clone()
    grid_sw[..., 1] += 1
    
    grid_se = grid_nw.clone() + 1
    
    nw = (grid_se - grid_).prod(3)
    ne = (grid_ - grid_sw).abs().prod(3)
    sw = (grid_ne - grid_).abs().prod(3)
    se = (grid_ - grid_nw).prod(3)

    indices_ = torch.cat([
        (
            (
                g[:, None, ..., 0] + g[:, None,..., 1] * w
            ).repeat_interleave(c, dim=1) 
            + torch.arange(c, device=g.device)[None, :, None, None] * (h*w) # add channel shifts
            + torch.arange(b, device=g.device)[:, None, None, None] * (c*h*w) # add batch size shifts
        ).view(-1) 
        for g in [grid_nw, grid_ne, grid_sw, grid_se]
    ])

    masks = torch.cat([
        (
            (g[..., 0] >= 0) & (g[..., 0] < w) & (g[..., 1] >= 0) & (g[..., 1] < h)
        )[:, None].repeat_interleave(c, dim=1).view(-1)
        for g in [grid_nw, grid_ne, grid_sw, grid_se]
    ])
    
    values_ = torch.cat([
        (m[:, None].repeat_interleave(c, dim=1) * grad_out).view(-1)
        for m in [nw, ne, sw, se]
    ])

    indices = indices_[masks]
    values = values_[masks]
    
    grad_in.put_(indices, values, accumulate=True)

    return grad_in

# def replace_bn_with_in(module):
#     mod = module
    
#     if isinstance(module, nn.BatchNorm2d) or isinstance(module, apex.parallel.SyncBatchNorm):
#         mod = nn.InstanceNorm2d(module.num_features, affine=True)

#         gamma = module.weight.data.squeeze().detach().clone()
#         beta = module.bias.data.squeeze().detach().clone()
        
#         mod.weight.data = gamma
#         mod.bias.data = beta

#     else:
#         for name, child in module.named_children():
#             mod.add_module(name, replace_bn_with_in(child))

#     del module
#     return mod

@torch.no_grad()
def keypoints_to_heatmaps(keypoints, img):
    HEATMAPS_VAR = 1e-2
    s = img.shape[2]

    keypoints = keypoints[..., :2] # use 2D projection of keypoints

    return kp2gaussian(keypoints, img.shape[2:], HEATMAPS_VAR)

def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out

def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed

def prepare_visual(data_dict, tensor_name, preprocessing_op=None):
    visuals = []

    if tensor_name in data_dict.keys():
        tensor = data_dict[tensor_name].detach().cpu()
            
        if preprocessing_op is not None:
            tensor = preprocessing_op(tensor)

        if tensor.shape[1] == 1:
            tensor = torch.cat([tensor] * 3, dim=1)

        elif tensor.shape[1] == 2:
            b, _, h, w = tensor.shape

            tensor = torch.cat([tensor, torch.empty(b, 1, h, w, dtype=tensor.dtype).fill_(-1)], dim=1)
        tensor = F.interpolate(tensor, (256, 256))
        visuals += [tensor]

    return visuals

def draw_stickman_body(keypoints, image_size, images=None):
    ### Define drawing options ###
    edges_parts  = [[i] for i in range(25)]


    closed_parts = [True, True, True, False, False, False, False]

    colors_parts = [
        (  255,  255,  255),  
        (  255,    0,    0), (    0,  255,    0),
        (    0,    0,  255), (    0,    0,  255), 
        (  255,    0,  255), (    0,  255,  255),  (    0,  25,  255),  (    0,  255,  25),  (    15,  25,  255),  (    0,  50,  25),  (    0,  255,  255),  (    100,  25,  255),  (    0,  105,  25),  (    100,  255,  255),  (    20,  25,  255),  (    200,  0,  255),  (    0,  220,  25),  (    0,  255,  100),  (    0,  205,  255),  (    10,  255,  255),  (    0,  2,  255),  (    0,  22,  25),  (    100,  20,  255),  (    255,  10,  255)]

    ### Start drawing ###
    stickmen = []

    for i  in range(keypoints.shape[0]):
        if keypoints[i] is None:
            stickmen.append(torch.zeros(3, image_size, image_size))
            continue

        if isinstance(keypoints[i], torch.Tensor):
            xy = (keypoints[i, :, :2].detach().cpu().numpy() + 1) / 2 * image_size
        
        elif keypoints[i].max() < 1.0:
            xy = keypoints[i, :, :2] * image_size

        else:
            xy = keypoints[i, :, :2]

        xy = xy[None, :, None].astype(np.int32)
        stickman = np.ones((image_size, image_size, 3), np.uint8) if images is None else tensor2image(images[i])
        for edges, closed, color in zip(edges_parts, closed_parts, colors_parts):
            if len(edges) > 1:
                stickman = cv2.polylines(stickman, xy[:, edges], closed, color, thickness=2)
            else:
                stickman = cv2.circle(stickman, (int(xy[:, edges][0][0][0][0]), int(xy[:, edges][0][0][0][1])), 10, color, -1)

        stickman = torch.FloatTensor(stickman.transpose(2, 0, 1)) / 255.
        stickmen.append(stickman)

    stickmen = torch.stack(stickmen)
    stickmen = (stickmen - 0.5) * 2. 

    return stickmen

def draw_stickman_fa(keypoints, image_size, images=None):
    ### Define drawing options ###
    edges_parts  = [
        list(range( 0, 17)), # face
        list(range(17, 22)), list(range(22, 27)), # eyebrows (right left)
        list(range(27, 31)) + [30, 33], list(range(31, 36)), # nose
        list(range(36, 42)), list(range(42, 48)), # right eye, left eye
        list(range(48, 60)), list(range(60, 68))] # lips

    closed_parts = [
        False, False, False, False, False, True, True,  True, True]

    colors_parts = [
        (  255,  255,  255), 
        (  255,    0,    0), (    0,  255,    0),
        (    0,    0,  255), (    0,    0,  255), 
        (  255,    0,  255), (    0,  255,  255),
#        (   255,  255,  255),
        (  255,  255,    0), (  255,  255,    0)]

    ### Start drawing ###
    stickmen = []

    for i  in range(keypoints.shape[0]):
        if keypoints[i] is None:
            stickmen.append(torch.zeros(3, image_size, image_size))
            continue

        if isinstance(keypoints[i], torch.Tensor):
            xy = (keypoints[i, :, :2].detach().cpu().numpy() + 1) / 2 * image_size
        
        elif keypoints[i].max() < 1.0:
            xy = keypoints[i, :, :2] * image_size

        else:
            xy = keypoints[i, :, :2]

        xy = xy[None, :, None].astype(np.int32)

        stickman = np.ones((image_size, image_size, 3), np.uint8) if images is None else tensor2image(images[i])

        for edges, closed, color in zip(edges_parts, closed_parts, colors_parts):
            stickman = cv2.polylines(stickman, xy[:, edges], closed, color, thickness=2)

        stickman = torch.FloatTensor(stickman.transpose(2, 0, 1)) / 255.
        stickmen.append(stickman)

    stickmen = torch.stack(stickmen)
    stickmen = (stickmen - 0.5) * 2. 

    return stickmen


def draw_stickman(keypoints, image_size, images=None):
    ### Define drawing options ###
    edges_parts  = [
        list(range( 0, 17)), # face
        list(range(17, 22)), list(range(22, 27)), # eyebrows (right left)
        list(range(27, 31)) + [30, 33], list(range(31, 36)), # nose
        list(range(36, 42))+[68], list(range(42, 48))+[69], # right eye, left eye
        list(range(48, 60)), list(range(60, 68))] # lips

    closed_parts = [
        False, False, False, False, False, True, True,  True, True]

    colors_parts = [
        (  255,  255,  255), 
        (  255,    0,    0), (    0,  255,    0),
        (    0,    0,  255), (    0,    0,  255), 
        (  255,    0,  255), (    0,  255,  255),
#        (   255,  255,  255),
        (  255,  255,    0), (  255,  255,    0)]

    ### Start drawing ###
    stickmen = []

    for i  in range(keypoints.shape[0]):
        if keypoints[i] is None:
            stickmen.append(torch.zeros(3, image_size, image_size))
            continue

        if isinstance(keypoints[i], torch.Tensor):
            xy = (keypoints[i, :, :2].detach().cpu().numpy() + 1) / 2 * image_size
        
        elif keypoints[i].max() < 1.0:
            xy = keypoints[i, :, :2] * image_size

        else:
            xy = keypoints[i, :, :2]

        xy = xy[None, :, None].astype(np.int32)

        stickman = np.ones((image_size, image_size, 3), np.uint8) if images is None else tensor2image(images[i])

        for edges, closed, color in zip(edges_parts, closed_parts, colors_parts):
            stickman = cv2.polylines(stickman, xy[:, edges], closed, color, thickness=2)

        stickman = torch.FloatTensor(stickman.transpose(2, 0, 1)) / 255.
        stickmen.append(stickman)

    stickmen = torch.stack(stickmen)
    stickmen = (stickmen - 0.5) * 2. 

    return stickmen

def draw_keypoints(img, kp):
    to_image = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    
    h, w = img.shape[-2:]
    kp = (kp + 1) / 2
    kp[..., 0] *= w
    kp[..., 1] *= h
    kp = kp.detach().cpu().numpy().astype(int)

    img_out = []
    for i in range(kp.shape[0]):
        img_i = np.asarray(to_image(img[i].cpu())).copy()
        for j in range(kp.shape[1]):
            cv2.circle(img_i, tuple(kp[i, j]), radius=2, color=(255, 0, 0), thickness=-1)
        img_out.append(to_tensor(img_i))
    img_out = torch.stack(img_out)
    
    return img_out


def vis_parsing_maps(parsing_annotations, im, with_image=False, stride=1):
    # Colors for all 20 parts
    part_colors = [[255, 140, 255], [0, 30, 255],
                   [255, 0, 85],[0, 255, 255],  [255, 0, 170],
                   [170, 255, 0],
                   [0, 255, 85],   [255, 0, 0],
                   [0, 255, 0], [85, 255, 0], [0, 255, 170],
                   [255, 85, 0], [0, 255, 255],
                  [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [255, 170, 0],[0, 85, 255],
                   [255, 255, 0], [255, 255, 170],
                   [255, 85, 255],[255, 255, 35],
                   [255, 0, 255], [80, 215, 255], [140, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_annotations.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.ones((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    if with_image:
        vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    else:
        vis_im = vis_parsing_anno_color
    return vis_im