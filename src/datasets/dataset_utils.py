# Copyright 2024 TOYOTA MOTOR CORPORATION

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import torch
#from pytorch3d.ops.points_normals import estimate_pointcloud_normals
import cv2

def get_2d_coord_np(width, height, low=0, high=1, fmt="CHW"):
    """
    Args:
        width:
        height:
    Returns:
        xy: (2, height, width)
    """
    # coords values are in [low, high]  [0,1] or [-1,1]
    x = np.linspace(0, width-1, width, dtype=np.float32)
    y = np.linspace(0, height-1, height, dtype=np.float32)
    xy = np.asarray(np.meshgrid(x, y))
    if fmt == "HWC":
        xy = xy.transpose(1, 2, 0)
    elif fmt == "CHW":
        pass
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return xy

def sample_model_pcloud(pcloud_dict, samplenum, model_vis, sample_vis=True):
    pcloud, normal, feature, = pcloud_dict['vertices'], pcloud_dict['normals'], pcloud_dict['vfeatures'],
    vis, mask = model_vis, pcloud_dict['vmasks']
    valid_indices = np.where(mask>0)[0]
    l_all = len(valid_indices)
    if l_all <= 1.0:
        return None, None
    if l_all >= samplenum:
        replace_rnd = False
    else:
        replace_rnd = True
    choose = np.random.choice(valid_indices, samplenum, replace=replace_rnd)
    p_select = pcloud[choose, :]
    n_select = normal[choose, :]
    f_select = feature[choose, :]

    if sample_vis:
        vis_select = vis[choose]
    else:
        vis_select = np.ones(feature.shape[0])
    pcloud_dict['p_select'] = p_select
    pcloud_dict['n_select'] = n_select
    pcloud_dict['f_select'] = f_select
    pcloud_dict['vis_select'] = vis_select


def sample_model_pcloud_legacy(pcloud, normal, feature,  vis,  mask, samplenum, sample_method='basic',sample_vis=True):
    if sample_method == 'basic':
        l_all = pcloud.shape[0]
        if l_all <= 1.0:
            return None, None
        if l_all >= samplenum:
            replace_rnd = False
        else:
            replace_rnd = True
        choose = np.random.choice(l_all, samplenum, replace=replace_rnd)  # can selected more than one times
        p_select = pcloud[choose, :]
        f_select = feature[choose, :]
        vis_select = vis[choose]
    elif sample_method == 'mask':
        valid_indices = np.where(mask>0)[0]
        l_all = len(valid_indices)
        if l_all <= 1.0:
            return None, None
        if l_all >= samplenum:
            replace_rnd = False
        else:
            replace_rnd = True
        choose = np.random.choice(valid_indices, samplenum, replace=replace_rnd)
        p_select = pcloud[choose, :]
        #normals = estimate_normals(pcloud)
        n_select = normal[choose, :]
        f_select = feature[choose, :]
        if sample_vis:
            vis_select = vis[choose]
        else:
            vis_select = np.ones(feature.shape[0])
    else:
        p_select = None
        raise NotImplementedError
    return p_select, n_select, f_select, vis_select

def estimate_normals( points):
    # input: np.array
    points = torch.tensor(points).float()[None,]
    normals = estimate_pointcloud_normals(points)
    return normals.detach().numpy()[0]

def regularize_normals(points, normals, positive=True):
    r"""Regularize the normals towards the positive/negative direction to the origin point.

    positive: the origin point is on positive direction of the normals.
    negative: the origin point is on negative direction of the normals.
    """
    dot_products = -(points * normals).sum(axis=1, keepdims=True)
    direction = dot_products > 0
    if positive:
        normals = normals * direction - normals * (1 - direction)
    else:
        normals = normals * (1 - direction) - normals * direction
    return normals

def sample_input_PC(Depth, camK, coor2d, random_points=500):
    '''
    :param Depth: bs x 1 x h x w
    :param camK:
    :param coor2d:
    :return:
    '''

    obj_mask = torch.zeros(Depth.shape).to(Depth.device)
    obj_mask[Depth>0.01] = 1
    bs, H, W = Depth.shape[0], Depth.shape[2], Depth.shape[3]
    assert(bs==1)
    x_label = coor2d[:, 0, :, :]
    y_label = coor2d[:, 1, :, :]

    rand_num = random_points
    samplenum = rand_num
    PC = torch.zeros([bs, samplenum, 3], dtype=torch.float32, device=Depth.device)
    for i in range(bs):
        dp_now = Depth[i, ...].squeeze()   # 256 x 256
        x_now = x_label[i, ...]   # 256 x 256
        y_now = y_label[i, ...]
        obj_mask_now = obj_mask[i, ...].squeeze()  # 256 x 256
        dp_mask = (dp_now > 0.0)
        fuse_mask = obj_mask_now.float() * dp_mask.float()
        camK_now = camK #[i, ...]

        # analyze camK
        fx = camK_now[0, 0]
        fy = camK_now[1, 1]
        ux = camK_now[0, 2]
        uy = camK_now[1, 2]
        x_now = (x_now - ux) * dp_now / fx
        y_now = (y_now - uy) * dp_now / fy
        p_n_now = torch.cat([x_now[fuse_mask > 0].view(-1, 1),
                             y_now[fuse_mask > 0].view(-1, 1),
                             dp_now[fuse_mask > 0].view(-1, 1)], dim=1)

        # basic sampling
        l_all = p_n_now.shape[0]
        #print('points to sample ', p_n_now.shape[0])
        if l_all <= 50.0: #1.0
            return None,  None
        if l_all >= samplenum:
            replace_rnd = False
        else:
            replace_rnd = True
        choose = np.random.choice(l_all, samplenum, replace=replace_rnd)  # can selected more than one times
        p_select = p_n_now[choose, :]
        #n_select = estimate_normals(p_select)
        #n_select = regularize_normals(p_select.detach().numpy(),n_select)

        # reprojection
        if p_select.shape[0] > samplenum:
            p_select = p_select[p_select.shape[0]-samplenum:p_select.shape[0], :]
        PC[i, ...] = p_select[:, :3]
        #assert len(f_select.shape)==2
        fuse_mask_idx = torch.where(fuse_mask.reshape(-1))[0][choose]

    return PC / 1000.0,  fuse_mask_idx


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=False):
    """
    adapted from CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    center: ndarray: (cx, cy)
    scale: (w, h)
    rot: angle in deg
    output_size: int or (w, h)
    """
    if isinstance(center, (tuple, list)):
        center = np.array(center, dtype=np.float32)

    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale], dtype=np.float32)

    if isinstance(output_size, (int, float)):
        output_size = (output_size, output_size)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def crop_resize_by_warp_affine(img, center, scale, output_size, rot=0, interpolation=cv2.INTER_LINEAR):
    """
    output_size: int or (w, h)
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation)

    return dst_img

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_model_objects_shapenet(cat):
    name_path = os.path.join('configs/shapenet_names_textured','{}.txt'.format(cat))
    shapenames = open(name_path).read().splitlines()
    shapenames = [shapename.split('/')[-1] for shapename in shapenames]
    return shapenames

def get_sym_matched_indices(cat):
    if cat=='mug':
        sym_indices = [10, 11, 18, 19, 20, 21, 28, 29, 30, 31, 38, 39]
        sym_matched_indices = [10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30]
    elif cat=='bottle' or cat=='can' or cat=='bowl':
        sym_indices = list(range(40))
        sym_matched_indices = [0]*10+[10]*10+[20]*10+[30]*10
    elif cat=='laptop' or cat=='camera' or cat=='chair' or cat=='table':
        sym_indices = []
        sym_matched_indices = []
    assert(len(sym_indices)==len(sym_matched_indices))
    return sym_indices, sym_matched_indices
