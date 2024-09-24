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

import torch

def compute_focal_loss(conf, conf_gt, weight=None):
    """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
    Args:
        conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
        conf_gt (torch.Tensor): (N, HW0, HW1)
        weight (torch.Tensor): (N, HW0, HW1)
    """
    conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
    alpha = 0.5
    gamma = 2.0
    loss_pos = (
            -alpha
            * torch.pow(1 - conf[conf_gt == 1], gamma)
            * (conf[conf_gt == 1]).log()
    )
    loss_neg = (
            -(1 - alpha)
            * torch.pow(conf[conf_gt == 0], gamma)
            * (1 - conf[conf_gt == 0]).log()
    )
    if weight is not None:
        loss_pos = loss_pos * weight[conf_gt == 1]
        loss_neg = loss_neg * weight[conf_gt == 0]
    if loss_pos.shape[0] == 0:
        loss_mean = loss_neg.mean()
    elif loss_neg.shape[0] == 0:
        loss_mean = loss_pos.mean()
    else:
        loss_pos_mean = loss_pos.mean()
        loss_neg_mean = loss_neg.mean()
        loss_mean =  loss_pos_mean + loss_neg_mean
    return loss_mean

def compute_mean_loss(conf, conf_gt):
    """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
    Args:
        conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
        conf_gt (torch.Tensor): (N, HW0, HW1)
        weight (torch.Tensor): (N, HW0, HW1)
    """
    assert(len(conf_gt.shape)==1)
    conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
    loss = -(conf[conf_gt == 1]).log().sum() - (1 - conf[conf_gt == 0]).log().sum()
    return loss/conf_gt.shape[0]

def do_conf_loss_model(conf_matrix_gt,ind0,scores0):
    gt_score = conf_matrix_gt.max(2)[0][0][ind0[0]]
    #cls_loss = compute_focal_loss(scores0[0],gt_score)
    cls_loss = compute_mean_loss(scores0[0], gt_score)
    return cls_loss

def do_conf_loss_input(conf_matrix_gt,ind1,scores1):
    gt_score = conf_matrix_gt.max(1)[0][0][ind1[0]]
    #cls_loss = compute_focal_loss(scores1[0],gt_score)
    cls_loss = compute_mean_loss(scores1[0], gt_score)
    return cls_loss