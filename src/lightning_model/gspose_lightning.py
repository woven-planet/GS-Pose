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
import open3d as o3d
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from src.models.gspose_model import GsPoseModel
from src.models.optimizers.optimizers import (
    build_optimizer,
    build_scheduler,
)
import time

from .nndistance.modules.nnd import NNDModule
from src.utils.umeyama_scale import solve_umeyama_ransac_scale

class PL_GsPose(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        if "GsPose" in self.hparams:
            self.matcher = GsPoseModel(self.hparams["GsPose"])
            self.thr = self.hparams["GsPose"]['matching']['thr']
        else:
            self.matcher = GsPoseModel(self.hparams["FC6D"])
            self.thr = self.hparams["FC6D"]['matching']['thr']
            '''
            test_info = {
                'test_dataset': 'data/nocs',
                'result_path': 'data/shapenet',
                'test_cat': ${train_cat},
            }
            '''
        '''
        self.n_vals_plot = max(
            self.hparams["trainer"]["n_val_pairs_to_plot"]
            // self.hparams["trainer"]["world_size"],
            1,
        )
        '''
        self.chamfer_dist = NNDModule()
        dinov2_vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.dino_model = dinov2_vit.cuda().eval()
        self.img_height, self.img_width = 480, 480
        self.test_dataset = 'nocs'
        assert(self.test_dataset in ['nocs','wild6d'])


    def training_step(self, batch, batch_idx):

        self.sample_input_feature(batch)
        self.get_ground_truth(batch)
        self.matcher(batch)

        # Update tensorboard on rank0 every n steps
        if (
            self.trainer.global_rank == 0
            and self.global_step % self.trainer.log_every_n_steps == 0
        ):
            for k, v in batch["loss_scalars"].items():
                self.logger.experiment[0].add_scalar(f"train/{k}", v, self.global_step)
            self.logger.experiment[0].add_scalar(
                f"train/max conf_matrix", batch["conf_matrix"].max(), self.global_step
            )
        return {"loss": batch["loss"]}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment[0].add_scalar(
                "train/avg_loss_on_epoch", avg_loss, global_step=self.current_epoch
            )

    def validation_step(self, *args, **kwargs):
        pass

    def validation_epoch_end(self, outputs):
        return 1


    def test_step(self, batch, batch_idx):

        if 'input_keypoints3d' not in batch or len(list(batch['input_keypoints3d'][0].shape))==0:
            return  #dont consider bad ones
        self.sample_input_feature(batch)
        self.get_ground_truth(batch)

        pred = self.matcher(batch)
        if pred == None:
            return
        model_kpts3d =  batch['model_keypoints3d'][0]
        input_kpts3d = batch['input_keypoints3d'][0]
        gt_kpts3d_translated = (batch['vertices_trans'][0]+torch.tensor([[0,0,0.2]]).cuda())
        idx1, idx2 = pred['matches'][0][:,0], pred['matches'][0][:,1]
        model_selected = model_kpts3d[idx1].cpu().detach().numpy()
        input_selected = input_kpts3d[idx2].cpu().detach().numpy()

        if len(list(idx1)) < 10:
            print("ignore!")
            return  #dont consider bad ones

        ume_result = solve_umeyama_ransac_scale(model_selected,input_selected, model_kpts3d.cpu().detach().numpy())
        if ume_result==None:
            return
        Rotation, Translation, Scales =  ume_result
        pred_scales = torch.max(model_kpts3d, 0)[0].cpu() * torch.tensor(Scales)
        estimated_teaser_pose = np.eye(4)
        estimated_teaser_pose[:3,:3] = Rotation
        estimated_teaser_pose[:3,3] = Translation

        res = {}
        res.update({
            'pred_RTs': estimated_teaser_pose,
            'pred_scales': pred_scales.cpu(),
            'pkl_path': batch['pkl_path'][0],
            'pred_bboxes': batch['pred_bboxes'][0].cpu(),
            'pred_scores': batch['pred_scores'][0].cpu(),
        })

        return res

    def test_epoch_end(self, outputs):
        import pickle
        pkl_paths = [output['pkl_path'] for output in outputs ]
        unique_pkl_paths = list(set(pkl_paths))
        # test_dataset_path, result_path, test_cat
        result_dir = os.path.join(self.result_path,self.test_cat)
        os.makedirs(result_dir,exist_ok=True)
        for unique_pkl_path in unique_pkl_paths:
            indices = [i for i in range(len(pkl_paths)) if pkl_paths[i] == unique_pkl_path]
            res = {}
            pred_RTs = np.stack( [outputs[i]['pred_RTs'] for i in indices], 0 )
            pred_scales = torch.stack( [outputs[i]['pred_scales'] for i in indices], 0 ).cpu().detach().numpy()
            pred_bboxes =  torch.stack( [outputs[i]['pred_bboxes'] for i in indices], 0 ).cpu().detach().numpy()
            pred_scores = torch.stack([outputs[i]['pred_scores'] for i in indices], 0).cpu().detach().numpy()
            res['pred_RTs'] = pred_RTs
            res['pred_scales'] = pred_scales
            res['pred_bboxes'] = pred_bboxes
            res['pred_scores'] = pred_scores
            #with open('/drive/projects/CPPF/data/nocs_prediction/results_'+unique_pkl_path+'.pkl','wb') as f:
            with open( os.path.join(result_dir,'results_'+unique_pkl_path+'.pkl'), 'wb') as f:
                pickle.dump(res,f)
        return 0

    def get_ground_truth(self, data):
        mp_select_all = data['model_keypoints3d']
        ip_select_all = data['input_keypoints3d']
        vertices_trans_all = data['vertices_trans']
        bs, model_sample_num, _ = mp_select_all.shape
        _, input_sample_num, _ = ip_select_all.shape
        assign_matrix = torch.zeros((bs, model_sample_num, input_sample_num)).to(mp_select_all.device)

        for batch_idx in range(bs):
            ip_select = torch.tensor(ip_select_all[batch_idx]).cuda().float()
            vertices_trans = torch.tensor(vertices_trans_all[batch_idx]).cuda().float()
            dist1, _, idx1, idx2 = self.chamfer_dist(ip_select[None], vertices_trans[None])

            indices1 = torch.arange(idx1.shape[1], device=idx1.device)[None]
            indices2 = torch.arange(idx2.shape[1], device=idx2.device)[None]
            mutual1 = indices1 == idx2.gather(1, idx1.long())
            matched_input_idx = torch.arange(idx1.shape[1],device=ip_select.device)[None,][mutual1]
            matched_model_idx = idx1[mutual1]
            assign_matrix[batch_idx, matched_model_idx.long(), matched_input_idx.long()] = 1
        data['conf_matrix_gt'] = torch.tensor(assign_matrix).to(ip_select.device)


    def sample_input_feature(self, batch):
        bs = batch['input_color'].shape[0]
        with torch.no_grad():
            features_dict = self.dino_model.forward_features(batch['input_color'])
            feature = features_dict['x_norm_patchtokens'].reshape((bs,60, 60, -1))
            feature = nn.Upsample(size=(self.img_height, self.img_width), mode='bilinear')(feature.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        feature_select = []
        fuse_mask = batch['input_fuse_mask'].long()
        [feature_select.append(feature[idx].reshape((-1,384))[fuse_mask[idx]]) for idx in range(bs)]
        feature_select = torch.stack(feature_select,0).to(fuse_mask.device)
        batch['input_features'] =  feature_select.transpose(1,2)

    def configure_optimizers(self):
        optimizer = build_optimizer(self, self.hparams)
        scheduler = build_scheduler(self.hparams, optimizer)
        return [optimizer], [scheduler]

if __name__=='__main__':

    a=PL_GsPose()