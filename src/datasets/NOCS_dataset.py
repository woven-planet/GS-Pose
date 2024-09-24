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

try:
    import ujson as json
except ImportError:
    import json
from torch.utils.data import Dataset
from pathlib import Path
import open3d as o3d
from PIL import Image
import torch.nn as nn
import pickle
import os
import cv2
from tqdm import tqdm
from .dataset_utils import *
import torchvision.transforms as T

class NOCSDataset(Dataset):
    def __init__(
        self,
        nocs_root,
        cat,
        module_sample_num=3000,
        input_sample_num=1000,
        split="train",
    ):
        super(Dataset, self).__init__()
        self.nocs_root = Path(nocs_root)
        self.split = split
        self.patch_h, self.patch_w = (60,60)
        self.synset_names = ['BG',  # 0
                        'bottle',  # 1
                        'bowl',  # 2
                        'camera',  # 3
                        'can',  # 4
                        'laptop',  # 5
                        'mug'  # 6
                        ]
        self.sample_vis = False
        self.colorjitter = None
        self.transform = T.Compose([
            T.Resize((self.patch_h * 14, self.patch_w * 14)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self.model_sample_num = module_sample_num
        self.input_sample_num = input_sample_num
        self.cam = np.array([
                        [591.0125, 0, 322.525],
                        [0, 590.16775, 244.11084],
                        [0, 0, 1]])
        self.img_height, self.img_width = 480, 640
        self.coord_2d = get_2d_coord_np(self.img_width, self.img_height)

        # load gt labels
        self.seg_dir = self.nocs_root / 'nocs_seg'
        self.nocs_dir = self.nocs_root
        result_pkl_list =  self.seg_dir.glob('results_*.pkl')
        self.result_pkl_list = sorted(result_pkl_list)
        final_results = []
        for pkl_path in tqdm(self.result_pkl_list):
            with open(pkl_path, 'rb') as f:
                result = pickle.load(f)
                if not 'gt_handle_visibility' in result:
                    result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
                    print('can\'t find gt_handle_visibility in the pkl.')
                else:
                    assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(
                        result['gt_handle_visibility'], result['gt_class_ids'])
            if type(result) is list:
                final_results += result
            elif type(result) is dict:
                final_results.append(result)
            else:
                assert False
        self.final_results = final_results
        self.eval_cat = [cat]
        self.eval_cat_id = self.synset_names.index(cat)
        self.bbox_mask = False

        # filter specific category
        self.model_ind_list = []
        for frame_idx, res in enumerate(self.final_results):
            cls_ids = res['pred_class_ids']
            bboxs = res['pred_bboxes']
            for instance_idx, bbox in enumerate(bboxs):
                cls_id = cls_ids[instance_idx]
                cls_name = self.synset_names[cls_id]
                if (cls_name not in self.eval_cat) or (cls_id not in res['gt_class_ids']) :
                    continue
                self.model_ind_list.append([frame_idx,instance_idx])
        self.crop_img_size = 480
        self.model_ind_list = self.model_ind_list#[::200] #[::200]#[960:]


        if cat=='mug':
            #template_model = '/drive/data/mug/7a8ea24474846c5c2f23d8349a133d2b/dino_3d.pkl'
            template_model = 'data/shapenet/mug/5582a89be131867846ebf4f1147c3f0f/dino_3d.pkl'
        elif cat=='bottle':
            template_model = 'data/shapenet/bottle/6ebe74793197919e93f2361527e0abe5/dino_3d.pkl'
        elif cat=='can':
            template_model = 'data/shapenet/can/5bd768cde93ec1acabe235874aea9b9b/dino_3d.pkl'
        elif cat=='bowl':
            template_model = 'data/shapenet/bowl/8b90aa9f4418c75452dd9cc5bac31c96/dino_3d.pkl'
        elif cat=='camera':
            #template_model = '/drive/data_gspose/camera/63c10cfd6f0ce09a241d076ab53023c1/dino_3d.pkl'
            template_model = 'data/shapenet/camera/97690c4db20227d248e23e2c398d8046/dino_3d.pkl'
        elif cat=='laptop':
            template_model = 'data/shapenet/laptop/7e5b970c83dd97b7823eead1c8e7b3b4/dino_3d.pkl'

        with open(template_model,'rb') as handle:
            model_meta = pickle.load(handle)
        self.obj_meta = model_meta


    def __getitem__(self, index):

        # load gt
        frame_idx, instance_idx = self.model_ind_list[index]
        res = self.final_results[frame_idx]
        img = Image.open(os.path.join(self.nocs_dir, res['image_path'][5:] + '_color.png'))
        depth = Image.open(os.path.join(self.nocs_dir, res['image_path'][5:] + '_processed.png'))
        mask = res['pred_masks'][:,:,instance_idx].copy()
        bbox = res['pred_bboxes'][instance_idx]
        pred_score = res['pred_scores'][instance_idx]
        gt_pose = res['gt_RTs'][res['gt_class_ids'].tolist().index(self.eval_cat_id)]  #6
        gt_scale = res['gt_scales'][res['gt_class_ids'].tolist().index(self.eval_cat_id)] #6

        # crop
        mask_coord_2d = np.where(mask > 0)
        t, l = np.min(mask_coord_2d, axis=1)
        b, r = np.max(mask_coord_2d, axis=1)
        bbox_xyxy = np.array([l, t, r, b])
        x1, y1, x2, y2 = bbox_xyxy
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bbox_center = np.array([cx, cy])  # (w/2, h/2)
        scale = max(y2 - y1, x2 - x1) * 1.2  # *1.4
        scale = min(scale, max(self.img_height, self.img_width))
        scale = int(scale)

        img = crop_resize_by_warp_affine(
            np.array(img), bbox_center, scale, self.crop_img_size, interpolation=cv2.INTER_LINEAR, #cv2.INTER_NEAREST
        ).transpose(2, 0, 1)
        depth = crop_resize_by_warp_affine(
            np.array(depth), bbox_center, scale, self.crop_img_size, interpolation=cv2.INTER_NEAREST
        )
        mask = np.array(mask,np.uint8)
        mask = crop_resize_by_warp_affine(
            mask, bbox_center, scale, self.crop_img_size, interpolation=cv2.INTER_LINEAR, #cv2.INTER_NEAREST,
        )
        coord_2d = crop_resize_by_warp_affine(
            self.coord_2d.transpose(1, 2, 0), bbox_center, scale, self.crop_img_size, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)

        # get model input
        model_vis = None #np.ones((model_vertices.shape[0],1))
        sample_model_pcloud(self.obj_meta, self.model_sample_num, model_vis, sample_vis=self.sample_vis)
        mp_select, mn_select, mf_select,  = self.obj_meta['p_select'], self.obj_meta['n_select'], self.obj_meta['f_select'],

        norm_gt_scales = np.cbrt(np.linalg.det(gt_pose[:3, :3]))
        gt_pose[:3, :3] = gt_pose[:3, :3] / norm_gt_scales
        gt_scale = gt_scale * norm_gt_scales
        nocs_scale = np.linalg.norm(gt_scale)
        vertices_trans = nocs_scale * mp_select @ (gt_pose[:3, :3].T) + gt_pose[:3, 3]

        # get input
        depth = torch.tensor(np.array(depth))
        depth[mask < 0.5] = 0
        depth = depth[None, None]
        coord_2d = torch.tensor(coord_2d[None]).to(depth.device)
        cam = torch.tensor(self.cam).to(depth.device)
        ip_select,  fuse_mask = sample_input_PC(depth, cam, coord_2d, self.input_sample_num)
        if ip_select!=None:
            ip_select = ip_select.cpu().detach().numpy().reshape(-1, 3)

        img_masked = img.transpose(1, 2, 0)
        img = Image.fromarray( np.array(img_masked,np.uint8) )
        color_input = self.transform(img)[:3]

        data = {}
        data.update(
            {
                "model_keypoints3d": np.array(mp_select, np.float32),
                "model_features": np.array(mf_select, np.float32).transpose(),
                "input_keypoints3d": np.array(ip_select, np.float32),
                'input_color': color_input,
                'input_fuse_mask': np.array(fuse_mask, np.float32),
                'vertices_trans': np.array(vertices_trans, np.float32),
                'gt_pose': np.array(gt_pose, np.float32),
                'gt_scale': np.array(gt_scale, np.float32),
                'pkl_path': res['image_path'].replace('/', '_')[5:],
                'pred_bboxes': np.array(bbox, np.float32),
                'pred_scores': np.array(pred_score, np.float32),
            }
        )

        return data

    def __len__(self):
        return len(self.model_ind_list)

