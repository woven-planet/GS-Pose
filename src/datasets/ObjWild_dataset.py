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
from scipy.spatial.transform import Rotation as R
import os.path as osp
import _pickle as cPickle

class ObjWildDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        cat,
        module_sample_num=3000,
        input_sample_num=1000,
        split="train",
    ):
        super(Dataset, self).__init__()
        self.dataset_root = Path(dataset_root)
        self.select_class = cat
        file_path = 'test_list_{}.txt'.format(cat)
        self.img_list = [line.rstrip('\n').replace('data/UCSD_POSE_RGBD/test_set',str(self.dataset_root)).replace('rgbd','images') for
                    line in open(os.path.join(self.dataset_root, file_path))]
        self.cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        self.img_height, self.img_width = 640, 480
        self.crop_img_size = 480
        self.coord_2d = get_2d_coord_np(self.img_width, self.img_height)
        self.model_sample_num = module_sample_num
        self.input_sample_num = input_sample_num

        self.patch_h, self.patch_w = (60,60)
        self.sample_vis = False
        self.colorjitter = None
        self.transform = T.Compose([
            T.Resize((self.patch_h * 14, self.patch_w * 14)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])


        if cat=='mug':
            template_model = '/drive/data/mug/7a8ea24474846c5c2f23d8349a133d2b/dino_3d.pkl'
        elif cat=='bottle':
            template_model = '/drive/data/bottle/6ebe74793197919e93f2361527e0abe5/dino_3d.pkl'
        elif cat=='can':
            template_model = '/drive/data/can/5bd768cde93ec1acabe235874aea9b9b/dino_3d.pkl'
        elif cat=='bowl':
            template_model = '/drive/data/bowl/8b90aa9f4418c75452dd9cc5bac31c96/dino_3d.pkl'
        elif cat=='camera':
            template_model = '/drive/data/camera/63c10cfd6f0ce09a241d076ab53023c1/dino_3d.pkl'
        elif cat=='laptop':
            template_model = '/drive/data_vanilla/laptop/laptop_redfox/dino_3d.pkl'

        with open(template_model,'rb') as handle:
            model_meta = pickle.load(handle)
        self.obj_meta = model_meta
        self.img_list = self.img_list[::20]


    def __getitem__(self, index):
        img_path = self.img_list[index]
        mask_path = img_path.replace('.jpg', '-mask-hq.png')
        frame_idx = int(img_path.split('/')[-1].split('.jpg')[0])
        gt_name = self.select_class + '-' + img_path.split('/')[-4] + '-' + img_path.split('/')[-3] + '.pkl'
        gt_path = osp.join(self.dataset_root,'pkl_annotations',self.select_class, gt_name)
        gts = cPickle.load(open(gt_path, 'rb'))

        # deal with misssing data
        if (not osp.exists(mask_path)) or (not osp.exists(gt_path)) or (frame_idx >= len(gts['annotations'])):
            index=0
            img_path = self.img_list[index]
            mask_path = img_path.replace('.jpg', '-mask.png')
            frame_idx = int(img_path.split('/')[-1].split('.jpg')[0])
            gt_name = self.select_class + '-' + img_path.split('/')[-4] + '-' + img_path.split('/')[-3] + '.pkl'
            gt_path = osp.join(self.dataset_root, 'pkl_annotations', self.select_class, gt_name)
            gts = cPickle.load(open(gt_path, 'rb'))
        gts = gts['annotations'][frame_idx]

        # read images
        depth_path = img_path.replace('.jpg', '-depth.png')
        img = Image.open(img_path)
        mask = cv2.imread(mask_path)[:, :, 2]
        mask = mask / 255.
        depth = Image.open(depth_path)
        meta = json.load(open(osp.join(self.dataset_root, self.select_class, img_path.split('/')[-4], img_path.split('/')[-3], 'metadata')))
        cam = np.array(meta['K']).reshape(3, 3).T

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


        # get input
        depth = torch.tensor(np.array(depth))
        depth[mask < 0.5] = 0
        depth = depth[None, None]
        coord_2d = torch.tensor(coord_2d[None]).to(depth.device)
        cam = torch.tensor(cam).to(depth.device)
        ip_select, in_select, fuse_mask = sample_input_PC(depth, cam, coord_2d, self.input_sample_num)
        if ip_select!=None:
            ip_select = ip_select.cpu().detach().numpy().reshape(-1, 3)
        img_masked = img.transpose(1, 2, 0)
        img_masked[mask<0.5] = 255
        img = Image.fromarray( np.array(img_masked,np.uint8) )
        #img.show()
        color_input = self.transform(img)[:3]

        # get model input
        model_vertices, model_normals, model_features, model_masks = self.obj_meta['vertices'], self.obj_meta['normals'], self.obj_meta['vfeatures'], self.obj_meta['vmasks']
        model_vertices = np.array(model_vertices).reshape((-1,3))
        model_vis = None
        mp_select, mn_select, mf_select, mvis_select = sample_model_pcloud(model_vertices, model_normals, model_features,
                                                                           model_vis, model_masks,
                                                                           self.model_sample_num, 'mask', sample_vis=self.sample_vis)

        gt_pose = np.identity(4)
        norm_gt_scales = np.cbrt(np.linalg.det(gts['rotation']))
        gt_pose[:3,:3] = gts['rotation'] / norm_gt_scales
        gt_pose[:3,3] = gts['translation']

        if self.select_class in ['laptop','mug']:
            rot_cali = R.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix()
            gt_pose[:3, :3] = gt_pose[:3, :3] @ rot_cali

        gt_scale = gts['size']
        nocs_scale = np.linalg.norm(gt_scale)
        vertices_trans = nocs_scale * mp_select @ (gt_pose[:3, :3].T) + gt_pose[:3, 3]
        frame_name = gts['name'].replace('/', '_')

        data = {}
        data.update(
            {
                "model_keypoints3d": np.array(mp_select, np.float32),  # []
                "model_features": np.array(mf_select, np.float32).transpose(),  # []
                "model_normals": np.array(mn_select, np.float32),
                "input_keypoints3d": np.array(ip_select, np.float32),  # []
                "input_normals": np.array(in_select, np.float32),
                'input_color': color_input,
                'input_fuse_mask': np.array(fuse_mask, np.float32),
                'vertices_trans': np.array(vertices_trans, np.float32),
                'rgb_path': img_path,
                'cam': cam,
                'gt_pose': np.array(gt_pose, np.float32),
                'gt_scale': np.array(gt_scale, np.float32),
                'pkl_path': frame_name,
                'pred_bboxes': np.array([y1, x1, y2, x2], np.float32),
                'pred_scores': np.array([1], np.float32),
                'gt_class_ids': gts['class_id']
            }
        )
        return data

    def __len__(self):
        return len(self.img_list)


if __name__=='__main__':
    dataset = ObjWildDataset(dataset_root='/drive/objwild/test_set',cat='bottle')
    output =dataset.__getitem__(1865)
