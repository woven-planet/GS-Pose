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

from loguru import logger
try:
    import ujson as json
except ImportError:
    import json
from torch.utils.data import Dataset
from pathlib import Path
import open3d as o3d
from PIL import Image
import pickle
from .dataset_utils import *
import torchvision.transforms as T

class ShapeNetDataset(Dataset):
    def __init__(
        self,
        root,
        cat,
        module_sample_num=3000,
        input_sample_num=1000,
        split="train",
        train_symmetry=True,
        train_obj_num = 1,
    ):
        super(Dataset, self).__init__()
        self.cat=cat
        self.root = Path(root) / self.cat
        self.split = split
        self.patch_h, self.patch_w = (60,60)
        if self.split=='train':
            self.sample_vis=True
            self.colorjitter = None
            self.low_res = False
            self.transform = T.Compose([
                T.Resize((self.patch_h * 14, self.patch_w * 14)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            model_names = get_model_objects_shapenet(self.cat)[:train_obj_num]
        elif self.split=='val':
            model_names =  get_model_objects_shapenet(self.cat)[:1]
            self.sample_vis = False
            self.colorjitter = None
            self.low_res = False
            self.transform = T.Compose([
                T.Resize((self.patch_h * 14, self.patch_w * 14)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        self.objmodel_dict = {}
        self.model_ind_list = []
        if train_symmetry:
            self.sym_indices, self.sym_matched_indices = get_sym_matched_indices(cat)
        else:
            self.sym_indices, self.sym_matched_indices = [],[]
        for model_name in model_names:
            with open(self.root / model_name / 'dino_3d.pkl', 'rb') as handle:
                model_meta = pickle.load(handle)
            self.objmodel_dict[model_name] = model_meta
            color_paths = ( self.root / model_name / 'color').rglob( '*.png' )
            for color_path in color_paths:
                ind = int(color_path.name[:-4])
                self.model_ind_list.append( [model_name,ind] )
        self.model_sample_num = module_sample_num
        self.input_sample_num = input_sample_num
        self.cam = np.array([
            [411,0,240],
            [0,411,240],
            [0,0,1]
        ])
        self.img_height, self.img_width = 480, 480
        self.coord_2d = get_2d_coord_np(self.img_width, self.img_height)

    def __getitem__(self, index):

        model_name, frame_idx = self.model_ind_list[index]
        if frame_idx in self.sym_indices:
            frame_sym_idx = self.sym_matched_indices[ self.sym_indices.index(frame_idx) ]
            model_vis = np.loadtxt(self.root / model_name / 'visibility' / (str(frame_sym_idx) + '.txt'))
            pose = np.loadtxt(self.root / model_name / 'poses_ba' / (str(frame_sym_idx) + '.txt'))
        else:
            model_vis = np.loadtxt( self.root / model_name / 'visibility' / (str(frame_idx)+'.txt') )
            pose = np.loadtxt( self.root / model_name / 'poses_ba' / (str(frame_idx)+'.txt') )

        # sample model points
        obj_meta = self.objmodel_dict[model_name]
        model_vertices, model_normals, model_features, model_masks = obj_meta['vertices'], obj_meta['normals'], obj_meta['vfeatures'], obj_meta['vmasks']
        model_vertices = np.array(model_vertices).reshape((-1,3))
        mp_select, mn_select, mf_select, mvis_select = sample_model_pcloud_legacy(model_vertices, model_normals, model_features,
                                                                           model_vis, model_masks,
                                                                           self.model_sample_num, 'mask', sample_vis=self.sample_vis)
        vertices_trans = mp_select @ pose[:3, :3].T + pose[:3, 3]

        # sample inputs
        depth = Image.open( self.root / model_name / 'depth' / (str(frame_idx)+'.png') )
        color_input_org = Image.open( self.root / model_name / 'color' / (str(frame_idx)+'.png') )

        if self.low_res ==True:
            # resize to low res
            color = color_input_org.resize((120,120),Image.NEAREST)
            depth = depth.resize((120,120),Image.NEAREST)
            color_input_org = color.resize((480,480),Image.BILINEAR)
            depth = depth.resize((480,480),Image.NEAREST)

        depth = torch.tensor(np.array(depth))[None, None]
        if self.colorjitter:
            color_input = self.colorjitter(color_input_org)
            color_input = np.array(color_input, np.uint8)#.transpose((1,2,0))
            color_input_org = np.array(color_input_org, np.uint8)
            color_input[color_input_org==255]=255
            color_input = Image.fromarray(color_input)
            color_input = self.transform(color_input)[:3]
        else:
            color_input = self.transform(color_input_org)[:3]

        cam =  torch.tensor(self.cam).to(depth.device)
        coord_2d = torch.tensor(self.coord_2d[None]).to(depth.device)
        ip_select,  fuse_mask = sample_input_PC(depth, cam, coord_2d, self.input_sample_num)
        ip_select = ip_select.cpu().detach().numpy().reshape(-1,3)

        if frame_idx in self.sym_indices and self.split=='train':
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(vertices_trans[np.where(mvis_select>0)])
            color = np.zeros(vertices_trans[np.where(mvis_select>0)].shape)
            color[:,2]=1
            pcd1.colors = o3d.utility.Vector3dVector(color)
            #o3d.visualization.draw_geometries([pcd])

            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(ip_select)
            color = np.zeros(ip_select.shape)
            color[:,0]=1
            pcd2.colors = o3d.utility.Vector3dVector(color)
            threshold = 0.02
            trans_init = np.eye(4)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd1, pcd2, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())

            pcd1.points = o3d.utility.Vector3dVector(vertices_trans)
            color = np.zeros(vertices_trans.shape)
            color[:,2]=1
            pcd1.colors = o3d.utility.Vector3dVector(color)
            pcd1 = pcd1.transform(reg_p2p.transformation)
            vertices_trans = np.array(pcd1.points)
            #o3d.visualization.draw_geometries([pcd1,pcd2])

        data = {}
        data.update(
            {
                "model_keypoints3d": np.array(mp_select, np.float32),  # []
                "model_features":  np.array(mf_select, np.float32).transpose(),  # []
                "model_normals": np.array(mn_select, np.float32),
                "input_keypoints3d": np.array(ip_select,  np.float32),  # []
                #"input_normals": np.array(in_select, np.float32),
                'input_color': color_input,
                'input_fuse_mask': np.array(fuse_mask, np.float32),
                'vertices_trans': np.array(vertices_trans, np.float32),
                'gt_pose': np.array(pose, np.float32),
            }
        )
        return data

    def __len__(self):
        return len(self.model_ind_list)


if __name__=='__main__':
    pass


