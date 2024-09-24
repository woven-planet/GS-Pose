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

import trimesh
import os
import hydra
from pathlib import Path
import shutil
import pymeshlab
import mesh2sdf
import open3d as o3d

def preprocess(shapenet_root, target_root, category, resample=True):

    name_path = hydra.utils.to_absolute_path('configs/shapenet_names_textured/{}.txt'.format(category))
    shapenames = open(name_path).read().splitlines()
    os.makedirs(target_root/category, exist_ok=True)

    for shapename in shapenames[:200]:
        shapenet_path = shapenet_root / shapename
        target_path = target_root / category / shapename.split('/')[1]
        if not os.path.exists(target_path):
            shutil.copytree(shapenet_path, target_path)

        os.chdir(target_path/'models')
        obj_file_path = target_path / 'models' / 'model_normalized.obj'
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(obj_file_path))
        ms.apply_filter('transform_rotate', rotaxis=1, rotcenter=0, angle=-90) # adjust this value to nocs objects coordinates for each category
        ms.transform_translate_center_set_origin(traslmethod='Center on Scene BBox')
        ms.apply_filter('transform_scale_normalize', unitflag=True)
        ms.save_current_mesh(str(target_path / 'models' / 'model_normalized.obj'))

        if resample:
            mesh = trimesh.load(obj_file_path, force='mesh')
            size = 128 #256
            gt_pcd = mesh.as_open3d
            vertices = mesh.vertices
            bbmax = vertices.max(0)
            bbmin = vertices.min(0)
            center = (bbmin + bbmax) * 0.5
            scalars = (bbmax - bbmin).max()
            vertices = (vertices-center) / (scalars*1.1)
            level = 2.0 / size
            sdf, mesh_repair = mesh2sdf.compute(vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
            mesh_repair.vertices = mesh_repair.vertices*(scalars*1.1*0.95)+center
            re_pcd = mesh_repair.as_open3d
            #o3d.visualization.draw_geometries([gt_pcd, re_pcd])
            mesh_repair.export( str(obj_file_path)[:-4]+'_resample.obj' )


if __name__=='__main__':
    shapenet_root = Path('/drive/ShapeNetCore.v2')
    target_root = Path('data/shapenet')
    category = 'bottle'
    preprocess(shapenet_root,target_root,category)
