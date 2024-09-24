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

import  open3d as o3d
import trimesh
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
from sklearn.decomposition import PCA
from psbody.mesh.visibility import visibility_compute
from psbody.mesh import Mesh
import os
import cv2
from sklearn.cluster import KMeans
from pathlib import Path
from render_utils import *
import pyrender
from sklearn.decomposition import PCA
import pickle

def shapenet_render(obj_root, use_resample = True):
    # hyperparameters
    cam = np.eye(3)
    fx,fy,cx,cy=411,411,240,240
    cam[0,0], cam[1,1], cam[0,2], cam[1,2] = fx, fy, cx, cy

    obj_paths = sorted(set(obj_root.rglob('*/model_normalized.obj')))
    dino_model = 'vits14'
    FeatExtractor = DinoExtractor(dino_model='vits14')
    dim_dict = {'vits14':384,'vitb14':768,'vitl14':1024,'vitg14':1536}
    feature_dim = dim_dict[dino_model]
    use_feature_cache = False

    for obj_path in obj_paths:
        print('processing ',obj_path)
        #if obj_path.parent.parent.name !='6d036fd1c70e5a5849493d905c02fa86':
        #if obj_path.parent.parent.name != '1967344f80da29618d342172201b8d8c':
        #    continue
        trimesh_obj_render = trimesh.load(obj_path, force='mesh')
        if use_resample:
            trimesh_obj = trimesh.load(str(obj_path)[:-4] + '_resample.obj', force='mesh')
        else:
            trimesh_obj = trimesh.load(obj_path, force='mesh')

        vertices, _ = normalize_verts(trimesh_obj_render.vertices)
        trimesh_obj_render.vertices = vertices
        vertices, scale = normalize_verts( trimesh_obj.vertices )
        faces = trimesh_obj.faces
        trimesh_obj.vertices = vertices

        normals = estimate_normals(vertices)
        ren_poses = sample_view_points_ysym(4,10,visualize=False,r=1)
        scene = pyrender.Scene(bg_color=[255, 255, 255],ambient_light=[500, 500, 500])#, ambient_light=[200, 200, 200])
        r = pyrender.OffscreenRenderer(480, 480)
        mesh = pyrender.Mesh.from_trimesh(trimesh_obj_render)
        for primitive in mesh.primitives:
            primitive.material.alphaMode='OPAQUE'
        scene_obj_handle = scene.add(mesh)
        factor = np.array([[1, 1, 1, 1],
                           [-1, -1, -1, -1],
                           [-1, -1, -1, -1],
                           [1, 1, 1, 1]])
        camera_ren = pyrender.IntrinsicsCamera(0, 0, 0, 0)
        camera_ren.fx = fx
        camera_ren.fy = fy
        camera_ren.cx = cx
        camera_ren.cy = cy
        camera_node = scene.add(camera_ren)

        vertices_features = np.zeros((vertices.shape[0], feature_dim))
        vertices_mask = np.zeros(vertices.shape[0])

        render_save_root = obj_path.parent.parent
        rgb_path = os.path.join(render_save_root, 'color')
        depth_path = os.path.join(render_save_root, 'depth')
        intri_path = os.path.join(render_save_root, 'intrin_ba')
        pose_path = os.path.join(render_save_root, 'poses_ba')
        feature_path = os.path.join(render_save_root, 'feature')
        visibility_path = os.path.join(render_save_root, 'visibility')
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(depth_path, exist_ok=True)
        os.makedirs(intri_path, exist_ok=True)
        os.makedirs(pose_path, exist_ok=True)
        os.makedirs(feature_path, exist_ok=True)
        os.makedirs(visibility_path,exist_ok=True)

        #trimesh_obj.visual = trimesh.visual.ColorVisuals()
        #trimesh_obj.export(render_save_root/obj_path.name)
        #np.savez(render_save_root/'mesh.npz', vertices=trimesh_obj.vertices, faces=trimesh_obj.faces)
        for idx in range(len(ren_poses)):
            pose_obj = np.linalg.inv(ren_poses[idx])
            scene.set_pose(scene_obj_handle, pose_obj * factor)
            color, depth = r.render(scene)
            img = Image.fromarray(np.array(color, dtype=np.uint8))
            depth = Image.fromarray(np.array(depth * 1000, np.uint16))
            depth.save(os.path.join(depth_path, str(idx) + '.png'))
            img.save(os.path.join(rgb_path, str(idx) + '.png'))
            np.savetxt(os.path.join(intri_path, str(idx) + '.txt'), cam)
            np.savetxt(os.path.join(pose_path, str(idx) + '.txt'), pose_obj)
            if use_feature_cache:
                features = torch.load( os.path.join(feature_path, str(idx) + '.pt') )
            else:
                features = FeatExtractor.build_feature_extractor(img)
                torch.save(features, os.path.join(feature_path, str(idx) + '.pt'))
            features = torch.tensor(features.reshape((60, 60, -1)))

            # check visibility and save
            vertices_trans = trimesh_obj.vertices @ pose_obj[:3, :3].T + pose_obj[:3, 3]
            m = Mesh(v=vertices_trans, f=trimesh_obj.faces)
            (vis, n_dot) = visibility_compute(v=m.v, f=m.f, cams=np.array([[0.0, 0.0, 0.0]]))
            np.savetxt(os.path.join(visibility_path, str(idx) + '.txt'), vis)
            vis_vis = False
            if vis_vis:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(vertices_trans)
                color = np.zeros(vertices_trans.shape)
                color[vis[0] == 1, 0] = 1 #r
                color[vis[0] == 0, 2] = 1 #b
                pcd.colors = o3d.utility.Vector3dVector(color)
                mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                o3d.visualization.draw_geometries([mesh_frame,pcd])

            # fuse the features for the point cloud
            feature = nn.Upsample(size=(480, 480), mode='bilinear')(features.permute(2, 0, 1).unsqueeze(0))[0].permute(1, 2,0)
            c2d = (cam @ vertices_trans.transpose()).transpose()
            coords_2d = np.stack((c2d[:, 0] / c2d[:, 2], c2d[:, 1] / c2d[:, 2]), axis=-1)
            coords_2d = np.array(coords_2d, np.int32)
            feature_2d = feature[coords_2d[:, 1], coords_2d[:, 0]].detach().numpy()
            verts_mask = vis[0]
            vertices_mask += verts_mask
            vertices_features[verts_mask == 1] += feature_2d[verts_mask == 1]

        for verts_idx in range(vertices.shape[0]):
            if vertices_mask[verts_idx] > 0:
                vertices_features[verts_idx] /= vertices_mask[verts_idx]

        save_data = {'vertices':trimesh_obj.vertices,
                     'normals': normals,
                     'faces':trimesh_obj.faces,
                     'vfeatures':vertices_features,
                     'vmasks':vertices_mask }
        with open(render_save_root/'dino_3d.pkl', 'wb') as handle:
            pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #vfeatures_selected = vertices_features[vertices_mask > 0]
        #vertices_selected = vertices[vertices_mask > 0]
        #vis_pcloud_feature(vertices_selected,vfeatures_selected)

if __name__=='__main__':
    obj_root = Path('data/shapenet/bottle')
    shapenet_render(obj_root)