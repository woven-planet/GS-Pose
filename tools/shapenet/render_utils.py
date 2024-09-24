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

import numpy as np
import os
import math
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import torch
import torchvision.transforms as T
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def normalize_verts(vertices):
    # vertices: n_pts*3
    scale = np.linalg.norm(np.max(vertices,axis=0)-np.min(vertices,axis=0))
    return vertices/scale, scale


def d2r(degree):
    return math.pi/180.0*degree


def sample_view_points(elevation_num=4,azimuth_num=10,visualize=False):

    #thetas = np.linspace(math.pi/180.0*40, math.pi/180.0*140, 10) #4) #elevation
    thetas = np.linspace(math.pi/180.0*40, math.pi/180.0*100, elevation_num) #elevation
    phis = np.linspace(0, 2 * math.pi, azimuth_num) #azimuth
    r = 0.85 #1
    poses = []
    for theta in thetas:
        for phi in phis:
            x = r * math.sin(theta) * math.cos(phi)
            y = r * math.sin(theta) * math.sin(phi)
            z = r * math.cos(theta)
            euler_angle_Z = phi + math.pi/2
            euler_angle_X = theta + math.pi
            rot = R.from_euler('xz', [ euler_angle_X,euler_angle_Z])
            pose = np.eye(4)
            pose[:3,:3] = rot.as_matrix()
            pose[:3,3] = np.array([x,y,z])
            poses.append(pose)
    if visualize:
        cam_vis_list = []
        for idx in range(len(poses)):
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            mesh_frame = mesh_frame.transform(poses[idx])
            cam_vis_list.append(mesh_frame)
            if idx == 0:
                mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
                cam_vis_list.append(mesh_frame)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        o3d.visualization.draw_geometries(cam_vis_list + [sphere])
    return poses

def sample_view_points_ysym(elevation_num=4,azimuth_num=10,visualize=False, r=0.95):

    thetas = np.linspace(math.pi/180.0*40, math.pi/180.0*100, elevation_num) #elevation
    phis = np.linspace(-math.pi, math.pi, azimuth_num)  # azimuth
    #r =  #0.95 #0.85 #1
    poses = []
    for theta in thetas:
        for phi in phis:
            x = r * math.sin(theta) * math.cos(phi) #x
            z = r * math.sin(theta) * math.sin(phi) #y
            y = r * math.cos(theta) #z
            euler_angle_Y = -phi + math.pi/2
            euler_angle_X = theta + math.pi/2
            #rot = R.from_euler('zy', [ euler_angle_X,euler_angle_Y]) #xz
            rot = R.from_euler('xy', [euler_angle_X,euler_angle_Y])  # xz
            pose = np.eye(4)
            pose[:3,:3] = rot.as_matrix()
            pose[:3,3] = np.array([x,y,z])
            poses.append(pose)
    if visualize:
        cam_vis_list = []
        for idx in range(len(poses)):
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            mesh_frame = mesh_frame.transform(poses[idx])
            cam_vis_list.append(mesh_frame)
            if idx == 0:
                mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
                cam_vis_list.append(mesh_frame)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        o3d.visualization.draw_geometries(cam_vis_list + [sphere])
    return poses

class DinoExtractor():
    def __init__(self, patch_size=(60,60), dino_model='vitl14'):
        self.patch_h, self.patch_w = patch_size
        dim_dict = {'vits14':384,'vitb14':768,'vitl14':1024,'vitg14':1536}
        self.feat_dim = dim_dict[dino_model]
        self.transform = T.Compose([
            # T.GaussianBlur(9, sigma=(0.1, 2.0)),
            T.Resize(( self.patch_h * 14, self.patch_w * 14)),
            T.CenterCrop(( self.patch_h * 14, self.patch_w * 14)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        dinov2_vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_'+dino_model)
        self.dino_model = dinov2_vit.cuda().eval()

    def build_feature_extractor(self, img):
        imgs_tensor = self.transform(img)[:3].cuda().unsqueeze(0)
        with torch.no_grad():
            features_dict = self.dino_model.forward_features(imgs_tensor)
            features = features_dict['x_norm_patchtokens']
        return features.cpu()


def vis_pcloud_feature(vertices_selected,vfeatures_selected):
    pca = PCA(n_components=3)
    pca.fit(vfeatures_selected)
    pca_features = pca.transform(vfeatures_selected)
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(pca_features)
    labels = kmeans.predict(pca_features)
    for i in range(3):
        pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].mean()) / (
                pca_features[:, i].std() ** 2) + 0.5
    cluster_means_img = [pca_features[labels == i].mean(axis=0) for i in range(n_clusters)]
    visualize_pca = True
    if visualize_pca:
        for i in range(n_clusters):
            pca_features[labels == i] = [cluster_means_img[i][:3]]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices_selected)
    pcd.colors = o3d.utility.Vector3dVector(pca_features)
    o3d.visualization.draw_geometries([pcd])

def estimate_normals(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)
    return normals