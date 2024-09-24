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
import open3d as o3d
from scipy import ndimage
from pathlib import Path
from PIL import Image

def backproject(
    depth: np.ndarray, intrinsics: np.ndarray, instance_mask: np.ndarray = None
):
    """Back-projection, use opencv camera coordinate frame."""

    non_zero_mask = (depth > 0) & (depth < np.inf)
    if instance_mask is None:
        final_instance_mask = non_zero_mask
    else:
        final_instance_mask = instance_mask & non_zero_mask
    idxs = np.stack(np.where(final_instance_mask))

    z = depth[idxs[0], idxs[1]]
    yx = idxs * z
    xyz = np.stack((yx[1], yx[0], z), axis=-1).astype(depth.dtype)
    pts = xyz @ np.linalg.inv(intrinsics).T

    return pts, idxs

def detect_floor_and_wall(depth: np.ndarray, K: np.ndarray, bg_mask: np.ndarray = None):
    bg_pts, bg_depth_idxs = backproject(depth, K, bg_mask)
    bg_depth_idxs1 = np.stack(bg_depth_idxs, axis=-1)

    # remove floor / wall
    bg_pcl1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bg_pts))
    _, plane_ind1 = bg_pcl1.segment_plane(
        distance_threshold=8, ransac_n=3, num_iterations=1000)

    # remove wall / floor
    bg_pcl2 = bg_pcl1.select_by_index(plane_ind1, invert=True)
    _, plane_ind2 = bg_pcl2.segment_plane(
        distance_threshold=8, ransac_n=3, num_iterations=1000)

    # plane_pcl = bg_pcl1.select_by_index(plane_ind1)
    # o3d.io.write_point_cloud('./plane1.ply', plane_pcl)
    # plane_pcl = bg_pcl2.select_by_index(plane_ind2)
    # o3d.io.write_point_cloud('./plane2.ply', plane_pcl)
    # o3d.io.write_point_cloud('./bg2.ply', bg_pcl2)

    bg_depth_idxs2 = np.delete(bg_depth_idxs1, plane_ind1, axis=0)
    plane_idxs = np.concatenate(
        (bg_depth_idxs1[plane_ind1], bg_depth_idxs2[plane_ind2]), axis=0).T
    # plane_idxs = bg_depth_idxs1[plane_ind1].T

    return plane_idxs


def remove_outliers(depth_path: Path):
    name = depth_path.stem.split('_')[0]
    out_path = depth_path.with_name(f'{name}_processed.png')

    # if out_path.exists():
    #     return

    # bottle, bowl, camera, can, laptop, mug
    std_ratio1 = np.asarray([1.3, 2.5, 1.5, 1.3, 1.3, 2.5])
    std_ratio2 = np.asarray([4.5, 5.0, 5.0, 4.5, 4.5, 5.0])
    # radius = np.asarray([0.06, 0.04, 0.045, 0.04, 0.1, 0.045]) * 1000
    # nb = np.asarray([200, 250, 200, 250, 300, 100]) * 1000

    K = np.array([[591.0125, 0, 322.525],
                  [0, 590.16775, 244.11084],
                  [0, 0, 1]],
                 np.float32)

    depth = np.asarray(Image.open(depth_path).convert('I;16'))

    mask_path = depth_path.with_name(f'{name}_mask.png')
    mask = np.asarray(Image.open(mask_path).convert('L'))

    bg_mask = (mask == 255)
    structure = ndimage.generate_binary_structure(2, 2)
    dilated_bg_mask = ndimage.binary_dilation(bg_mask, structure, iterations=8)

    # Image.fromarray(bg_mask.astype(np.uint8) * 255).save('./bg_mask.png')
    # Image.fromarray(dilated_bg_mask.astype(np.uint8) * 255).save('./dilated_bg_mask.png')
    # img_path = depth_path.with_name(f'{name}_color.png')
    # img = np.asarray(Image.open(img_path).convert('RGB')).copy()
    # img[~dilated_bg_mask] = [255, 0, 0]
    # Image.fromarray(img).save('./test.png')

    plane_idxs = detect_floor_and_wall(depth, K, dilated_bg_mask)
    valid_mask = ~bg_mask
    valid_mask[plane_idxs[0], plane_idxs[1]] = False

    meta_path = depth_path.with_name(f'{name}_meta.txt')
    meta = meta_path.read_text()
    lines = meta.splitlines()
    all_inst_ids = np.unique(mask).tolist()

    choose_idxs = []
    for line in lines:
        info = line.strip().split(' ')
        inst_id, cls_id, model_id = int(info[0]), int(info[1]), info[2]

        # background objects and non-existing objects
        if cls_id == 0 or (inst_id not in all_inst_ids):
            continue

        inst_mask = mask == inst_id
        inst_pts, depth_idxs = backproject(depth, K, inst_mask & valid_mask)

        depth_idxs = np.stack(depth_idxs, axis=-1)
        pcl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inst_pts))
        pcl, ind1 = pcl.remove_statistical_outlier(80, std_ratio1[cls_id-1])
        pcl, ind2 = pcl.remove_statistical_outlier(2000, std_ratio2[cls_id-1])

        if len(ind2) == 0:
            continue

        # pcl, ind4 = pcl.remove_radius_outlier(10, 10)
        # # pcl, _ = pcl.remove_radius_outlier(300, radius[cls_id-1])
        # pcl, _ = pcl.remove_radius_outlier(nb[cls_id - 1], 0.025)
        # pcl, _ = pcl.remove_radius_outlier(100, 0.025)

        # labels = np.asarray(pcl.cluster_dbscan(eps=20, min_points=100))
        labels = np.asarray(pcl.cluster_dbscan(eps=60, min_points=200))
        max_label = labels.max()
        min_label = labels.min()
        if max_label != min_label:
            cluster_el_cnts = [(labels == l).sum() for l in range(max_label+1)]
            biggest_cluster_idx = np.argmax(cluster_el_cnts)

            biggest_cluster = pcl.select_by_index(
                np.where(labels == biggest_cluster_idx)[0])
            biggest_cluster_pts = np.asarray(biggest_cluster.points)
            biggest_cluster_center = np.mean(biggest_cluster_pts, axis=0)

            final_cluster_list = [biggest_cluster_idx]
            for label_idx in range(max_label + 1):
                if label_idx == biggest_cluster_idx:
                    continue

                cluster = pcl.select_by_index(np.where(labels == label_idx)[0])
                cluster_pts = np.asarray(cluster.points)
                cluster_center = np.mean(cluster_pts, axis=0)

                dist = np.linalg.norm(biggest_cluster_center - cluster_center)
                # biggest_cluster_pts = np.asarray(biggest_cluster.points)
                # dist = np.linalg.norm(biggest_cluster_pts - cluster_center[None], axis=1).min()
                if dist < 120:
                    # biggest_cluster_pts = np.concatenate((biggest_cluster_pts, cluster_pts), axis=0)
                    # biggest_cluster_center = np.mean(biggest_cluster_pts, axis=0)
                    # biggest_cluster = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(biggest_cluster_pts))

                    final_cluster_list.append(label_idx)

            ind3 = [i for idx in final_cluster_list for i in np.where(labels == idx)[0]]
        else:
            ind3 = list(range(labels.shape[0]))

        choose_idxs.append(depth_idxs[ind1][ind2][ind3])

    choose_idxs = np.concatenate(choose_idxs, axis=0).T
    depthmap = np.zeros_like(depth, dtype=np.uint16)
    depthmap[choose_idxs[0], choose_idxs[1]
             ] = depth[choose_idxs[0], choose_idxs[1]]
    Image.fromarray(depthmap, 'I;16').save(out_path)


def worker(depth_path):
    try:
        print(depth_path)
        remove_outliers(depth_path)
    except Exception as e:
        # print(depth_path)
        print(f'Some error occured while processing {depth_path}')
        print(e)


def main(data_dir: str):
    from tqdm import tqdm
    from tqdm.contrib.concurrent import process_map, thread_map
    from natsort import natsorted

    data_path = Path(data_dir)
    depth_paths = natsorted(data_path.rglob('real_*/**/*_depth.png'))

    for depth_path in tqdm(depth_paths, desc='Remove depth outliers'):
         # print(depth_path)
         depth_path = Path(depth_path)
         remove_outliers(depth_path)


if __name__ == '__main__':
    nocs_path = 'data/nocs'
    main(nocs_path)