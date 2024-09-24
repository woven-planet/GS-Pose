"""
    Umeyama codes are modified from
    https://github.com/hughw19/NOCS_CVPR2019
"""

import numpy as np
import torch
from skimage.measure import ransac
import cv2
import itertools
import open3d as o3d

def umeyama(src, dst, estimate_scale=True):
    """Estimate N-D similarity transformation with or without scaling.
    Taken from skimage!

    homo_src = np.hstack((src, np.ones((len(src), 1))))
    homo_dst = np.hstack((src, np.ones((len(src), 1))))

    homo_dst = T @ homo_src, where T is the returned transformation

    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.eye(4), 1
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T, scale



class RigidBodyUmeyama():

    def estimate(self, world_corr1, world_corr2):
        self.T, self.lam = umeyama(world_corr1, world_corr2)



    def residuals(self, world_corr1, world_corr2):
        world_corr2_est = self.transform(world_corr1)
        res = torch.nn.PairwiseDistance(p=2)(torch.Tensor(world_corr2_est),
                                             torch.Tensor(world_corr2))
        return res.numpy()

    def transform(self, world_corr1):
        w1_homo = np.vstack((world_corr1.T, np.ones((1, (len(world_corr1))))))
        transformed = self.T @ w1_homo
        return (transformed[:3, :]).T

def solve_umeyama_ransac_scale(world_corr1, world_corr2, world_corr1_all):
    rbt_model, inliers = ransac(data=(world_corr1, world_corr2),
                        model_class=RigidBodyUmeyama,
                        min_samples=4, #4,
                        residual_threshold=0.05,
                        # max_trials=10000)
                        max_trials=1000)
                               # max_trials=1)
    if rbt_model==None:
        return None
    else:
        R = rbt_model.T[:3, :3] / rbt_model.lam
        t = rbt_model.T[:3, 3:][:,0]
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t

        w2_homo = np.vstack((world_corr2.T, np.ones((1, (len(world_corr2))))))
        transformed = np.linalg.inv(rbt_model.T) @ w2_homo
        world_corr2_new = (transformed[:3, :]).T

        num=10
        corrd2_max = np.partition(abs(world_corr2_new), -num, axis=0)[-num:, :].min(0)
        coord1_max = np.partition(abs(world_corr1), -1, axis=0)[-1:, :].min(0)
        Scales = corrd2_max / coord1_max
        scale = rbt_model.lam * Scales

        threshold = 0.05 #0.05
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(world_corr2)
        color = np.zeros(world_corr2.shape)
        color[:,0]=1
        pcd1.colors = o3d.utility.Vector3dVector(color)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(world_corr1*scale)
        color = np.zeros(world_corr1.shape)
        color[:,2]=1
        pcd2.colors = o3d.utility.Vector3dVector(color)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd1, pcd2, threshold, np.linalg.inv(T),
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        T_new = np.linalg.inv(reg_p2p.transformation)
        R,t = T_new[:3,:3], T_new[:3,3]

        return R, t, scale