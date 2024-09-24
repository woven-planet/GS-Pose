"""
    Evaluation-related codes are modified from
    https://github.com/qq456cvb/CPPF
"""


import argparse
import glob
import os
from tqdm import tqdm
import pickle
import numpy as np
from util import compute_degree_cm_mAP
from scipy.spatial.transform import Rotation


synset_names = ['BG',  # 0
                'bottle',  # 1
                'bowl',  # 2
                'camera',  # 3
                'can',  # 4
                'laptop',  # 5
                'mug'  # 6
                ]

if __name__ == '__main__':

    pred_dir = 'data/nocs/nocs_seg'
    pred_results_path = 'data/results/'
    result_pkl_list = glob.glob(os.path.join(pred_dir, 'results_*.pkl'))
    result_pkl_list = sorted(result_pkl_list)[::10]
    assert len(result_pkl_list)

    final_results = []
    for pkl_path in tqdm(result_pkl_list):
        with open(pkl_path, 'rb') as f:
            result = pickle.load(f)
            del result['pred_masks']
            gt_pose_calib = result['gt_RTs']
            gt_scales = result['gt_scales']
            for i in range(gt_pose_calib.shape[0]):
                norm_gt_scales = np.cbrt(np.linalg.det(gt_pose_calib[i, :3, :3]))
                gt_pose_calib[i, :3, :3] = gt_pose_calib[i, :3, :3] / norm_gt_scales
                gt_scales[i] = gt_scales[i] * norm_gt_scales
            result['gt_RTs'] = gt_pose_calib
            result['gt_scales'] = gt_scales
        pred_RTs = []
        pred_bboxes = []
        pred_class_ids = []
        pred_scales = []
        pred_scores = []
        for cat_name in synset_names:
            cat_id = synset_names.index(cat_name)
            cat_pkl_path = os.path.join(pred_results_path,cat_name,pkl_path.split('/')[-1])
            if os.path.exists(cat_pkl_path):
                with open(cat_pkl_path, 'rb') as f:
                    pred_result = pickle.load(f)
                pred_RTs.extend( pred_result['pred_RTs'] )
                pred_bboxes.extend( pred_result['pred_bboxes'] )
                pred_class_ids.extend( (np.ones(pred_result['pred_bboxes'].shape[0], np.int64)*cat_id).tolist() )
                pred_scales.extend( pred_result['pred_scales']*2 )
                pred_scores.extend( pred_result['pred_scores'] )

        result['pred_RTs'] = pred_RTs
        result['pred_bboxes'] = pred_bboxes
        result['pred_class_ids'] = np.array(pred_class_ids).astype(int)
        result['pred_scales'] = pred_scales
        result['pred_scores'] = np.array(pred_scores)


        gt_handle_visibility = result['gt_handle_visibility']
        gt_class_ids = result['gt_class_ids']
        result['gt_up_syms'] = np.zeros_like(result['gt_handle_visibility'], dtype=bool)
        for i, (cls_id, vis) in enumerate(zip(gt_class_ids, gt_handle_visibility)):
            if vis == 0:  # handle not seen, assume up summetry
                assert synset_names[cls_id] == 'mug'
                result['gt_up_syms'][i] = True
            elif synset_names[cls_id] in ['bowl', 'bottle', 'can']:
                result['gt_up_syms'][i] = True

        assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(
            result['gt_handle_visibility'], result['gt_class_ids'])

        if type(result) is list:
            final_results += result
        elif type(result) is dict:
            final_results.append(result)
        else:
            assert False


    iou_3d_aps, pose_aps, pose_pred_matches, pose_gt_matches = compute_degree_cm_mAP(final_results, synset_names,
                                                                                     os.path.join(pred_results_path,'vis'),
                                                                                     degree_thresholds=[5,10,15],
                                                                                     shift_thresholds= [5,10,15],
                                                                                     iou_3d_thresholds=np.linspace(0, 1,
                                                                                                                   101),
                                                                                     iou_pose_thres=0.1,
                                                                                     use_matches_for_pose=True)