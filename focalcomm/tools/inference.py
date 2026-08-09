# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
from tqdm import tqdm
import numpy as np
import torch
import open3d as o3d
from torch.utils.data import DataLoader

import focalcomm.data_utils
import focalcomm.hypes_yaml.yaml_utils as yaml_utils
from focalcomm.tools import train_utils, inference_utils
from focalcomm.data_utils.datasets import build_dataset, GT_RANGE
from focalcomm.utils import eval_utils
from focalcomm.visualization import vis_utils, simple_vis
import matplotlib.pyplot as plt


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result' 
                             'in npy_test file')
    parser.add_argument('--dataset_mode', type=str, default="")
    parser.add_argument('--epoch', default=None,
                        help="epoch number to load model")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', "nofusion"]
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)
    # print(hypes)
    if opt.dataset_mode:
        hypes['dataset_mode'] = opt.dataset_mode

    # print(hypes['dataset_mode'])

    print('Dataset Building')
    # breakpoint()
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    
    # Set fixed seed for reproducible sampling
    # Set seeds for deterministic behavior
    seed = 2025  # You can choose any seed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Only take first N samples
    num_test_samples = len(opencood_dataset)
    # num_test_samples = 50
    subset_indices = list(range(min(num_test_samples, len(opencood_dataset))))
    # Reverse the order of indices
    # subset_indices.sort(reverse=True)
    
    print(f"Testing on {len(subset_indices)} samples")
    
    data_loader = DataLoader(torch.utils.data.Subset(opencood_dataset, subset_indices),
                           batch_size=1,
                           num_workers=4,
                           collate_fn=opencood_dataset.collate_batch_test,
                           shuffle=False,  # Keep shuffle=False for consistency
                           pin_memory=False,
                           drop_last=False)
    print('Creating Model')
    # breakpoint()
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model, epoch=opt.epoch)
    model.eval()
    
    # if hasattr(model, 'enable_him_visualization'):
    #     viz_path = os.path.join(opt.model_dir, 'him_visualization')
    #     model.enable_him_visualization(viz_path)
    #     print(f"HIM visualization enabled at: {viz_path}")

    # Create the dictionary for evaluation
    # Use the dataset's class names instead of always using SUPER_CLASS_MAP
    result_stat = {}
    class_names = opencood_dataset.class_names if hasattr(opencood_dataset, 'class_names') else list(focalcomm.data_utils.SUPER_CLASS_MAP.keys())
    for class_name in class_names:
        result_stat[class_name] = {}
        for iou_threshold in [0.3, 0.5, 0.7]:
            result_stat[class_name][iou_threshold] = \
                {'tp': [], 'fp': [], 'gt': 0, 'score': []}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(100):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())

    for i, batch_data in tqdm(enumerate(data_loader)):
        # print(i)
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'nofusion':
                pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                    inference_utils.inference_nofusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                print("intermediate fusion");

                infer_result = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset)
                # breakpoint()
                pred_box_tensor = infer_result["pred_box_tensor"]
                pred_score = infer_result["pred_score"]
                gt_box_tensor = infer_result["gt_box_tensor"]
                gt_label_tensor = infer_result["gt_label_tensor"]
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')
            # breakpoint()
            for class_id, class_name in enumerate(result_stat.keys()):
                class_id += 1
                if pred_box_tensor is None or pred_score is None:
                    print(f"No valid predictions for sample {i}")
                    continue
                for iou_threshold in result_stat[class_name].keys():
                    keep_index_pred = pred_score[:, -1] == class_id
                    keep_index_gt = gt_label_tensor == class_id
                    eval_utils.caluclate_tp_fp(pred_box_tensor[keep_index_pred, ...],
                                               pred_score[keep_index_pred, 0],
                                               gt_box_tensor[keep_index_gt, ...],
                                               result_stat[class_name],
                                               iou_threshold)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   gt_box_tensor,
                                                   batch_data['ego'][
                                                       'origin_lidar'][0],
                                                   i,
                                                   npy_save_path)
            # opt.save_vis = True
            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis_bev')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)
                # pred_box_tensor
                # breakpoint()
                simple_vis.visualize(infer_result,
                                    batch_data['ego'][
                                        'origin_lidar'][0],
                                   GT_RANGE,
                                    vis_save_path,
                                    method='bev',
                                    left_hand=False)
                # Visualize raw LiDAR only
                # simple_vis.visualize_lidar_raw(
                #     batch_data['ego']['origin_lidar'],  # List of LiDAR point clouds
                #     GT_RANGE,
                #     os.path.join(os.path.dirname(vis_save_path), 'lidar_raw_%05d.png' % i),
                #     method='bev',
                #     left_hand=False
                # )
                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  pred_score[:,-1],
                                                  gt_label_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'],
                                                  None,
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset)

            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader_with_map(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'],
                        None,
                        vis_pcd,
                        mode='constant'
                        )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir)
    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
