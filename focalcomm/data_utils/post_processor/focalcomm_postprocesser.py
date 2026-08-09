# Author: Dereje Shenkut <derejeshenkut@gmail.com>

"""
3D Anchor Generator for Voxel
"""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

from focalcomm.data_utils.post_processor.base_postprocessor \
    import BasePostprocessor
from focalcomm.utils import box_utils


class FocalCommPostprocessor(BasePostprocessor):
    def __init__(self, params, class_names, train):
        super(FocalCommPostprocessor, self).__init__(params, class_names, train)
        self.params = params  # This only contains postprocess section
        self.class_names = class_names
        self.train = train
        
        # Get parameters from postprocess section
        self.point_cloud_range = params['cav_lidar_range']
        self.nms_thresh = params['nms_thresh']
        self.max_num = params['max_num']
        self.order = params['order']
        self.score_threshold = params['score_threshold']

        # These would need to be passed in from outside or set to default values
        self.voxel_size = params['voxel_size']  # Default from yaml
        self.feature_map_stride = params['feature_map_stride']  # Default from yaml
        self.num_classes = params['num_classes']  # Default from yaml
        self.post_center_range = params['post_center_range']  # Default from yaml

    def decode_predictions(self, pred_dicts, filter_by_range=True):
        """
        Decode raw predictions into boxes, scores, and labels format.
        
        Args:
            pred_dicts (dict): Dictionary containing raw predictions
            filter_by_range (bool): Whether to filter predictions by post-processing range
        
        Returns:
            list[dict]: List of dictionaries containing decoded predictions
        """
        # Get batch size and apply sigmoid to heatmap
        batch_size = pred_dicts["heatmap"].shape[0]
        batch_score = pred_dicts["heatmap"].sigmoid()
        
        # Combine scores with query heatmap scores and one-hot labels
        query_labels = pred_dicts["query_labels"]
        one_hot = F.one_hot(
            query_labels, num_classes=self.num_classes
        ).permute(0, 2, 1)
        batch_score = batch_score * pred_dicts["query_heatmap_score"] * one_hot

        # Decode center coordinates to ego vehicle coordinate system
        batch_center = pred_dicts["center"].clone()
        batch_center[:, 0, :] = batch_center[:, 0, :] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        batch_center[:, 1, :] = batch_center[:, 1, :] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]

        # Decode dimensions (exponential for positive values)
        batch_dim = pred_dicts["dim"].exp()

        # Decode rotation (convert from sin/cos to angle)
        batch_rot = pred_dicts["rot"]
        rots, rotc = batch_rot[:, 0:1, :], batch_rot[:, 1:2, :]
        batch_rot = torch.atan2(rots, rotc)

        # Combine all box parameters
        batch_height = pred_dicts["height"]
        final_box_preds = torch.cat(
            [batch_center, batch_height, batch_dim, batch_rot], 
            dim=1
        ).permute(0, 2, 1)  # (B, N, 7)

        # Get final scores and labels
        final_scores = batch_score.max(1, keepdims=False).values  # (B, N)
        final_labels = batch_score.max(1, keepdims=False).indices  # (B, N)

        predictions_dicts = []
        
        if filter_by_range:
            # Get post-processing configuration
            post_center_range = torch.tensor(
                self.post_center_range
            ).to(final_box_preds.device).float()

            # Create range and score masks
            range_mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
            range_mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)
            if isinstance(self.score_threshold, list):
                # Per-class thresholding
                score_mask = torch.zeros_like(final_scores, dtype=torch.bool)
                for class_idx, threshold in enumerate(self.score_threshold):
                    class_mask = final_labels == class_idx
                    score_class_mask = final_scores > threshold
                    score_mask = score_mask | (class_mask & score_class_mask)
            else:
                # Original single threshold
                score_mask = final_scores > self.score_threshold

            for i in range(batch_size):
                # Combine masks
                valid_mask = range_mask[i] & score_mask[i]
                
                predictions_dict = {
                    'pred_boxes': final_box_preds[i][valid_mask],    # (N, 7)
                    'pred_scores': final_scores[i][valid_mask],      # (N,)
                    'pred_labels': final_labels[i][valid_mask].int() + 1  # (N,)
                }
                predictions_dicts.append(predictions_dict)
        else:
            for i in range(batch_size):
                predictions_dict = {
                    'pred_boxes': final_box_preds[i],    # (N, 7)
                    'pred_scores': final_scores[i],      # (N,)
                    'pred_labels': final_labels[i].int() + 1  # (N,)
                }
                predictions_dicts.append(predictions_dict)

        return predictions_dicts

    def post_process(self, data_dict, output_dict, projection=True):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Debug version with prints but keeping all filtering steps.
        """
        pred_box3d_list = []
        pred_scores_list = []
        pred_labels_list = []

        for cav_id, cav_content in data_dict.items():
            if cav_id not in output_dict:
                continue

            # Get transformation matrix to ego space
            transformation_matrix = cav_content['transformation_matrix']

            # Decode predictions
            pred_dicts = output_dict[cav_id]

            decoded_preds = self.decode_predictions(pred_dicts, filter_by_range=True)  
            assert len(decoded_preds) == 1
            pred_dict = decoded_preds[0]            
            boxes3d = pred_dict['pred_boxes']
            scores = pred_dict['pred_scores']
            labels = pred_dict['pred_labels']


            if len(boxes3d) != 0:
                # Convert boxes to corner format (N, 8, 3)
                boxes3d_corner = box_utils.boxes_to_corners_3d(boxes3d, 
                                                             order=self.order)
                
                # Project boxes to ego space if required
                if projection:
                    projected_boxes3d = box_utils.project_box3d(boxes3d_corner,
                                                              transformation_matrix)
                    # print(f"After projection: shape {projected_boxes3d.shape}")
                else:
                    projected_boxes3d = boxes3d_corner

                pred_box3d_list.append(projected_boxes3d)
                pred_scores_list.append(scores)
                pred_labels_list.append(labels)

        if len(pred_box3d_list) == 0:
            return None, None

        # Concatenate predictions from all CAVs
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        scores = torch.cat(pred_scores_list)
        pred_label_tensor = torch.cat(pred_labels_list)  # Original labels
        
        # Apply first filter to all tensors
        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)
        
        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]
        pred_label_tensor = pred_label_tensor[keep_index]  # Filter labels

        # Apply NMS filter to all tensors
        keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                         scores,
                                         self.params['nms_thresh'])
        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]
        pred_label_tensor = pred_label_tensor[keep_index]  # Filter labels

        # Apply range mask to all tensors
        mask = box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
        pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        scores = scores[mask]
        pred_label_tensor = pred_label_tensor[mask]  # Filter labels

        assert scores.shape[0] == pred_box3d_tensor.shape[0]
        assert scores.shape[0] == pred_label_tensor.shape[0]  # New check

        # Combine scores and labels
        score_labels = torch.cat([
            scores.unsqueeze(1),
            pred_label_tensor.unsqueeze(1)
        ], dim=1)

        return pred_box3d_tensor, score_labels

    def generate_label(self, **kwargs):
        """
        Generate raw ground-truth bounding box info in TransFusion style.
        This will return an Nx8 or Nx9 array of [x, y, z, dx, dy, dz, yaw, (vx, vy), class].
        TransFusion's get_targets expects gt_bboxes_3d and gt_labels_3d in the batch_dict,
        so we will store them accordingly.

        Parameters
        ----------
        kwargs : dict
            Should at least contain:
            - gt_box_center: shape (max_num_boxes, 8) -> (x, y, z, dx, dy, dz, yaw, class)
              or shape (max_num_boxes, 9 or 10) if velocity is included
            - mask: shape (max_num_boxes,), marking valid boxes with 1
        Returns
        -------
        label_dict : dict
            A dictionary containing ground-truth boxes with shape (N, 8) or (N, 9)
            in the key "gt_boxes". The last column is the class label.
            e.g. gt_boxes[i] = [x, y, z, dx, dy, dz, yaw, class]
        """

        # Pull out the GT boxes and mask
        gt_box_center_all = kwargs['gt_box_center']  # e.g. shape [max_num, 8] or [max_num, 9]
        mask = kwargs['mask']                       # shape [max_num]

        # Filter out invalid boxes
        valid_indices = (mask == 1)
        gt_box_center_valid = gt_box_center_all[valid_indices]

        label_dict = {
            'gt_boxes': gt_box_center_valid  # shape: (N, 8) or (N, 9)
        }

        return label_dict


    def generate_gt_bbx(self, data_dict):
        """
        The base postprocessor will generate 3d groundtruth bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        Returns
        -------
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor, shape (N, 8, 3).
        """
        gt_box3d_list = []
        label_list = []
        # used to avoid repetitive bounding box
        object_id_list = []

        for cav_id, cav_content in data_dict.items():
            # used to project gt bounding box to ego space
            transformation_matrix = cav_content['transformation_matrix']

            object_bbx_center = cav_content['gt_boxes'][0]
            # object_bbx_mask = cav_content['object_bbx_mask']
            object_ids = cav_content['object_ids'][0]
            # object_bbx_center = object_bbx_center[object_bbx_mask == 1]
            # breakpoint()
            labels = object_bbx_center[:, -1]
            # convert center to corner
            # print(self.params)
            object_bbx_corner = \
                box_utils.boxes_to_corners_3d(object_bbx_center,
                                              self.params['order'])
            projected_object_bbx_corner = \
                box_utils.project_box3d(object_bbx_corner.float(),
                                        transformation_matrix)
            gt_box3d_list.append(projected_object_bbx_corner)
            label_list.append(labels)

            # append the corresponding ids
            object_id_list += object_ids

        # gt bbx 3d
        gt_box3d_list = torch.vstack(gt_box3d_list)
        label_list = torch.cat(label_list)
        # some of the bbx may be repetitive, use the id list to filter
        # breakpoint()
        gt_box3d_selected_indices = \
            [object_id_list.index(x) for x in set(object_id_list)]
        gt_box3d_tensor = gt_box3d_list[gt_box3d_selected_indices]
        gt_label_tensor = label_list[gt_box3d_selected_indices]

        # filter the gt_box to make sure all bbx are in the range
        mask = \
            box_utils.get_mask_for_boxes_within_range_torch(gt_box3d_tensor)
        gt_box3d_tensor = gt_box3d_tensor[mask, :, :]
        gt_label_tensor = gt_label_tensor[mask]

        return gt_box3d_tensor, gt_label_tensor

    def generate_object_center(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 8), where 8 represents
        x, y, z, l, w, h, yaw, class or x, y, z, h, w, l, yaw, class.

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 8).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        from focalcomm.data_utils.datasets import GT_RANGE

        tmp_object_dict = {}
        for cav_content in cav_contents:
            tmp_object_dict.update(cav_content['params']['vehicles'])

        output_dict = {}
        filter_range = self.params['cav_lidar_range'] \
            if self.train else GT_RANGE
        box_utils.project_world_objects(tmp_object_dict,
                                        output_dict,
                                        reference_lidar_pose,
                                        filter_range,
                                        self.params['order'])

        object_np = np.zeros((self.params['max_num'], 8))

        mask = np.zeros(self.params['max_num'])
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            object_ids.append(object_id)

        return object_np, mask, object_ids
    
    # def generate_object_center_dairv2x(self, cav_contents, reference_lidar_pose):
    #     """
    #     Generate object centers for DAIR-V2X format.
    #     Simplified version that converts DAIR-V2X format to expected format.
    #     """
    #     # First ensure vehicles data is set properly
    #     tmp_object_list = []
    #     cav_content = cav_contents[0]
    #     tmp_object_list = cav_content['params']['vehicles'] 
    #     from focalcomm.data_utils.datasets import GT_RANGE



    #     output_dict = {}

    #     box_utils.project_world_objects_dairv2x(tmp_object_list,
    #                                     output_dict,
    #                                     reference_lidar_pose,
    #                                     GT_RANGE,
    #                                     self.params['order'])

    #     object_np = np.zeros((self.params['max_num'], 8))
    #     mask = np.zeros(self.params['max_num'])
    #     object_ids = []

    #     for i, (object_id, object_bbx) in enumerate(output_dict.items()):
    #         object_np[i] = object_bbx[0, :]
    #         mask[i] = 1
    #         object_ids.append(object_id)

    #     return object_np, mask, object_ids      
    #     # return object_np, mask, object_ids

    # def generate_gt_bbx_by_iou(self, data_dict):
    #     """
    #     This function is only used by DAIR-V2X + late fusion dataset

    #     DAIR-V2X + late fusion dataset's label are from veh-side and inf-side
    #     and do not have unique object id.

    #     So we will filter the same object by IoU

    #     The base postprocessor will generate 3d groundtruth bounding box.

    #     For early and intermediate fusion,
    #         data_dict only contains ego.

    #     For late fusion,
    #         data_dcit contains all cavs, so we need transformation matrix.
    #         To generate gt boxes, transformation_matrix should be clean

    #     Parameters
    #     ----------
    #     data_dict : dict
    #         The dictionary containing the origin input data of model.

    #     Returns
    #     -------
    #     gt_box3d_tensor : torch.Tensor
    #         The groundtruth bounding box tensor, shape (N, 8, 3).
    #     """
    #     gt_box3d_list = []

    #     for cav_id, cav_content in data_dict.items():
    #         # used to project gt bounding box to ego space
    #         # object_bbx_center is clean.
    #         transformation_matrix = cav_content['transformation_matrix_clean']

    #         object_bbx_center = cav_content['object_bbx_center']
    #         object_bbx_mask = cav_content['object_bbx_mask']
    #         object_ids = cav_content['object_ids']
    #         object_bbx_center = object_bbx_center[object_bbx_mask == 1]

    #         # convert center to corner
    #         object_bbx_corner = \
    #             box_utils.boxes_to_corners_3d(object_bbx_center,
    #                                           self.params['order'])
    #         projected_object_bbx_corner = \
    #             box_utils.project_box3d(object_bbx_corner.float(),
    #                                     transformation_matrix)
    #         gt_box3d_list.append(projected_object_bbx_corner)

    #     # if only ego agent
    #     if len(data_dict) == 1:
    #         gt_box3d_tensor = torch.vstack(gt_box3d_list)
    #     # both veh-side and inf-side label
    #     else:
    #         veh_corners_np = gt_box3d_list[0].cpu().numpy()
    #         inf_corners_np = gt_box3d_list[1].cpu().numpy()
    #         inf_polygon_list = list(common_utils.convert_format(inf_corners_np))
    #         veh_polygon_list = list(common_utils.convert_format(veh_corners_np))
    #         iou_thresh = 0.05 


    #         gt_from_inf = []
    #         for i in range(len(inf_polygon_list)):
    #             inf_polygon = inf_polygon_list[i]
    #             ious = common_utils.compute_iou(inf_polygon, veh_polygon_list)
    #             if (ious > iou_thresh).any():
    #                 continue
    #             gt_from_inf.append(inf_corners_np[i])
            
    #         if len(gt_from_inf):
    #             gt_from_inf = np.stack(gt_from_inf)
    #             gt_box3d = np.vstack([veh_corners_np, gt_from_inf])
    #         else:
    #             gt_box3d = veh_corners_np

    #         gt_box3d_tensor = torch.from_numpy(gt_box3d).to(device=gt_box3d_list[0].device)

    #     # mask_boxes_outside_range_numpy has filtering of z-dim
    #     # gt_box3d_np = gt_box3d_tensor.cpu().numpy()
    #     # gt_box3d_np = box_utils.mask_boxes_outside_range_numpy(gt_box3d_np,
    #     #                                             self.params['gt_range'],
    #     #                                             self.params['order'])
    #     # gt_box3d_tensor = torch.from_numpy(gt_box3d_np).to(device=gt_box3d_list[0].device)

    #     # need discussion. not filter z-dim.
    #     mask = \
    #         box_utils.get_mask_for_boxes_within_range_torch(gt_box3d_tensor, self.params['gt_range'])
    #     gt_box3d_tensor = gt_box3d_tensor[mask, :, :]


    #     return gt_box3d_tensor
    
    @staticmethod
    def visualize(pred_box_tensor, gt_box_tensor, pred_label_tensor, gt_label_tensor, origin_lidar, map_lidar, show_vis, save_path,
                  dataset=None):
        """
        Visualize the prediction, ground truth with point cloud together.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
            (N, 8, 3) prediction.

        gt_box_tensor : torch.Tensor
            (N, 8, 3) groundtruth bbx

        origin_lidar : torch.Tensor
            PointCloud, (N, 4).

        show_vis : bool
            Whether to show visualization.

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        """
        from focalcomm.visualization import vis_utils
        vis_utils.visualize_single_sample_output_gt(pred_box_tensor,
                                                    gt_box_tensor,
                                                    pred_label_tensor,
                                                    gt_label_tensor,
                                                    origin_lidar,
                                                    map_lidar,
                                                    show_vis,
                                                    save_path)