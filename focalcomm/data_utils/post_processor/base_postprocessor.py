# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Template for AnchorGenerator
"""

import numpy as np
import torch

from focalcomm.utils import box_utils


class BasePostprocessor(object):
    """
    Template for Anchor generator.

    Parameters
    ----------
    anchor_params : dict
        The dictionary containing all anchor-related parameters.
    train : bool
        Indicate train or test mode.

    Attributes
    ----------
    bbx_dict : dictionary
        Contain all objects information across the cav, key: id, value: bbx
        coordinates (1, 7)
    """

    def __init__(self, params, class_names, train=True):
        self.params = params
        # self.lidar_range = lidar_range
        self.class_names = class_names
        self.bbx_dict = {}
        self.train = train
        print(self.params)
        # self.gaussian_overlap = anchor_params.get('gaussian_overlap', 0.1)
        # self.min_radius = anchor_params.get('min_radius', 2)
        # self.grid_size = anchor_params['anchor_args'].get('grid_size', [96, 256])
        # self.pc_range = anchor_params['anchor_args']['cav_lidar_range']

    def generate_anchor_box(self):
        # needs to be overloaded
        return None

    def generate_label(self, *argv):
        return None
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

            object_bbx_center = cav_content['object_bbx_center']
            object_bbx_mask = cav_content['object_bbx_mask']
            object_ids = cav_content['object_ids']
            object_bbx_center = object_bbx_center[object_bbx_mask == 1]

            labels = object_bbx_center[:, -1]
            # convert center to corner
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
        # during training using cav_lidar_range to set learning targets
        # during inference using gt_range to set ground truth
        filter_range = self.params['anchor_args']['cav_lidar_range'] \
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



    def generate_object_center_dairv2x(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 8), where 8 represents
        x, y, z, l, w, h, yaw, class or x, y, z, h, w, l, yaw, clas.

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

        # tmp_object_dict = {}
        tmp_object_list = []
        cav_content = cav_contents[0]
        tmp_object_list = cav_content['params']['vehicles'] #世界坐标系下

        output_dict = {}
        filter_range = self.params['cav_lidar_range']


        box_utils.project_world_objects_dairv2x(tmp_object_list,
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


    def generate_object_center_dairv2x_single(self,
                               cav_contents,
                               suffix=""):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """

        # tmp_object_dict = {}
        tmp_object_list = []
        cav_content = cav_contents[0]
        tmp_object_list = cav_content['params'][f'vehicles{suffix}'] # ego 坐标系下

        output_dict = {}
        filter_range = self.params['anchor_args']['cav_lidar_range']


        box_utils.load_single_objects_dairv2x(tmp_object_list,
                                        output_dict,
                                        filter_range,
                                        self.params['order'])

        object_np = np.zeros((self.params['max_num'], 7))
        mask = np.zeros(self.params['max_num'])
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            object_ids.append(object_id)

        return object_np, mask, object_ids


    # def generate_gt_bbx_by_iou(self, data_dict):
        # """
        # This function is only used by DAIR-V2X + late fusion dataset

        # DAIR-V2X + late fusion dataset's label are from veh-side and inf-side
        # and do not have unique object id.

        # So we will filter the same object by IoU

        # The base postprocessor will generate 3d groundtruth bounding box.

        # For early and intermediate fusion,
        #     data_dict only contains ego.

        # For late fusion,
        #     data_dcit contains all cavs, so we need transformation matrix.
        #     To generate gt boxes, transformation_matrix should be clean

        # Parameters
        # ----------
        # data_dict : dict
        #     The dictionary containing the origin input data of model.

        # Returns
        # -------
        # gt_box3d_tensor : torch.Tensor
        #     The groundtruth bounding box tensor, shape (N, 8, 3).
        # """
        # gt_box3d_list = []

        # for cav_id, cav_content in data_dict.items():
        #     # used to project gt bounding box to ego space
        #     # object_bbx_center is clean.
        #     transformation_matrix = cav_content['transformation_matrix_clean']

        #     object_bbx_center = cav_content['object_bbx_center']
        #     object_bbx_mask = cav_content['object_bbx_mask']
        #     object_ids = cav_content['object_ids']
        #     object_bbx_center = object_bbx_center[object_bbx_mask == 1]

        #     # convert center to corner
        #     object_bbx_corner = \
        #         box_utils.boxes_to_corners_3d(object_bbx_center,
        #                                       self.params['order'])
        #     projected_object_bbx_corner = \
        #         box_utils.project_box3d(object_bbx_corner.float(),
        #                                 transformation_matrix)
        #     gt_box3d_list.append(projected_object_bbx_corner)

        # # if only ego agent
        # if len(data_dict) == 1:
        #     gt_box3d_tensor = torch.vstack(gt_box3d_list)
        # # both veh-side and inf-side label
        # else:
        #     veh_corners_np = gt_box3d_list[0].cpu().numpy()
        #     inf_corners_np = gt_box3d_list[1].cpu().numpy()
        #     inf_polygon_list = list(common_utils.convert_format(inf_corners_np))
        #     veh_polygon_list = list(common_utils.convert_format(veh_corners_np))
        #     iou_thresh = 0.05 


        #     gt_from_inf = []
        #     for i in range(len(inf_polygon_list)):
        #         inf_polygon = inf_polygon_list[i]
        #         ious = common_utils.compute_iou(inf_polygon, veh_polygon_list)
        #         if (ious > iou_thresh).any():
        #             continue
        #         gt_from_inf.append(inf_corners_np[i])
            
        #     if len(gt_from_inf):
        #         gt_from_inf = np.stack(gt_from_inf)
        #         gt_box3d = np.vstack([veh_corners_np, gt_from_inf])
        #     else:
        #         gt_box3d = veh_corners_np

        #     gt_box3d_tensor = torch.from_numpy(gt_box3d).to(device=gt_box3d_list[0].device)

        # # mask_boxes_outside_range_numpy has filtering of z-dim
        # # gt_box3d_np = gt_box3d_tensor.cpu().numpy()
        # # gt_box3d_np = box_utils.mask_boxes_outside_range_numpy(gt_box3d_np,
        # #                                             self.params['gt_range'],
        # #                                             self.params['order'])
        # # gt_box3d_tensor = torch.from_numpy(gt_box3d_np).to(device=gt_box3d_list[0].device)

        # # need discussion. not filter z-dim.
        # mask = \
        #     box_utils.get_mask_for_boxes_within_range_torch(gt_box3d_tensor, self.params['gt_range'])
        # gt_box3d_tensor = gt_box3d_tensor[mask, :, :]


        # return gt_box3d_tensor