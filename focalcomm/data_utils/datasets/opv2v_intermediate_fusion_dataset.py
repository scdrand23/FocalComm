# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
# Modified by: Dereje Shenkut <derejeshenkut@gmail.com> for FocalComm

import os
import random
import math
from collections import OrderedDict
import cv2
import h5py
import numpy as np
import torch
from PIL import Image
import focalcomm.data_utils.datasets
import focalcomm.data_utils.post_processor as post_processor
from focalcomm.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset
from focalcomm.data_utils.augmentor.data_augmentor import DataAugmentor
from focalcomm.data_utils.pre_processor import build_preprocessor
from focalcomm.hypes_yaml.yaml_utils import load_yaml
from focalcomm.utils.camera_utils import load_camera_data
from focalcomm.utils.common_utils import read_json
from focalcomm.utils.pcd_utils import downsample_lidar_minimum, mask_points_by_range
from focalcomm.utils.transformation_utils import x1_to_x2
from focalcomm.utils import box_utils
from focalcomm.utils import pcd_utils


class OPV2VIntermediateFusionDataset(OPV2VBaseDataset):
    """
    This class is for OPV2V intermediate fusion for FocalComm.
    """
    def __init__(self, params, visualize, train=True):
        super(OPV2VFocalCommIntermediateDataset, self).__init__(params, visualize, train)
        
        # FocalComm specific initialization
        self.visible = params['fusion']['args']['visible'] if 'visible' in params['fusion']['args'] else False
        
        # Communication range for OPV2V
        self.comm_range = params.get('comm_range', 70)  # Default 70m for OPV2V
        
        # intermediate and supervise single
        self.supervise_single = params['fusion']['args'].get('supervise_single', False)
        
        if 'proj_first' in params['fusion']['args'] and params['fusion']['args']['proj_first']:
            self.proj_first = True
        else:
            self.proj_first = False

        # anchor-free method specific post processor
        if hasattr(self.post_processor, 'generate_label'):
            self.anchor_free = True
        else:
            self.anchor_free = False

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
            else:
                idx -= ele

        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database,
                                                   timestamp_index)
        # calculate distance to ego for each cav
        ego_cav_base = None

        for cav_id, cav_content in scenario_database.items():
            if cav_id == 'ego_vehicle':
                ego_cav_base = cav_content[timestamp_key]
                break
        
        if not ego_cav_base:
            # Handle case where there's no ego_vehicle (OPV2V doesn't always have explicit ego)
            # Take the first vehicle as ego
            ego_cav_base = list(scenario_database.values())[0][timestamp_key]

        assert ego_cav_base is not None

        ego_pose = load_yaml(ego_cav_base['yaml'])['true_ego_pos']
        ego_pose = x1_to_x2(ego_pose, ego_pose) # np.eye(4)

        data = OrderedDict()
        # load files for all CAVs
        for i, (cav_id, cav_content) in enumerate(scenario_database.items()):
            if cav_id == 'scene_len':
                continue
            
            if i >= self.max_cav:
                break

            distance = self.calc_dist_to_ego(cav_content[timestamp_key]['yaml'], ego_pose)
            
            # if distance > self.comm_range, we will just skip this agent
            if distance > self.comm_range:
                continue

            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = True if cav_id == 'ego_vehicle' or i == 0 else False

            # load the yaml file
            cav_yaml = load_yaml(cav_content[timestamp_key]['yaml'])
            data[cav_id].update(cav_yaml)

            # load lidar files
            if self.load_lidar_file:
                # load lidar file as npy from CrossMAP version
                if self.use_hdf5:
                    lidar_file = h5py.File(cav_content[timestamp_key]['lidar'].replace(".pcd", ".hdf5"), 'r', swmr=True)
                    lidar_np = lidar_file['lidar'][()]
                    lidar_file.close()
                else:
                    lidar_np = pcd_utils.read_pcd(cav_content[timestamp_key]['lidar'])

                data[cav_id]['lidar_np'] = lidar_np

            # load camera files
            if self.load_camera_file:
                camera_files = []
                # Check different camera file extensions and merge
                if 'camera0' in cav_content[timestamp_key]:
                    camera_files.extend(cav_content[timestamp_key]['camera0'])
                if 'camera1' in cav_content[timestamp_key]:
                    camera_files.extend(cav_content[timestamp_key]['camera1'])
                if 'camera2' in cav_content[timestamp_key]:
                    camera_files.extend(cav_content[timestamp_key]['camera2'])
                
                data[cav_id]['camera_data'] = load_camera_data(camera_files)

        return data

    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Project the lidar and bbx to ego space first, and then load data for
        the selected cav.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list, length 6
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = x1_to_x2(selected_cav_base['params']['lidar_pose'], ego_pose)

        # retrieve objects under ego coordinates
        object_bbx_center, object_bbx_mask, object_ids = \
            self.generate_object_center([selected_cav_base],
                                        ego_pose)

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = mask_points_by_range(lidar_np,
                                       self.params['preprocess']['cav_lidar_range'])
        # remove points that hit ego vehicle
        lidar_np = mask_ego_points(lidar_np)

        # project the lidar to ego space
        if self.proj_first:
            lidar_np[:, 0:3] = box_utils.project_points_by_matrix_torch(lidar_np[:, 0:3],
                                                                   transformation_matrix)

        lidar_np = mask_points_by_range(lidar_np,
                                       self.params['preprocess']['cav_lidar_range'])
        
        # Check for empty lidar after filtering
        if len(lidar_np) == 0:
            # Create minimal dummy data to avoid empty tensors
            lidar_np = np.zeros((1, 4), dtype=np.float32)  # Single point at origin
            
        processed_lidar = self.pre_processor.preprocess(lidar_np)

        # velocity
        velocity = selected_cav_base['params'].get('velocity', [0, 0])
        # normalize veloccity by average speed 30 km/h = 8.3 m/s
        velocity = [velocity[0] / 8.3, velocity[1] / 8.3]

        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'projected_lidar': lidar_np,
             'processed_features': processed_lidar,
             'velocity': velocity})

        return selected_cav_processed

    def calc_dist_to_ego(self, cav_yaml_path, ego_pose):
        """
        Calculate the distance to ego vehicle.

        Parameters
        ----------
        cav_yaml_path : str
            The path to the CAV yaml file.
        ego_pose : np.ndarray
            The ego vehicle pose.

        Returns
        -------
        distance : float
            The distance to ego vehicle.
        """
        cav_yaml = load_yaml(cav_yaml_path)
        cav_pose = cav_yaml['lidar_pose']
        
        # Calculate Euclidean distance
        distance = np.sqrt((cav_pose[0] - ego_pose[0, 3])**2 + 
                          (cav_pose[1] - ego_pose[1, 3])**2)
        
        return distance

    def return_timestamp_key(self, scenario_database, timestamp_index):
        """
        Given the timestamp index, return the corresponding timestamp key.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all cavs' information.
        timestamp_index : int
            The index of the timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database['ego_vehicle'].keys()) if 'ego_vehicle' in scenario_database \
                        else list(list(scenario_database.values())[0].keys())
        
        return timestamp_keys[timestamp_index]

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        agents_image_list = []
        processed_features_list = []
        object_ids_list = []
        object_bbx_center_list = []
        
        if self.visualize:
            projected_lidar_list = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)
            if distance > self.comm_range:
                continue

            selected_cav_processed = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose)

            object_bbx_center_list.append(selected_cav_processed['object_bbx_center'])
            object_ids_list.append(selected_cav_processed['object_ids'])
            agents_image_list.append(selected_cav_processed['processed_features'])
            
            if self.load_camera_file:
                agents_image_list.append(
                    selected_cav_processed['camera_data'])

            if self.visualize:
                projected_lidar_list.append(
                    selected_cav_processed['projected_lidar'])

        # exclude all repetitive objects
        unique_indices = \
            [object_ids_list.index(x) for x in set(object_ids_list)]
        object_bbx_center_list = np.vstack(object_bbx_center_list)
        object_bbx_center_list = object_bbx_center_list[unique_indices]

        object_ids_list = set(object_ids_list)

        # make sure bounding boxes across all frames have the same number
        object_bbx_center_list, mask = self.pad_object_to_max_num(object_bbx_center_list)

        processed_data_dict['ego']['object_bbx_center'] = object_bbx_center_list
        processed_data_dict['ego']['object_bbx_mask'] = mask
        processed_data_dict['ego']['object_ids'] = object_ids_list
        processed_data_dict['ego']['anchor_box'] = self.post_processor.generate_anchor_box()
        processed_data_dict['ego']['processed_lidar'] = \
            self.merge_features_to_dict(agents_image_list)
        processed_data_dict['ego']['ego_lidar_pose'] = ego_lidar_pose

        if self.visualize:
            processed_data_dict['ego']['origin_lidar'] = \
                np.vstack(projected_lidar_list)

        # generate targets label for FocalComm
        if self.anchor_free:
            # For anchor-free detection (FocalComm), object_bbx_center should be (N, 8)
            # with format [x, y, z, dx, dy, dz, yaw, class_label]
            label_dict = self.post_processor.generate_label(
                gt_box_center=object_bbx_center_list,
                mask=mask,
                # Pass any additional parameters needed
            )
            processed_data_dict['ego'].update(label_dict)

        return processed_data_dict

    def collate_batch_train(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # during training, we only care about ego's lidar and label
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        processed_lidar_list = []
        object_ids = []
        ego_lidar_pose = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            processed_lidar_list.append(ego_dict['processed_lidar'])
            object_ids.append(ego_dict['object_ids'])
            ego_lidar_pose.append(ego_dict['ego_lidar_pose'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

        # convert to numpy, (B, max_num, 8)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask})

        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(processed_lidar_list)
        output_dict['ego'].update({'processed_lidar': processed_lidar_torch_dict})

        # For anchor-free FocalComm
        if self.anchor_free:
            # object_bbx_center should already be (B, N, 8) with class labels
            label_torch_dict = self.post_processor.collate_batch(batch)
            output_dict['ego'].update(label_torch_dict)

        # object id
        output_dict['ego'].update({'object_ids': object_ids})
        output_dict['ego'].update({'ego_lidar_pose': ego_lidar_pose})

        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict

    def collate_batch_test(self, batch):
        """
        Customized collate function for pytorch dataloader during testing
        for intermediate fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # currently, we only support batch size of 1 during testing
        assert len(batch) <= 1, "Batch size > 1 is not supported during testing!"
        
        output_dict = self.collate_batch_train(batch)
        if output_dict is None:
            return None

        # add transformation matrix
        transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()
        output_dict['ego'].update({'transformation_matrix': transformation_matrix_torch})

        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        pred_score : torch.Tensor
            The tensor of prediction score after NMS.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)

        return pred_box_tensor, pred_score

    def pad_object_to_max_num(self, object_bbx_center):
        """
        Pad the object list to max_num.

        Parameters
        ----------
        object_bbx_center : np.ndarray
            Shape: (n, 8) for anchor-free

        Returns
        -------
        object_bbx_center_padded : np.ndarray
            Shape: (max_num, 8)
        mask : np.ndarray
            Shape: (max_num,)
        """
        max_num = self.params['postprocess']['max_num']
        n_obj = len(object_bbx_center)
        
        if n_obj > max_num:
            # Randomly select max_num objects
            indices = np.random.choice(n_obj, max_num, replace=False)
            object_bbx_center = object_bbx_center[indices]
            mask = np.ones(max_num)
        else:
            # Pad with zeros
            object_bbx_center_padded = np.zeros((max_num, 8))
            mask = np.zeros(max_num)
            object_bbx_center_padded[:n_obj] = object_bbx_center
            mask[:n_obj] = 1
            object_bbx_center = object_bbx_center_padded
            
        return object_bbx_center, mask

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """Merge features from different CAVs."""
        merged_feature_dict = OrderedDict()
        
        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)
        
        return merged_feature_dict


def mask_ego_points(lidar_np):
    """
    Remove points that are within the ego vehicle itself.
    
    Parameters
    ----------
    lidar_np : np.ndarray
        Lidar points in shape (N, 4)
        
    Returns
    -------
    lidar_np : np.ndarray
        Filtered lidar points
    """
    # Define ego vehicle dimensions (typical car dimensions)
    ego_x_range = [-2.5, 2.5]  # 5m length
    ego_y_range = [-1.0, 1.0]  # 2m width
    ego_z_range = [-1.5, 0.5]  # Height range
    
    # Create mask for points outside ego vehicle
    mask = ~((lidar_np[:, 0] > ego_x_range[0]) & 
             (lidar_np[:, 0] < ego_x_range[1]) &
             (lidar_np[:, 1] > ego_y_range[0]) & 
             (lidar_np[:, 1] < ego_y_range[1]) &
             (lidar_np[:, 2] > ego_z_range[0]) & 
             (lidar_np[:, 2] < ego_z_range[1]))
    
    return lidar_np[mask]