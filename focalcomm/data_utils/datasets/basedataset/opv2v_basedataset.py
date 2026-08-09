# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
# Modified by: Dereje Shenkut <derejeshenkut@gmail.com> for FocalComm

import os
from collections import OrderedDict
import cv2
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import json
import random
import focalcomm.utils.pcd_utils as pcd_utils
from focalcomm.data_utils.augmentor.data_augmentor import DataAugmentor
from focalcomm.hypes_yaml.yaml_utils import load_yaml
from focalcomm.utils.camera_utils import load_camera_data
from focalcomm.utils.transformation_utils import x1_to_x2
from focalcomm.data_utils.pre_processor import build_preprocessor
from focalcomm.data_utils.post_processor import build_postprocessor

SCENARIO_BLACKLIST = [
    "2021_08_24_11_37_54"  # Skip this problematic scenario
]

class OPV2VBaseDataset(Dataset):
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.use_hdf5 = True

        
        if params['preprocess'].get('num_class', 3) == 1:
            self.class_names = ['vehicle'] 
        elif params['preprocess'].get('num_class', 3) == 2:
            self.class_names = ['vehicle', 'truck']  
        else:
            self.class_names = ['vehicle', 'pedestrian', 'truck']  

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train, self.class_names)
        
        if 'data_augment' in params: 
            self.data_augmentor = DataAugmentor(params['data_augment'], train)
        else:
            self.data_augmentor = None

        if self.train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']
        self.root_dir = root_dir 
        
        print("Dataset dir:", root_dir)

        if 'train_params' not in params or \
                'max_cav' not in params['train_params']:
            self.max_cav = 5
        else:
            self.max_cav = params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        self.label_type = params['label_type'] 
        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
                                            else self.generate_object_center_camera
        self.generate_object_center_single = self.generate_object_center # will it follows 'self.generate_object_center' when 'self.generate_object_center' change?

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        # by default, we load lidar, camera and metadata. But users may
        # define additional inputs/tasks
        self.add_data_extension = \
            params['add_data_extension'] if 'add_data_extension' \
                                            in params else []

        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False

        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x)) and x not in SCENARIO_BLACKLIST])
        
        self.scenario_folders = scenario_folders
        self.reinitialize()


    def reinitialize(self):
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(self.scenario_folders):
            self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            if self.train:
                cav_list = [x for x in os.listdir(scenario_folder)
                            if os.path.isdir(
                        os.path.join(scenario_folder, x))]
            else:
                cav_list = ['ego_vehicle']
            assert len(cav_list) > 0

            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j >= self.max_cav:
                    print('too many cavs')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml') and 'additional' not in x])

                timestamps = self.extract_timestamps(yaml_files)

                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()

                    yaml_file = os.path.join(cav_path,
                                            timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path,
                                             timestamp + '.pcd')
                    
                    self.scenario_database[i][cav_id][timestamp]['yaml'] = \
                        yaml_file
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                        lidar_file
                    
                    if self.load_camera_file:
                        camera_files = \
                            sorted([os.path.join(cav_path, x)
                                    for x in os.listdir(cav_path) if
                                    x.endswith('.png') and timestamp in x])
                        self.scenario_database[i][cav_id][timestamp]['camera0'] = \
                            camera_files
                        camera_files = \
                            sorted([os.path.join(cav_path, x)
                                    for x in os.listdir(cav_path) if
                                    x.endswith('.jpg') and timestamp in x])
                        self.scenario_database[i][cav_id][timestamp]['camera1'] = \
                            camera_files
                        camera_files = \
                            sorted([os.path.join(cav_path, x)
                                    for x in os.listdir(cav_path) if
                                    x.endswith('.jpeg') and timestamp in x])
                        self.scenario_database[i][cav_id][timestamp]['camera2'] = \
                            camera_files
                
                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the
                # scene.
                if j == 0:
                    # we regard the time stamp length as the scene length
                    self.scenario_database[i]['scene_len'] = len(timestamps)
                    self.len_record.append(len(timestamps))

    def generate_object_center_lidar(self, cav_contents, reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 8) for anchor-free detection:
        [x, y, z, dx, dy, dz, yaw, class_label].

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
        return self.post_processor.generate_object_center(
            cav_contents, reference_lidar_pose
        )

    def generate_object_center_camera(self, cav_contents, reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 8) for anchor-free detection.
        Camera view version.
        """
        return self.post_processor.generate_object_center(
            cav_contents, reference_lidar_pose
        )

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
        pass

    def __len__(self):
        return sum(self.len_record)

    def __getitem__(self, idx):
        pass

    def extract_timestamps(self, yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps

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
        pass

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 8) or (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw, (class)

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask

    def get_unique_scenario_name(self, scenario_id):
        """Get the unique scenario folder name for OPV2V"""
        scenario_name = self.scenario_folders[scenario_id].split('/')[-1]
        return scenario_name