# -*- coding: utf-8 -*-
# Author: Dereje Shenkut <derejeshenkut@gmail.com>
# Wrapper to use DAIR-V2X with FocalCommIntermediateDataset (anchor-free)

"""
DAIR-V2X wrapper for FocalComm anchor-free intermediate fusion
"""
from focalcomm.data_utils.datasets.basedataset.dairv2x_basedataset import DAIRV2XBaseDataset
from collections import OrderedDict
import numpy as np
import torch
from focalcomm.utils.transformation_utils import x1_to_x2, x_to_world, get_pairwise_transformation
from focalcomm.utils import box_utils
from focalcomm.utils.camera_utils import (
    sample_augmentation, 
    img_transform, 
    normalize_img, 
    img_to_tensor)

from focalcomm.utils.common_utils import merge_features_to_dict
from focalcomm.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)
import math

class DAIRV2XIntermediateFusionDataset(DAIRV2XBaseDataset):
    """
    Minimal wrapper to make DAIR-V2X work with FocalComm's anchor-free approach.
    Inherits from DAIRV2XBaseDataset and adds FocalCommIntermediateDataset functionality.
    
    DAIR-V2X cooperative labels only contain vehicle types (car, van, truck, bus).
    Standard evaluation uses single-class vehicle detection (num_class: 1).
    Class mapping is handled by the postprocessor based on num_class in the config.
    """
    
    def __init__(self, params, visualize, train=True):
        super().__init__(params, visualize, train)
        
        # Copy the anchor-free specific initialization from FocalCommIntermediateDataset
        self.proj_first = True
        if 'proj_first' in params['fusion']['args'] and \
                not params['fusion']['args']['proj_first']:
            self.proj_first = False

        self.kd_flag = params.get('kd_flag', False)
        
    def get_item_single_car(self, selected_cav_base, ego_cav_base):
        """
        Process a single CAV's information for the train/test pipeline.


        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
            including 'params', 'camera_data'
        ego_pose : list, length 6
            The ego vehicle lidar pose under world coordinate.
        ego_pose_clean : list, length 6
            only used for gt box generation

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}
        ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose']

        # calculate the transformation matrix
        transformation_matrix = \
            x1_to_x2(selected_cav_base['params']['lidar_pose'],
                    ego_pose) # T_ego_cav
        # transformation_matrix_clean = \
        #     x1_to_x2(selected_cav_base['params']['lidar_pose_clean'],
        #             ego_pose_clean)
        
        # lidar
        if self.load_lidar_file or self.visualize:
            # process lidar
            lidar_np = selected_cav_base['lidar_np']
            lidar_np = shuffle_points(lidar_np)
            # remove points that hit itself
            lidar_np = mask_ego_points(lidar_np)
            # project the lidar to ego space
            # x,y,z in ego space
            projected_lidar = \
                box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                            transformation_matrix)
            if self.proj_first:
                lidar_np[:, :3] = projected_lidar

            if self.visualize:
                # filter lidar
                selected_cav_processed.update({'projected_lidar': projected_lidar})

            if self.kd_flag:
                lidar_proj_np = copy.deepcopy(lidar_np)
                lidar_proj_np[:,:3] = projected_lidar

                selected_cav_processed.update({'projected_lidar': lidar_proj_np})

            processed_lidar = self.pre_processor.preprocess(lidar_np)
            selected_cav_processed.update({'processed_features': processed_lidar})

        # generate targets label single GT, note the reference pose is itself.
        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
            [selected_cav_base], selected_cav_base['params']['lidar_pose']
        )


        if self.load_camera_file:
            camera_data_list = selected_cav_base["camera_data"]

            params = selected_cav_base["params"]
            imgs = []
            rots = []
            trans = []
            intrins = []
            extrinsics = []
            post_rots = []
            post_trans = []

            for idx, img in enumerate(camera_data_list):
                camera_to_lidar, camera_intrinsic = self.get_ext_int(params, idx)

                intrin = torch.from_numpy(camera_intrinsic)
                rot = torch.from_numpy(
                    camera_to_lidar[:3, :3]
                )  # R_wc, we consider world-coord is the lidar-coord
                tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc

                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)

                img_src = [img]

                # depth
                if self.load_depth_file:
                    depth_img = selected_cav_base["depth_data"][idx]
                    img_src.append(depth_img)
                else:
                    depth_img = None

                # data augmentation
                resize, resize_dims, crop, flip, rotate = sample_augmentation(
                    self.data_aug_conf, self.train
                )
                img_src, post_rot2, post_tran2 = img_transform(
                    img_src,
                    post_rot,
                    post_tran,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                # for convenience, make augmentation matrices 3x3
                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2

                # decouple RGB and Depth

                img_src[0] = normalize_img(img_src[0])
                if self.load_depth_file:
                    img_src[1] = img_to_tensor(img_src[1]) * 255

                imgs.append(torch.cat(img_src, dim=0))
                intrins.append(intrin)
                extrinsics.append(torch.from_numpy(camera_to_lidar))
                rots.append(rot)
                trans.append(tran)
                post_rots.append(post_rot)
                post_trans.append(post_tran)
                

            selected_cav_processed.update(
                {
                "image_inputs": 
                    {
                        "imgs": torch.stack(imgs), # [Ncam, 3or4, H, W]
                        "intrins": torch.stack(intrins),
                        "extrinsics": torch.stack(extrinsics),
                        "rots": torch.stack(rots),
                        "trans": torch.stack(trans),
                        "post_rots": torch.stack(post_rots),
                        "post_trans": torch.stack(post_trans),
                    }
                }
            )


        selected_cav_processed.update(
            {
                "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                "object_bbx_mask": object_bbx_mask,
                "object_ids": object_ids,
                'transformation_matrix': transformation_matrix,
                # 'transformation_matrix_clean': transformation_matrix_clean
            }
        )


        return selected_cav_processed



    def __getitem__(self, idx):
        """
        Adapted from FocalCommIntermediateDataset to work with DAIR-V2X base data.
        """
        # Get base data from DAIR-V2X
        base_data_dict = self.retrieve_base_data(idx)
        
        # Following the same structure as FocalCommIntermediateDataset
        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}
        
        # Generate augmentation parameters
        # flip, noise_rotation, noise_scale = self.generate_augment()
        
        # Find ego vehicle
        ego_id = -1
        ego_lidar_pose = []
        ego_cav_base = None

        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                ego_cav_base = cav_content
                break
        
        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0
        
        # Get pairwise transformation
        # pairwise_t_matrix = self.get_pairwise_transformation(base_data_dict, self.max_cav)
        
        agents_image_inputs = []
        processed_features = []
        object_stack = []
        object_id_stack = []
        too_far = []
        lidar_pose_list = []
        # lidar_pose_clean_list = []
        cav_id_list = []
        projected_lidar_clean_list = [] # disconet
        
        if self.visualize or self.kd_flag:
            projected_lidar_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                        ego_lidar_pose[0]) ** 2 + (
                                selected_cav_base['params'][
                                    'lidar_pose'][1] - ego_lidar_pose[
                                    1]) ** 2)

            # if distance is too far, we will just skip this agent
            if distance > self.params['comm_range']:
                too_far.append(cav_id)
                continue

            # lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
            lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
            cav_id_list.append(cav_id)   

        for cav_id in too_far:
            base_data_dict.pop(cav_id)

        pairwise_t_matrix = \
            get_pairwise_transformation(base_data_dict,
                                            self.max_cav,
                                            self.proj_first)

        lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
        # lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]
        
        # merge preprocessed features from different cavs into the same dict
        cav_num = len(cav_id_list)
        
        
        for _i, cav_id in enumerate(cav_id_list):
            selected_cav_base = base_data_dict[cav_id]
            selected_cav_processed = self.get_item_single_car(
                selected_cav_base,
                ego_cav_base)
                
            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']
            if self.load_lidar_file:
                processed_features.append(
                    selected_cav_processed['processed_features'])
            if self.load_camera_file:
                agents_image_inputs.append(
                    selected_cav_processed['image_inputs'])

            if self.visualize or self.kd_flag:
                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])


        # if self.kd_flag:
        #     stack_lidar_np = np.vstack(projected_lidar_stack)
        #     stack_lidar_np = mask_points_by_range(stack_lidar_np,
        #                                 self.params['preprocess'][
        #                                     'cav_lidar_range'])
        #     stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)
        #     processed_data_dict['ego'].update({'teacher_processed_lidar':
        #     stack_feature_processed})

        
        # exclude all repetitive objects    
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

    
        gt_boxes = object_stack
        valid_object_ids = [object_id_stack[i] for i in unique_indices]


        if self.load_lidar_file:
            merged_feature_dict = merge_features_to_dict(processed_features)
            processed_data_dict['ego'].update({'processed_lidar': merged_feature_dict})
        if self.load_camera_file:
            merged_image_inputs_dict = merge_features_to_dict(agents_image_inputs, merge='stack')
            processed_data_dict['ego'].update({'image_inputs': merged_image_inputs_dict})
        
        processed_data_dict['ego'].update({
            'gt_boxes': gt_boxes,  # Only valid boxes
            'object_ids': valid_object_ids,
            'cav_num': cav_num,
            'pairwise_t_matrix': pairwise_t_matrix,
            'lidar_poses': lidar_poses, 
            # 'lidar_poses_clean': lidar_poses_clean,
        })
        
        if self.visualize:
            processed_data_dict['ego'].update(
                {'origin_lidar': np.vstack(projected_lidar_stack)})

        processed_data_dict['ego'].update({'sample_idx': idx,
                                                'cav_id_list': cav_id_list})      
        return processed_data_dict
    



    

    
    def collate_batch_train(self, batch):
        """Collate batch for anchor-free training."""
        # Use the same logic as FocalCommIntermediateDataset
        output_dict = {'ego': {}}
        
        gt_boxes_list = []
        object_ids_list = []
        processed_lidar_list = []
        image_inputs_list = []
        record_len = []
        
        lidar_pose_list = []
        origin_lidar = []
        # lidar_pose_clean_list = []

        # pairwise transformation matrix
        pairwise_t_matrix_list = []

        # disconet
        teacher_processed_lidar_list = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            gt_boxes_list.append(ego_dict['gt_boxes'])
            object_ids_list.append(ego_dict['object_ids'])
            lidar_pose_list.append(ego_dict['lidar_poses']) # ego_dict['lidar_pose'] is np.ndarray [N,6]
            # lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])
            if self.load_lidar_file:
                processed_lidar_list.append(ego_dict['processed_lidar'])
            if self.load_camera_file:
                image_inputs_list.append(ego_dict['image_inputs']) # different cav_num, ego_dict['image_inputs'] is dict.
            
            record_len.append(ego_dict['cav_num'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

            if self.kd_flag:
                teacher_processed_lidar_list.append(ego_dict['teacher_processed_lidar'])

        gt_boxes_list = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x 
                         for x in gt_boxes_list]
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        
        if self.load_lidar_file:
            merged_feature_dict = merge_features_to_dict(processed_lidar_list)
            processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(merged_feature_dict)
            output_dict['ego'].update({'processed_lidar': processed_lidar_torch_dict})

        if self.load_camera_file:
            merged_image_inputs_dict = merge_features_to_dict(image_inputs_list, merge='cat')

            output_dict['ego'].update({'image_inputs': merged_image_inputs_dict})
        
        lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
        # lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

        output_dict['ego'].update({
            'gt_boxes': gt_boxes_list,  # List of variable-size tensors
            'processed_lidar': processed_lidar_torch_dict,
            'record_len': record_len,
            'object_ids': object_ids_list,
            'lidar_pose': lidar_pose,
            # 'lidar_pose_clean': lidar_pose_clean,
            'pairwise_t_matrix': pairwise_t_matrix
        })
        
        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        if self.kd_flag:
            teacher_processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(teacher_processed_lidar_list)
            output_dict['ego'].update({'teacher_processed_lidar':teacher_processed_lidar_torch_dict})

        
        return output_dict
    
    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)
        if output_dict is None:
            return None
        transformation_matrix_torch = \
            torch.from_numpy(np.identity(4)).float()
        transformation_matrix_clean_torch = \
            torch.from_numpy(np.identity(4)).float()

        output_dict['ego'].update({'transformation_matrix':
                                    transformation_matrix_torch,
                                    'transformation_matrix_clean':
                                    transformation_matrix_clean_torch,})

        output_dict['ego'].update({
            "sample_idx": batch[0]['ego']['sample_idx'],
            "cav_id_list": batch[0]['ego']['cav_id_list']
        })
        return output_dict
    
    def post_process(self, data_dict, output_dict):
        """Process model outputs to bounding boxes."""
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor, gt_label_tensor = \
            self.post_processor.generate_gt_bbx(data_dict)
        
        return pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor
    

    
