# Anonymous author: xyz@gmail.com

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from focalcomm.models.sub_modules.mean_vfe import MeanVFE
from focalcomm.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from focalcomm.models.sub_modules.height_compression import HeightCompression
from focalcomm.models.sub_modules.focalcomm_transfusion_head import TransFusionHead
from focalcomm.models.fuse_modules.fuse_utils import regroup
from focalcomm.models.sub_modules.himv3 import HardInstanceMinerV3
from focalcomm.models.fuse_modules.qaffv3 import QAFFV3
from focalcomm.models.sub_modules.naive_compress import NaiveCompressor


class FocalCommV3(nn.Module):
    def __init__(self, args):
        super(FocalCommV3, self).__init__()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.batch_size = args['batch_size']
        self.max_cav = args['max_cav']
        self.mean_vfe = MeanVFE(args['mean_vfe'], 4)
        self.backbone_3d = VoxelBackBone8x(args['backbone_3d'],4, args['grid_size'])
        self.height_compression = HeightCompression(args['height_compression'])
        him_args, qaff_args, head_cfg_args = args['him'], args['qaff'], args['dense_head']
        
        him_args['point_cloud_range'] = args['lidar_range']
        him_args['voxel_size'] = args['voxel_size']
        him_args['feature_map_stride'] = args.get('feature_map_stride', 8)
        
        self.him = HardInstanceMinerV3(**him_args)
        self.compression = False
        # print(args['compression'])
        if "compression" in args:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression']) 
        self.qaff = QAFFV3(**qaff_args)
        self.head = TransFusionHead(head_cfg_args)
        self.num_classes = args['num_classes']
        self.use_him = True
        self.sample_id = 0 
        self.viz_dir = "him_visualization"
        os.makedirs(self.viz_dir, exist_ok=True)
        
    def enable_him_visualization(self, viz_path=None):
        if viz_path is None:
            viz_path = os.path.join(os.getcwd(), "him_debug_visualization", datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        self.him.enable_visualization(viz_path)
        print(f"HIM visualization enabled at: {viz_path}")
        return viz_path

 
    def forward(self, data_dict):
        
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        # breakpoint()
        gt_boxes = data_dict.get('gt_boxes', None) if self.training else None
        
        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'batch_size': torch.sum(record_len).cpu().numpy(),
                      'record_len': record_len}
        batch_dict = self.mean_vfe(batch_dict)       
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.height_compression(batch_dict)       
        spatial_features = batch_dict['spatial_features']
        if self.compression:
            spatial_features = self.naive_compressor(spatial_features)       
        regroup_feature, mask = regroup(spatial_features,
                                      record_len,
                                      self.max_cav)        
        B, K, C, H, W = regroup_feature.shape
        # breakpoint()
        # Process all agent features through HIM for collaborative hard instance mining
        him_outputs = self.him(spatial_features, gt_boxes=gt_boxes, record_len=record_len)
        # breakpoint()
        query_features = him_outputs['query_features']
        # breakpoint()
        fused_features = self.qaff(
            query_features=regroup(query_features, record_len, self.max_cav)[0],
            agent_features=regroup_feature,
            record_len=record_len
        )
        
        batch_dict['spatial_features_2d'] = fused_features
        preds_dict = self.head(batch_dict)
        # breakpoint()
        if 'stage_predictions' in him_outputs:
            preds_dict['stage_predictions'] = him_outputs['stage_predictions']
            
        if 'stage_masks' in him_outputs:
            preds_dict['stage_masks'] = him_outputs['stage_masks']
        
        preds_dict['record_len'] = record_len
        
        self.sample_id += 1
        return preds_dict

