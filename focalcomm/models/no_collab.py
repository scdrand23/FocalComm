import torch
import torch.nn as nn
from focalcomm.models.sub_modules.mean_vfe import MeanVFE
from focalcomm.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from focalcomm.models.sub_modules.height_compression import HeightCompression
from focalcomm.models.sub_modules.focalcomm_transfusion_head import TransFusionHead
from focalcomm.models.fuse_modules.fuse_utils import regroup


class NoCollab(nn.Module):
    """
    No Collaboration baseline model - uses only ego vehicle data
    """
    def __init__(self, args):
        super(NoCollab, self).__init__()
        self.batch_size = args['batch_size']
        self.max_cav = args['max_cav']
        self.mean_vfe = MeanVFE(args['mean_vfe'], 4)
        self.backbone_3d = VoxelBackBone8x(args['backbone_3d'], 4, args['grid_size'])
        self.height_compression = HeightCompression(args['height_compression'])
        self.head = TransFusionHead(args['dense_head'])

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        
        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'batch_size': torch.sum(record_len).cpu().numpy(),
                      'record_len': record_len}
        
        batch_dict = self.mean_vfe(batch_dict)       
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.height_compression(batch_dict)       
        
        spatial_features = batch_dict['spatial_features']
        regroup_feature, mask = regroup(spatial_features,
                                      record_len,
                                      self.max_cav)        
        B, K, C, H, W = regroup_feature.shape         

        # Use only ego features (no collaboration)
        ego_features = regroup_feature[:, 0] 
        
        batch_dict['spatial_features_2d'] = ego_features
        preds_dict = self.head(batch_dict)
        
        # Pass record_len for consistency with loss computation
        preds_dict['record_len'] = record_len
        
        return preds_dict