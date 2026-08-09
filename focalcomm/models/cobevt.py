# Anchor-free version by anynomous author

import torch
import torch.nn as nn
from focalcomm.models.sub_modules.mean_vfe import MeanVFE
from focalcomm.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from focalcomm.models.sub_modules.height_compression import HeightCompression
from focalcomm.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from focalcomm.models.sub_modules.focalcomm_transfusion_head import TransFusionHead
from focalcomm.models.fuse_modules.fuse_utils import regroup
from focalcomm.models.fuse_modules.fusion_in_one import CoBEVT
from focalcomm.utils.transformation_utils import normalize_pairwise_tfm
from focalcomm.models.sub_modules.naive_compress import NaiveCompressor
from focalcomm.models.sub_modules.downsample_conv import DownsampleConv

class Cobevt(nn.Module):
    """
    Co-Event Fusion implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(Cobevt, self).__init__()

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.batch_size = args['batch_size']
        self.max_cav = args['max_cav']
        self.mean_vfe = MeanVFE(args['mean_vfe'], 4)
        self.backbone_3d = VoxelBackBone8x(args['backbone_3d'],4, args['grid_size'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 256)
        self.voxel_size = args['voxel_size']
        self.num_levels = len(args['base_bev_backbone']['layer_nums'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.compression = False
        if "compression" in args:
            self.compression = True
            self.naive_compressor = NaiveCompressor(64, args['compression'])

        self.height_compression = HeightCompression(args['height_compression'])

        self.fusion_net = CoBEVT(args['cobevt'])

        # self.fusion_net = AttFusion(args['coalign'])
        head_args = args['dense_head']
        self.head = TransFusionHead(head_args)


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
        
        # Dimension flow: voxel features (N, 4) -> feature vectors (N, C)
        batch_dict = self.mean_vfe(batch_dict)
        
        # Dimension flow: voxel features (N, C) -> 3D sparse features -> dense features (B, C, H, W, Z)
        batch_dict = self.backbone_3d(batch_dict)
        
        # Dimension flow: 3D features (B, C, H, W, Z) -> 2D BEV features (B, C', H, W)
        batch_dict = self.height_compression(batch_dict)
        
        # calculate pairwise affine transformation matrix
        _, _, H0, W0 = batch_dict['spatial_features'].shape # original feature map shape H0, W0
        normalized_affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])

        spatial_features = batch_dict['spatial_features']

        # Dimension flow: optional compression (B, C', H, W) -> (B, C_compressed, H, W)
        if self.compression:
            spatial_features = self.naive_compressor(spatial_features)

        # Dimension flow: multi-agent fusion (B, C, H, W) -> fused features (B, C, H, W) b =4, record_Len = [2, 3, 4, 5] [k,4, 4] 
        fused_features = self.fusion_net(spatial_features, record_len, normalized_affine_matrix)
        
        # Dimension flow: optional downsampling (B, C, H, W) -> (B, C_out, H', W')
        if self.shrink_flag:
            fused_features = self.shrink_conv(fused_features)

        # Add spatial_features_2d for TransFusion head compatibility
        batch_dict['spatial_features_2d'] = fused_features.contiguous()
        # breakpoint()
        
        # Dimension flow: BEV features (B, C, H, W) -> detection outputs (cls, reg, etc.)
        output_dict = self.head(batch_dict)

        return output_dict
