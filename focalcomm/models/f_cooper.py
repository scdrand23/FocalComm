import torch
import torch.nn as nn
from focalcomm.models.sub_modules.mean_vfe import MeanVFE
from focalcomm.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from focalcomm.models.sub_modules.height_compression import HeightCompression
from focalcomm.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from focalcomm.models.sub_modules.focalcomm_transfusion_head import TransFusionHead
from focalcomm.models.fuse_modules.fuse_utils import regroup
from focalcomm.models.fuse_modules.fusion_in_one import AttFusion
from focalcomm.utils.transformation_utils import normalize_pairwise_tfm
from focalcomm.models.sub_modules.naive_compress import NaiveCompressor
from focalcomm.models.sub_modules.downsample_conv import DownsampleConv
from focalcomm.models.fuse_modules.f_cooper_fuse import SpatialFusion


class FCooper(nn.Module):
    """
    V2XViT implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(FCooper, self).__init__()

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.batch_size = args['batch_size']
        self.max_cav = args['max_cav']
        self.mean_vfe = MeanVFE(args['mean_vfe'], 4)
        self.backbone_3d = VoxelBackBone8x(args['backbone_3d'],4, args['grid_size'])
        # self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 256)
        self.voxel_size = args['voxel_size']
        # self.num_levels = len(args['base_bev_backbone']['layer_nums'])

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

        self.fusion_net = SpatialFusion()

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
        # n, 4 -> n, c
        batch_dict = self.mean_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.height_compression(batch_dict)
        # calculate pairwise affine transformation matrix


        spatial_features = batch_dict['spatial_features']



        if self.shrink_flag:
            spatial_features = self.shrink_conv(spatial_features)

        if self.compression:
            spatial_features = self.naive_compressor(spatial_features, "encoder")


        fused_features = self.fusion_net(spatial_features, record_len)
        batch_dict['spatial_features_2d'] = fused_features.contiguous()
        # breakpoint()
        output_dict = self.head(batch_dict)

        return output_dict
