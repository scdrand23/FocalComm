import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from einops import rearrange, repeat

from focalcomm.models.sub_modules.mean_vfe import MeanVFE
from focalcomm.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from focalcomm.models.sub_modules.height_compression import HeightCompression
from focalcomm.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from focalcomm.models.sub_modules.downsample_conv import DownsampleConv
from focalcomm.models.sub_modules.naive_compress import NaiveCompressor

from focalcomm.models.sub_modules.focalcomm_transfusion_head import TransFusionHead

from focalcomm.models.fuse_modules.ermp_fusion_modules import ERMVPFusionEncoder
from focalcomm.models.fuse_modules.fuse_utils import regroup
from focalcomm.models.sub_modules.sampler import SortSampler
from focalcomm.models.sub_modules.cluster import merge_tokens,cluster_dpc_knn,index_points
from einops import repeat
import math

def get_selected_cav_feature(x, record_len,selected_cav_id_list):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    out = []
    idx = 0
    for xx in split_x:
        xx = xx[selected_cav_id_list[idx]].unsqueeze(0)
        out.append(xx)
        idx = idx + 1 
    return torch.cat(out, dim=0)

def get_ego_feature(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    out = []
    for xx in split_x:
        xx = xx[0].unsqueeze(0)
        out.append(xx)
    return torch.cat(out, dim=0)

def get_fused_ego_feature(x):
    B,N,C,H,W = x.shape
    out = []
    for b in range(B):
       xx = x[b][0].unsqueeze(0)
       out.append(xx)
    return torch.cat(out, dim=0)

    
class ERMVP(nn.Module):
    def __init__(self, args):
        super(ERMVP, self).__init__()
        # torch.backends.cudnn.benchmark = True
        # torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cudnn.allow_tf32 = True
        self.max_cav = args['max_cav']
        # Voxel VFE
        self.mean_vfe = MeanVFE(args['mean_vfe'], 4)
        
        # 3D Sparse Backbone
        self.backbone_3d = VoxelBackBone8x(args['backbone_3d'], 4, args['grid_size'])
        
        # Height compression
        self.height_compression = HeightCompression(args['height_compression'])
        
        # 2D backbone
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 256)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

  

        self.topk_ratio = args['comm']['topk_ratio']
        self.cluster_sample_ratio = args['comm']['cluster_sample_ratio']
        # Determine input dim based on whether shrink_header is used
        sampler_input_dim = args['shrink_header']['dim'][0] if self.shrink_flag else sum(args['base_bev_backbone']['num_upsample_filter'])
        self.sampler = SortSampler(topk_ratio=self.topk_ratio, input_dim=sampler_input_dim, score_pred_net='2layer-fc-256')
        self.fusion_net = ERMVPFusionEncoder(args['ermvp_fusion'])
        
        self.dense_head = TransFusionHead(args['dense_head'])


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
        
        # VFE: n, 4 -> n, c
        batch_dict = self.mean_vfe(batch_dict)
        
        # 3D Sparse Backbone
        batch_dict = self.backbone_3d(batch_dict)
        
        # Height compression: 3D -> 2D
        batch_dict = self.height_compression(batch_dict)
        
        # 2D Backbone
        batch_dict = self.backbone(batch_dict)
        
        # Get spatial features from height compression
        spatial_features_2d = batch_dict['spatial_features']
        
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # [1,384,60,180]

        #         # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
            
        # Skip communication/clustering part - use features directly
        ego_features = get_ego_feature(spatial_features_2d,record_len)
        
        regroup_feature, mask = regroup(spatial_features_2d,
                                        record_len,
                                        self.max_cav)
        # [1,1,1,1,2]
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # [1,60,180,1,2]
        com_mask = repeat(com_mask,
                          'b h w c l -> b (h new_h) (w new_w) c l',
                          new_h=regroup_feature.shape[3],
                          new_w=regroup_feature.shape[4])
        # breakpoint()
        fused_feature = self.fusion_net(regroup_feature, com_mask)
        batch_dict['spatial_features_2d'] = fused_feature.contiguous()
        # breakpoint()
        output_dict = self.dense_head(batch_dict)
        return output_dict


        
        