import torch
import torch.nn as nn
from focalcomm.models.sub_modules.mean_vfe import MeanVFE
from focalcomm.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from focalcomm.models.sub_modules.height_compression import HeightCompression
from focalcomm.models.sub_modules.focalcomm_transfusion_head import TransFusionHead
from focalcomm.models.fuse_modules.fuse_utils import regroup
from focalcomm.models.sub_modules.him import HardInstanceMiner
from focalcomm.models.sub_modules.naive_compress import NaiveCompressor
from focalcomm.models.fuse_modules.qaff import QAFF


class FocalComm(nn.Module):
    def __init__(self, args: dict):
        super(FocalComm, self).__init__()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.batch_size = args['batch_size']
        self.max_cav = args['max_cav']
        self.mean_vfe = MeanVFE(args['mean_vfe'], 4)
        self.backbone_3d = VoxelBackBone8x(args['backbone_3d'], 4, args['grid_size'])
        self.height_compression = HeightCompression(args['height_compression'])
        him_args, qaff_args, head_cfg_args = args['him'], args['qaff'], args['dense_head']
        self.him = HardInstanceMiner(**him_args)
        self.compression = False
        if "compression" in args:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])
        self.qaff = QAFF(**qaff_args)
        self.head = TransFusionHead(head_cfg_args)
        self.num_classes = args['num_classes']

    def forward(self, data_dict: dict) -> dict:
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
        if self.compression:
            spatial_features = self.naive_compressor(spatial_features)
        regroup_feature, mask = regroup(spatial_features,
                                        record_len,
                                        self.max_cav)
        ego_features = regroup_feature[:, 0]
        him_outputs = self.him(ego_features)
        query_features = him_outputs['query_features']
        fused_features = self.qaff(
            query_features=query_features,
            agent_features=regroup_feature,
            record_len=record_len
        )
        batch_dict['spatial_features_2d'] = fused_features
        preds_dict = self.head(batch_dict)
        if him_outputs and 'stage_heatmaps' in him_outputs:
            stage_predictions = []
            for stage_hm in him_outputs['stage_heatmaps']:
                stage_predictions.append({'dense_heatmap': stage_hm})
            preds_dict['stage_predictions'] = stage_predictions
            if 'stage_peaks' in him_outputs:
                preds_dict['stage_masks'] = him_outputs['stage_peaks']
        preds_dict['record_len'] = record_len
        return preds_dict
