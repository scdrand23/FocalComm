import torch
import torch.nn as nn
import torch.nn.functional as F
from focalcomm.models.sub_modules.focalcomm_transfusion_head import TransFusionHead

class HardInstanceMiner(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, num_classes=3, num_stages=3, 
                 attention_decay_factor=2, base_threshold=0.5, head_cfg_args=None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.attention_decay_factor = attention_decay_factor
        self.base_threshold = base_threshold
        
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        self.stage_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_stages)
        ])
        
        self.stage_heads = nn.ModuleList([
            TransFusionHead(head_cfg_args) for _ in range(num_stages)
        ])
        
        self.query_proj = nn.Conv2d(hidden_dim * num_stages, in_channels * num_stages, kernel_size=1)
        
    def forward(self, ego_features):
        B, C, H, W = ego_features.shape
        
        if C != self.hidden_dim:
            features = self.input_proj(ego_features)
        else:
            features = ego_features
            
        stage_outputs = []
        stage_heatmaps = []
        stage_peaks = []
        accumulated_mask = torch.zeros(B, self.num_classes, H, W, device=features.device)
        
        for stage_idx in range(self.num_stages):
            stage_feat = self.stage_convs[stage_idx](features)
            stage_outputs.append(stage_feat)
            
            batch_dict_stage = {'spatial_features_2d': stage_feat}
            stage_pred = self.stage_heads[stage_idx](batch_dict_stage)
            
            if 'dense_heatmap' in stage_pred:
                heatmap = stage_pred['dense_heatmap']
                stage_heatmaps.append(heatmap)
                
                conf = torch.sigmoid(heatmap)
                threshold = self.base_threshold * (self.attention_decay_factor ** stage_idx)
                peaks = (conf > threshold).float()
                stage_peaks.append(peaks)
                
                accumulated_mask = torch.maximum(accumulated_mask, peaks)
                
                attention_weight = 1.0 - accumulated_mask
                features = features * attention_weight.max(dim=1, keepdim=True)[0]
        
        query_features = torch.cat(stage_outputs, dim=1)
        query_features = self.query_proj(query_features)
        
        return {
            'query_features': query_features,
            'stage_heatmaps': stage_heatmaps,
            'stage_peaks': stage_peaks,
            'accumulated_mask': accumulated_mask
        }