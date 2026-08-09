import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveCompressor(nn.Module):
    def __init__(self, in_channels, compress_ratio=0.25, importance_threshold=0.1):
        super(SelectiveCompressor, self).__init__()
        self.in_channels = in_channels
        self.compress_ratio = compress_ratio
        self.importance_threshold = importance_threshold
        
        self.compress_channels = int(in_channels * compress_ratio)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, self.compress_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.compress_channels),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(self.compress_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.importance_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, features, hard_instance_mask=None):
        B, C, H, W = features.shape
        
        importance_map = torch.sigmoid(self.importance_conv(features))
        
        if hard_instance_mask is not None:
            if hard_instance_mask.dim() == 3:
                hard_instance_mask = hard_instance_mask.unsqueeze(1)
            elif hard_instance_mask.dim() == 2:
                hard_instance_mask = hard_instance_mask.unsqueeze(0).unsqueeze(0)
            
            hard_mask_resized = F.interpolate(
                hard_instance_mask.float(), 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
            importance_map = importance_map * (1 + hard_mask_resized)
            importance_map = torch.clamp(importance_map, 0, 1)
        
        spatial_mask = (importance_map > self.importance_threshold).float()
        
        compressed = self.encoder(features)
        
        compressed_masked = compressed * spatial_mask
        
        reconstructed = self.decoder(compressed_masked)
        
        info = {
            'original_channels': C,
            'compressed_channels': self.compress_channels,
            'compression_ratio': self.compress_ratio,
            'active_ratio': spatial_mask.mean().item(),
            'importance_map': importance_map
        }
        
        return compressed_masked, reconstructed, info

class MultiScaleSelectiveCompressor(nn.Module):
    def __init__(self, channel_list, compress_ratios=None, base_ratio=0.25):
        super(MultiScaleSelectiveCompressor, self).__init__()
        
        if compress_ratios is None:
            compress_ratios = [base_ratio * (2 ** i) for i in range(len(channel_list))]
            compress_ratios = [min(r, 0.5) for r in compress_ratios]
        
        self.compressors = nn.ModuleList()
        for channels, ratio in zip(channel_list, compress_ratios):
            self.compressors.append(SelectiveCompressor(channels, ratio))
    
    def forward(self, feature_list, hard_instance_masks=None):
        compressed_list = []
        reconstructed_list = []
        info_list = []
        
        if hard_instance_masks is None:
            hard_instance_masks = [None] * len(feature_list)
        
        for feat, mask, compressor in zip(feature_list, hard_instance_masks, self.compressors):
            compressed, reconstructed, info = compressor(feat, mask)
            compressed_list.append(compressed)
            reconstructed_list.append(reconstructed)
            info_list.append(info)
        
        return compressed_list, reconstructed_list, info_list