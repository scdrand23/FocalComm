import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class QAFF(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.1, num_stages=3, compress_ratio=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_stages = num_stages
        self.compress_ratio = compress_ratio
        
        self.stage_self_attns = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_stages)
        ])
        
        self.stage_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.proj_k = nn.Linear(hidden_dim, hidden_dim)
        self.proj_v = nn.Linear(hidden_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
       
        self.norm_stages = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_stages)])
        self.norm_combined = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)

        self.agent_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.sample_id = 0


    def forward(self, query_features, agent_features, record_len):
        B, K, C, H, W = agent_features.shape
        
        query_features = query_features.unsqueeze(1).expand(-1, K, -1, -1, -1)
        
        valid_mask = torch.arange(K, device=record_len.device)[None, :] < record_len[:, None]
        attn_mask = ~valid_mask
        
        stage_queries = torch.chunk(query_features, self.num_stages, dim=2)
        
        stage_outputs = []
        for stage_idx in range(self.num_stages):
            stage_q = rearrange(stage_queries[stage_idx], 'b k c h w -> (b h w) k c')
            stage_q = self.norm_stages[stage_idx](stage_q)
            
            stage_out, _ = self.stage_self_attns[stage_idx](
                query=stage_q,
                key=stage_q,
                value=stage_q,
                key_padding_mask=attn_mask.repeat_interleave(H*W, dim=0)
            )
            stage_outputs.append(stage_out)
        
        stage_weights = []
        for stage_out in stage_outputs:
            stage_feat = stage_out.mean(dim=1)
            weight = self.stage_attention(stage_feat)
            stage_weights.append(weight)
        
        stage_weights = torch.stack(stage_weights, dim=-1)
        stage_weights = F.softmax(stage_weights, dim=-1)
        
        combined_queries = sum(stage_outputs[i] * stage_weights[..., i:i+1] 
                              for i in range(self.num_stages))
        combined_queries = self.norm_combined(combined_queries)
        
        agent_feats_flat = rearrange(agent_features, 'b k c h w -> (b h w) k c')
        agent_feats_flat = self.norm_kv(agent_feats_flat)
        
        keys = self.proj_k(agent_feats_flat)
        values = self.proj_v(agent_feats_flat)
        
        cross_out, _ = self.cross_attn(
            query=combined_queries,
            key=keys,
            value=values,
            key_padding_mask=attn_mask.repeat_interleave(H*W, dim=0)
        )
        
        cross_out = self.norm_out(cross_out)
        cross_out = self.mlp(cross_out)
        
        cross_out = rearrange(cross_out, '(b h w) k c -> b k c h w', b=B, h=H, w=W)
        
        agent_feats_pooled = cross_out.mean(dim=(3, 4))
        agent_weights = self.agent_attention(agent_feats_pooled)
        
        agent_weights = agent_weights * valid_mask.float().unsqueeze(-1) + (~valid_mask).float().unsqueeze(-1) * (-1e9)
        agent_weights = F.softmax(agent_weights, dim=1)
        
        fused_out = (cross_out * agent_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        
        self.sample_id += 1
        return fused_out
