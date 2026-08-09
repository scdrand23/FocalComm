import torch
import torch.nn as nn
import torch.nn.functional as F
from focalcomm.models.sub_modules.focalcomm_transfusion_head import TransFusionHead
from focalcomm.pcdet_utils.iou3d_nms import iou3d_nms_utils
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class HardInstanceMinerV3(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, num_classes=3, num_stages=3, 
                 iou_threshold=0.3, masking_strategy='pooling', head_cfg_args=None,
                 point_cloud_range=None, voxel_size=None, feature_map_stride=8, loss_module=None,
                 conf_threshold=0.3, dilation_kernel=3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.iou_threshold = iou_threshold
        self.masking_strategy = masking_strategy
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = feature_map_stride
        self.loss_module = loss_module
        self.conf_threshold = conf_threshold
        self.dilation_kernel = dilation_kernel
        
        self.debug_save_enabled = False
        self.debug_save_path = None
        self.sample_counter = 0
        
        self.stage_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_stages)
        ])
        
        self.head = TransFusionHead(head_cfg_args)
        
        self.query_proj = nn.Conv2d(hidden_dim * num_stages, hidden_dim * num_stages, kernel_size=1)
    
    def enable_visualization(self, save_path):
        self.debug_save_enabled = True
        self.debug_save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        print(f"HIMv3 visualization enabled at: {save_path}")
    
    def create_stage_visualizations(self, stage_idx, features, masked_features, stage_feat, 
                                  spatial_mask, accumulated_mask, stage_pred, record_len, sample_id):
        if not self.debug_save_enabled:
            return
        
        if not hasattr(self, 'stage_data'):
            self.stage_data = {}
            
        if sample_id not in self.stage_data:
            self.stage_data[sample_id] = {
                'stages': [],
                'agent_data': []
            }
            
        BN, C, H, W = features.shape
        total_agents = sum([x.item() if torch.is_tensor(x) else x for x in record_len])
        
        stage_agent_data = []
        agent_idx = 0
        for batch_idx, num_agents in enumerate(record_len):
            num_agents = num_agents.item() if torch.is_tensor(num_agents) else num_agents
            for agent_in_batch in range(num_agents):
                agent_data = {
                    'input_feat': features[agent_idx].mean(dim=0).detach().cpu().numpy(),
                    'masked_feat': masked_features[agent_idx].mean(dim=0).detach().cpu().numpy(),
                    'stage_feat': stage_feat[agent_idx].mean(dim=0).detach().cpu().numpy(),
                    'spatial_mask': spatial_mask[agent_idx].squeeze().detach().cpu().numpy(),
                    'accumulated_mask': accumulated_mask[agent_idx].max(dim=0)[0].detach().cpu().numpy() if not self.training else None
                }
                stage_agent_data.append(agent_data)
                agent_idx += 1
        
        self.stage_data[sample_id]['stages'].append(stage_agent_data)
        
        if len(self.stage_data[sample_id]['stages']) == self.num_stages:
            self.create_grid_visualization(sample_id, total_agents)
    
    def create_grid_visualization(self, sample_id, num_agents):
        """Create a grid visualization: rows=agents, cols=stages"""
        sample_dir = os.path.join(self.debug_save_path, f"sample_{sample_id:05d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Ensure num_agents is a Python int
        num_agents = int(num_agents)
        
        # Create feature grid: agents x stages
        fig_feat = plt.figure(figsize=(4 * self.num_stages, 3 * num_agents))
        fig_mask = plt.figure(figsize=(4 * self.num_stages, 3 * num_agents))
        fig_supp = plt.figure(figsize=(4 * self.num_stages, 3 * num_agents))
        
        for agent_idx in range(num_agents):
            for stage_idx in range(self.num_stages):
                agent_data = self.stage_data[sample_id]['stages'][stage_idx][agent_idx]
                
                # Features plot
                plt.figure(fig_feat.number)
                plt.subplot(num_agents, self.num_stages, agent_idx * self.num_stages + stage_idx + 1)
                plt.imshow(agent_data['stage_feat'], cmap='viridis', aspect='auto')
                plt.title(f'Agent {agent_idx}, Stage {stage_idx}')
                if stage_idx == 0:
                    plt.ylabel(f'Agent {agent_idx}')
                if agent_idx == 0:
                    plt.xlabel(f'Stage {stage_idx}')
                plt.xticks([])
                plt.yticks([])
                
                # Masks plot  
                plt.figure(fig_mask.number)
                plt.subplot(num_agents, self.num_stages, agent_idx * self.num_stages + stage_idx + 1)
                plt.imshow(agent_data['spatial_mask'], cmap='Reds', aspect='auto')
                plt.title(f'Mask A{agent_idx}, S{stage_idx}')
                if stage_idx == 0:
                    plt.ylabel(f'Agent {agent_idx}')
                if agent_idx == 0:
                    plt.xlabel(f'Stage {stage_idx}')
                plt.xticks([])
                plt.yticks([])
                
                # Suppression plot
                plt.figure(fig_supp.number)
                plt.subplot(num_agents, self.num_stages, agent_idx * self.num_stages + stage_idx + 1)
                suppression = agent_data['input_feat'] - agent_data['masked_feat']
                plt.imshow(suppression, cmap='RdBu', aspect='auto')
                plt.title(f'Supp A{agent_idx}, S{stage_idx}')
                if stage_idx == 0:
                    plt.ylabel(f'Agent {agent_idx}')
                if agent_idx == 0:
                    plt.xlabel(f'Stage {stage_idx}')
                plt.xticks([])
                plt.yticks([])
        
        plt.figure(fig_feat.number)
        plt.suptitle(f'Sample {sample_id:05d} - Stage Features Grid', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, f'features_grid.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        plt.figure(fig_mask.number) 
        plt.suptitle(f'Sample {sample_id:05d} - Spatial Masks Grid', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, f'masks_grid.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        plt.figure(fig_supp.number)
        plt.suptitle(f'Sample {sample_id:05d} - Feature Suppression Grid', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, f'suppression_grid.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Clear stored data for this sample
        del self.stage_data[sample_id]
    
    
    def create_summary_visualization(self, input_features, stage_query_features, final_query, 
                                   accumulated_mask, stage_predictions, record_len, sample_id):
        if not self.debug_save_enabled:
            return
            
        summary_dir = os.path.join(self.debug_save_path, f"sample_{sample_id:05d}", "summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(24, 16))
        
        BN = sum(record_len)
        
        for stage_idx, stage_features in enumerate(stage_query_features):
            stage_avg = stage_features.mean(dim=[0, 1]).detach().cpu().numpy()
            
            plt.subplot(3, self.num_stages + 1, stage_idx + 1)
            plt.imshow(stage_avg.reshape(1, -1), cmap='viridis', aspect='auto')
            plt.title(f'Stage {stage_idx} Query Features')
            plt.xlabel('Feature Channel')
            plt.colorbar()
        
        final_avg = final_query.mean(dim=[0, 1]).detach().cpu().numpy()
        plt.subplot(3, self.num_stages + 1, self.num_stages + 1)
        plt.imshow(final_avg.reshape(1, -1), cmap='plasma', aspect='auto')
        plt.title('Final Query Features')
        plt.xlabel('Feature Channel')
        plt.colorbar()
        
        for stage_idx, stage_pred in enumerate(stage_predictions):
            if 'heatmap' in stage_pred:
                heatmap_avg = torch.sigmoid(stage_pred['heatmap']).mean(dim=0).detach().cpu().numpy()
                
                plt.subplot(3, self.num_stages + 1, self.num_stages + 2 + stage_idx)
                plt.imshow(heatmap_avg[0] if len(heatmap_avg.shape) > 2 else heatmap_avg, 
                          cmap='hot', aspect='auto')
                plt.title(f'Stage {stage_idx} Heatmap')
                plt.colorbar()
        
        accumulated_avg = accumulated_mask.max(dim=1)[0].mean(dim=0).detach().cpu().numpy()
        plt.subplot(3, self.num_stages + 1, 2 * self.num_stages + 2)
        plt.imshow(accumulated_avg, cmap='Reds', aspect='auto')
        plt.title('Final Accumulated Mask')
        plt.colorbar()
        
        agent_idx = 0
        agent_activations = []
        for batch_idx, num_agents in enumerate(record_len):
            for agent_in_batch in range(num_agents):
                agent_activation = []
                for stage_features in stage_query_features:
                    activation = stage_features[agent_idx].mean().item()
                    agent_activation.append(activation)
                agent_activations.append(agent_activation)
                agent_idx += 1
        
        plt.subplot(3, self.num_stages + 1, 2 * self.num_stages + 3)
        agent_activations = np.array(agent_activations)
        for i, activations in enumerate(agent_activations):
            plt.plot(range(self.num_stages), activations, 
                    marker='o', label=f'Agent {i}', alpha=0.7)
        plt.title('Agent-wise Stage Activations')
        plt.xlabel('Stage Index')
        plt.ylabel('Mean Activation')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, f'sample_{sample_id:05d}_him_summary.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
    
    def identify_tp_from_loss_assignment(self, stage_pred_single, valid_gt_boxes, valid_gt_labels, H, W):
        """
        Use the same assignment logic as the loss function to identify True Positives
        """
        if self.loss_module is None or len(valid_gt_boxes) == 0:
            return torch.zeros((self.num_classes, H, W), device=stage_pred_single["heatmap"].device)
        
        # Prepare predictions in the format expected by loss function
        pred_dict_single = {}
        for key in ['heatmap', 'center', 'height', 'dim', 'rot']:
            if key in stage_pred_single:
                pred_dict_single[key] = stage_pred_single[key].unsqueeze(0)  # Add batch dim
        
        # Use loss function's assignment logic
        assignment_result = self.loss_module.get_targets_single(
            valid_gt_boxes, valid_gt_labels, pred_dict_single
        )
        
        # Extract positive assignment indices
        labels = assignment_result[0].squeeze(0)  # Remove batch dim
        pos_inds = torch.nonzero(labels < self.loss_module.num_classes, as_tuple=False).squeeze(-1)
        
        # Create spatial TP mask
        tp_mask = torch.zeros((self.num_classes, H, W), device=stage_pred_single["heatmap"].device)
        
        if len(pos_inds) > 0:
            # Get positive assignments info
            pos_labels = labels[pos_inds]
            bbox_targets = assignment_result[2].squeeze(0)[pos_inds]  # Encoded targets
            
            # Convert encoded targets back to spatial locations
            for pos_idx, (label, target) in enumerate(zip(pos_labels, bbox_targets)):
                if label < self.num_classes:
                    # target[0:2] contains center coordinates in feature map space
                    center_x_idx = int(torch.clamp(target[0], 0, W-1))
                    center_y_idx = int(torch.clamp(target[1], 0, H-1))
                    
                    # Apply masking strategy around the matched location
                    if self.masking_strategy == 'point':
                        tp_mask[label, center_y_idx, center_x_idx] = 1
                    elif self.masking_strategy == 'pooling':
                        for dy in range(-1, 2):
                            for dx in range(-1, 2):
                                ny, nx = center_y_idx + dy, center_x_idx + dx
                                if 0 <= ny < H and 0 <= nx < W:
                                    tp_mask[label, ny, nx] = 1
        
        return tp_mask
    
    
    def forward(self, features, gt_boxes=None, record_len=None):
        BN, C, H, W = features.shape        
        BN = sum(record_len)
        stage_outputs = []
        stage_predictions = []
        stage_masks = []
        query_features = []
        accumulated_mask = torch.zeros((BN, self.num_classes, H, W), device=features.device)
        # breakpoint()
        for stage_idx in range(self.num_stages):

            # Convert accumulated per-class mask to a spatial mask so it can
            # broadcast over the feature-channel dimension (C=hidden_dim)
            # accumulated_mask: [BN, Nc, H, W]
            # spatial_mask   : [BN, 1, H, W]
            spatial_mask = accumulated_mask.max(dim=1, keepdim=True)[0]

            # Broadcast spatial mask over all feature channels
            masked_features = features * (1 - spatial_mask)
            
            stage_feat = self.stage_convs[stage_idx](masked_features)
            query_features.append(stage_feat)
            # breakpoint()
            batch_dict_stage = {'spatial_features_2d': stage_feat}
            stage_pred = self.head(batch_dict_stage)
            stage_predictions.append(stage_pred)
            # ------------------------------
            # Inference-time TP mask update
            # ------------------------------
            # When the model is in eval() mode we do not have GT boxes, therefore
            # the progressive masking must rely on a surrogate.  We treat any
            # pixel whose class-wise heat-map confidence exceeds 0.3 as a TP and
            # dilate it with a 3×3 max-pool.  This keeps the progressive
            # hard-instance logic consistent between training and inference.
            if not self.training:
                with torch.no_grad():
                    hm_key = 'dense_heatmap' if 'dense_heatmap' in stage_pred else 'heatmap'
                    conf_src = stage_pred[hm_key]
                    # If using proposal-level heatmap (B,Nc,P), reshape to [B,Nc,H,W]
                    if conf_src.dim() == 3:
                        # assume P == H*W
                        conf_src = conf_src.view(conf_src.size(0), conf_src.size(1), H, W)
                    conf_mask = (conf_src.sigmoid() > self.conf_threshold).float()
                    conf_mask = F.max_pool2d(conf_mask, self.dilation_kernel, 1, self.dilation_kernel//2)
                    accumulated_mask = torch.maximum(accumulated_mask, conf_mask)
            
            if not self.training:
                self.create_stage_visualizations(
                    stage_idx, features, masked_features, stage_feat,
                    spatial_mask, accumulated_mask, stage_pred, record_len, self.sample_counter
                )
            
            if self.training and gt_boxes is not None:
                batch_tp_masks = []
                # Process each agent in the flattened batch
                agent_idx = 0
                for batch_idx, num_agents in enumerate(record_len):
                    # gt boxes is (B, N, 8) - same for all agents in this batch
                    if gt_boxes[batch_idx] is not None and len(gt_boxes[batch_idx]) > 0:
                        valid_gt_mask = gt_boxes[batch_idx][:, 3] > 0
                        valid_gt_boxes = gt_boxes[batch_idx][valid_gt_mask, :7]
                        valid_gt_labels = gt_boxes[batch_idx][valid_gt_mask, 7].long() - 1
                        
                        # Process each agent for this batch sample
                        for _ in range(num_agents):
                            # Extract single agent predictions
                            stage_pred_single = {}
                            for key in ['heatmap', 'center', 'height', 'dim', 'rot']:
                                if key in stage_pred:
                                    stage_pred_single[key] = stage_pred[key][agent_idx]
                            
                            # Use loss assignment logic to identify TPs
                            tp_mask = self.identify_tp_from_loss_assignment(
                                stage_pred_single, valid_gt_boxes, valid_gt_labels, H, W
                            )
                            
                            batch_tp_masks.append(tp_mask)
                            agent_idx += 1
                    else:
                        # No valid GT boxes for this batch sample - add zero masks for all agents
                        for _ in range(num_agents):
                            batch_tp_masks.append(torch.zeros((self.num_classes, H, W), device=features.device))
                            agent_idx += 1
                # breakpoint()
                stage_mask = torch.stack(batch_tp_masks, dim=0)
                stage_masks.append(stage_mask)
                accumulated_mask = torch.maximum(accumulated_mask, stage_mask)
            else:
                stage_mask = None
            
            stage_outputs.append({
                'predictions': stage_pred,
                'mask': stage_mask if self.training else None,
                'accumulated_mask': accumulated_mask.clone()
            })
        
        combined_queries = torch.cat(query_features, dim=1)
        final_query = self.query_proj(combined_queries)
        
        if not self.training:
            self.create_summary_visualization(
                features, query_features, final_query, accumulated_mask, 
                stage_predictions, record_len, self.sample_counter
            )
            self.sample_counter += 1
        
        # breakpoint()
        return {
            'stage_outputs': stage_outputs,
            'stage_predictions': stage_predictions,
            'stage_masks': stage_masks if self.training else None,
            'query_features': final_query,
            'accumulated_mask': accumulated_mask
        }
