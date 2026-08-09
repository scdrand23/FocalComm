import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from focalcomm.utils.hungarian_assigner import HungarianAssigner3D
from focalcomm.utils import loss_utils
from focalcomm.utils import centernet_utils
from focalcomm.models.sub_modules.transfusion_utils import clip_sigmoid


class FocalCommLoss(nn.Module):
    def __init__(self, args):
        super(FocalCommLoss, self).__init__()
        self.args = args
        self.loss_dict = {}
        self.grid_size = args['grid_size']
        self.point_cloud_range = args['point_cloud_range']
        self.voxel_size = args['voxel_size']
        self.num_classes = args['num_classes']
        self.feature_map_stride = args['feature_map_stride']
        self.bbox_assigner = HungarianAssigner3D(**args['target_assigner_config']['hungarian_assigner'])
        self.loss_heatmap = loss_utils.GaussianFocalLoss()
        self.loss_bbox = loss_utils.L1Loss()
        loss_cls = args['loss_config']['loss_cls']
        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = loss_utils.SigmoidFocalClassificationLoss(gamma=loss_cls['gamma'], alpha=loss_cls['alpha'])
        self.loss_cls_weight = args['loss_config']['loss_weights']['cls_weight']
        self.loss_bbox = loss_utils.L1Loss()
        self.loss_bbox_weight = args['loss_config']['loss_weights']['bbox_weight']
        self.loss_heatmap = loss_utils.GaussianFocalLoss()
        self.loss_heatmap_weight = args['loss_config']['loss_weights']['hm_weight']
        if 'him_hm_weight' in args['loss_config']['loss_weights']:
            self.him_hm_weight = args['loss_config']['loss_weights']['him_hm_weight']
        self.code_size = 8
        self.iou = 0  # Add this line to store IoU
        self.per_class_iou = {}  # Store per-class IoU

    def forward(self, pred_dicts, gt_boxes_list):
        """
        Args:
            batch_dict (dict): Contains:
                - gt_boxes: (B, N, 8) ground truth boxes with class labels
                - spatial_features_2d: BEV features from backbone
                - ...other prediction outputs from TransFusionHead
            target_dict (dict): Contains additional ground truth info if needed
        """
        gt_boxes = gt_boxes_list
        pred_dicts = pred_dicts
        # breakpoint()
        # # gt_bboxes_3d = gt_boxes[...,:-1]  # Remove class label
        # # gt_labels_3d = gt_boxes[...,-1].long() - 1  # Convert to 0-based class index
        # breakpoint()
        loss, tb_dict = self.loss(gt_boxes, pred_dicts)
        # batch_dict['loss'] = loss
        # batch_dict['tb_dict'] = tb_dict
        
        # Store all losses
        self.loss_dict.update({
            'total_loss': loss.item(),
            'cls_loss': tb_dict['loss_cls'],
            'reg_loss': tb_dict['loss_bbox'],
            'heatmap_loss': tb_dict['loss_heatmap'],
            'matched_ious': tb_dict['matched_ious'],
            'loss_trans': tb_dict['loss_trans']
        })
        self.iou = tb_dict['matched_ious']
        return loss, tb_dict



    def get_targets(self, gt_bboxes, pred_dicts):
        # print(f"\n=== Starting get_targets with {len(gt_bboxes)} batches ===")
        assign_results = []
        all_per_class_ious = {i: [] for i in range(self.num_classes)}  # Track IoUs per class
        
        for batch_idx, gt_bbox in enumerate(gt_bboxes):
            gt_bboxes_3d = gt_bbox[...,:-1]  # Remove class label
            gt_labels_3d = gt_bbox[...,-1].long() - 1  # Convert to 0-based class index
            pred_dict = {}
            for key in pred_dicts.keys():
                pred_dict[key] = pred_dicts[key][batch_idx : batch_idx + 1]
            valid_idx = []
            # filter empty boxes
            for i in range(len(gt_bbox)):
                if gt_bbox[i][3] > 0 and gt_bbox[i][4] > 0:
                    valid_idx.append(i)
            # print(f"Batch {batch_idx}: Found {len(valid_idx)} valid boxes out of {len(gt_bbox)}")
            assign_result = self.get_targets_single(gt_bboxes_3d[valid_idx], gt_labels_3d[valid_idx], pred_dict)
            assign_results.append(assign_result)
            
            # Extract per-class IoUs from the last element if it exists
            if len(assign_result) > 7:  # Check if per-class IoUs were returned
                batch_per_class_ious = assign_result[7]
                for cls_idx, ious in batch_per_class_ious.items():
                    all_per_class_ious[cls_idx].extend(ious)

        # print("=== Concatenating results ===")
        res_tuple = tuple(map(list, zip(*assign_results)))
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        num_pos = np.sum(res_tuple[4])
        matched_ious = np.mean(res_tuple[5])
        heatmap = torch.cat(res_tuple[6], dim=0)
        
        # Calculate per-class IoU averages
        per_class_iou_avg = {}
        for cls_idx, ious in all_per_class_ious.items():
            if len(ious) > 0:
                per_class_iou_avg[cls_idx] = np.mean(ious)
            else:
                per_class_iou_avg[cls_idx] = 0.0
        
        # Store in instance variable for external access
        self.per_class_iou = per_class_iou_avg
        
        # print(f"Final stats: {num_pos} positive samples, mean IoU: {matched_ious:.4f}")
        return labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap
        


    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        # print("\n=== Starting get_targets_single ===")
        # print(gt_bboxes_3d.shape, gt_labels_3d.shape)
        num_proposals = preds_dict["center"].shape[-1]
        # print(f"Processing {num_proposals} proposals")
        
        # Convert all tensors to float32
        score = copy.deepcopy(preds_dict["heatmap"].detach()).float()
        center = copy.deepcopy(preds_dict["center"].detach()).float()
        height = copy.deepcopy(preds_dict["height"].detach()).float()
        dim = copy.deepcopy(preds_dict["dim"].detach()).float()
        rot = copy.deepcopy(preds_dict["rot"].detach()).float()
        
        # Convert gt_bboxes to float32 if needed
        gt_bboxes_3d = gt_bboxes_3d.float()
        
        if "vel" in preds_dict.keys():
            vel = copy.deepcopy(preds_dict["vel"].detach()).float()
            # print("Velocity information available")
        else:
            vel = None
            # print("No velocity information")

        # print("Decoding bounding boxes...")
        boxes_dict = self.decode_bbox(score, rot, dim, center, height)
        bboxes_tensor = boxes_dict[0]["pred_boxes"].float()
        gt_bboxes_tensor = gt_bboxes_3d.to(score.device)

        # print("Running Hungarian assignment...")
        assigned_gt_inds, ious = self.bbox_assigner.assign(
            bboxes_tensor, gt_bboxes_tensor, gt_labels_3d,
            score, self.point_cloud_range,
        )
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1
        # print(f"Assignment complete: {len(pos_inds)} positive, {len(neg_inds)} negative matches")

        if gt_bboxes_3d.numel() == 0:
            assert pos_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes_3d).view(-1, 9)
            # print("No ground truth boxes found")
        else:
            pos_gt_bboxes = gt_bboxes_3d[pos_assigned_gt_inds.long(), :]

        # print("Creating targets for loss computation...")
        bbox_targets = torch.zeros([num_proposals, self.code_size]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.code_size]).to(center.device)
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  
            labels += self.num_classes
            # print(f"Using {self.num_classes} classes")

        if len(pos_inds) > 0:
            # print("Processing positive samples...")
            # print(pos_gt_bboxes.shape)
            pos_bbox_targets = self.encode_bbox(pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[pos_assigned_gt_inds]
            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            # print("Processing negative samples...")
            label_weights[neg_inds] = 1.0

        # Track per-class IoUs
        per_class_ious = {i: [] for i in range(self.num_classes)}
        if len(pos_inds) > 0:
            pos_ious = ious[pos_inds]
            pos_labels = gt_labels_3d[pos_assigned_gt_inds]
            for iou_val, label in zip(pos_ious, pos_labels):
                if 0 <= label < self.num_classes:
                    per_class_ious[label.item()].append(iou_val.item())
        
        # print("Computing dense heatmap targets...")
        device = labels.device
        target_assigner_cfg = self.args['target_assigner_config']
        
        # Get the actual feature map size for non-square grid
        x_size = self.grid_size[0] // self.feature_map_stride  # e.g., 1000 // 8 = 125
        y_size = self.grid_size[1] // self.feature_map_stride  # e.g., 400 // 8 = 50
        
        # Create heatmap with correct dimensions (note the order: [num_classes, y, x])
        heatmap = gt_bboxes_3d.new_zeros(self.num_classes, y_size, x_size)
        
        # print(f"Processing {len(gt_bboxes_3d)} boxes for heatmap...")
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / self.voxel_size[0] / self.feature_map_stride
            length = length / self.voxel_size[1] / self.feature_map_stride
            if width > 0 and length > 0:
                radius = centernet_utils.gaussian_radius(length.view(-1), width.view(-1), target_assigner_cfg['gaussian_overlap'])[0]
                radius = max(target_assigner_cfg['min_radius'], int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / self.feature_map_stride
                coor_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / self.feature_map_stride

                center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                center_int = center.to(torch.int32)
                centernet_utils.draw_gaussian_to_heatmap(heatmap[gt_labels_3d[idx]], center_int, radius)

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        # print(f"Completed get_targets_single with mean IoU: {mean_iou:.4f}")
        return (labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], int(pos_inds.shape[0]), float(mean_iou), heatmap[None], per_class_ious)
    
    def loss(self, gt_boxes, pred_dicts, **kwargs):
        labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap = \
            self.get_targets(gt_boxes, pred_dicts)
        loss_dict = dict()
        loss_all = 0

        # Handle stage predictions from ego-guided HIM
        if 'stage_predictions' in pred_dicts:
            stage_loss = self.compute_multi_agent_stage_loss(
                pred_dicts['stage_predictions'],  # List of dicts with 'dense_heatmap'
                heatmap,
                pred_dicts.get('record_len', torch.tensor([1])),  # Default to 1 if not provided
                pred_dicts.get('stage_masks', None)
            )
            loss_dict['stage_heatmap_loss'] = stage_loss.item() * self.him_hm_weight
            loss_all += stage_loss * self.him_hm_weight
        # breakpoint()
        # print("Computing heatmap loss...")
        # Normalize by positive pixels for proper gradient scaling
        heatmap_loss_per_pixel = self.loss_heatmap(
            clip_sigmoid(pred_dicts["dense_heatmap"]),
            heatmap,
        )
        loss_heatmap = heatmap_loss_per_pixel.sum() / max(heatmap.eq(1).float().sum().item(), 1)
        loss_dict["loss_heatmap"] = loss_heatmap.item() * self.loss_heatmap_weight
        loss_all += loss_heatmap * self.loss_heatmap_weight
        # print(f"Heatmap loss: {loss_dict['loss_heatmap']:.4f}")

        # print("Computing classification loss...")
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = pred_dicts["heatmap"].permute(0, 2, 1).reshape(-1, self.num_classes)

        one_hot_targets = torch.zeros(*list(labels.shape), self.num_classes+1, dtype=cls_score.dtype, device=labels.device)
        one_hot_targets.scatter_(-1, labels.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., :-1]
        loss_cls = self.loss_cls(
            cls_score, one_hot_targets, label_weights
        ).sum() / max(num_pos, 1)

        # print("Computing bbox regression loss...")
        head_names = ['center', 'height', 'dim', 'rot']
        preds = torch.cat([pred_dicts[head_name] for head_name in head_names], dim=1).permute(0, 2, 1)
        
        code_weights = self.args['loss_config']['code_weights']
        reg_weights = bbox_weights * bbox_weights.new_tensor(code_weights)

        loss_bbox = self.loss_bbox(preds, bbox_targets) 
        loss_bbox = (loss_bbox * reg_weights).sum() / max(num_pos, 1)

        loss_dict["loss_cls"] = loss_cls.item() * self.loss_cls_weight
        loss_dict["loss_bbox"] = loss_bbox.item() * self.loss_bbox_weight
        loss_all = loss_all + loss_cls * self.loss_cls_weight + loss_bbox * self.loss_bbox_weight

        loss_dict[f"matched_ious"] = loss_cls.new_tensor(matched_ious)
        loss_dict['loss_trans'] = loss_all

        # print(f"Final losses - Cls: {loss_dict['loss_cls']:.4f}, Bbox: {loss_dict['loss_bbox']:.4f}, Total: {loss_all.item():.4f}")
        return loss_all, loss_dict

    def encode_bbox(self, bboxes):
        code_size = 8
        targets = torch.zeros([bboxes.shape[0], code_size]).to(bboxes.device)
        
        # Encode position, dimensions, and height
        targets[:, 0] = (bboxes[:, 0] - self.point_cloud_range[0]) / (self.feature_map_stride * self.voxel_size[0])
        targets[:, 1] = (bboxes[:, 1] - self.point_cloud_range[1]) / (self.feature_map_stride * self.voxel_size[1])
        targets[:, 3:6] = bboxes[:, 3:6].log()
        targets[:, 2] = bboxes[:, 2]
        
        # Encode rotation
        targets[:, 6] = torch.sin(bboxes[:, 6])
        targets[:, 7] = torch.cos(bboxes[:, 6])
        
        return targets


    def decode_bbox(self, heatmap, rot, dim, center, height, vel=None, filter=False):
        
        post_process_cfg = self.args['post_processing']
        score_thresh = post_process_cfg['score_thresh']
        post_center_range = post_process_cfg['post_center_range']
        post_center_range = torch.tensor(post_center_range).cuda().float()
        # class label
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values

        # breakpoint()
        # print("\nBefore decoding:")
        # print(f"Raw center range: [{center.min():.3f}, {center.max():.3f}]")
        # print(f"Raw height range: [{height.min():.3f}, {height.max():.3f}]")
        # print(f"Raw dim range: [{dim.min():.3f}, {dim.max():.3f}]")
        # print(f"Raw rot range: [{rot.min():.3f}, {rot.max():.3f}]")

        center[:, 0, :] = center[:, 0, :] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        center[:, 1, :] = center[:, 1, :] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        dim = dim.exp()
        # dim = dim[:, [2,1,0], :] 
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)

        # Always use the version without velocity
        final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)

        # print("\nAfter decoding:")
        # print(f"Final boxes shape: {final_box_preds.shape}")
        # print(f"First decoded box: {final_box_preds[0,0]}")
        # print(f"Final center range: [{final_box_preds[...,0:2].min():.3f}, {final_box_preds[...,0:2].max():.3f}]")
        # print(f"Final dim range: [{final_box_preds[...,3:6].min():.3f}, {final_box_preds[...,3:6].max():.3f}]")


        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            boxes3d = final_box_preds[i]
            scores = final_scores[i]
            labels = final_preds[i]
            predictions_dict = {
                'pred_boxes': boxes3d,
                'pred_scores': scores,
                'pred_labels': labels
            }
            predictions_dicts.append(predictions_dict)

        if filter is False:
            return predictions_dicts

        thresh_mask = final_scores > score_thresh        
        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            cmask = mask[i, :]
            cmask &= thresh_mask[i]

            boxes3d = final_box_preds[i, cmask]
            scores = final_scores[i, cmask]
            labels = final_preds[i, cmask]
            predictions_dict = {
                'pred_boxes': boxes3d,
                'pred_scores': scores,
                'pred_labels': labels,
            }

            predictions_dicts.append(predictions_dict)

        return predictions_dicts


    def get_bboxes(self, preds_dicts):

        batch_size = preds_dicts["heatmap"].shape[0]
        batch_score = preds_dicts["heatmap"].sigmoid()
        query_labels = preds_dicts["query_labels"]
        one_hot = F.one_hot(
            query_labels, num_classes=self.num_classes
        ).permute(0, 2, 1)
        batch_score = batch_score * preds_dicts["query_heatmap_score"] * one_hot
        batch_center = preds_dicts["center"]
        batch_height = preds_dicts["height"]
        batch_dim = preds_dicts["dim"]
        batch_rot = preds_dicts["rot"]
        batch_vel = None
        if "vel" in preds_dicts:
            batch_vel = preds_dicts["vel"]

        ret_dict = self.decode_bbox(
            batch_score, batch_rot, batch_dim,
            batch_center, batch_height, batch_vel,
            filter=True,
        )
        for k in range(batch_size):
            ret_dict[k]['pred_labels'] = ret_dict[k]['pred_labels'].int() + 1

        return ret_dict 
    
    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        """Log training progress.
        
        Args:
            epoch (int): Current epoch
            batch_id (int): Current batch index
            batch_len (int): Total number of batches
            writer: Tensorboard writer instance
            pbar: Optional progress bar
        """
        # Create log string with all available losses
        log_parts = [f"[epoch {epoch}][{batch_id+1}/{batch_len}]"]
        
        # Add all losses from loss_dict to log string
        for key, value in self.loss_dict.items():
            if isinstance(value, (int, float)):
                log_parts.append(f"{key}: {value:.4f}")
        
        log_str = " || ".join(log_parts)

        # Display progress
        if pbar is None:
            print(log_str)
        else:
            pbar.set_description(log_str)

        # TensorBoard logging - log all available losses
        step = epoch * batch_len + batch_id
        for key, value in self.loss_dict.items():
            if isinstance(value, (int, float)):
                # Group losses and stats into separate categories
                if 'loss' in key.lower():
                    writer.add_scalar(f'Loss/{key}', value, step)
                else:
                    writer.add_scalar(f'Stats/{key}', value, step)


    def visualize_predictions_bev(points, gt_boxes, pred_boxes, save_path):
        """
        Visualize BEV (Bird's Eye View) predictions vs ground truth
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10,10))
        # Plot points
        plt.scatter(points[:, 0], points[:, 1], c='gray', s=1, alpha=0.5)
        
        # Plot GT boxes (green)
        for box in gt_boxes:
            plt.plot([box[0], box[0]+box[3]], [box[1], box[1]], 'g-', label='GT')
            plt.plot([box[0], box[0]], [box[1], box[1]+box[4]], 'g-')
            plt.plot([box[0]+box[3], box[0]+box[3]], [box[1], box[1]+box[4]], 'g-')
            plt.plot([box[0], box[0]+box[3]], [box[1]+box[4], box[1]+box[4]], 'g-')
        
        # Plot predicted boxes (red)
        for box in pred_boxes:
            plt.plot([box[0], box[0]+box[3]], [box[1], box[1]], 'r--', label='Pred')
            plt.plot([box[0], box[0]], [box[1], box[1]+box[4]], 'r--')
            plt.plot([box[0]+box[3], box[0]+box[3]], [box[1], box[1]+box[4]], 'r--')
            plt.plot([box[0], box[0]+box[3]], [box[1]+box[4], box[1]+box[4]], 'r--')
        
        plt.axis('equal')
        plt.savefig(save_path)
        plt.close()

    def visualize_progressive_masking(self, stage_heatmaps, gt_heatmap, final_mask, stage_losses, sample_id):
        """
        Visualize how progressive masking works
        """
        try:
            import matplotlib.pyplot as plt
            import os
            
            viz_dir = "progressive_masking_viz"
            os.makedirs(viz_dir, exist_ok=True)
            
            B, N, H, W = gt_heatmap.shape
            num_stages = len(stage_heatmaps)
            
            # Create visualization for each class
            for class_idx in range(N):
                fig, axes = plt.subplots(2, num_stages + 2, figsize=(18, 8))
                
                # Top row: GT progression
                # Original GT
                gt_np = self.normalize_for_display(gt_heatmap[0, class_idx])
                axes[0, 0].imshow(gt_np, cmap='hot')
                axes[0, 0].set_title('Original GT')
                axes[0, 0].axis('off')
                
                # Progressive GT targets
                accumulated_mask = torch.zeros_like(gt_heatmap[0, class_idx])
                for stage_idx in range(num_stages):
                    if stage_idx == 0:
                        stage_gt = gt_heatmap[0, class_idx]
                    else:
                        stage_gt = gt_heatmap[0, class_idx] * (1 - accumulated_mask)
                    
                    gt_np = self.normalize_for_display(stage_gt)
                    axes[0, stage_idx + 1].imshow(gt_np, cmap='hot')
                    axes[0, stage_idx + 1].set_title(f'GT Stage {stage_idx+1}\nLoss: {stage_losses[stage_idx]:.4f}')
                    axes[0, stage_idx + 1].axis('off')
                    
                    # Update accumulated mask (simplified for visualization)
                    stage_pred = stage_heatmaps[stage_idx][0].mean(dim=0)[class_idx]  # Average across agents
                    pred_positive = (stage_pred.sigmoid() > 0.3).float()
                    tp_mask = pred_positive * gt_heatmap[0, class_idx]
                    expanded_tp = F.max_pool2d(tp_mask.unsqueeze(0).unsqueeze(0), 3, 1, 1).squeeze()
                    accumulated_mask = torch.clamp(accumulated_mask + expanded_tp, 0, 1)
                
                # Final accumulated mask
                mask_np = self.normalize_for_display(final_mask[0, class_idx])
                axes[0, -1].imshow(mask_np, cmap='Blues')
                axes[0, -1].set_title('Final TP Mask')
                axes[0, -1].axis('off')
                
                # Bottom row: Stage predictions
                for stage_idx in range(num_stages):
                    stage_pred = stage_heatmaps[stage_idx][0].mean(dim=0)[class_idx]  # Average across agents
                    pred_np = self.normalize_for_display(stage_pred.sigmoid())
                    axes[1, stage_idx + 1].imshow(pred_np, cmap='hot')
                    axes[1, stage_idx + 1].set_title(f'Pred Stage {stage_idx+1}')
                    axes[1, stage_idx + 1].axis('off')
                
                # Hide unused subplot
                axes[1, 0].axis('off')
                axes[1, -1].axis('off')
                
                plt.suptitle(f'Progressive Masking - Class {class_idx+1} - Sample {sample_id}', fontsize=16)
                plt.tight_layout()
                plt.savefig(f'{viz_dir}/progressive_class_{class_idx+1}_{sample_id:06d}.png', dpi=150, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            print(f"Visualization error: {e}")

    def normalize_for_display(self, tensor):
        if tensor.dtype == torch.bool:
            tensor = tensor.float()
        
        tensor = tensor.detach().cpu().numpy()
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val > min_val:
            tensor = (tensor - min_val) / (max_val - min_val)
        return tensor

    def visualize_focalformer_progressive_loss(self, stage_predictions, gt_heatmap, 
                                             accumulated_tp_mask, stage_losses, sample_id):
        """Visualize FocalFormer-style progressive GT masking"""
        try:
            import matplotlib.pyplot as plt
            import os
            
            viz_dir = "focalformer_progressive_viz"
            os.makedirs(viz_dir, exist_ok=True)
            
            B, N, H, W = gt_heatmap.shape
            num_stages = len(stage_predictions)
            
            for class_idx in range(N):
                fig, axes = plt.subplots(3, num_stages + 1, figsize=(6*(num_stages+1), 15))
                
                # Row 1: Progressive GT targets
                accumulated_mask_viz = torch.zeros_like(gt_heatmap[0, class_idx])
                
                # Original GT
                gt_np = self.normalize_for_display(gt_heatmap[0, class_idx])
                axes[0, 0].imshow(gt_np, cmap='hot')
                axes[0, 0].set_title('Original GT')
                axes[0, 0].axis('off')
                
                for stage_idx in range(num_stages):
                    if stage_idx == 0:
                        stage_gt = gt_heatmap[0, class_idx]
                    else:
                        stage_gt = gt_heatmap[0, class_idx] * (1 - accumulated_mask_viz)
                    
                    gt_np = self.normalize_for_display(stage_gt)
                    axes[0, stage_idx + 1].imshow(gt_np, cmap='hot')
                    axes[0, stage_idx + 1].set_title(f'GT Stage {stage_idx+1}\nLoss: {stage_losses[stage_idx]:.4f}')
                    axes[0, stage_idx + 1].axis('off')
                    
                    # Update accumulated mask for visualization
                    dense_heatmap = stage_predictions[stage_idx]['dense_heatmap']
                    pred_confidence = dense_heatmap[0, class_idx].sigmoid()
                    soft_pred = torch.sigmoid((pred_confidence - 0.3) * 2.0)
                    tp_mask = soft_pred * gt_heatmap[0, class_idx]
                    expanded_tp = F.max_pool2d(tp_mask.unsqueeze(0).unsqueeze(0), 3, 1, 1).squeeze()
                    accumulated_mask_viz = torch.clamp(accumulated_mask_viz + expanded_tp * 0.7, 0, 0.9)
                
                # Row 2: Stage predictions
                for stage_idx in range(num_stages):
                    pred_heatmap = stage_predictions[stage_idx]['dense_heatmap'][0, class_idx]
                    pred_np = self.normalize_for_display(pred_heatmap.sigmoid())
                    axes[1, stage_idx + 1].imshow(pred_np, cmap='hot')
                    axes[1, stage_idx + 1].set_title(f'Pred Stage {stage_idx+1}')
                    axes[1, stage_idx + 1].axis('off')
                
                axes[1, 0].axis('off')  # Empty cell
                
                # Row 3: Accumulated TP masks
                for stage_idx in range(num_stages):
                    if stage_idx < len(stage_predictions):
                        # Show progressive accumulation
                        temp_mask = torch.zeros_like(gt_heatmap[0, class_idx])
                        for i in range(stage_idx + 1):
                            dense_hm = stage_predictions[i]['dense_heatmap']
                            pred_conf = dense_hm[0, class_idx].sigmoid()
                            soft_pred = torch.sigmoid((pred_conf - 0.3) * 2.0)
                            tp = soft_pred * gt_heatmap[0, class_idx]
                            expanded = F.max_pool2d(tp.unsqueeze(0).unsqueeze(0), 3, 1, 1).squeeze()
                            temp_mask = torch.clamp(temp_mask + expanded * 0.7, 0, 0.9)
                        
                        mask_np = self.normalize_for_display(temp_mask)
                        axes[2, stage_idx + 1].imshow(mask_np, cmap='Blues')
                        axes[2, stage_idx + 1].set_title(f'Acc TP Mask {stage_idx+1}')
                        axes[2, stage_idx + 1].axis('off')
                
                axes[2, 0].axis('off')  # Empty cell
                
                plt.suptitle(f'FocalFormer Progressive Loss - Class {class_idx+1} - Sample {sample_id}', fontsize=16)
                plt.tight_layout()
                plt.savefig(f'{viz_dir}/focalformer_class_{class_idx+1}_{sample_id:06d}.png', 
                           dpi=150, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            print(f"FocalFormer visualization error: {e}")

    def compute_multi_agent_stage_loss(self, stage_predictions, heatmap, record_len, stage_masks=None):
        """
        Progressive loss for ego-guided HIM
        Args:
            stage_predictions: List of stage prediction dicts from HIM, each containing 'dense_heatmap' of shape [B, num_classes, H, W]
            heatmap: [B, num_classes, H, W] target heatmap (only for ego vehicles)
            record_len: [B] valid agents per batch
            stage_masks: List of [B, num_classes, H, W] accumulated positive masks from HIM
        """
        total_loss = 0
        B, N, H, W = heatmap.shape
        
        for stage_idx, stage_pred_dict in enumerate(stage_predictions):
            ego_pred = stage_pred_dict['dense_heatmap']  # [B, num_classes, H, W] - already ego only
            
            # Apply stage mask if provided
            if stage_masks is not None and stage_idx < len(stage_masks):
                ego_mask = stage_masks[stage_idx]  # [B, num_classes, H, W] - already ego only
                
                # For progressive loss: focus on regions NOT covered by previous stages
                if stage_idx == 0:
                    active_mask = torch.ones_like(ego_mask)
                else:
                    # Regions not confidently detected in previous stages
                    active_mask = 1.0 - ego_mask.max(dim=1, keepdim=True)[0].float()
                
                # Apply mask to both prediction and target
                masked_pred = ego_pred * active_mask
                masked_target = heatmap * active_mask
            else:
                masked_pred = ego_pred
                masked_target = heatmap
            
            # Compute focal loss with proper normalization
            stage_loss = self.loss_heatmap(
                clip_sigmoid(masked_pred),
                masked_target
            ).sum() / max(masked_target.eq(1).float().sum().item(), 1)  # Normalize by positive pixels
            
            total_loss += stage_loss
        
        return total_loss / len(stage_predictions)

    def _multi_agent_stage_loss(self, stage_predictions, heatmap, record_len, 
                                     stage_masks=None, stage_thresholds=None):
        """
        Enhanced loss that teaches hard instance self-identification using GT
        """
        total_loss = 0
        threshold_reg_loss = 0
        self_identification_loss = 0

        B, N, H, W = heatmap.shape

        # Compute what each stage SHOULD identify based on GT
        remaining_gt = heatmap.clone()
        stage_gt_targets = []

        for stage_idx, stage_pred_dict in enumerate(stage_predictions):
            ego_pred = stage_pred_dict['dense_heatmap']  # [B, num_classes, H, W]

            # What this stage SHOULD identify: current remaining GT
            stage_gt_targets.append(remaining_gt.clone())

            # 1. Standard detection loss on remaining targets
            stage_detection_loss = self.loss_heatmap(
                clip_sigmoid(ego_pred),
                remaining_gt
            ).sum() / max(remaining_gt.eq(1).float().sum().item(), 1)

            # 2. Self-identification supervision
            if stage_masks is not None and stage_idx < len(stage_masks):
                predicted_mask = stage_masks[stage_idx]  # Model's masking decision
                conf = torch.sigmoid(ego_pred)

                # Compute what SHOULD be masked based on GT
                # High confidence predictions that align with GT should be masked
                pred_binary = (conf > 0.3).float()
                gt_positive = (remaining_gt > 0.5).float()

                # True positives: correctly detected instances
                true_positives = pred_binary * gt_positive

                # Expand TPs slightly to account for localization uncertainty
                kernel = torch.ones(1, 1, 3, 3, device=true_positives.device) / 9
                tp_expanded = F.conv2d(
                    true_positives.view(-1, 1, H, W),
                    kernel,
                    padding=1,
                    groups=1
                ).view(B, N, H, W)
                tp_expanded = torch.clamp(tp_expanded, 0, 1)

                # GT-based target mask: what SHOULD be masked
                gt_target_mask = tp_expanded

                # Supervise the masking decision
                mask_supervision_loss = F.binary_cross_entropy(
                    predicted_mask,
                    gt_target_mask,
                    reduction='mean'
                )

                self_identification_loss += mask_supervision_loss

                # Update remaining GT for next stage
                # Remove confidently detected instances
                detected_instances = predicted_mask * gt_positive
                remaining_gt = remaining_gt * (1 - detected_instances)

            # 3. Threshold regularization
            if stage_thresholds is not None and stage_idx < len(stage_thresholds):
                threshold = stage_thresholds[stage_idx]

                # Encourage reasonable threshold values
                threshold_reg = (
                    torch.relu(0.15 - threshold) +  # Not too low
                    torch.relu(threshold - 0.85)     # Not too high
                )

                # Encourage progressive thresholds (later stages should be more sensitive)
                if stage_idx > 0:
                    prev_threshold = stage_thresholds[stage_idx - 1]
                    # Later stages should have lower or similar thresholds
                    ordering_penalty = torch.relu(threshold - prev_threshold - 0.1)
                    threshold_reg += ordering_penalty

                threshold_reg_loss += threshold_reg

            total_loss += stage_detection_loss

        # Combine all losses
        final_loss = (
            total_loss +                              # Detection loss
            0.3 * self_identification_loss +          # Mask supervision
            0.1 * threshold_reg_loss                  # Threshold regularization
        )

        return final_loss / len(stage_predictions)