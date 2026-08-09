# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

# Modified by Dereje Shenkut <derejeshenkut@gmail.com>


import argparse
import os
import statistics
import gc
import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import focalcomm.hypes_yaml.yaml_utils as yaml_utils
from focalcomm.tools import train_utils
from focalcomm.tools import multi_gpu_utils
from focalcomm.data_utils.datasets import build_dataset
from focalcomm.tools import train_utils


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_true',
                        help='whether to use distributed training')
    opt = parser.parse_args()
    return opt

# Convergence test 
def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    # Initialize distributed mode
    # multi_gpu_utils.init_distributed_mode(opt)
    torch.manual_seed(2025)  # You can choose any seed value
    
    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)
    # breakpoint()
    # # Create a small subset of scenario_folders for both datasets
    opencood_train_dataset.scenario_folders = opencood_train_dataset.scenario_folders[:4]
    opencood_validate_dataset.scenario_folders = opencood_validate_dataset.scenario_folders[:2]
    
    # Reinitialize datasets with new scenario folders
    opencood_train_dataset.reinitialize()
    opencood_validate_dataset.reinitialize()

    # Setup distributed sampling
    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset, shuffle=True)
        sampler_val = DistributedSampler(opencood_validate_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(
            opencood_train_dataset,
            batch_sampler=batch_sampler_train,
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch_train
        )
        val_loader = DataLoader(
            opencood_validate_dataset,
            sampler=sampler_val,
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch_train,
            drop_last=False
        )
    else:
        train_loader = DataLoader(
            opencood_train_dataset,
            batch_size=hypes['train_params']['batch_size'],
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch_train,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            opencood_validate_dataset,
            batch_size=hypes['train_params']['batch_size'],
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch_train,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )
    
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module
    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer, onecycle_scheduler = train_utils.setup_optimizer(hypes, model)
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        # Clear memory before each epoch
        gc.collect()
        torch.cuda.empty_cache()
        
        if opt.distributed:
            sampler_train.set_epoch(epoch)
        
        total_loss = 0
        iou_list = []
        pbar = tqdm.tqdm(total=len(train_loader), leave=True)

        for i, batch_data in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)
            # breakpoint()
            # Handle half precision training
            if not opt.half:
                output_dict = model(batch_data['ego'])
                final_loss, loss_dict = criterion(output_dict, batch_data['ego']['gt_boxes'])
            else:
                with torch.cuda.amp.autocast():
                    output_dict = model(batch_data['ego'])
                    final_loss, loss_dict = criterion(output_dict, batch_data['ego']['gt_boxes'])
            
            # Loss logging
            total_loss += final_loss.item()
            if hasattr(criterion, 'iou') and i%10 == 0:
                iou_list.append(criterion.iou)
                # Print per-class IoU if available
                if hasattr(criterion, 'per_class_iou') and criterion.per_class_iou:
                    class_names = opencood_train_dataset.class_names  # ['vehicle', 'pedestrian', 'truck']
                    if not isinstance(class_names, list):
                        class_names = list(class_names)
                    iou_str = " | ".join([f"{class_names[i]}: {criterion.per_class_iou.get(i, 0.0):.3f}" 
                                        for i in range(len(class_names))])
                    print(f"IoU: {criterion.iou:.3f} | Per-class: {iou_str}")
                else:
                    print(f"IoU: {criterion.iou}")
            
            # Backward pass with gradient clipping
            if not opt.half:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()

            # Periodic memory cleanup
            if i % 20 == 0:
                torch.cuda.empty_cache()
                
            pbar.update(1)

        # Save model checkpoint
        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                      os.path.join(saved_path, f'net_epoch{epoch + 1}.pth'))

        # Log epoch statistics
        avg_loss = total_loss / len(train_loader)
        avg_iou = sum(iou_list) / len(iou_list) if iou_list else 0
        
        # Print overall stats
        print(f'\nEpoch {epoch}: Loss = {avg_loss:.4f}, IoU = {avg_iou:.4f}')
        
        # Print per-class IoU summary if available
        if hasattr(criterion, 'per_class_iou') and criterion.per_class_iou:
            class_names = opencood_train_dataset.class_names
            if not isinstance(class_names, list):
                class_names = list(class_names)
            print(f'Per-class IoU: ', end='')
            for i in range(len(class_names)):
                class_iou = criterion.per_class_iou.get(i, 0.0)
                print(f'{class_names[i]}: {class_iou:.3f}', end=' | ')
                writer.add_scalar(f'Train/IoU_{class_names[i]}', class_iou, epoch)
            print()  # New line
        
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        writer.add_scalar('Train/IoU', avg_iou, epoch)
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)

        # Validation loop
        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_losses = []
            valid_ious = []

            with torch.no_grad():
                for batch_data in val_loader:
                    model.eval()
                    batch_data = train_utils.to_device(batch_data, device)
                    output_dict = model(batch_data['ego'])
                    final_loss, _ = criterion(output_dict, batch_data['ego']['gt_boxes'])
                    valid_losses.append(final_loss.item())
                    if hasattr(criterion, 'iou'):
                        valid_ious.append(criterion.iou.item() if torch.is_tensor(criterion.iou) else criterion.iou)

            avg_valid_loss = statistics.mean(valid_losses)
            avg_valid_iou = statistics.mean(valid_ious) if valid_ious else 0
            print(f'Validation - Loss: {avg_valid_loss:.4f}, IoU: {avg_valid_iou:.4f}')
            writer.add_scalar('Validate/Loss', avg_valid_loss, epoch)
            writer.add_scalar('Validate/IoU', avg_valid_iou, epoch)

        # Dataset reinitialization
        # PERFORMANCE WARNING: This rebuilds the entire dataset from scratch every epoch!
        # This causes massive timing variations and should be optimized.
        # Commenting out for now - CAV shuffling can be handled more efficiently
        opencood_train_dataset.reinitialize()
        
        # TODO: Implement efficient CAV shuffling without full dataset rebuild
        # if epoch > 0:  # Skip first epoch to preserve original dataset order
        #     opencood_train_dataset.shuffle_cav_selection()  # More efficient approach

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
