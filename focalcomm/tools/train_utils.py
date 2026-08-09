# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


import glob
import importlib
import yaml
import os
import re
from datetime import datetime
from focalcomm.utils.optim_wrapper import *
import torch
import torch.optim as optim
import torch.nn as nn
from functools import partial


def load_saved_model(saved_path, model, epoch=None):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        if os.path.exists(os.path.join(saved_path, 'latest.pth')):
            return 10000
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    if epoch is None:
        initial_epoch = findLastCheckpoint(saved_path)
    else:
        initial_epoch = int(epoch)

    if initial_epoch > 0:
        model_file = os.path.join(saved_path,
                         'net_epoch%d.pth' % initial_epoch) \
            if initial_epoch != 10000 else os.path.join(saved_path,
                         'latest.pth')
        print('resuming by loading epoch %d' % initial_epoch)
        
        # Load checkpoint
        checkpoint = torch.load(model_file, map_location='cpu')
        
        # Filter out accumulated_mask from checkpoint
        if isinstance(checkpoint, dict):
            filtered_checkpoint = {k: v for k, v in checkpoint.items() 
                                if 'accumulated_mask' not in k}
        
        # Load filtered state dict
        model.load_state_dict(filtered_checkpoint, strict=False)
        del checkpoint

    return initial_epoch, model


def setup_train(hypes):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')

    full_path = os.path.join(current_path, folder_name)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
            except FileExistsError:
                pass
        # save the yaml file
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

    return full_path



def create_model(hypes):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']

    model_filename = "focalcomm.models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('backbone not found in models folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (model_filename,
                                                       target_model_name))
        exit(0)
    
    instance = model(backbone_config)
    
    # Check if we need to load pretrained V2XViT weights
    if 'pretrained' in hypes['model'] and hypes['model']['pretrained']:
        pretrained_path = hypes['model']['pretrained_path']
        print(f'Loading pretrained V2XViT weights from {pretrained_path}')
        
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
        model_dict = instance.state_dict()
        
        # Filter out multi-stage components and incompatible layers
        filtered_dict = {k: v for k, v in pretrained_dict.items() 
                        if k in model_dict and not any(x in k for x in 
                        ['stage_blocks', 'heatmap_heads', 'accumulated_mask'])}
        
        # Update model weights
        model_dict.update(filtered_dict)
        instance.load_state_dict(model_dict)
        
        # Optionally freeze pretrained parts if specified
        if hypes['model'].get('freeze_backbone', False):
            for name, param in instance.named_parameters():
                if not any(x in name for x in 
                    ['stage_blocks', 'heatmap_heads', 'accumulated_mask']):
                    param.requires_grad = False
    
    return instance


def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss']['core_method']
    loss_func_config = hypes['loss']['args']

    loss_filename = "focalcomm.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes, model):
    """Create optimizer based on configuration.
    
    Args:
        hypes (dict): Configuration dictionary
        model (nn.Module): Model to optimize
    """
    optim_cfg = hypes['optimization']
    
    if optim_cfg['optimizer'] in ['adam_onecycle', 'adam_cosineanneal']:
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        # Configure the optimizer function
        betas = tuple(optim_cfg.get('betas', [0.9, 0.99]))
        optimizer_func = partial(
            optim.Adam, 
            betas=betas
        )
        
        # Create optimizer with param groups
        optimizer_wrapper = OptimWrapper.create(
            optimizer_func=optimizer_func,
            lr=optim_cfg['lr'],
            layer_groups=get_layer_groups(model),
            wd=optim_cfg['weight_decay'],
            momentum=optim_cfg.get('momentum', 0.9),   # read momentum
            true_wd=True,
            bn_wd=True
        )
        optimizer = optimizer_wrapper.optimizer
        
        # Create scheduler
        if optim_cfg['optimizer'] == 'adam_onecycle':
            scheduler = OneCycleScheduler(
                optimizer=optimizer,
                max_lr=optim_cfg['lr'],
                total_steps=optim_cfg['num_epochs'],
                pct_start=optim_cfg.get('pct_start', 0.4),
                div_factor=optim_cfg.get('div_factor', 10),
                final_div_factor=optim_cfg.get('final_div_factor', 1e4),
                moms=tuple(optim_cfg.get('moms', [0.9, 0.8])),
                decay_step_list=optim_cfg.get('decay_step_list', None),
                lr_decay=optim_cfg.get('lr_decay', 0.1),
                lr_clip=optim_cfg.get('lr_clip', 1e-7)
            )
        else:
            # adam_cosineanneal path
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=optim_cfg['num_epochs']
            )
        
        return optimizer, scheduler

    else:
        # Original optimizer logic if user picks 'adam', 'sgd', etc.
        method_dict = hypes['optimizer']
        optimizer_method = getattr(optim, method_dict['core_method'], None)
        if not optimizer_method:
            raise ValueError(f"{method_dict['core_method']} is not supported.")
            
        optimizer = optimizer_method(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=method_dict['lr'],
            **method_dict.get('args', {})
        )
        return optimizer, None


def setup_lr_schedular(hypes, optimizer):
    """
    Set up the learning rate schedular.

    Parameters
    ----------
    hypes : dict
        The training configurations.
    optimizer : torch.optimizer
    """
    # If using OneCycle or CosineAnnealing, scheduler is already created
    if hypes['optimization']['optimizer'] in ['adam_onecycle', 'adam_cosineanneal']:
        return None  # Scheduler already created in setup_optimizer
        
    # Traditional scheduler setup for other optimizers
    if 'lr_scheduler' not in hypes:
        return None  # No scheduler needed
        
    lr_schedule_config = hypes['lr_scheduler']

    if lr_schedule_config['core_method'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer,
                              milestones=milestones,
                              gamma=gamma)

    else:
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    return scheduler


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str):
            return inputs
        return inputs.to(device)
