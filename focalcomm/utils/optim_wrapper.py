import torch
import torch.nn as nn
from functools import partial
import math

class OptimWrapper:
    def __init__(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    @classmethod
    def create(cls, optimizer_func, lr, layer_groups, wd=0.01, momentum=0.9, true_wd=True, bn_wd=True):
        """Create an optimizer with layer-wise learning rates and weight decay."""
        def split_bn_bias(layer_groups):
            for layer in layer_groups:
                for module in layer.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        if not bn_wd:
                            module.weight.requires_grad_(False)
                            module.bias.requires_grad_(False)
                    
        split_bn_bias(layer_groups)
        
        param_groups = []
        for layer in layer_groups:
            param_groups.append({
                'params': layer.parameters(),
                'lr': lr,
                'weight_decay': wd if true_wd else 0.0,
                'momentum': momentum
            })
            
        optimizer = optimizer_func(param_groups)
        return cls(optimizer)

class OneCycleScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_lr, total_steps, 
                 pct_start=0.4, 
                 div_factor=10., 
                 final_div_factor=1e4,
                 moms=(0.9, 0.8052631),
                 decay_step_list=None,
                 lr_decay=0.1,
                 lr_clip=0.0000001,
                 last_epoch=-1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.moms = moms
        self.decay_step_list = decay_step_list
        self.lr_decay = lr_decay
        self.lr_clip = lr_clip
        
        self.step_size_up = int(total_steps * pct_start)
        self.step_size_down = total_steps - self.step_size_up
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Check if we're in decay steps
        if self.decay_step_list and self.last_epoch in self.decay_step_list:
            return [max(self.lr_clip, lr * self.lr_decay) for lr in self.base_lrs]
            
        if self.last_epoch <= self.step_size_up:
            # Learning rate annealing up
            pct = self.last_epoch / self.step_size_up
            cos_out = math.cos(math.pi * pct + math.pi) / 2 + 0.5
            return [max(self.lr_clip,
                    self.max_lr / self.div_factor + 
                    cos_out * (self.max_lr - self.max_lr / self.div_factor))
                    for _ in self.base_lrs]
        else:
            # Learning rate annealing down
            pct = (self.last_epoch - self.step_size_up) / self.step_size_down
            cos_out = math.cos(math.pi * pct) / 2 + 0.5
            return [max(self.lr_clip,
                    self.max_lr * cos_out + 
                    self.max_lr / (self.div_factor * self.final_div_factor))
                    for _ in self.base_lrs] 