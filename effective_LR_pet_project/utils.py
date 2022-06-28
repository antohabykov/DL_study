import random
import torch
import numpy as np
import sys

def weight_norm(model):
    """
    Computes and returns norm for all current weights
    """
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    params = torch.cat(params)
    norm = torch.norm(params)

    return norm

def eff_lr(model, optimizer, init_lr):
    """
    Takes current optimizer and computes the new one with updated by weight_norm() lr

    Returns updated optimizer and current lr for the history
    """
    lr = init_lr / weight_norm(model) #list with one tensor
    current_lr = lr.item()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer, current_lr

def get_optimizer(model_parameters, config):
    """Defines optimizer according to the given config"""
    if config["optim"] == 'SGD':
        optimizer = torch.optim.SGD(model_parameters,
                                    lr=config["lr"],
                                    weight_decay=config["reg"])
    elif config["optim"] == 'Adam':
        optimizer = torch.optim.Adam(model_parameters,
                                     lr=config["lr"],
                                     weight_decay=config["reg"])
    else:
        print('Un-known optimizer', config["optim"])
        sys.exit()

    return optimizer

def get_scheduler(optimizer, config):
    """
    Defines learning rate scheduler according to the given config
    """
    if config["sched"] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config['milestones'],
            gamma=config['gamma'])
    elif config["sched"] == "CyclicLR":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, 
            base_lr=config['eta_min'], 
            max_lr=config['lr'],
            step_size_up=config['step_size_up'],
            mode=config['mode'])
    elif config["sched"] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config["T_max"], 
            eta_min=config['eta_min'])
    elif config["sched"] == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=config['T_0'], 
            T_mult=config['T_mult'], 
            eta_min=config['eta_min'])
    else:
        scheduler = None
    return scheduler

def set_seed(seed):
    """ 
    Set initial seed for reproduction
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False