from __future__ import  print_function, division
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class DictToObj:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_state(m):
    '''
      This code returns model states
    '''
    if m is None:
        return None
    state = {}
    for k, v in m.state_dict().items():
        state[k] = v.clone()
    return state

def load_model_states(path):
    '''
     Load previously learned model
    '''
    checkpoint = torch.load(path, map_location='cpu')
    m_states = checkpoint['model_states']
    m_params = checkpoint['args']
    if 'env_ob_rms' in checkpoint:
        env_ob_rms = checkpoint['env_ob_rms']
    else:
        env_ob_rms = None

    return m_states, DictToObj(**m_params), env_ob_rms

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    '''
        Decreases the learning rate linearly
    '''
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr