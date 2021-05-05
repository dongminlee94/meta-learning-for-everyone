import torch
import numpy as np
from numbers import Number
from collections import OrderedDict


def hard_target_update(main, target):
      target.load_state_dict(main.state_dict())

def soft_target_update(main, target, tau=0.005):
    for main_param, target_param in zip(main.parameters(), target.parameters()):
        target_param.data.copy_(tau*main_param.data + (1.0-tau)*target_param.data)

def elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return torch.from_numpy(elem_or_tuple).float()

def filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v

def np_to_pytorch_batch(np_batch):
    return {
        k: elem_or_tuple_to_variable(x)
        for k, x in filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }

def unpack_batch(batch):
        ''' Unpack a batch and return individual elements '''
        obs = batch['obs'][None, ...]
        action = batch['action'][None, ...]
        reward = batch['reward'][None, ...]
        next_obs = batch['next_obs'][None, ...]
        done = batch['done'][None, ...]
        return [obs, action, reward, next_obs, done]
