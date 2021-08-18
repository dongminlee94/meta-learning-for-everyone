"""
troch_utils support PyTorch related methods
"""

import numpy as np
import torch


def ndarray_to_tensor(elements):
    """Change numpy elements to torch variable"""
    if isinstance(elements, tuple):
        return tuple(ndarray_to_tensor(e) for e in elements)
    return torch.from_numpy(elements).float()


def filter_batch(np_batch):
    """Filter bool type into int type in numpy batch"""
    for key, value in np_batch.items():
        if value.dtype == np.bool:
            yield key, value.astype(int)
        else:
            yield key, value


def np_to_torch_batch(np_batch):
    """Change numpy batch to torch batch"""
    return {
        k: ndarray_to_tensor(x)
        for k, x in filter_batch(np_batch)
        if x.dtype != np.dtype("O")  # ignore object (e.g. dictionaries)
    }


def unpack_batch(batch):
    """Unpack a batch and return individual elements"""
    cur_obs = batch["cur_obs"][None, ...]
    actions = batch["actions"][None, ...]
    rewards = batch["rewards"][None, ...]
    next_obs = batch["next_obs"][None, ...]
    dones = batch["dones"][None, ...]
    return [cur_obs, actions, rewards, next_obs, dones]
