"""
troch_utils support PyTorch related methods
"""

from typing import Dict, List

import numpy as np
import torch


def np_to_torch_batch(np_batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Change numpy batch to torch batch"""
    batch_dict = {}
    for key, value in np_batch.items():
        if value.dtype == bool:
            value.astype(int)
        batch_dict[key] = torch.Tensor(value)
    return batch_dict


def unpack_batch(batch: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    """Unpack a batch and return individual elements"""
    cur_obs = batch["cur_obs"][None, ...]
    actions = batch["actions"][None, ...]
    rewards = batch["rewards"][None, ...]
    next_obs = batch["next_obs"][None, ...]
    dones = batch["dones"][None, ...]
    return [cur_obs, actions, rewards, next_obs, dones]
