"""
RL^2 trainer based on half-cheetah environment
"""

import argparse

import numpy as np
import torch

from rl2.algorithm.ppo import PPO
from rl2.algorithm.rl2 import RL2
from rl2.configs.cheetah_dir import config as dir_config
from rl2.configs.cheetah_vel import config as vel_config
from rl2.envs import ENVS

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="dir", help="Set an environment to use")
parser.add_argument(
    "--exp-name", type=str, default="exp_5", help="Set an experiment name"
)
parser.add_argument("--file-name", type=str, default="0.0", help="Set a file name")
parser.add_argument("--gpu-index", type=int, default=0, help="Set a GPU index")


if __name__ == "__main__":
    args = parser.parse_args()

    # Create a multi-task environment and sample tasks
    if args.env == "dir":
        config = dir_config
        env = ENVS[config["env_name"]]()
    elif args.env == "vel":
        config = vel_config
        env = ENVS[config["env_name"]](
            num_tasks=config["train_tasks"] + config["test_tasks"]
        )
    tasks = env.get_all_task_idx()

    # Set a random seed
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    observ_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    trans_dim = observ_dim + action_dim + 2
    hidden_dim = config["hidden_dim"]

    device = (
        torch.device("cuda", index=args.gpu_index)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    agent = PPO(
        env_name=config["env_name"],
        trans_dim=trans_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        device=device,
        **config["ppo_params"],
    )

    rl2 = RL2(
        env=env,
        env_name=config["env_name"],
        agent=agent,
        trans_dim=trans_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        train_tasks=list(tasks[: config["train_tasks"]]),
        test_tasks=list(tasks[-config["test_tasks"] :]),
        exp_name=args.exp_name,
        file_name=args.file_name,
        device=device,
        **config["rl2_params"],
    )

    # Run RL^2 training
    rl2.meta_train()
