"""
MAML trainer based on half-cheetah environment
"""

import argparse

import numpy as np
import torch

from maml.algorithm.maml import MAML
from maml.algorithm.ppo import PPO
from maml.configs.cheetah_dir import config as dir_config
from maml.configs.cheetah_vel import config as vel_config
from maml.envs import ENVS

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="vel", help="Set an environment to use")
parser.add_argument(
    "--exp-name", type=str, default="exp_1", help="Set an experiment name"
)
parser.add_argument("--file-name", type=str, default=None, help="Set a file name")
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
    hidden_units = list(map(int, config["hidden_units"].split(",")))

    device = (
        torch.device("cuda", index=args.gpu_index)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    agent = PPO(
        observ_dim=observ_dim,
        action_dim=action_dim,
        hidden_units=hidden_units,
        env_target=config["env_name"],
        device=device,
        **config["ppo_params"],
    )

    maml = MAML(
        env=env,
        agent=agent,
        observ_dim=observ_dim,
        action_dim=action_dim,
        train_tasks=list(tasks[: config["train_tasks"]]),
        eval_tasks=list(tasks[-config["eval_tasks"] :]),
        exp_name=args.exp_name,
        file_name=args.file_name,
        device=device,
        **config["maml_params"],
    )

    # Run PEARL training
    maml.meta_train()
