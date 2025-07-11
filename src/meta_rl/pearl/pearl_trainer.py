import os
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv

from meta_rl.envs import ENVS
from meta_rl.pearl.algorithm.meta_learner import MetaLearner
from meta_rl.pearl.algorithm.sac import SAC

if __name__ == "__main__":
    # Load hyperparameters for experimental environment settings
    with open(os.path.join("configs", "experiment_config.yaml"), "r", encoding="utf-8") as file:
        experiment_config: Dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)

    # Load hyperparameters for target reward settings
    with open(
        os.path.join("configs", experiment_config["env_name"] + "_target_config.yaml"),
        "r",
        encoding="utf-8",
    ) as file:
        env_target_config: Dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)

    # Create multi-task environment and sample tasks
    env: HalfCheetahEnv = ENVS["cheetah-" + experiment_config["env_name"]](
        num_tasks=env_target_config["train_tasks"] + env_target_config["test_tasks"],
    )
    tasks: List[int] = env.get_all_task_idx()

    # Set random seed values
    env.reset(seed=experiment_config["seed"])
    np.random.seed(experiment_config["seed"])
    torch.manual_seed(experiment_config["seed"])

    observ_dim: int = env.observation_space.shape[0]
    action_dim: int = env.action_space.shape[0]
    hidden_dim: int = env_target_config["hidden_dim"]

    device: torch.device = (
        torch.device("cuda", index=experiment_config["gpu_index"])
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    agent = SAC(
        observ_dim=observ_dim,
        action_dim=action_dim,
        latent_dim=env_target_config["latent_dim"],
        hidden_dim=hidden_dim,
        encoder_input_dim=observ_dim + action_dim + 1,
        encoder_output_dim=env_target_config["latent_dim"] * 2,
        device=device,
        **env_target_config["sac_params"],
    )

    meta_learner = MetaLearner(
        env=env,
        env_name=experiment_config["env_name"],
        agent=agent,
        observ_dim=observ_dim,
        action_dim=action_dim,
        train_tasks=tasks[: env_target_config["train_tasks"]],
        test_tasks=tasks[-env_target_config["test_tasks"] :],
        save_exp_name=experiment_config["save_exp_name"],
        save_file_name=experiment_config["save_file_name"],
        load_exp_name=experiment_config["load_exp_name"],
        load_file_name=experiment_config["load_file_name"],
        load_ckpt_num=experiment_config["load_ckpt_num"],
        device=device,
        **env_target_config["pearl_params"],
    )

    # Start PEARL training
    meta_learner.meta_train()
