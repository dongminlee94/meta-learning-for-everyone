"""
RL^2 trainer based on half-cheetah environment
"""

import numpy as np
import torch
import yaml

from src.rl2.algorithm.meta_learner import MetaLearner
from src.rl2.algorithm.ppo import PPO
from src.rl2.envs import ENVS

if __name__ == "__main__":
    # Experiment configuration setup
    with open("./configs/experiment_config.yaml", "r") as file:
        experiment_config = yaml.load(file, Loader=yaml.FullLoader)

    # Target reward configuration setup
    if experiment_config["env_name"] == "cheetah-dir":
        with open("./configs/dir_target_config.yaml", "r") as file:
            env_target_config = yaml.load(file, Loader=yaml.FullLoader)
    elif experiment_config["env_name"] == "cheetah-vel":
        with open("./configs/vel_target_config.yaml", "r") as file:
            env_target_config = yaml.load(file, Loader=yaml.FullLoader)

    # Create a multi-task environment and sample tasks
    env = ENVS[experiment_config["env_name"]](
        num_tasks=env_target_config["train_tasks"] + env_target_config["test_tasks"]
    )
    tasks = env.get_all_task_idx()

    # Set a random seed
    env.seed(experiment_config["seed"])
    np.random.seed(experiment_config["seed"])
    torch.manual_seed(experiment_config["seed"])

    observ_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    trans_dim = observ_dim + action_dim + 2
    hidden_dim = env_target_config["hidden_dim"]

    device = (
        torch.device("cuda", index=experiment_config["gpu_index"])
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    agent = PPO(
        trans_dim=trans_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        device=device,
        **env_target_config["ppo_params"],
    )

    meta_learner = MetaLearner(
        env=env,
        env_name=experiment_config["env_name"],
        agent=agent,
        trans_dim=trans_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        train_tasks=list(tasks[: env_target_config["train_tasks"]]),
        test_tasks=list(tasks[-env_target_config["test_tasks"] :]),
        exp_name=experiment_config["exp_name"],
        file_name=experiment_config["file_name"],
        device=device,
        **env_target_config["rl2_params"],
    )

    # Run RL^2 training
    meta_learner.meta_train()
