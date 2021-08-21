"""
PEARL trainer based on half-cheetah environment
"""

import numpy as np
import torch
import yaml

from src.envs import ENVS
from src.pearl.algorithm.meta_learner import MetaLearner
from src.pearl.algorithm.sac import SAC

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
    hidden_dim = env_target_config["hidden_dim"]

    device = (
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
        train_tasks=list(tasks[: env_target_config["train_tasks"]]),
        test_tasks=list(tasks[-env_target_config["test_tasks"] :]),
        exp_name=experiment_config["exp_name"],
        file_name=experiment_config["file_name"],
        device=device,
        **env_target_config["pearl_params"],
    )

    # Run PEARL training
    meta_learner.meta_train()
