"""
PEARL trainer based on half-cheetah environment
"""

import argparse

import torch
from algorithm.pearl import PEARL
from algorithm.sac import SAC
from configs.cheetah_dir import config as dir_config
from configs.cheetah_vel import config as vel_config
from envs import ENVS

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env", type=str, default="dir", help="Env to use: default cheetah-dir"
)
parser.add_argument("--gpu-index", type=int, default=0, help="Set a GPU index")
parser.add_argument("--filename", type=str, default="exp-1", help="Set a file name")


if __name__ == "__main__":
    args = parser.parse_args()

    # Create a multi-task environment and sample tasks
    if args.env == "dir":
        config = dir_config
        env = ENVS[config["env_name"]]()
    elif args.env == "vel":
        config = vel_config
        env = ENVS[config["env_name"]](**config["env_params"])
    env.seed(config["seed"])
    tasks = env.get_all_task_idx()

    observ_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_units = list(map(int, config["hidden_units"].split(",")))

    device = (
        torch.device("cuda", index=args.gpu_index)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    agent = SAC(
        observ_dim=observ_dim,
        action_dim=action_dim,
        latent_dim=config["latent_size"],
        hidden_units=hidden_units,
        encoder_input_dim=observ_dim + action_dim + 1,
        encoder_output_dim=config["latent_size"] * 2,
        device=device,
        **config["sac_params"],
    )

    pearl = PEARL(
        env=env,
        agent=agent,
        observ_dim=observ_dim,
        action_dim=action_dim,
        train_tasks=list(tasks[: config["n_train_tasks"]]),
        test_tasks=list(tasks[-config["n_test_tasks"] :]),
        filename=args.filename,
        device=device,
        **config["pearl_params"],
    )

    # Run PEARL training
    pearl.meat_train()
