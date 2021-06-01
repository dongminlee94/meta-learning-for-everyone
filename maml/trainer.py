"""
MAML trainer based on half-cheetah environment
"""

import argparse

import torch
from algorithm.maml import MAML
from algorithm.ppo import PPO
from configs.cheetah_dir import config as dir_config
from configs.cheetah_vel import config as vel_config
from envs import ENVS

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env", type=str, default="dir", help="Env to use: default cheetah-dir"
)
parser.add_argument("--gpu_index", type=int, default=0, help="Set a GPU index")
args = parser.parse_args()


if __name__ == "__main__":
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
        train_tasks=list(tasks[: config["n_train_tasks"]]),
        eval_tasks=list(tasks[-config["n_eval_tasks"] :]),
        device=device,
        **config["maml_params"],
    )

    # Run meta-train
    maml.meta_train()

    # Run meta-test
    # test_results = meta_learner.meta_test()
