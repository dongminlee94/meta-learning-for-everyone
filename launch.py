import os
import gym
import time
import yaml
import torch
import argparse
import datetime
import numpy as np

from meta_rl.envs import envs
from meta_rl.envs.wrappers import NormalizedBoxEnv
from meta_rl.policies.gaussian_policy import GaussianPolicy
from meta_rl.baselines.linear_baseline import LinearFeatureBaseline

parser = argparse.ArgumentParser(description='Meta-RL algorithms with PyTorch in MuJoCo environments')
parser.add_argument('--config', type=str, required=True,
                    help='path to the configuration file.')
parser.add_argument('--algo', type=str, default='maml-trpo', 
                    help='select an algorithm among maml-trpo, metasgd-trpo, reptile-trpo, promp-trpo')
parser.add_argument('--seed', type=int, default=0,
                    help='seed for random number generators')
parser.add_argument('--num_workers', type=int, default=2,
                    help='number of workers for trajectories sampling')
parser.add_argument('--use_cuda', action='store_true',
                    help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
                         'is not guaranteed. Using CPU is encouraged.')
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.algo == 'maml-trpo':
    from meta_rl.algorithms.maml_trpo import MAMLTRPO
# elif args.algo == 'metasgd-trpo':
#     from agents.metasgd_trpo import Agent
# elif args.algo == 'reptile-trpo':
#     from agents.metasgd_trpo import Agent
# elif args.algo == 'promp-trpo':
#     from agents.promp_trpo import Agent


def main():
    """Main."""
    # Load configurations
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Initialize an environment
    env = NormalizedBoxEnv(envs[config['env_name']](config['env_params']))
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(env)
    print(obs_dim)
    print(act_dim)

    # Create sample tasks
    tasks = env.get_all_task_idx()
    train_tasks=list(tasks[:2])
    eval_tasks=list(tasks[-2:])
    
    # Set a random seed
    if args.seed is not None:
        env.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    baseline = LinearFeatureBaseline(env)
    
    # if args.algo == 'maml-trpo':
    #     algorithm = MAMLTRPO(env, args, obs_dim, act_dim, tasks, baseline)
    # # elif args.algo == 'metasgd-trpo':
    # #     algorithm = MetaSGDTRPO()

    # algorithm.train()

if __name__ == '__main__':
    main()