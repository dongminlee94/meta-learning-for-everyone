"""
Meta-train and meta-test codes with RL^2 algorithm
"""

import datetime
import os
import time
from collections import deque
from typing import Any, Dict, List

import numpy as np
import torch
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from torch.utils.tensorboard import SummaryWriter

from meta_rl.rl2.algorithm.buffer import Buffer
from meta_rl.rl2.algorithm.ppo import PPO
from meta_rl.rl2.algorithm.sampler import Sampler


class MetaLearner:
    """RL^2 meta-learner class"""

    def __init__(
        self,
        env: HalfCheetahEnv,
        env_name: str,
        agent: PPO,
        trans_dim: int,
        action_dim: int,
        hidden_dim: int,
        train_tasks: List[int],
        test_tasks: List[int],
        save_exp_name: str,
        save_file_name: str,
        load_exp_name: str,
        load_file_name: str,
        load_ckpt_num: str,
        device: torch.device,
        **config,
    ) -> None:

        self.env = env
        self.env_name = env_name
        self.agent = agent
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks

        self.num_iterations: int = config["num_iterations"]
        self.meta_batch_size: int = config["meta_batch_size"]
        self.num_samples: int = config["num_samples"]

        self.batch_size: int = self.meta_batch_size * config["num_samples"]
        self.max_step: int = config["max_step"]

        self.sampler = Sampler(
            env=env,
            agent=agent,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_step=config["max_step"],
        )

        self.buffer = Buffer(
            trans_dim=trans_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_size=self.batch_size,
            device=device,
        )

        if not save_file_name:
            save_file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.result_path = os.path.join("results", save_exp_name, save_file_name)
        self.writer = SummaryWriter(log_dir=self.result_path)

        if load_exp_name and load_file_name:
            ckpt_path = os.path.join(
                "results",
                load_exp_name,
                load_file_name,
                "checkpoint_" + str(load_ckpt_num) + ".pt",
            )
            ckpt = torch.load(ckpt_path)

            self.agent.policy.load_state_dict(ckpt["policy"])
            self.agent.vf.load_state_dict(ckpt["vf"])
            self.buffer = ckpt["buffer"]

        # Set up early stopping condition
        self.dq: deque = deque(maxlen=config["num_stop_conditions"])
        self.num_stop_conditions: int = config["num_stop_conditions"]
        self.stop_goal: int = config["stop_goal"]
        self.is_early_stopping = False

    def meta_train(self) -> None:
        """RL^2 meta-training"""
        total_start_time = time.time()
        for iteration in range(self.num_iterations):
            start_time = time.time()

            print(f"=============== Iteration {iteration} ===============")
            # Sample data randomly from train tasks.
            indices = np.random.randint(len(self.train_tasks), size=self.meta_batch_size)
            for i, index in enumerate(indices):
                self.env.reset_task(index)
                self.agent.policy.is_deterministic = False

                print(f"[{i + 1}/{self.meta_batch_size}] collecting samples")
                trajs: List[Dict[str, np.ndarray]] = self.sampler.obtain_samples(
                    max_samples=self.num_samples,
                )
                self.buffer.add_trajs(trajs)

            # Get all samples for the train tasks
            batch = self.buffer.sample_batch()

            # Train the policy and the value function
            print(f"Start the meta-gradient update of iteration {iteration}")
            log_values = self.agent.train_model(self.batch_size, batch)

            # Evaluate on test tasks
            self.meta_test(iteration, total_start_time, start_time, log_values)

            if self.is_early_stopping:
                print(
                    f"\n================================================== \n"
                    f"The last {self.num_stop_conditions} meta-testing results are {self.dq}. \n"
                    f"And early stopping condition is {self.is_early_stopping}. \n"
                    f"Therefore, meta-training is terminated.",
                )
                break

    def visualize_within_tensorboard(self, test_results: Dict[str, Any], iteration: int) -> None:
        """Tensorboard visualization"""
        self.writer.add_scalar("test/return", test_results["return"], iteration)
        if self.env_name == "vel":
            self.writer.add_scalar("test/sum_run_cost", test_results["sum_run_cost"], iteration)
            for step in range(len(test_results["run_cost"])):
                self.writer.add_scalar(
                    "run_cost/iteration_" + str(iteration),
                    test_results["run_cost"][step],
                    step,
                )
        self.writer.add_scalar("train/total_loss", test_results["total_loss"], iteration)
        self.writer.add_scalar("train/policy_loss", test_results["policy_loss"], iteration)
        self.writer.add_scalar("train/value_loss", test_results["value_loss"], iteration)
        self.writer.add_scalar("time/total_time", test_results["total_time"], iteration)
        self.writer.add_scalar("time/time_per_iter", test_results["time_per_iter"], iteration)

    def meta_test(
        self,
        iteration: int,
        total_start_time: float,
        start_time: float,
        log_values: Dict[str, float],
    ) -> None:
        """RL^2 meta-testing"""
        test_results = {}
        test_return: float = 0
        test_run_cost = np.zeros(self.max_step)

        for index in self.test_tasks:
            self.env.reset_task(index)
            self.agent.policy.is_deterministic = True

            trajs: List[Dict[str, np.ndarray]] = self.sampler.obtain_samples(max_samples=self.max_step)
            test_return += np.sum(trajs[0]["rewards"]).item()

            if self.env_name == "vel":
                for i in range(self.max_step):
                    test_run_cost[i] += trajs[0]["infos"][i]

        # Collect meta-test results
        test_results["return"] = test_return / len(self.test_tasks)
        if self.env_name == "vel":
            test_results["run_cost"] = test_run_cost / len(self.test_tasks)
            test_results["sum_run_cost"] = np.sum(abs(test_results["run_cost"]))
        test_results["total_loss"] = log_values["total_loss"]
        test_results["policy_loss"] = log_values["policy_loss"]
        test_results["value_loss"] = log_values["value_loss"]
        test_results["total_time"] = time.time() - total_start_time
        test_results["time_per_iter"] = time.time() - start_time

        self.visualize_within_tensorboard(test_results, iteration)

        # Check if each element of self.dq satisfies early stopping condition
        if self.env_name == "dir":
            self.dq.append(test_results["return"])
            if all(list(map((lambda x: x >= self.stop_goal), self.dq))):
                self.is_early_stopping = True
        elif self.env_name == "vel":
            self.dq.append(test_results["sum_run_cost"])
            if all(list(map((lambda x: x <= self.stop_goal), self.dq))):
                self.is_early_stopping = True

        # Save the trained models
        if self.is_early_stopping:
            ckpt_path = os.path.join(self.result_path, "checkpoint_" + str(iteration) + ".pt")
            torch.save(
                {
                    "policy": self.agent.policy.state_dict(),
                    "vf": self.agent.vf.state_dict(),
                    "buffer": self.buffer,
                },
                ckpt_path,
            )
