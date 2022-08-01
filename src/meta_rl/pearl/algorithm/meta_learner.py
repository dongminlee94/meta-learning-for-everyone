import datetime
import os
import time
import warnings
from collections import deque
from typing import Any, Dict, List

warnings.filterwarnings("ignore")

import numpy as np
import torch
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from meta_rl.pearl.algorithm.buffers import MultiTaskReplayBuffer
from meta_rl.pearl.algorithm.sac import SAC
from meta_rl.pearl.algorithm.sampler import Sampler


class MetaLearner:
    def __init__(
        self,
        env: HalfCheetahEnv,
        env_name: str,
        agent: SAC,
        observ_dim: int,
        action_dim: int,
        train_tasks: List[int],
        test_tasks: List[int],
        save_exp_name: str,
        save_file_name: str,
        load_exp_name: str,
        load_file_name: str,
        load_ckpt_num: int,
        device: torch.device,
        **config,
    ) -> None:
        self.env = env
        self.env_name = env_name
        self.agent = agent
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        self.device = device

        self.num_iterations: int = config["num_iterations"]
        self.num_sample_tasks: int = config["num_sample_tasks"]

        self.num_init_samples: int = config["num_init_samples"]
        self.num_prior_samples: int = config["num_prior_samples"]
        self.num_posterior_samples: int = config["num_posterior_samples"]

        self.num_meta_grads: int = config["num_meta_grads"]
        self.meta_batch_size: int = config["meta_batch_size"]
        self.batch_size: int = config["batch_size"]
        self.max_step: int = config["max_step"]

        self.sampler = Sampler(env=env, agent=agent, max_step=config["max_step"], device=device)

        # 리플레이 버퍼를 분리하여 초기화
        # - RL 업데이트를 위한 버퍼
        # - 인코더 업데이트를 위한 버퍼
        self.rl_replay_buffer = MultiTaskReplayBuffer(
            observ_dim=observ_dim,
            action_dim=action_dim,
            tasks=train_tasks,
            max_size=config["max_buffer_size"],
        )
        self.encoder_replay_buffer = MultiTaskReplayBuffer(
            observ_dim=observ_dim,
            action_dim=action_dim,
            tasks=train_tasks,
            max_size=config["max_buffer_size"],
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
            self.agent.encoder.load_state_dict(ckpt["encoder"])
            self.agent.qf1.load_state_dict(ckpt["qf1"])
            self.agent.qf2.load_state_dict(ckpt["qf2"])
            self.agent.target_qf1.load_state_dict(ckpt["target_qf1"])
            self.agent.target_qf2.load_state_dict(ckpt["target_qf2"])
            self.agent.log_alpha = ckpt["log_alpha"]
            self.agent.alpha = ckpt["alpha"]
            self.rl_replay_buffer = ckpt["rl_replay_buffer"]
            self.encoder_replay_buffer = ckpt["encoder_replay_buffer"]

        # 조기 학습 중단 조건 설정
        self.dq: deque = deque(maxlen=config["num_stop_conditions"])
        self.num_stop_conditions: int = config["num_stop_conditions"]
        self.stop_goal: int = config["stop_goal"]
        self.is_early_stopping = False

    def collect_train_data(
        self,
        task_index: int,
        max_samples: int,
        update_posterior: bool,
        add_to_enc_buffer: bool,
    ) -> None:
        # 주어진 인덱스 태스크에 대한 경로 데이터 수집
        self.agent.encoder.clear_z()
        self.agent.policy.is_deterministic = False

        cur_samples = 0
        while cur_samples < max_samples:
            trajs, num_samples = self.sampler.obtain_samples(
                max_samples=max_samples - cur_samples,
                update_posterior=update_posterior,
                accum_context=False,
            )
            cur_samples += num_samples

            # RL 리플레이 버퍼에 수집한 데이터 저장
            self.rl_replay_buffer.add_trajs(task_index, trajs)
            if add_to_enc_buffer:
                # 인코더 리플레이 버퍼에 수집한 데이터 저장
                self.encoder_replay_buffer.add_trajs(task_index, trajs)

            if update_posterior:
                # 샘플한 context에 따른 posterior 업데이트
                context_batch = self.sample_context(np.array([task_index]))
                self.agent.encoder.infer_posterior(context_batch)

    def sample_context(self, indices: np.ndarray) -> torch.Tensor:
        # 인코더 버퍼에서 주어진 인덱스에 해당하는 태스크의 context 샘플
        context_batch = []
        for index in indices:
            batch = self.encoder_replay_buffer.sample_batch(task=index, batch_size=self.batch_size)
            context_batch.append(
                np.concatenate((batch["cur_obs"], batch["actions"], batch["rewards"]), axis=-1),
            )
        return torch.Tensor(context_batch).to(self.device)

    def sample_transition(self, indices: np.ndarray) -> List[torch.Tensor]:
        # RL 버퍼에서 주어진 인덱스에 해당하는 태스크의 경로 샘플
        cur_obs, actions, rewards, next_obs, dones = [], [], [], [], []
        for index in indices:
            batch = self.rl_replay_buffer.sample_batch(task=index, batch_size=self.batch_size)
            cur_obs.append(batch["cur_obs"])
            actions.append(batch["actions"])
            rewards.append(batch["rewards"])
            next_obs.append(batch["next_obs"])
            dones.append(batch["dones"])

        cur_obs = torch.Tensor(cur_obs).view(len(indices), self.batch_size, -1).to(self.device)
        actions = torch.Tensor(actions).view(len(indices), self.batch_size, -1).to(self.device)
        rewards = torch.Tensor(rewards).view(len(indices), self.batch_size, -1).to(self.device)
        next_obs = torch.Tensor(next_obs).view(len(indices), self.batch_size, -1).to(self.device)
        dones = torch.Tensor(dones).view(len(indices), self.batch_size, -1).to(self.device)
        return [cur_obs, actions, rewards, next_obs, dones]

    def meta_train(self) -> None:
        # 메타-트레이닝
        total_start_time: float = time.time()
        for iteration in range(self.num_iterations):
            start_time: float = time.time()

            # 첫번째 반복단계에 한해 모든 메타-트레이닝 태스크에 대한 경로를 수집하여 리플레이 버퍼에 저장
            if iteration == 0:
                print("Collecting initial pool of data for train and eval")
                for index in tqdm(self.train_tasks):
                    self.env.reset_task(index)
                    self.collect_train_data(
                        task_index=index,
                        max_samples=self.num_init_samples,
                        update_posterior=False,
                        add_to_enc_buffer=True,
                    )

            print(f"\n=============== Iteration {iteration} ===============")
            # 임의의 메타 트레이닝 태스크에 대한 새로운 경로를 버퍼에 저장
            for i in range(self.num_sample_tasks):
                index = np.random.randint(len(self.train_tasks))
                self.env.reset_task(index)
                self.encoder_replay_buffer.task_buffers[index].clear()

                #  샘플된 z ~ prior r(z)에 대한 경로 수집
                if self.num_prior_samples > 0:
                    print(f"[{i + 1}/{self.num_sample_tasks}] collecting samples with prior")
                    self.collect_train_data(
                        task_index=index,
                        max_samples=self.num_prior_samples,
                        update_posterior=False,
                        add_to_enc_buffer=True,
                    )

                # 인코더는 prior r(z)로 생성된 경로 데이터만을 사용하여 학습되나,
                # RL 정책의 학습에는 z ~ posterior q(z|c)로 생성된 경로도 사용
                if self.num_posterior_samples > 0:
                    print(f"[{i + 1}/{self.num_sample_tasks}] collecting samples with posterior")
                    self.collect_train_data(
                        task_index=index,
                        max_samples=self.num_posterior_samples,
                        update_posterior=True,
                        add_to_enc_buffer=False,
                    )

            # 샘플된 메타-배치 태스크들의 경로 데이터로 네트워크 업데이트
            print(f"Start meta-gradient updates of iteration {iteration}")
            for i in range(self.num_meta_grads):
                indices: np.ndarray = np.random.choice(self.train_tasks, self.meta_batch_size)

                # 인코더의 context와 은닉 상태 초기화
                self.agent.encoder.clear_z(num_tasks=len(indices))

                # Context 배치 샘플
                context_batch: torch.Tensor = self.sample_context(indices)

                # 경로 배치 샘플
                transition_batch: List[torch.Tensor] = self.sample_transition(indices)

                # 정책, Q-함수, 인코더 네트워크를 SAC 알고리즘에서 학습
                log_values: Dict[str, float] = self.agent.train_model(
                    meta_batch_size=self.meta_batch_size,
                    batch_size=self.batch_size,
                    context_batch=context_batch,
                    transition_batch=transition_batch,
                )

                # 인코더의 태스크변수 z의 Backpropagation 차단
                self.agent.encoder.task_z.detach()

            # 메타-테스트 태스크에서 학습성능 평가
            self.meta_test(iteration, total_start_time, start_time, log_values)

            if self.is_early_stopping:
                print(
                    f"\n==================================================\n"
                    f"The last {self.num_stop_conditions} meta-testing results are {self.dq}.\n"
                    f"And early stopping condition is {self.is_early_stopping}.\n"
                    f"Therefore, meta-training is terminated.",
                )
                break

    def collect_test_data(
        self,
        max_samples: int,
        update_posterior: bool,
    ) -> List[List[Dict[str, np.ndarray]]]:
        # 메타-테스트 태스크에 대한 경로 데이터 수집
        self.agent.encoder.clear_z()
        self.agent.policy.is_deterministic = True

        cur_trajs = []
        cur_samples = 0
        while cur_samples < max_samples:
            trajs, num_samples = self.sampler.obtain_samples(
                max_samples=max_samples - cur_samples,
                update_posterior=update_posterior,
                accum_context=True,
            )
            cur_trajs.append(trajs)
            cur_samples += num_samples

            # Context에 따른 posterior 업데이트
            self.agent.encoder.infer_posterior(self.agent.encoder.context)
        return cur_trajs

    def visualize_within_tensorboard(self, test_results: Dict[str, Any], iteration: int) -> None:
        # 메타-트레이닝 및 메타-테스팅 결과를 텐서보드에 기록
        self.writer.add_scalar(
            "test/return_before_infer",
            test_results["return_before_infer"],
            iteration,
        )
        self.writer.add_scalar("test/return_after_infer", test_results["return_after_infer"], iteration)
        if self.env_name == "vel":
            self.writer.add_scalar(
                "test/sum_run_cost_before_infer",
                test_results["sum_run_cost_before_infer"],
                iteration,
            )
            self.writer.add_scalar(
                "test/sum_run_cost_after_infer",
                test_results["sum_run_cost_after_infer"],
                iteration,
            )
            for step in range(len(test_results["run_cost_before_infer"])):
                self.writer.add_scalar(
                    "run_cost_before_infer/iteration_" + str(iteration),
                    test_results["run_cost_before_infer"][step],
                    step,
                )
                self.writer.add_scalar(
                    "run_cost_after_infer/iteration_" + str(iteration),
                    test_results["run_cost_after_infer"][step],
                    step,
                )
        self.writer.add_scalar("train/policy_loss", test_results["policy_loss"], iteration)
        self.writer.add_scalar("train/qf1_loss", test_results["qf1_loss"], iteration)
        self.writer.add_scalar("train/qf2_loss", test_results["qf2_loss"], iteration)
        self.writer.add_scalar("train/encoder_loss", test_results["encoder_loss"], iteration)
        self.writer.add_scalar("train/alpha_loss", test_results["alpha_loss"], iteration)
        self.writer.add_scalar("train/alpha", test_results["alpha"], iteration)
        self.writer.add_scalar("train/z_mean", test_results["z_mean"], iteration)
        self.writer.add_scalar("train/z_var", test_results["z_var"], iteration)
        self.writer.add_scalar("time/total_time", test_results["total_time"], iteration)
        self.writer.add_scalar("time/time_per_iter", test_results["time_per_iter"], iteration)

    def meta_test(
        self,
        iteration: int,
        total_start_time: float,
        start_time: float,
        log_values: Dict[str, float],
    ) -> None:
        # 메타-테스트
        test_results = {}
        return_before_infer = 0
        return_after_infer = 0
        run_cost_before_infer = np.zeros(self.max_step)
        run_cost_after_infer = np.zeros(self.max_step)

        for index in self.test_tasks:
            self.env.reset_task(index)
            trajs: List[List[Dict[str, np.ndarray]]] = self.collect_test_data(
                max_samples=self.max_step * 2,
                update_posterior=True,
            )

            return_before_infer += np.sum(trajs[0][0]["rewards"])
            return_after_infer += np.sum(trajs[1][0]["rewards"])
            if self.env_name == "vel":
                for i in range(self.max_step):
                    run_cost_before_infer[i] += trajs[0][0]["infos"][i]
                    run_cost_after_infer[i] += trajs[1][0]["infos"][i]

        test_results["return_before_infer"] = return_before_infer / len(self.test_tasks)
        test_results["return_after_infer"] = return_after_infer / len(self.test_tasks)
        if self.env_name == "vel":
            test_results["run_cost_before_infer"] = run_cost_before_infer / len(self.test_tasks)
            test_results["run_cost_after_infer"] = run_cost_after_infer / len(self.test_tasks)
            test_results["sum_run_cost_before_infer"] = sum(
                abs(run_cost_before_infer / len(self.test_tasks)),
            )
            test_results["sum_run_cost_after_infer"] = sum(
                abs(run_cost_after_infer / len(self.test_tasks)),
            )
        test_results["policy_loss"] = log_values["policy_loss"]
        test_results["qf1_loss"] = log_values["qf1_loss"]
        test_results["qf2_loss"] = log_values["qf2_loss"]
        test_results["encoder_loss"] = log_values["encoder_loss"]
        test_results["alpha_loss"] = log_values["alpha_loss"]
        test_results["alpha"] = log_values["alpha"]
        test_results["z_mean"] = log_values["z_mean"]
        test_results["z_var"] = log_values["z_var"]
        test_results["total_time"] = time.time() - total_start_time
        test_results["time_per_iter"] = time.time() - start_time

        self.visualize_within_tensorboard(test_results, iteration)

        # 학습 결과가 조기 중단 조건을 만족하는지 체크
        if self.env_name == "dir":
            self.dq.append(test_results["return_after_infer"])
            if all(list(map((lambda x: x >= self.stop_goal), self.dq))):
                self.is_early_stopping = True
        elif self.env_name == "vel":
            self.dq.append(test_results["sum_run_cost_after_infer"])
            if all(list(map((lambda x: x <= self.stop_goal), self.dq))):
                self.is_early_stopping = True

        # 학습 모델 저장
        if self.is_early_stopping:
            ckpt_path = os.path.join(self.result_path, "checkpoint_" + str(iteration) + ".pt")
            torch.save(
                {
                    "policy": self.agent.policy.state_dict(),
                    "encoder": self.agent.encoder.state_dict(),
                    "qf1": self.agent.qf1.state_dict(),
                    "qf2": self.agent.qf2.state_dict(),
                    "target_qf1": self.agent.target_qf1.state_dict(),
                    "target_qf2": self.agent.target_qf2.state_dict(),
                    "log_alpha": self.agent.log_alpha,
                    "alpha": self.agent.alpha,
                    "rl_replay_buffer": self.rl_replay_buffer,
                    "encoder_replay_buffer": self.encoder_replay_buffer,
                },
                ckpt_path,
            )
