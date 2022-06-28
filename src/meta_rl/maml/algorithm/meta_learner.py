import datetime
import os
import time
import warnings
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import torch
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from meta_rl.maml.algorithm.buffer import MultiTaskBuffer
from meta_rl.maml.algorithm.optimizer import DifferentiableSGD
from meta_rl.maml.algorithm.sampler import Sampler
from meta_rl.maml.algorithm.trpo import TRPO


class MetaLearner:
    def __init__(
        self,
        env: HalfCheetahEnv,
        env_name: str,
        agent: TRPO,
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

        self.num_iterations = config["num_iterations"]
        self.meta_batch_size = config["meta_batch_size"]
        self.num_samples = config["num_samples"]
        self.max_steps = config["max_steps"]

        self.num_adapt_epochs = config["num_adapt_epochs"]
        self.backtrack_iters = config["backtrack_iters"]
        self.backtrack_coeff = config["backtrack_coeff"]
        self.max_kl = config["max_kl"]

        self.sampler = Sampler(
            env=env,
            agent=agent,
            action_dim=action_dim,
            max_step=config["max_steps"],
            device=device,
        )

        self.buffers = MultiTaskBuffer(
            observ_dim=observ_dim,
            action_dim=action_dim,
            agent=agent,
            num_tasks=max(self.meta_batch_size, len(self.test_tasks)),
            num_episodes=(self.num_adapt_epochs + 1),  # [num of adapatation for train] + [validation]
            max_size=self.num_samples,
            device=device,
        )

        self.inner_optimizer = DifferentiableSGD(
            self.agent.policy,
            lr=config["inner_learning_rate"],
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

        # 조기 학습 중단 조건 설정
        self.dq: deque = deque(maxlen=config["num_stop_conditions"])
        self.num_stop_conditions: int = config["num_stop_conditions"]
        self.stop_goal: int = config["stop_goal"]
        self.is_early_stopping = False

    def collect_train_data(self, indices: np.ndarray, is_eval: bool = False) -> None:
        # 경사하강 기반 태스크 적응을 동반한 경로 데이터 수집
        backup_params = dict(self.agent.policy.named_parameters())

        mode = "test" if is_eval else "train"
        print(f"Collecting samples for meta-{mode}")
        for cur_task, task_index in enumerate(tqdm(indices)):

            self.env.reset_task(task_index)

            # 내부 루프 (inner loop)
            # 각각의 태스크에 대한 경사하강 기반의 태스크 적응
            for cur_adapt in range(self.num_adapt_epochs + 1):

                # 메타-테스트에 대해서는 deterministic한 정책으로 경로 생성
                self.agent.policy.is_deterministic = (
                    True if cur_adapt == self.num_adapt_epochs and is_eval else False
                )

                # 학습 경로의 수집과 정책의 적응을 반복한 후 평가 경로를 수집
                trajs = self.sampler.obtain_samples(max_samples=self.num_samples)
                self.buffers.add_trajs(cur_task, cur_adapt, trajs)

                if cur_adapt < self.num_adapt_epochs:
                    train_batch = self.buffers.get_trajs(cur_task, cur_adapt)

                    # 정책의 태스크 적응
                    inner_loss = self.agent.policy_loss(train_batch)
                    self.inner_optimizer.zero_grad(set_to_none=True)
                    require_grad = cur_adapt < self.num_adapt_epochs - 1
                    inner_loss.backward(create_graph=require_grad)

                    with torch.set_grad_enabled(require_grad):
                        self.inner_optimizer.step()

            # 태스크 적응 이후의 정책 파라미터 저장
            self.buffers.add_params(
                cur_task,
                self.num_adapt_epochs,
                dict(self.agent.policy.named_parameters()),
            )

            # 태스크 적응 이전의 정책으로 복원
            self.agent.update_model(self.agent.policy, backup_params)
            self.agent.policy.is_deterministic = False

    def meta_surrogate_loss(self, set_grad: bool) -> Tuple[torch.Tensor, ...]:
        # 수집된 메타-배치 태스크의 데이터를 바탕으로 메타러닝 손실 계산
        losses, kls, entropies = [], [], []
        backup_params = dict(self.agent.policy.named_parameters())

        # 메타-배치 태스크에 대한 손실 계산
        for cur_task in range(self.meta_batch_size):
            # 내부 루프 (inner loop)
            # 각각의 태스크에 대한 경사하강 기반의 태스크 적응
            for cur_adapt in range(self.num_adapt_epochs):

                require_grad = cur_adapt < self.num_adapt_epochs - 1 or set_grad

                # 버퍼에 저장된 태스크 학습 경로 얻기
                train_batch = self.buffers.get_trajs(cur_task, cur_adapt)

                # 액터-크리틱 알고리즘을 사용한 정책의 태스크 적응
                inner_loss = self.agent.policy_loss(train_batch)
                self.inner_optimizer.zero_grad(set_to_none=True)
                inner_loss.backward(create_graph=require_grad)

                with torch.set_grad_enabled(require_grad):
                    self.inner_optimizer.step()

            # Surrogate 손실 계산을 위해 line search 초기 정책으로 초기화
            valid_params = self.buffers.get_params(cur_task, self.num_adapt_epochs)
            self.agent.update_model(self.agent.old_policy, valid_params)

            # 메타-배치 태스크에 대한 메타러닝 손실로서 평가경로의 surrogage 손실 계산
            valid_batch = self.buffers.get_trajs(cur_task, self.num_adapt_epochs)
            loss = self.agent.policy_loss(valid_batch, is_meta_loss=True)
            losses.append(loss)

            # 배치 태스크에 대한 평가 경로의 평균 KL divergence 계산
            kl = self.agent.kl_divergence(valid_batch)
            kls.append(kl)

            # 배치 태스크에 대한 평가 경로의 평균 정책 엔트로피 계산
            entropy = self.agent.compute_policy_entropy(valid_batch)
            entropies.append(entropy)

            self.agent.update_model(self.agent.policy, backup_params)
        return torch.stack(losses).mean(), torch.stack(kls).mean(), torch.stack(entropies).mean()

    def meta_update(self) -> Dict[str, float]:
        # 외부 루프 (outer loop)
        # Line search를 시작하기 위한 첫 경사하강 스텝 계산
        loss_before, kl_before, _ = self.meta_surrogate_loss(set_grad=True)

        gradient = torch.autograd.grad(loss_before, self.agent.policy.parameters(), retain_graph=True)
        gradient = self.agent.flat_grad(gradient)
        Hvp = self.agent.hessian_vector_product(kl_before, self.agent.policy.parameters())

        search_dir = self.agent.conjugate_gradient(Hvp, gradient)
        descent_step = self.agent.compute_descent_step(Hvp, search_dir, self.max_kl)
        loss_before.detach_()

        # Line search 역추적을 통한 파라미터 업데이트
        backup_params = deepcopy(dict(self.agent.policy.named_parameters()))
        for i in range(self.backtrack_iters):
            ratio = self.backtrack_coeff**i

            for params, step in zip(self.agent.policy.parameters(), descent_step):
                params.data.add_(step, alpha=-ratio)

            loss_after, kl_after, policy_entropy = self.meta_surrogate_loss(set_grad=False)

            # KL 제약조건을 만족할 경우 정책 업데이트
            is_improved = loss_after < loss_before
            is_constrained = kl_after <= self.max_kl
            print(f"{i}-Backtracks | Loss {loss_after:.4f} < Loss_old {loss_before:.4f} : ", end="")
            print(f"{is_improved} | KL {kl_after:.4f} <= maxKL {self.max_kl:.4f} : {is_constrained}")

            if is_improved and is_constrained:
                print(f"Update meta-policy through {i+1} backtracking line search step(s)")
                break

            self.agent.update_model(self.agent.policy, backup_params)

            if i == self.backtrack_iters - 1:
                print("Keep current meta-policy skipping meta-update")

        self.buffers.clear()
        return dict(
            loss_after=loss_after.item(),
            kl_after=kl_after.item(),
            policy_entropy=policy_entropy.item(),
        )

    def meta_train(self) -> None:
        # 메타-트레이닝
        total_start_time = time.time()
        for iteration in range(self.num_iterations):
            start_time = time.time()

            print(f"\n=============== Iteration {iteration} ===============")
            # 메타-배치 태스크에 대한 데이터 수집
            indices = np.random.randint(len(self.train_tasks), size=self.meta_batch_size)
            self.collect_train_data(indices)

            # 경사하강 기반의 메타-업데이트
            log_values = self.meta_update()

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

    def visualize_within_tensorboard(self, results_summary: Dict[str, Any], iteration: int) -> None:
        # 메타-트레이닝 및 메타-테스트 결과를 텐서보드에 기록
        self.writer.add_scalar("train/loss_after", results_summary["loss_after"], iteration)
        self.writer.add_scalar("train/kl_after", results_summary["kl_after"], iteration)
        self.writer.add_scalar("train/policy_entropy", results_summary["policy_entropy"], iteration)

        self.writer.add_scalar(
            "test/return_before_grad",
            results_summary["return_before_grad"],
            iteration,
        )
        self.writer.add_scalar(
            "test/return_after_grad",
            results_summary["return_after_grad"],
            iteration,
        )
        if self.env_name == "vel":
            self.writer.add_scalar(
                "test/sum_run_cost_before_grad",
                results_summary["sum_run_cost_before_grad"],
                iteration,
            )
            self.writer.add_scalar(
                "test/sum_run_cost_after_grad",
                results_summary["sum_run_cost_after_grad"],
                iteration,
            )
            for step in range(len(results_summary["run_cost_before_grad"])):
                self.writer.add_scalar(
                    "run_cost_before_grad/iteration_" + str(iteration),
                    results_summary["run_cost_before_grad"][step],
                    step,
                )
                self.writer.add_scalar(
                    "run_cost_after_grad/iteration_" + str(iteration),
                    results_summary["run_cost_after_grad"][step],
                    step,
                )

        self.writer.add_scalar("time/total_time", results_summary["total_time"], iteration)
        self.writer.add_scalar("time/time_per_iter", results_summary["time_per_iter"], iteration)

    def meta_test(
        self,
        iteration: int,
        total_start_time: float,
        start_time: float,
        log_values: Dict[str, float],
    ) -> None:
        # 메타-테스트
        results_summary = {}
        returns_before_grad = []
        returns_after_grad = []
        run_costs_before_grad = []
        run_costs_after_grad = []

        results_summary["loss_after"] = log_values["loss_after"]
        results_summary["kl_after"] = log_values["kl_after"]
        results_summary["policy_entropy"] = log_values["policy_entropy"]

        results_summary["total_time"] = time.time() - total_start_time
        results_summary["time_per_iter"] = time.time() - start_time

        self.collect_train_data(np.array(self.test_tasks), is_eval=True)

        for task in range(len(self.test_tasks)):
            batch_before_grad = self.buffers.get_trajs(task, 0)
            batch_after_grad = self.buffers.get_trajs(task, self.num_adapt_epochs)

            rewards_before_grad = batch_before_grad["rewards"][: self.max_steps]
            rewards_after_grad = batch_after_grad["rewards"][: self.max_steps]
            returns_before_grad.append(torch.sum(rewards_before_grad).item())
            returns_after_grad.append(torch.sum(rewards_after_grad).item())

            if self.env_name == "vel":
                run_costs_before_grad.append(
                    batch_before_grad["infos"][: self.max_steps].cpu().numpy(),
                )
                run_costs_after_grad.append(
                    batch_after_grad["infos"][: self.max_steps].cpu().numpy(),
                )

        run_cost_before_grad = np.sum(run_costs_before_grad, axis=0)
        run_cost_after_grad = np.sum(run_costs_after_grad, axis=0)

        self.buffers.clear()

        results_summary["return_before_grad"] = sum(returns_before_grad) / len(self.test_tasks)
        results_summary["return_after_grad"] = sum(returns_after_grad) / len(self.test_tasks)
        if self.env_name == "vel":
            results_summary["run_cost_before_grad"] = run_cost_before_grad / len(self.test_tasks)
            results_summary["run_cost_after_grad"] = run_cost_after_grad / len(self.test_tasks)
            results_summary["sum_run_cost_before_grad"] = sum(
                abs(run_cost_before_grad / len(self.test_tasks)),
            )
            results_summary["sum_run_cost_after_grad"] = sum(
                abs(run_cost_after_grad / len(self.test_tasks)),
            )

        # 학습 결과가 조기 중단 조건을 만족하는지 체크
        self.dq.append(results_summary["return_after_grad"])
        if all(list(map((lambda x: x >= self.stop_goal), self.dq))):
            self.is_early_stopping = True

        # 학습 모델 저장
        if self.is_early_stopping:
            ckpt_path = os.path.join(self.result_path, "checkpoint_" + str(iteration) + ".pt")
            torch.save({"policy": self.agent.policy.state_dict()}, ckpt_path)

        self.visualize_within_tensorboard(results_summary, iteration)
