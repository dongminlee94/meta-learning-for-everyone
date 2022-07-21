import os
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv

from meta_rl.envs import ENVS
from meta_rl.maml.algorithm.meta_learner import MetaLearner
from meta_rl.maml.algorithm.trpo import TRPO

if __name__ == "__main__":
    # 실험 환경 설정에 대한 하이퍼파라미터들 불러오기
    with open(os.path.join("configs", "experiment_config.yaml"), "r", encoding="utf-8") as file:
        experiment_config: Dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)

    # 목표 보상 설정에 대한 하이퍼파라미터들 불러오기
    with open(
        os.path.join("configs", experiment_config["env_name"] + "_target_config.yaml"),
        "r",
        encoding="utf-8",
    ) as file:
        env_target_config: Dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)

    # 멀티-태스크 환경과 샘플 태스크들 생성
    env: HalfCheetahEnv = ENVS["cheetah-" + experiment_config["env_name"]](
        num_tasks=env_target_config["train_tasks"] + env_target_config["test_tasks"],
    )
    tasks: List[int] = env.get_all_task_idx()

    # 랜덤 시드 값 설정
    env.reset(seed=experiment_config["seed"])
    np.random.seed(experiment_config["seed"])
    torch.manual_seed(experiment_config["seed"])

    observ_dim: int = env.observation_space.shape[0]
    action_dim: int = env.action_space.shape[0]
    policy_hidden_dim: int = env_target_config["policy_hidden_dim"]
    vf_hidden_dim: int = env_target_config["value_function_hidden_dim"]

    device: torch.device = (
        torch.device("cuda", index=experiment_config["gpu_index"])
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    agent = TRPO(
        observ_dim=observ_dim,
        action_dim=action_dim,
        policy_hidden_dim=policy_hidden_dim,
        vf_hidden_dim=vf_hidden_dim,
        device=device,
        **env_target_config["pg_params"],
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
        **env_target_config["maml_params"],
    )

    # MAML 학습 시작
    meta_learner.meta_train()
