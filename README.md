# Meta-Reinforcement Learning Algorithms with PyTorch
## high-level structure
```
├── pybullet_envs: pybullet 물리 엔진을 이용한 환경들 구성
├── configs: 알고리즘을 실행할 때 사용할 configurations
├── trainer: main 함수 실행
└── algorithm: MAML 알고리즘을 동작하기 위한 폴더
    ├── metalearner: MAML 알고리즘 실행
    ├── ppo: meta-parameter 업데이트
    └── utils: metalearner와 sac에 필요한 도구들 모음
        ├── networks: 각종 neural networks
        ├── buffers: multi-task replay buffer를 구현
        ├── samplers: 환경과 interaction을 통해 sample들을 수집
        └── util: 각종 잡다하고 중복돼서 쓰이는 함수들 구현
```