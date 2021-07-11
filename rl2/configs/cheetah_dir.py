"""
RL^2 cheetah-dir experiment settings
"""

config = dict(
    env_name="cheetah-dir",
    train_tasks=2,
    test_tasks=2,
    # number of random seed
    seed=0,
    # number of hidden units in neural networks
    hidden_dim=256,
    rl2_params=dict(
        # number of training iterates
        train_iters=1,
        # number of transitions to collect per task
        train_samples=400,
        # number of meta-gradient iterations per iteration
        train_grad_iters=30,
        # maximum step for the environment
        max_step=200,
        # How many transitions to store (train_tasks * train_samples)
        max_buffer_size=800,
    ),
    ppo_params=dict(
        # discount factor
        gamma=0.99,
        # number of timesteps collected for each meta-gradient update
        batch_size=800,
        # number of minibatch within each epoch
        minibatch_size=128,
        # PPO clip parameter
        clip_param=0.3,
        # learning rate of PPO losses
        learning_rate=3e-4,
    ),
)
