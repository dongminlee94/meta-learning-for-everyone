"""
RL^2 cheetah-vel experiment settings
"""

config = dict(
    env_name="cheetah-vel",
    train_tasks=50,
    test_tasks=15,
    # number of random seed
    seed=0,
    # number of hidden units in neural networks
    hidden_dim=256,
    rl2_params=dict(
        # number of training iterations
        train_iters=500,
        # number of transitions to train
        train_samples=400,
        # number of transitions in the batch (train_tasks * train_samples)
        batch_size=800,
        # maximum step for the environment
        max_step=200,
        # number of transitions to test
        test_samples=200,
    ),
    ppo_params=dict(
        # discount factor
        gamma=0.99,
        # number of meta-gradient updates per iteration
        grad_iters=5,
        # number of minibatch within each epoch
        mini_batch_size=128,
        # PPO clip parameter
        clip_param=0.3,
        # learning rate of PPO losses
        learning_rate=3e-4,
    ),
)
