"""
PEARL cheetah-vel experiment settings
"""

config = dict(
    env_name="cheetah-dir",
    n_train_tasks=2,  # default: 2
    n_test_tasks=2,  # default: 2
    # number of random seed
    seed=0,
    # dimension of the latent context vector
    latent_size=5,  # default: 5
    # number of hidden units in neural networks
    hidden_units="300,300,300",  # default: '300,300,300'
    # path to pre-trained weights to load into networks
    path_to_weights=None,  # default: None
    pearl_params=dict(
        # number of data sampling / training iterates
        train_iters=200,  # default: 500
        # number of sampled tasks to collect data for each iteration
        train_task_iters=5,  # default: 5
        # number of transitions collected per task before training
        train_init_samples=2000,  # default: 2000
        # number of transitions to collect per task with z ~ prior
        train_prior_samples=1000,  # default: 1000
        # number of transitions to collect per task with z ~ posterior
        # that are only used to train the policy and NOT the encoder
        train_posterior_samples=1000,  # default: 1000
        # number of meta-gradient taken per iteration
        meta_grad_iters=2000,  # default: 2000
        # number of tasks to average the gradient across
        meta_batch_size=4,  # default: 4
        # number of transitions in the context batch
        batch_size=256,  # default: 256
        # maximum step for the environment
        max_step=200,  # default: 1000
        # How many transitions to store
        max_buffer_size=int(1e6),  # default: int(1e6)
        # number of independent testing
        test_iters=4,  # default: 4
        # number of transitions to test
        test_samples=200,  # default: 600
    ),
    sac_params=dict(
        # RL discount factor
        gamma=0.99,  # default: 0.99
        # weight on KL divergence term in encoder loss
        kl_lambda=0.1,  # default: 0.1
        # number of transitions in the RL batch
        batch_size=256,  # default: 256
        # Q-function network's learning rate
        qf_lr=3e-4,  # default: 3e-4
        # Encoder network's learning rate
        encoder_lr=3e-4,  # default: 3e-4
        # Policy network's learning rate
        policy_lr=3e-4,  # default: 3e-4
    ),
)
