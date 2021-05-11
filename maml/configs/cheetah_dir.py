# PEARL cheetah-vel experiment settings

config = dict(
    env_name='cheetah-dir',
    
    n_train_tasks=2,                # default: 2
    
    n_eval_tasks=2,                 # default: 2
    
    # dimension of the latent context vector
    latent_size=5,                  # default: 5

    # number of hidden units in neural networks
    hidden_units='300,300,300',     # default: '300,300,300'

    # path to pre-trained weights to load into networks
    path_to_weights=None,           # default: None
    
    env_params=dict(
        # number of distinct tasks in the domain, should equal sum of train and eval tasks
        n_tasks=2,                  # default: 2
        
        # shuffle the tasks after creating them     
        randomize_tasks=True,       # default: True
    ),
    
    pearl_params=dict(
        # number of data sampling / training iterates 
        num_iterations=10,                  # default: 500
        
        # number of transitions collected per task before training 
        num_init_samples=10,                # default: 2000
        
        # number of sampled tasks to collect data for each iteration
        num_task_samples=5,                 # default: 5
        
        # number of transitions to collect per task with z ~ prior 
        num_prior_samples=10,               # default: 1000
        
        # number of transitions to collect per task with z ~ posterior 
        # that are only used to train the policy and NOT the encoder
        num_posterior_samples=10,           # default: 1000
        
        # number of meta-training taken per iteration
        num_meta_training=10,               # default: 2000
        
        # number of tasks to average the gradient across 
        meta_batch_size=4,                  # default: 4

        # number of transitions in the context batch
        batch_size=256,                     # default: 256
        
        # maximum step for the environment
        max_step=200,                       # default: 200
        
        # How many transitions to store
        max_buffer_size=int(1e6),           # default: int(1e6)
        
        # number of independent evals 
        num_evals=4,                        # default: 4
        
        # number of transitions to eval on
        num_steps_per_eval=600,             # default: 600
        
        # how many exploration trajs to collect before beginning posterior sampling at test time
        num_exp_traj_eval=2,                # default: 2
    ),
    
    sac_params=dict(
        
        # RL discount factor
        gamma=0.99,                     # default: 0.99
        
        # weight on KL divergence term in encoder loss
        kl_lambda=.1,                   # default: .1
        
        # number of transitions in the RL batch
        batch_size=256,                 # default: 256
        
        # scale rewards before constructing Bellman update, 
        # effectively controls weight on the entropy of the policy
        reward_scale=5.,                # default: 5.
        
        # Q-function network's learning rate
        qf_lr=3e-4,                     # default: 3e-4
        
        # Encoder network's learning rate
        encoder_lr=3e-4,                # default: 3e-4
        
        # Policy network's learning rate
        policy_lr=3e-4,                 # default: 3e-4
    ),
)



