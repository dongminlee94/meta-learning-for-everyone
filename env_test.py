import sys
import gym 

from meta_rl.envs import envs

config = dict(
    # AntDir-v2, AntGoal-v2, HalfCheetahDir-v2, HalfCheetahVel-v2, HumanoidDir-v2
    env_name='AntDir-v2', 
    env_params=dict(
        n_tasks=2, # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
        randomize_tasks=True, # shuffle the tasks after creating them
    )
)

env = envs[config['env_name']](**config['env_params'])

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
print(obs_dim)
print(act_dim)

for episode in range(1):
    done = False
    obs = env.reset()

    while not done:
        # env.render()

        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        
        print('obs: {} | action: {} | reward: {} | next_obs: {} | done: {}\n'.format(
                obs, action, reward, next_obs, done))
        
        obs = next_obs

        sys.exit()

