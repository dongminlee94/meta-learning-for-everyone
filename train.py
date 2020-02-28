import gym
from envs.mujoco import *

# env = gym.make('AntVel-v2')
# env = gym.make('AntDir-v2')
# env = gym.make('AntPos-v1')
# env = gym.make('HalfCheetahVel-v2')
# env = gym.make('HalfCheetahDir-v2')
# env = gym.make('Humanoid-v2')
# env = gym.make('HumanoidDir-v2')

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

print(obs_dim)
print(act_dim)

# for episode in range(1):
#     done = False
#     obs = env.reset()

#     while not done:
#         # env.render()

#         action = env.action_space.sample()
#         next_obs, reward, done, _ = env.step(action)
        
#         print('obs: {} | action: {} | reward: {} | next_obs: {} | done: {}\n'.format(
#                 obs, action, reward, next_obs, done))
        
#         obs = next_obs

#         sys.exit()

