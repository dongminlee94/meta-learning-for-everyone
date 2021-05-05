import numpy as np
import torch


class Sampler(object):
    def __init__(
        self, 
        env, 
        agent, 
        max_step,
        device,
    ):
    
        self.env = env
        self.agent = agent
        self.max_step = max_step
        self.device = device

    def obtain_samples(self, max_samples, min_trajs, accum_context=True, use_rendering=False):
        ''' Obtain samples up to the number of maximum samples '''
        trajs = []
        cur_samples = 0
        num_trajs = 0

        while cur_samples < max_samples:
            traj = self.rollout(accum_context=accum_context, use_rendering=use_rendering)
            
            # Save the latent context that generated this trajectory
            traj['context'] = self.agent.encoder.z.detach().cpu().numpy()
            trajs.append(traj)
            cur_samples += len(traj['obs'])
            num_trajs += 1

            self.agent.encoder.sample_z()
            
            if min_trajs == 1:
                break
        return trajs, cur_samples

    def rollout(self, accum_context=True, use_rendering=False):
        ''' Rollout up to maximum trajectory length '''
        observs = []
        actions = []
        rewards = []
        dones = []

        obs = self.env.reset()
        cur_step = 0

        if use_rendering:
            self.env.render()

        while cur_step < self.max_step:
            action = self.agent.get_action(obs)
            next_obs, reward, done, _ = self.env.step(action)
            
            # Update the agent's current context
            if accum_context:
                self.update_context(obs, action, reward)
            
            observs.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            cur_step += 1
            obs = next_obs
            if done:
                break

        actions = np.array(actions)
        observs = np.array(observs)
        next_observs = np.vstack((observs[1:, :], np.expand_dims(next_obs, 0)))
        return dict(
            obs=observs,
            action=actions,
            reward=np.array(rewards).reshape(-1, 1),
            next_obs=next_observs,
            done=np.array(dones).reshape(-1, 1),
        )
    
    def update_context(self, obs, action, reward):
        ''' Append single transition to the current context '''
        obs = torch.from_numpy(obs[None, None, ...]).float().to(self.device)
        action = torch.from_numpy(action[None, None, ...]).float().to(self.device)
        reward = torch.from_numpy(np.array([reward])[None, None, ...]).float().to(self.device)
        transition = torch.cat([obs, action, reward], dim=-1).to(self.device)
        
        if self.agent.encoder.context is None:
            self.agent.encoder.context = transition
        else:
            self.agent.encoder.context = torch.cat([self.agent.encoder.context, transition], dim=1).to(self.device)