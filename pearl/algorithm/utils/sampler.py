import numpy as np
import torch

class Sampler(object):
    def __init__(
        self, 
        env, 
        agent, 
        max_step,
    ):
    
        self.env = env
        self.agent = agent
        self.max_step = max_step

    def obtain_samples(self, max_samples, min_trajs, accum_context=True, use_rendering=False):
        trajs = []
        cur_samples = 0
        num_trajs = 0

        while cur_samples < max_samples:
            traj = self.rollout(
                accum_context=accum_context,
                use_rendering=use_rendering,
            )
            
            # Save the latent context that generated this trajectory
            traj['context'] = self.agent.encoder.z.detach().cpu().numpy()
            trajs.append(traj)
            cur_samples += len(traj['observs'])
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
            action, _ = self.agent.get_action(obs)
            next_obs, reward, done, _ = self.env.step(action)
            
            # Update the agent's current context
            if accum_context:
                self.update_context([obs, action, reward])
            
            observs.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            cur_step += 1
            obs = next_obs
            if done:
                break

        actions = np.array(actions)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 1)
        
        observs = np.array(observs)
        if len(observs.shape) == 1:
            observs = np.expand_dims(observs, 1)
            next_obs = np.array([next_obs])
        
        next_observs = np.vstack(
            (
                observs[1:, :],
                np.expand_dims(next_obs, 0)
            )
        )
        return dict(
            observs=observs,
            actions=actions,
            rewards=np.array(rewards).reshape(-1, 1),
            next_observs=next_observs,
            dones=np.array(dones).reshape(-1, 1),
        )
    
    def update_context(self, transition: torch.Tensor):
        ''' Append single transition to the current context '''
        obs, action, reward = transition
        obs = torch.from_numpy(obs[None, None, ...]).float()
        action = torch.from_numpy(action[None, None, ...]).float()
        reward = torch.from_numpy(np.array([reward])[None, None, ...]).float()
        data = torch.cat([obs, action, reward], dim=-1)
        
        if self.agent.encoder.context is None:
            self.agent.encoder.context = data
        else:
            self.agent.encoder.context = torch.cat([self.agent.encoder.context, data], dim=1)