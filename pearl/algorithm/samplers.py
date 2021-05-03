import numpy as np
import torch

class InPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, encoder, max_path_length, update_context):
        self.env = env
        self.policy = policy
        self.encoder = encoder
        self.max_path_length = max_path_length

    def update_context(self, transition: torch.Tensor):
        ''' Append single transition to the current context '''
        o, a, r = transition
        o = torch.from_numpy(o[None, None, ...]).float()
        a = torch.from_numpy(a[None, None, ...]).float()
        r = torch.from_numpy(np.array([r])[None, None, ...]).float()
        data = torch.cat([o, a, r], dim=2)
        if self.encoder.context is None:
            self.encoder.context = data
        else:
            self.encoder.context = torch.cat([self.encoder.context, data], dim=1)

    def rollout(self, agent, accum_context=True, animated=False):
        """
        The following value for the following keys will be a 2D array, with the
        first dimension corresponding to the time dimension.
        - observations
        - actions
        - rewards
        - next_observations
        - terminals

        The next two elements will be lists of dictionaries, with the index into
        the list being the index into the time
        - agent_infos
        - env_infos

        :param env:
        :param agent:
        :param max_path_length:
        :param accum_context: if True, accumulate the collected context
        :param animated: if True, render video of rollout
        :return:
        """
        observations = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []
        o = self.env.reset()
        next_o = None
        path_length = 0

        if animated:
            self.env.render()

        while path_length < self.max_path_length:
            a, agent_info = agent.get_action(o)
            next_o, r, d, env_info = self.env.step(a)
            
            # update the agents's current context
            if accum_context:
                self.update_context([o, a, r])
            
            observations.append(o)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            agent_infos.append(agent_info)
            
            path_length += 1
            o = next_o
            if d:
                break

        actions = np.array(actions)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 1)
        
        observations = np.array(observations)
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, 1)
            next_o = np.array([next_o])
        
        next_observations = np.vstack(
            (
                observations[1:, :],
                np.expand_dims(next_o, 0)
            )
        )
        return dict(
            observations=observations,
            actions=actions,
            rewards=np.array(rewards).reshape(-1, 1),
            next_observations=next_observations,
            terminals=np.array(terminals).reshape(-1, 1),
            agent_infos=agent_infos,
            env_infos=env_infos,
        )

    def obtain_samples(self, max_samples, max_trajs, deterministic=False, accum_context=True, resample=1):
        paths = []
        n_steps_total = 0
        n_trajs = 0

        while n_steps_total < max_samples:
            path = rollout(self.policy, accum_context)
            
            # save the latent context that generated this trajectory
            path['context'] = self.policy.z.detach().cpu().numpy()
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1

            # don't we also want the option to resample z ever transition?
            if n_trajs % resample == 0:
                self.policy.sample_z()
            
            if max_trajs == 1:
                break
        return paths, n_steps_total