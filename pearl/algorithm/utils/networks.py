import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal
from typing import Callable, Tuple, List


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_units: list,
        hidden_activation: Callable = F.relu,
        init_w: float = 3e-3,
    ):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation
        
        # Set fully connected layers
        self.fcs = nn.ModuleList()
        in_dim = input_dim
        for i, hidden_unit in enumerate(hidden_units):
            fc = nn.Linear(in_dim, hidden_unit)
            in_dim = hidden_unit
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

        # Set the output layer
        self.last_fc = nn.Linear(in_dim, output_dim)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc in self.fcs:
            x = self.hidden_activation(fc(x))
        x = self.last_fc(x)
        return x


class FlattenMLP(MLP):
    ''' If there are multiple inputs, concatenate along dim 1 '''
    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        x = torch.cat(x, dim=-1)
        return super(FlattenMLP, self).forward(x)


class MLPEncoder(FlattenMLP):
    ''' Encode context via MLP '''
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        latent_dim: int,
        hidden_units: List[int],
    ):  
        super(MLPEncoder, self).__init__(
            input_dim=input_dim, 
            output_dim=output_dim,
            hidden_units=hidden_units
        )

        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.clear_z()

    def clear_z(self, num_tasks: int = 1):
        '''
        Reset q(z|c) to the prior r(z)
        Sample a new z from the prior r(z)
        Reset the context collected so far
        '''
        # Reset q(z|c) to the prior r(z)
        self.z_mu = torch.zeros(num_tasks, self.latent_dim)
        self.z_var = torch.ones(num_tasks, self.latent_dim)
        
        # Sample a new z from the prior r(z)
        self.sample_z()
        
        # Reset the context collected so far
        self.context = None

    def sample_z(self):
        ''' Sample z ~ r(z) or z ~ q(z|c) '''
        dists = []
        for mu, var in zip(torch.unbind(self.z_mu), torch.unbind(self.z_var)):
            dist = torch.distributions.Normal(mu, torch.sqrt(var))
            dists.append(dist)
        z = [dist.rsample() for dist in dists]
        self.z = torch.stack(z)

    def product_of_gaussians(self, mu: torch.Tensor, var: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ''' Compute mu, sigma of product of gaussians (POG) '''
        var = torch.clamp(var, min=1e-7)
        pog_var = 1. / torch.sum(torch.reciprocal(var), dim=0)
        pog_mu = pog_var * torch.sum(mu / var, dim=0)
        return pog_mu, pog_var

    def infer_posterior(self, context: torch.Tensor):
        ''' Compute q(z|c) as a function of input context and sample new z from it '''
        params = self.forward(context)
        params = params.view(context.size(0), -1, self.output_dim)

        # With probabilistic z, predict mean and variance of q(z | c)
        z_mu = torch.unbind(params[..., :self.latent_dim])
        z_var = torch.unbind(F.softplus(params[..., self.latent_dim:]))
        z_params = [self.product_of_gaussians(mu, var) for mu, var in zip(z_mu, z_var)]
        
        self.z_mu = torch.stack([z_param[0] for z_param in z_params])
        self.z_var = torch.stack([z_param[1] for z_param in z_params])
        self.sample_z()

    def compute_kl_div(self):
        ''' Compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(torch.zeros(self.latent_dim), torch.ones(self.latent_dim))

        posteriors = []
        for mu, var in zip(torch.unbind(self.z_mu), torch.unbind(self.z_var)):
            dist = torch.distributions.Normal(mu, torch.sqrt(var)) 
            posteriors.append(dist)
        
        kl_div = [torch.distributions.kl.kl_divergence(posterior, prior) for posterior in posteriors]
        return torch.sum(torch.stack(kl_div))

    def detach_z(self):
        ''' Disable backprop through z '''
        self.z = self.z.detach()


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class TanhGaussianPolicy(MLP):
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        hidden_units: List[int],
        init_w: float = 1e-3,
    ):
        super(TanhGaussianPolicy, self).__init__(
            input_dim=input_dim, 
            output_dim=output_dim,
            hidden_units=hidden_units,
            init_w=init_w
        )
        
        last_hidden_units = hidden_units[-1]
        self.last_fc_log_std = nn.Linear(last_hidden_units, output_dim)
        self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)

    def forward(
            self,
            x: torch.Tensor,
            deterministic: bool = False,
            reparameterize: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        for i, fc in enumerate(self.fcs):
            x = self.hidden_activation(fc(x))
        
        mu = self.last_fc(x)
        log_std = self.last_fc_log_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        if deterministic:
            pi = torch.tanh(mu)
        else:
            normal = Normal(mu, std)
            # If reparameterize, use reparameterization trick (mean + std * N(0,1))
            pi = normal.rsample() if reparameterize else normal.sample()
            
            # Compute log prob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic.  
            # To get an understanding of where it comes from, check out the original SAC paper
            # (https://arxiv.org/abs/1801.01290) and look in appendix C.
            # This is a more numerically-stable equivalent to Eq 21.
            # Derivation:
            #               log(1 - tanh(x)^2))
            #               = log(sech(x)^2))
            #               = 2 * log(sech(x)))
            #               = 2 * log(2e^-x / (e^-2x + 1)))
            #               = 2 * (log(2) - x - log(e^-2x + 1)))
            #               = 2 * (log(2) - x - softplus(-2x)))
            log_pi = normal.log_prob(pi)
            log_pi -= 2 * (np.log(2) - pi - F.softplus(-2*pi))
            log_pi = log_pi.sum(-1, keepdim=True)

        pi = torch.tanh(pi)
        return pi, log_pi


