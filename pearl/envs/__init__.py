import os
import importlib

import gym
from gym.envs.registration import registry, make, spec

envs = {}

def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)

def register_env(name):
    def register_env_fn(fn):
        if name in envs:
            raise ValueError("Cannot register duplicate env {}".format(name))
        if not callable(fn):
            raise TypeError("env {} must be callable".format(name))
        envs[name] = fn
        return fn
    return register_env_fn

# automatically import any envs in the envs/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('pearl.envs.' + module)
