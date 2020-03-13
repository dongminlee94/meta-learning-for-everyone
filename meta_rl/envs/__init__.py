import os
import importlib


envs = {}


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
        importlib.import_module('meta_rl.envs.' + module)
