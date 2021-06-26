"""
Registration code for Half-cheetah environments
"""

import importlib
import os

ENVS = {}


def register_env(name):
    """Register an environment"""

    def register_env_fn(filename):
        if name in ENVS:
            raise ValueError("Cannot register duplicate env {}".format(name))
        if not callable(filename):
            raise TypeError("env {} must be callable".format(name))
        ENVS[name] = filename
        return filename

    return register_env_fn


# automatically import any envs in the envs/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        module = file[: file.find(".py")]
        importlib.import_module("pearl.envs." + module)
