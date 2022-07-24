import importlib
import os
from typing import Callable

ENVS = {}


def register_env(name: str) -> Callable:
    def register_env_fn(filename: str) -> str:
        if name in ENVS:
            raise ValueError(f"Cannot register duplicate env {name}")
        if not callable(filename):
            raise TypeError(f"Env {name} must be callable")
        ENVS[name] = filename
        return filename

    return register_env_fn


# automatically import any envs in the envs/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        module = file[: file.find(".py")]
        importlib.import_module("meta_rl.envs." + module)
