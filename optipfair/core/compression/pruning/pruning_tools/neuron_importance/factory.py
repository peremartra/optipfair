import importlib
import os
from typing import Callable
from typing import Dict

__NEURON_IMPORTANCE_FN__: Dict[str, Callable] = dict()


def factory(name):
    return __NEURON_IMPORTANCE_FN__[name]


def register_neuron_importance_function(name):
    def register_neuron_importance_function_fn(fn):
        if name in __NEURON_IMPORTANCE_FN__:
            raise ValueError(f"Name {name} already registered!")
        if not isinstance(fn, Callable):
            raise ValueError(f"neuron importance function {fn} is not of type Callable")
        __NEURON_IMPORTANCE_FN__[name] = fn
        return fn

    return register_neuron_importance_function_fn

def import_modules():
    for file in os.listdir(os.path.dirname(__file__)):
        if (
            file.endswith(".py")
            and not file.startswith("_")
            and not file == "factory.py"
        ):
            module_name = file[: file.find(".py")]
            importlib.import_module(f"core.compression.pruning.pruning_tools.neuron_importance.{module_name}")
