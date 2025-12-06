import importlib
import os
from core.compression.pruning.base import BasePruner
from typing import Dict

__PRUNER__: Dict[str, BasePruner] = dict()


def factory(name, *args, **kwargs) -> BasePruner:
    return __PRUNER__[name](*args, **kwargs)


def register_pruner(name):
    def register_pruner_fn(cls):
        if name in __PRUNER__:
            raise ValueError(f"Name {name} already registered!")
        if not issubclass(cls, BasePruner):
            raise ValueError(f"Class {cls} is not a subclass of {BasePruner}")
        __PRUNER__[name] = cls
        return cls

    return register_pruner_fn


def import_modules():
    for file in os.listdir(os.path.dirname(__file__)):
        if (
            file.endswith(".py")
            and not file.startswith("_")
            and not file == "factory.py"
        ):
            module_name = file[: file.find(".py")]
            importlib.import_module(f"core.compression.pruning.{module_name}")
