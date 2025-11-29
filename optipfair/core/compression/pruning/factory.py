import importlib
import os

from core.compression.pruning.base import BasePruner

__EXTRACTOR__ = dict()


def factory(name, *args, **kwargs):
    return __EXTRACTOR__[name](*args, **kwargs)


def register_pruner(name):
    def register_pruner_fn(cls):
        if name in __EXTRACTOR__:
            raise ValueError(f"Name {name} already registered!")
        if not issubclass(cls, BasePruner):
            raise ValueError(f"Class {cls} is not a subclass of {BasePruner}")
        __EXTRACTOR__[name] = cls
        return cls

    return register_pruner_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_') and not file == 'factory.py' and not file == 'utils.py':
        module_name = file[:file.find('.py')]
        importlib.import_module(f'core.compression.pruning.{module_name}')
