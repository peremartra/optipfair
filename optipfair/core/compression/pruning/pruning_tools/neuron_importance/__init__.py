from core.compression.pruning.pruning_tools.neuron_importance.factory import (
    import_modules,
    register_neuron_importance_function,
    factory,
)

__all__ = ["register_neuron_importance_function", "factory"]

import_modules()
