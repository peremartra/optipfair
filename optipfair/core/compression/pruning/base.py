from abc import ABC, abstractmethod
from transformers import PreTrainedModel

class BasePruner(ABC):
    
    @abstractmethod
    def prune(self, *args, **kwargs) -> PreTrainedModel:
        raise NotImplementedError("method execute_prune needs to be implemented")
