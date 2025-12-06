from abc import ABC, abstractmethod
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class BasePruner(ABC):
    @abstractmethod
    def prune(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *args,
        **kwargs,
    ) -> PreTrainedModel:
        raise NotImplementedError("method execute_prune needs to be implemented")
