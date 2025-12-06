from core.compression.pruning.base import BasePruner
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class AttentionPruner(BasePruner):
    def prune(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *args,
        **kwargs,
    ) -> PreTrainedModel:
        """
        Placeholder for future implementation of attention head pruning.

        Args:
            model: Model to prune
            head_importance_method: Method to calculate head importance
            prune_percentage: Percentage of heads to prune

        Returns:
            model: Pruned model
        """
        # validate kwargs
        # parsed_kwargs = AttentionPrunerKwargs.model_validate(**kwargs)
        raise NotImplementedError("Attention pruning is not yet implemented.")
