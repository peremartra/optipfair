from core.compression.pruning.base import BasePruner
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import torch
from torch import nn
from core.compression.pruning.pruning_tools.calculate_attention_head_importance import compute_head_importance
from core.compression.pruning.types.attention.gqa_kwargs import GroupedQueryAttentionPrunerKwargs
from loguru import logger
from core.compression.pruning.factory import register_pruner


@register_pruner("gqa_attention")
class GroupQueryAttentionPruner(BasePruner):

    def prune_attention_heads(self, self_attn, prune_percent: float):
        """
        Reduces attention heads by pruning K/V heads and their associated query heads
        to maintain GQA compatibility.
        """
        config = self_attn.config
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = self_attn.head_dim
        num_query_groups = num_attention_heads // num_key_value_heads

        head_importance_scores = compute_head_importance(
            self_attn, num_attention_heads, head_dim
        )

        group_importance_scores = torch.zeros(num_key_value_heads).to(
            head_importance_scores.device
        )
        for i in range(num_key_value_heads):
            start = i * num_query_groups
            end = (i + 1) * num_query_groups
            group_importance_scores[i] = head_importance_scores[start:end].sum()

        num_kv_heads_to_prune = int(prune_percent * num_key_value_heads)
        if num_kv_heads_to_prune >= num_key_value_heads:
            num_kv_heads_to_prune = num_key_value_heads - 1
        k_kv_heads = num_key_value_heads - num_kv_heads_to_prune

        _, kv_indices_to_keep = torch.topk(
            group_importance_scores, k_kv_heads, largest=True, sorted=True
        )
        kv_indices_to_keep = kv_indices_to_keep.sort().values

        q_mask, k_mask, v_mask = [], [], []

        for idx in kv_indices_to_keep:
            q_start = idx.item() * num_query_groups * head_dim
            q_end = q_start + num_query_groups * head_dim
            q_mask.extend(range(q_start, q_end))

            kv_start = idx.item() * head_dim
            kv_end = kv_start + head_dim
            k_mask.extend(range(kv_start, kv_end))
            v_mask.extend(range(kv_start, kv_end))

        device, dtype = self_attn.q_proj.weight.device, self_attn.q_proj.weight.dtype

        new_q_proj = nn.Linear(
            config.hidden_size, len(q_mask), bias=config.attention_bias
        ).to(device, dtype)
        new_k_proj = nn.Linear(
            config.hidden_size, len(k_mask), bias=config.attention_bias
        ).to(device, dtype)
        new_v_proj = nn.Linear(
            config.hidden_size, len(v_mask), bias=config.attention_bias
        ).to(device, dtype)

        new_o_proj = nn.Linear(
            len(q_mask), config.hidden_size, bias=config.attention_bias
        ).to(device, dtype)

        new_q_proj.weight.data = self_attn.q_proj.weight.data[q_mask, :]
        new_k_proj.weight.data = self_attn.k_proj.weight.data[k_mask, :]
        new_v_proj.weight.data = self_attn.v_proj.weight.data[v_mask, :]
        new_o_proj.weight.data = self_attn.o_proj.weight.data[:, q_mask]

        self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj = (
            new_q_proj,
            new_k_proj,
            new_v_proj,
            new_o_proj,
        )

        k_heads = k_kv_heads * num_query_groups
        self_attn.config.num_attention_heads = k_heads
        self_attn.config.num_key_value_heads = k_kv_heads

        return self_attn, k_heads, k_kv_heads



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
        parsed_kwargs = GroupedQueryAttentionPrunerKwargs.model_validate(kwargs)

        original_kv_heads = model.config.num_key_value_heads
        if parsed_kwargs.num_kv_heads_to_keep >= original_kv_heads:
            logger.info("Target K/V heads is >= original. No pruning will be performed.")
            return model

        prune_percent = (original_kv_heads - parsed_kwargs.num_kv_heads_to_keep) / original_kv_heads

        new_num_attention_heads, new_num_key_value_heads = None, None
        logger.info(
            f"\n--- Starting Attention Head Pruning (Targeting {parsed_kwargs.num_kv_heads_to_keep} K/V heads) ---"
        )

        for idx, layer in enumerate(model.model.layers):
            self_attn = layer.self_attn
            if self_attn.config.num_attention_heads == self_attn.config.num_key_value_heads:
                logger.info(
                    f"Skipping attention pruning for layer {idx}: Not GQA or already fully pruned."
                )
                continue

            _, k_heads, k_kv_heads = self.prune_attention_heads(self_attn, prune_percent)
            if new_num_attention_heads is None:
                new_num_attention_heads = k_heads
                new_num_key_value_heads = k_kv_heads

        if new_num_attention_heads is not None:
            model.config.num_attention_heads = new_num_attention_heads
            model.config.num_key_value_heads = new_num_key_value_heads
            logger.info(
                f"Attention pruning complete. New heads Q:{new_num_attention_heads}, KV:{new_num_key_value_heads}"
            )
        else:
            logger.info("No attention heads were pruned.")
        return model