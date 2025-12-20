import torch


def compute_head_importance(self_attn, num_attention_heads, head_dim):
    """
    Computes importance scores for each attention head based on the L2 norm of its
    corresponding weights in the output projection layer (o_proj).

    Args:
    - self_attn: The attention block (e.g., model.layers[0].self_attn).
    - num_attention_heads: The total number of query heads.
    - head_dim: The dimension of each attention head.

    Returns:
    - importance_scores: A tensor containing the importance score for each head.
    """
    o_proj_weight = self_attn.o_proj.weight.data.float()

    # The input to o_proj is the concatenated output of all heads.
    # We can view the o_proj weight matrix as being composed of blocks,
    # where each block processes the output of one head.
    # o_proj_weight has shape [hidden_size, num_attention_heads * head_dim]

    importance_scores = torch.zeros(num_attention_heads).to(o_proj_weight.device)

    for i in range(num_attention_heads):
        # Extract the block of weights in o_proj corresponding to the i-th head
        start_idx = i * head_dim
        end_idx = (i + 1) * head_dim

        # The shape of this block is [hidden_size, head_dim]
        head_o_weight_block = o_proj_weight[:, start_idx:end_idx]

        # The importance is the L2 norm of this block
        importance_scores[i] = torch.norm(head_o_weight_block, p=2)

    return importance_scores
