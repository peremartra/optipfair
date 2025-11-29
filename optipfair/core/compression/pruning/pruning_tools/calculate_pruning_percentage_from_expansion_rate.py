def calculate_pruning_percentage_from_expansion_rate(
    current_intermediate_size: int,
    current_hidden_size: int,
    target_expansion_rate: float,
) -> float:
    """
    Calculate the pruning percentage needed to achieve a target expansion rate.

    Args:
        current_intermediate_size: Current size of the intermediate layer
        current_hidden_size: Current size of the hidden layer
        target_expansion_rate: Target expansion rate in percentage (e.g., 140 for 140%)

    Returns:
        pruning_percentage: Percentage of neurons to prune
    """
    current_expansion_rate = (current_intermediate_size / current_hidden_size) * 100
    target_intermediate_size = (target_expansion_rate / 100) * current_hidden_size

    if target_intermediate_size >= current_intermediate_size:
        raise ValueError(
            f"Target expansion rate ({target_expansion_rate}%) would increase the model size. "
            f"Current expansion rate is {current_expansion_rate:.2f}%."
        )

    pruning_percentage = (
        1 - (target_intermediate_size / current_intermediate_size)
    ) * 100
    return pruning_percentage
