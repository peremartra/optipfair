def round_to_divisor(value: int, divisor: int) -> int:
    """
    Round value to the nearest multiple of divisor.

    Args:
        value: Value to round
        divisor: Divisor to round to

    Returns:
        Rounded value (nearest multiple of divisor)

    Example:
        >>> round_to_divisor(8100, 128)
        8064
        >>> round_to_divisor(8200, 128)
        8192
    """
    return round(value / divisor) * divisor
