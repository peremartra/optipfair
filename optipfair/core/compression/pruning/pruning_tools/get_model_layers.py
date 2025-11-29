from loguru import logger
from typing import List, Any
from transformers import PreTrainedModel


def get_model_layers(model: PreTrainedModel) -> List[Any]:
    """
    Extract transformer layers from a pre-trained model.
    Currently supports LLaMA, Mistral, and similar model architectures.

    Args:
        model: Pre-trained model

    Returns:
        List of decoder layers that contain MLP blocks
    """
    # Try different attribute paths based on common model architectures
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # LLaMA, Mistral, and similar architectures
        return list(model.model.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # GPT-2 and similar architectures
        return list(model.transformer.h)
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        # BERT and similar architectures
        return list(model.encoder.layer)
    elif hasattr(model, "layers"):
        # Direct layers attribute
        return list(model.layers)

    logger.warning("Could not find layers in the model")
    return []
