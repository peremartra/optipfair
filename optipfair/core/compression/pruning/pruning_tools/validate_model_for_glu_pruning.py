from loguru import logger
import torch
from transformers import PreTrainedModel
from core.compression.pruning.pruning_tools.get_model_layers import get_model_layers


def validate_model_for_glu_pruning(model: PreTrainedModel) -> bool:
    """
    Validate that a model is compatible with GLU pruning.

    Args:
        model: Model to validate

    Returns:
        bool: True if the model is compatible, False otherwise
    """
    # Check if the model has the expected structure
    try:
        layers = get_model_layers(model)
        if not layers:
            logger.warning("Could not find decoder layers in the model")
            return False

        # Check the first layer for GLU components
        first_layer = layers[0]
        if not hasattr(first_layer, "mlp"):
            logger.warning("Model layers do not have 'mlp' attribute")
            return False

        mlp = first_layer.mlp
        required_attributes = ["gate_proj", "up_proj", "down_proj"]
        for attr in required_attributes:
            if not hasattr(mlp, attr):
                logger.warning(f"MLP does not have required attribute: {attr}")
                return False

            # Verify these are linear layers
            layer = getattr(mlp, attr)
            if not isinstance(layer, torch.nn.Linear):
                logger.warning(f"{attr} is not a Linear layer")
                return False

        # Verify gate_proj and up_proj have the same dimensions
        if mlp.gate_proj.in_features != mlp.up_proj.in_features:
            logger.warning("gate_proj and up_proj have different input dimensions")
            return False

        if mlp.gate_proj.out_features != mlp.up_proj.out_features:
            logger.warning("gate_proj and up_proj have different output dimensions")
            return False

        if mlp.down_proj.in_features != mlp.gate_proj.out_features:
            logger.warning(
                "down_proj input dimensions don't match gate_proj output dimensions"
            )
            return False

        return True

    except Exception as e:
        logger.warning(f"Error validating model for GLU pruning: {str(e)}")
        return False
