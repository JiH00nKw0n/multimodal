from src.models.base import *
from transformers import PretrainedConfig, PreTrainedModel
from src.common.registry import registry
from transformers import (
    CLIPConfig,
    CLIPModel
)
__all__ = [
    "BaseDualEncoderModel",
    "BaseFrozenDualEncoderModel",
    "BaseDualEncoderConfig",
    "PretrainedConfig",
    "PreTrainedModel",
    "CLIPModel",
    "CLIPConfig",
]

registry.register_model("CLIPModel")(CLIPModel)
registry.register_model_config("CLIPConfig")(CLIPConfig)
