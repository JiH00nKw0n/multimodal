from typing import Optional
from src.common.registry import registry
from src.models.configuration_base import (
    BaseConfig
)
from transformers.utils import logging

logger = logging.get_logger(__name__)
__all__ = ["FuseMixConfig"]


@registry.register_model_config("FuseMixConfig")
class FuseMixConfig(BaseConfig):
    model_type = "fusemix"

    def __init__(
            self,
            text_pretrained_model_name_or_path: Optional[str] = None,
            vision_pretrained_model_name_or_path: Optional[str] = None,
            projection_dim: Optional[int] = 512,
            logit_scale_init_value: Optional[int] = 0.07,
            pool_type: Optional[str] = None,
            drop_out: Optional[float] = 0.0,
            num_fusion_layer: Optional[int] = 4,
            expansion_factor: Optional[int] = 4,
            **kwargs
    ):
        super().__init__(**kwargs)

        if text_pretrained_model_name_or_path is None:
            raise ValueError()

        if vision_pretrained_model_name_or_path is None:
            raise ValueError()

        self.text_config = dict(
            {"pretrained_name_or_path": text_pretrained_model_name_or_path}, **kwargs
        )
        self.vision_config = dict(
            {"pretrained_name_or_path": vision_pretrained_model_name_or_path}, **kwargs
        )

        self.pool_type = pool_type
        self.projection_dim = projection_dim
        self.drop_out = drop_out
        self.num_fusion_layer = num_fusion_layer
        self.expansion_factor = expansion_factor
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
