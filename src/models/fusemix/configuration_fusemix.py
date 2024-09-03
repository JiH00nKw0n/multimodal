import os
from typing import Optional, Union
import torch
from transformers import PretrainedConfig, AutoConfig

from src.common.registry import registry

from transformers.utils import logging

logger = logging.get_logger(__name__)
__all__ = ["FuseMixConfig"]


@registry.register_model_config("FuseMixConfig")
class FuseMixConfig(PretrainedConfig):
    model_type = "fusemix"

    def __init__(
            self,
            text_pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
            vision_pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
            projection_dim: Optional[int] = 512,
            temperature: Optional[int] = 0.07,
            pool_type: Optional[str] = None,
            drop_out: Optional[float] = 0.0,
            num_fusion_layer: Optional[int] = 4,
            expansion_factor: Optional[int] = 4,
            **kwargs
    ):
        if kwargs.get('torch_dtype') == 'fp16':
            kwargs['torch_dtype'] = torch.float16

        text_config = kwargs.pop('text_config', None)
        vision_config = kwargs.pop('vision_config', None)

        super().__init__(**kwargs)

        self.text_config = text_config if text_config is not None else dict(
            {"pretrained_model_name_or_path": text_pretrained_model_name_or_path}, **kwargs
        )
        self.vision_config = vision_config if vision_config is not None else dict(
            {"pretrained_model_name_or_path": vision_pretrained_model_name_or_path}, **kwargs
        )

        self.pool_type = pool_type
        self.projection_dim = projection_dim
        self.drop_out = drop_out
        self.num_fusion_layer = num_fusion_layer
        self.expansion_factor = expansion_factor
        self.temperature = temperature
        self.initializer_factor = 1.0


AutoConfig.register('fusemix', FuseMixConfig)
