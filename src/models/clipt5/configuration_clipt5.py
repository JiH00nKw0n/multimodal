"""CLIPT5 model configuration"""

import os
from typing import TYPE_CHECKING, Optional, Union

import torch

from src.common import registry

if TYPE_CHECKING:
    from transformers.utils import TensorType

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)
__all__ = ["CLIPT5Config"]


@registry.register_model_config("CLIPT5Config")
class CLIPT5Config(PretrainedConfig):
    model_type = "clip_t5"

    def __init__(
            self,
            text_pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
            vision_pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
            pool_type: Optional[str] = None,
            projection_dim: Optional[int] = 512,
            logit_scale_init_value: Optional[float] = 2.6592,
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
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
