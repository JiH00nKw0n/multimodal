"""Base model configuration"""

import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from transformers.processing_utils import ProcessorMixin
    from transformers.utils import TensorType

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)
__all__ = ["BaseConfig"]


class BaseConfig(PretrainedConfig):
    r"""
    [`BaseConfig`] is the configuration class to store the configuration of a [`BaseModel`]. It is used to instantiate
    a Base model according to the specified arguments, defining the text model and vision model configs. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the Base

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`BaseTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`BaseVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The initial value of the *logit_scale* parameter. Default is used as per the original Base implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from src.models import BaseConfig, BaseModel

    >>> # Initializing a BaseConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = BaseConfig()

    >>> # Initializing a BaseModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = BaseModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a BaseConfig from a BaseTextConfig and a BaseVisionConfig
    >>> from src.models import BaseTextConfig, BaseVisionConfig

    >>> # Initializing a BaseText and BaseVision configuration
    >>> config_text = BaseTextConfig()
    >>> config_vision = BaseVisionConfig()

    >>> config = BaseConfig.from_text_vision_configs(config_text, config_vision)
    ```"""

    model_type = "base"

    def __init__(
            self,
            text_pretrained_model_name_or_path: Optional[str] = None,
            vision_pretrained_model_name_or_path: Optional[str] = None,
            pool_type: Optional[str] = None,
            projection_dim: Optional[int] = 512,
            logit_scale_init_value: Optional[float] = 2.6592,
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
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
