"""Base model configuration"""

import os
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union, Dict

if TYPE_CHECKING:
    from transformers.processing_utils import ProcessorMixin
    from transformers.utils import TensorType

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig

logger = logging.get_logger(__name__)

__all__ = ["BaseTextConfig", "BaseVisionConfig", "BaseConfig"]


# Copied from transformers.models.Base.configuration_Base.BaseTextConfig with Base -> Base
class BaseTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BaseTextModel`]. It is used to instantiate a Base
    text encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the text encoder of the Base

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        pretrained_name_or_path (`Union[str, os.PathLike]`)
            Path to load pretrained text model
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 49406):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 49407):
            End of stream token id.

    Example:

    ```python
    >>> from src.models import BaseTextConfig, BaseTextModel

    >>> # Initializing a BaseTextConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = BaseTextConfig()

    >>> # Initializing a BaseTextModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = BaseTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    def __init__(
            self,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            **kwargs,
    ):
        config_dict = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        parent_instance = super(BaseTextConfig, self)

        init_kwargs = {}
        for key, value in config_dict.items():
            if hasattr(parent_instance, key):
                init_kwargs[key] = value
            else:
                setattr(self, key, value)

        super(BaseTextConfig, self).__init__(**init_kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        config_dict = config_dict["text_config"]

        return cls.from_dict(config_dict, **kwargs)


class BaseVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BaseVisionModel`]. It is used to instantiate a
    Base vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the Base

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from src.models import BaseVisionConfig, BaseVisionModel

    >>> # Initializing a BaseVisionConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = BaseVisionConfig()

    >>> # Initializing a BaseVisionModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = BaseVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    def __init__(
            self,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            **kwargs,
    ):
        config_dict = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        parent_instance = super(BaseVisionConfig, self)

        init_kwargs = {}
        for key, value in config_dict.items():
            if hasattr(parent_instance, key):
                init_kwargs[key] = value
            else:
                setattr(self, key, value)

        super(BaseVisionConfig, self).__init__(**init_kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        config_dict = config_dict["vision_config"]

        return cls.from_dict(config_dict, **kwargs)


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
            text_config: Optional[Dict] = None,
            vision_config: Optional[Dict] = None,
            projection_dim: Optional[int] = 512,
            logit_scale_init_value: Optional[float] = 2.6592,
            **kwargs
    ):
        # If `_config_dict` exist, we use them for the backward compatibility.
        # We pop out these 2 attributes before calling `super().__init__` to avoid them being saved (which causes a lot
        # of confusion!).
        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)
        pool_type = kwargs.pop("pool_type", None)

        super().__init__(**kwargs)

        # Instead of simply assigning `[text|vision]_config_dict` to `[text|vision]_config`, we use the values in
        # `[text|vision]_config_dict` to update the values in `[text|vision]_config`. The values should be same in most
        # cases, but we don't want to break anything regarding `_config_dict` that existed before commit `8827e1b2`.
        if text_config_dict is not None:
            if text_config is None:
                text_config = {}

            # This is the complete result when using `text_config_dict`.
            _text_config_dict = BaseTextConfig(**text_config_dict).to_dict()

            # Give a warning if the values exist in both `_text_config_dict` and `text_config` but being different.
            for key, value in _text_config_dict.items():
                if key in text_config and value != text_config[key] and key not in ["transformers_version"]:
                    # If specified in `text_config_dict`
                    if key in text_config_dict:
                        message = (
                            f"`{key}` is found in both `text_config_dict` and `text_config` but with different values. "
                            f'The value `text_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`text_config_dict` is provided which will be used to initialize `BaseTextConfig`. The "
                            f'value `text_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `text_config` with the ones in `_text_config_dict`.
            text_config.update(_text_config_dict)

        if vision_config_dict is not None:
            if vision_config is None:
                vision_config = {}

            # This is the complete result when using `vision_config_dict`.
            _vision_config_dict = BaseVisionConfig(**vision_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _vision_config_dict:
                _vision_config_dict["id2label"] = {
                    str(key): value for key, value in _vision_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_vision_config_dict` and `vision_config` but being different.
            for key, value in _vision_config_dict.items():
                if key in vision_config and value != vision_config[key] and key not in ["transformers_version"]:
                    # If specified in `vision_config_dict`
                    if key in vision_config_dict:
                        message = (
                            f"`{key}` is found in both `vision_config_dict` and `vision_config` but with different "
                            f'values. The value `vision_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`vision_config_dict` is provided which will be used to initialize `BaseVisionConfig`. "
                            f'The value `vision_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `vision_config` with the ones in `_vision_config_dict`.
            vision_config.update(_vision_config_dict)

        if text_config is None:
            raise ValueError("One of text_config or text_config_dict must not `None`.")

        if vision_config is None:
            raise ValueError("One of vision_config or vision_config_dict must not `None`.")

        self.text_config = BaseTextConfig(**text_config)
        self.vision_config = BaseVisionConfig(**vision_config)

        self.pool_type = pool_type
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: PretrainedConfig, vision_config: PretrainedConfig, **kwargs):
        r"""
        Instantiate a [`BaseConfig`] (or a derived class) from Base text model configuration and Base vision model
        configuration.

        Returns:
            [`BaseConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)

    @classmethod
    def from_text_vision_pretrained(
            cls,
            text_pretrained_model_name_or_path: Union[str, os.PathLike],
            vision_pretrained_model_name_or_path: Union[str, os.PathLike],
            **kwargs
    ):
        from transformers import AutoConfig

        configs_input = dict({
            'text_config': AutoConfig.from_pretrained(text_pretrained_model_name_or_path).to_dict(),
            'vision_config': AutoConfig.from_pretrained(vision_pretrained_model_name_or_path).to_dict(),
        }, **kwargs)
        return cls.from_text_vision_configs(**configs_input)