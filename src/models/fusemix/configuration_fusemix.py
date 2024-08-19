import os
from typing import Union, Optional, Dict
from src.common.registry import registry
from src.models.configuration_base import (
    BaseConfig, BaseTextConfig, BaseVisionConfig
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


@registry.register_model_config("FuseMixTextConfig")
class FuseMixTextConfig(BaseTextConfig):

    def __init__(self, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        super().__init__(pretrained_model_name_or_path, **kwargs)


@registry.register_model_config("FuseMixVisionConfig")
class FuseMixVisionConfig(BaseVisionConfig):
    def __init__(self, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        super().__init__(pretrained_model_name_or_path, **kwargs)


@registry.register_model_config("FuseMixConfig")
class FuseMixConfig(BaseConfig):
    def __init__(
            self,
            text_config: Optional[Dict] = None,
            vision_config: Optional[Dict] = None,
            projection_dim: Optional[int] = 512,
            logit_scale_init_value: Optional[int] = 0.07,
            pool_type: Optional[str] = None,
            drop_out: Optional[float] = 0.0,
            num_fusion_layer: Optional[int] = 4,
            expansion_factor: Optional[int] = 4,
            **kwargs
    ):
        # If `_config_dict` exist, we use them for the backward compatibility.
        # We pop out these 2 attributes before calling `super().__init__` to avoid them being saved (which causes a lot
        # of confusion!).
        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)

        super().__init__(**kwargs)

        # Instead of simply assigning `[text|vision]_config_dict` to `[text|vision]_config`, we use the values in
        # `[text|vision]_config_dict` to update the values in `[text|vision]_config`. The values should be same in most
        # cases, but we don't want to break anything regarding `_config_dict` that existed before commit `8827e1b2`.
        if text_config_dict is not None:
            if text_config is None:
                text_config = {}

            # This is the complete result when using `text_config_dict`.
            _text_config_dict = FuseMixTextConfig(**text_config_dict).to_dict()

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
            _vision_config_dict = FuseMixVisionConfig(**vision_config_dict).to_dict()
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

        self.text_config = FuseMixTextConfig(**text_config)
        self.vision_config = FuseMixVisionConfig(**vision_config)

        self.projection_dim = projection_dim
        self.pool_type = pool_type
        self.drop_out = drop_out
        self.num_fusion_layer = num_fusion_layer
        self.expansion_factor = expansion_factor
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
