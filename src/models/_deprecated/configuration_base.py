from transformers import PretrainedConfig
from typing import Optional, Dict
from src.common import registry

__all__ = ["BaseDualEncoderConfig"]


@registry.register_model_config("BaseDualEncoderConfig")
class BaseDualEncoderConfig(PretrainedConfig):
    model_type = "base"

    def __init__(
            self,
            text_config: Optional[Dict] = None,
            vision_config: Optional[Dict] = None,
            projection_dim: int = 1024,
            logit_scale_init_value: float = 2.6592,
            **kwargs
    ):
        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)

        super().__init__(**kwargs)

        if text_config_dict is not None:
            raise NotImplementedError
        if vision_config_dict is not None:
            raise NotImplementedError
        if text_config is None:
            text_config = {}
            raise NotImplementedError
        if vision_config is None:
            vision_config = {}
            raise NotImplementedError

        self.text_config = text_config
        self.vision_config = vision_config
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
