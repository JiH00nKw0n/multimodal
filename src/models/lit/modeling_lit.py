import torch
from torch import nn
from transformers import add_start_docstrings

from src.models.modeling_base import (
    BaseOutput, BasePreTrainedModel, BaseTextModel, BaseVisionModel, base_loss
)
from src.models.configuration_base import BaseConfig
from src.models.lit.configuration_lit import (
    LiTTextConfig, LiTVisionConfig, LiTConfig
)


class LiTPreTrainedModel(BasePreTrainedModel):
    config_class = BaseConfig
    base_model_prefix = "lit"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        raise NotImplementedError


LIT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LiTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(LIT_START_DOCSTRING)
class LiTModel(LiTPreTrainedModel):

    config_class = LiTConfig

    def __init__(self, config: LiTConfig):
        super().__init__(config)

        if not isinstance(config.text_config, LiTTextConfig):
            raise TypeError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, LiTVisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # Initialize weights and apply final processing
        super().init_weights()

        text_model = BaseTextModel._from_config(text_config, attn_implementation=config._attn_implementation)
        self.text_model = text_model.text_model

        vision_model = BaseVisionModel._from_config(vision_config, attn_implementation=config._attn_implementation)
        self.vision_model = vision_model.vision_model

        super()._backward_compatibility_gradient_checkpointing()

        # Lock the image Tower
        for param in self.vision_model.parameters():
            param.requires_grad = False
