from typing import Optional, Union, Tuple

import torch
from torch import nn
from transformers import add_start_docstrings
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings

from src.utils import pool
from src.models.modeling_base import (
    BaseOutput, BasePreTrainedModel, BaseTextModel, BaseVisionModel, base_loss,
    BASE_INPUTS_DOCSTRING, BASE_TEXT_INPUTS_DOCSTRING, BASE_VISION_INPUTS_DOCSTRING
)
from src.common import registry
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


@registry.register_model("LiTTextModel")
@add_start_docstrings(
    """The text model from Base without any head or projection on top.""",
    LIT_START_DOCSTRING,
)
class LiTTextModel(BasePreTrainedModel):
    config_class = LiTTextConfig

    def __init__(self, config: LiTTextConfig):
        super().__init__(config)

    @add_start_docstrings_to_model_forward(BASE_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=LiTTextConfig)
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


@registry.register_model("BaseVisionModel")
@add_start_docstrings(
    """The vision model from CLIP without any head or projection on top.""",
    LIT_START_DOCSTRING,
)
class BaseVisionModel(BasePreTrainedModel):
    config_class = LiTVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: LiTVisionConfig):
        super().__init__(config)

    @add_start_docstrings_to_model_forward(BASE_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=LiTVisionConfig)
    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


@registry.register_model("LiTModel")
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

        self.pool_type = config.pool_type
        self.vision_embed_dim = vision_config.hidden_size
        self.text_embed_dim = text_config.hidden_size

        self.text_projection = nn.Linear(self.text_embed_dim, self.vision_embed_dim, bias=False)
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

    @add_start_docstrings_to_model_forward(BASE_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.pool_type is not None:
            pooled_output = pool(
                last_hidden_states=text_outputs.last_hidden_states,
                attention_mask=attention_mask,
                pool_type=self.pool_type
            )
        else:
            pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    @add_start_docstrings_to_model_forward(BASE_VISION_INPUTS_DOCSTRING)
    def get_image_features(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features.detach()

    @add_start_docstrings_to_model_forward(BASE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseOutput, config_class=LiTConfig)
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            return_loss: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseOutput]:

        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        # Lock image Tower
        image_embeds = image_embeds.detach()

        if self.pool_type is not None:
            text_embeds = pool(
                last_hidden_states=text_outputs.last_hidden_states,
                attention_mask=attention_mask,
                pool_type=self.pool_type
            )
        else:
            text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device)) * logit_scale.to(
            text_embeds.device
        )
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = base_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return BaseOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
