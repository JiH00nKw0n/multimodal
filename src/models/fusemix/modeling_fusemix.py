from typing import Optional, Union, Tuple

import torch
from torch import nn
from transformers import add_start_docstrings, AutoModel, PreTrainedModel, AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings

from src.utils import pool
from src.models.modeling_base import (
    BaseOutput, base_loss,
    BASE_INPUTS_DOCSTRING, BASE_TEXT_INPUTS_DOCSTRING, BASE_VISION_INPUTS_DOCSTRING
)
from src.common import registry
from src.models.configuration_base import BaseConfig
from src.models.fusemix.configuration_fusemix import (
    FuseMixConfig
)
__all__ = ["FuseMixModel"]


class FuseMixMLP(nn.Module):
    def __init__(self, hidden_size: int, config: FuseMixConfig):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config
        self.expansion_factor = self.config.expansion_factor

        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(self.hidden_size, self.expansion_factor * self.hidden_size)
        self.fc2 = nn.Linear(self.expansion_factor * self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.config.drop_out)
        self.layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _hidden_states = self.fc1(hidden_states)
        _hidden_states = self.activation_fn(_hidden_states)
        _hidden_states = self.dropout(_hidden_states)
        _hidden_states = self.fc2(_hidden_states)
        _hidden_states = self.layer_norm(_hidden_states)
        return hidden_states + _hidden_states


class FuseMixLayer(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            config: FuseMixConfig,
            proj_bias: Optional[bool] = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config
        self.projection_dim = self.config.projection_dim
        self.proj_bias = proj_bias

        self.layers = nn.ModuleList(
            [FuseMixMLP(self.hidden_size, self.config) for _ in range(self.config.num_fusion_layer)])
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.projection_dim, self.proj_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _hidden_states = hidden_states
        for idx, layer in enumerate(self.layers):
            _hidden_states = layer(hidden_states)
        _hidden_states = self.layer_norm(_hidden_states)
        _hidden_states = self.fc(_hidden_states)

        return _hidden_states


class FuseMixPreTrainedModel(PreTrainedModel):
    config_class = FuseMixConfig
    base_model_prefix = "fusemix"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


FUSEMIX_START_DOCSTRING = r"""
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


@registry.register_model("FuseMixModel")
@add_start_docstrings(FUSEMIX_START_DOCSTRING)
class FuseMixModel(FuseMixPreTrainedModel):
    config_class = FuseMixConfig

    def __init__(self, config: FuseMixConfig):
        super().__init__(config)

        text_config = AutoConfig.from_pretrained(**config.text_config)
        vision_config = AutoConfig.from_pretrained(**config.vision_config)

        self.pool_type = config.pool_type
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_projection = FuseMixLayer(self.text_embed_dim, config)
        self.vision_projection = FuseMixLayer(self.vision_embed_dim, config)

        self.temperature = nn.Parameter(torch.tensor(self.config.temperature))

        # Initialize weights and apply final processing
        super().init_weights()

        self.text_model = AutoModel.from_pretrained(**config.text_config)

        self.vision_model = AutoModel.from_pretrained(**config.vision_config)

        super()._backward_compatibility_gradient_checkpointing()

        # Lock the image Model
        for param in self.vision_model.parameters():
            param.requires_grad = False

        # Lock the text Model
        for param in self.text_model.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(BASE_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
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
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.pool_type is not None:
            pooled_output = pool(
                last_hidden_state=text_outputs.last_hidden_state,
                attention_mask=attention_mask,
                pool_type=self.pool_type
            )
        else:
            pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features.detach()

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
    # @replace_return_docstrings(output_type=BaseOutput, config_class=FuseMixConfig)
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
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
            token_type_ids=token_type_ids,
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
                last_hidden_state=text_outputs.last_hidden_state,
                attention_mask=attention_mask,
                pool_type=self.pool_type
            ).detach()
        else:
            text_embeds = text_outputs[1].detach()

        image_embeds = self.vision_projection(image_embeds)
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        with torch.no_grad():
            self.logit_scale.clamp_(0.001, 0.5)

        # cosine similarity as logits
        temperature = self.temperature
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device)) * temperature.to(
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
