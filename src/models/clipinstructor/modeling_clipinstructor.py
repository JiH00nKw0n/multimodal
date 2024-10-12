from dataclasses import dataclass
from typing import Optional, Tuple, Any, Union

import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    CLIPVisionConfig,
    T5EncoderModel,
    T5ForConditionalGeneration,
    CLIPVisionModelWithProjection, T5Config
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import ModelOutput, logging

from src.common import registry
from src.models.clipinstructor.configuration_clipinstructor import CLIPInstructorConfig
from src.utils import pool

logger = logging.get_logger(__name__)

__all__ = [
    "CLIPInstructorOutput", "CLIPInstructorPreTrainedModel", "CLIPInstructorModel"
]



# Copied from transformers.models.clip.modeling_clip.contrastive_loss
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# Copied from transformers.models.clip.modeling_clip.clip_loss with clip->base
def base_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


# Copied from transformers.models.clip.modeling_clip.CLIPOutput with CLIP -> Base
@dataclass
class CLIPInstructorOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# Copied from transformers.models.clip.modeling_clip.CLIPPreTrainedModel with CLIP -> Base
class CLIPInstructorPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CLIPInstructorConfig
    base_model_prefix = "clip_instructor"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


@registry.register_model("CLIPInstructorModel")
class CLIPInstructorModel(CLIPInstructorPreTrainedModel):
    config_class = CLIPInstructorConfig

    def __init__(self, config: CLIPInstructorConfig):
        super().__init__(config)

        text_config = T5Config.from_pretrained(**config.text_config)
        vision_config = CLIPVisionConfig.from_pretrained(**config.vision_config)

        self.pool_type = config.pool_type
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_projection = nn.Linear(
            self.text_embed_dim, self.projection_dim, bias=False, dtype=config.torch_dtype
        )
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value, dtype=config.torch_dtype))

        # Initialize weights and apply final processing
        super().init_weights()

        self.text_model = T5EncoderModel.from_pretrained(**config.text_config)

        self.vision_model = CLIPVisionModelWithProjection.from_pretrained(**config.vision_config)

        super()._backward_compatibility_gradient_checkpointing()

        # Lock the image Model
        for param in self.vision_model.parameters():
            param.requires_grad = False

    def get_text_features(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
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
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = pool(
            last_hidden_state=text_outputs.last_hidden_state,
            attention_mask=attention_mask,
            pool_type=self.pool_type
        )

        text_features = self.text_projection(pooled_output)

        return text_features.detach()

    def get_image_features(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            interpolate_pos_encoding: bool = False,
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
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        return vision_outputs.image_embeds.detach()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            interpolate_pos_encoding: bool = False,
            return_loss: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPInstructorOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs.image_embeds.detach()

        text_embeds = pool(
            last_hidden_state=text_outputs.last_hidden_state,
            attention_mask=attention_mask,
            pool_type=self.pool_type
        )

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

        return CLIPInstructorOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
