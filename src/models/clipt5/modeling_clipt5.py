import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
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
from src.models.clipt5.configuration_clipt5 import CLIPT5Config
from src.utils import pool

logger = logging.get_logger(__name__)

__all__ = [
    "CLIPT5Output", "CLIPT5PreTrainedModel", "CLIPT5EncoderModel", "CLIPT5ForConditionalGeneration"
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
class CLIPT5Output(ModelOutput):
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


@dataclass
class CLIPT5ForConditionalGenerationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    contrastive_loss: Optional[torch.FloatTensor] = None
    lm_logits: torch.FloatTensor = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    vision_model_output: BaseModelOutputWithPooling = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# Copied from transformers.models.clip.modeling_clip.CLIPPreTrainedModel with CLIP -> Base
class CLIPT5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CLIPT5Config
    base_model_prefix = "clip_t5"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


@registry.register_model("CLIPT5EncoderModel")
class CLIPT5EncoderModel(CLIPT5PreTrainedModel):
    config_class = CLIPT5Config

    def __init__(self, config: CLIPT5Config):
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

        return vision_outputs.image_embeds.detach()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            return_loss: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPT5Output]:

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

        return CLIPT5Output(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


@registry.register_model("CLIPT5ForConditionalGeneration")
class CLIPT5ForConditionalGeneration(CLIPT5PreTrainedModel):
    config_class = CLIPT5Config

    def __init__(self, config: CLIPT5Config):
        super().__init__(config)

        self.text_config = T5Config.from_pretrained(**config.text_config)
        self.vision_config = CLIPVisionConfig.from_pretrained(**config.vision_config)

        self.pool_type = config.pool_type
        self.projection_dim = config.projection_dim
        self.text_embed_dim = self.text_config.hidden_size
        self.vision_embed_dim = self.vision_config.hidden_size

        self.text_projection = nn.Linear(
            self.text_embed_dim, self.projection_dim, bias=False, dtype=config.torch_dtype
        )
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value, dtype=config.torch_dtype))

        # Initialize weights and apply final processing
        super().init_weights()

        self.text_model = T5ForConditionalGeneration.from_pretrained(**config.text_config)
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

        text_outputs = self.text_model.encoder(
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

        return vision_outputs.image_embeds.detach()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            return_loss: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPT5ForConditionalGenerationOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        use_cache = use_cache if use_cache is not None else self.text_config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.text_config.num_decoder_layers:
                warnings.warn("""
                The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`.
                Currently, `decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
                If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers, num_heads)`.
                """, FutureWarning)
                decoder_head_mask = head_mask

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs.image_embeds.detach()

        encoder_outputs = self.text_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_embeds = pool(
            last_hidden_state=encoder_outputs.last_hidden_state,
            attention_mask=attention_mask,
            pool_type=self.pool_type
        )

        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        decoder_outputs = None
        lm_logits = None

        if self.training:
            encoder_hidden_states = text_embeds.unsqueeze(1)
            batch_size, _ = attention_mask.shape
            encoder_attention_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)

            if self.text_model.model_parallel:
                torch.cuda.set_device(self.text_model.decoder.first_device)

            if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
                # get decoder inputs from shifting lm labels to the right
                decoder_input_ids = self.text_model._shift_right(labels)

            # Set device for model parallelism
            if self.text_model.model_parallel:
                torch.cuda.set_device(self.decoder.first_device)
                text_embeds = text_embeds.to(self.text_model.decoder.first_device)
                if decoder_input_ids is not None:
                    decoder_input_ids = decoder_input_ids.to(self.text_model.decoder.first_device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.text_model.decoder.first_device)
                if decoder_attention_mask is not None:
                    decoder_attention_mask = decoder_attention_mask.to(self.text_model.decoder.first_device)

            # Decode
            decoder_outputs = self.text_model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = decoder_outputs[0]

            # Set device for model parallelism
            if self.text_model.model_parallel:
                torch.cuda.set_device(self.text_model.encoder.first_device)
                self.text_model.lm_head = self.lm_head.to(self.text_model.encoder.first_device)
                sequence_output = sequence_output.to(self.text_model.lm_head.weight.device)

            if self.text_config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                sequence_output = sequence_output * (self.text_model.model_dim ** -0.5)

            lm_logits = self.text_model.lm_head(sequence_output)

        lm_loss = None
        if labels is not None and self.training:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device)) * logit_scale.to(
            text_embeds.device
        )
        logits_per_image = logits_per_text.t()

        cont_loss = None
        if return_loss:
            cont_loss = base_loss(logits_per_text)

        loss = None
        if cont_loss is not None and lm_loss is not None:
            loss = (cont_loss + lm_loss) / 2.0

        if return_dict:
            decoder_hidden_states = decoder_outputs.hidden_states if decoder_outputs is not None else None
            decoder_attentions = decoder_outputs.attentions if decoder_outputs is not None else None
            cross_attentions = decoder_outputs.cross_attentions if decoder_outputs is not None else None
        else:
            decoder_hidden_states = decoder_outputs[1] if decoder_outputs is not None else None
            decoder_attentions = decoder_outputs[2] if decoder_outputs is not None else None
            cross_attentions = decoder_outputs[3] if decoder_outputs is not None else None

        if not return_dict:
            output = ((lm_logits, logits_per_image, logits_per_text, text_embeds, image_embeds,)
                      + decoder_outputs[1:] + encoder_outputs)
            return ((loss, lm_loss, cont_loss,) + output) if loss is not None else output

        return CLIPT5ForConditionalGenerationOutput(
            loss=loss,
            lm_loss=lm_loss,
            contrastive_loss=cont_loss,
            lm_logits=lm_logits,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            vision_model_output=vision_outputs,
            decoder_hidden_states=decoder_hidden_states,
            decoder_attentions=decoder_attentions,
            cross_attentions=cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
