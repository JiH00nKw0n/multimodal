from typing import Optional
import torch
from torch.utils.data import Dataset, RandomSampler
from transformers.trainer_utils import has_length
from transformers.utils import logging
from transformers import Trainer
from src.common.registry import registry
from src.models.modeling_base import contrastive_loss

logger = logging.get_logger(__name__)

__all__ = ['BaseTrainer']


def neg_clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t()[:len(similarity)])
    return (caption_loss + image_loss) / 2.0


@registry.register_trainer('BaseTrainer')
class BaseTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        Basic RandomSampler.
        """
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.group_by_length:
            raise ValueError("Argument `group_by_length` must be `False`.")
        else:
            return RandomSampler(self.train_dataset)


@registry.register_trainer('NegCLIPTrainer')
class NegCLIPTrainer(BaseTrainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        Basic RandomSampler.
        """
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.group_by_length:
            raise ValueError("Argument `group_by_length` must be `False`.")
        else:
            return RandomSampler(self.train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = dict(inputs, **{
            'return_dict': True,
            'return_loss': False,
        })
        outputs = model(**inputs)
        torch.cuda.empty_cache()
        logits_per_image = outputs.logits_per_image

        loss = neg_clip_loss(logits_per_image)
        torch.cuda.empty_cache()
        
        return (loss, outputs) if return_outputs else loss
