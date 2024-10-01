import torch
from torch.utils.data import Dataset, RandomSampler
from typing import Optional
from transformers.trainer_utils import has_length

from src.common.registry import registry
from src.models.modeling_base import contrastive_loss
from src.runners.base import BaseTrainer


def neg_clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative CLIP loss by combining the caption and image loss.

    Args:
        similarity (`torch.Tensor`): The similarity scores between images and captions.

    Returns:
        `torch.Tensor`: The average loss calculated from the caption loss and the image loss.
    """
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t()[:len(similarity)])
    return (caption_loss + image_loss) / 2.0


@registry.register_trainer('RandomSamplerTrainer')
class RandomSamplerTrainer(BaseTrainer):
    """
    A subclass of the `BaseTrainer` that overrides the method for 
    getting the sampler used for the training dataset. The sampler used 
    in this class is a basic `RandomSampler`.
    """

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        Returns a `RandomSampler` for the training dataset.

        Returns:
            `Optional[torch.utils.data.Sampler]`: A `RandomSampler` for the dataset if available, otherwise `None`.

        Raises:
            ValueError: If `group_by_length` is set to `True`, since this trainer doesn't support grouping by length.
        """
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.group_by_length:
            raise ValueError("Argument `group_by_length` must be `False`.")
        else:
            return RandomSampler(self.train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss for the current batch of inputs using the negative CLIP loss function.

        Args:
            model: The model to be used for generating the logits.
            inputs: A dictionary of inputs that includes features such as images and captions.
            return_outputs (`bool`, *optional*, defaults to `False`): If `True`, returns both the loss and the model outputs.

        Returns:
            `torch.Tensor` or `Tuple[torch.Tensor, Any]`: If `return_outputs` is `True`, returns a tuple of (loss, outputs).
            Otherwise, returns only the loss.
        """
        inputs = dict(inputs, **{
            'return_dict': True,
            'return_loss': True,
        })
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


@registry.register_trainer('NegCLIPRandomSamplerTrainer')
class NegCLIPRandomSamplerTrainer(RandomSamplerTrainer):
    """
    A subclass of the `RandomSamplerTrainer` which includes the functionality for 
    computing the negative CLIP loss during training.

    Inherits:
        `RandomSamplerTrainer`: Inherits the random sampler functionality.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss for the current batch of inputs using the negative CLIP loss function.

        Args:
            model: The model to be used for generating the logits.
            inputs: A dictionary of inputs that includes features such as images and captions.
            return_outputs (`bool`, *optional*, defaults to `False`): If `True`, returns both the loss and the model outputs.

        Returns:
            `torch.Tensor` or `Tuple[torch.Tensor, Any]`: If `return_outputs` is `True`, returns a tuple of (loss, outputs). 
            Otherwise, returns only the loss.
        """
        inputs = dict(inputs, **{
            'return_dict': True,
            'return_loss': False,
        })
        outputs = model(**inputs)
        loss = neg_clip_loss(outputs.logits_per_image)

        return (loss, outputs) if return_outputs else loss