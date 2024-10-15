from typing import Optional, Dict

import torch
from torch.utils.data import Dataset, RandomSampler
from transformers import is_torch_xla_available
from transformers.trainer_utils import has_length

from src.common.registry import registry
from src.models.modeling_base import contrastive_loss
from src.runners.base import BaseTrainer

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


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


@registry.register_trainer('RandomSamplerWithMultiLossTrainer')
class RandomSamplerWithMultiLossTrainer(RandomSamplerTrainer):
    """
    A subclass of the `RandomSamplerTrainer` that overrides the method for
    getting the multi-loss used for the training dataset.
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
            'return_loss': True,
        })
        outputs = model(**inputs)
        loss = outputs.loss
        lm_loss = outputs.lm_loss
        cont_loss = outputs.contrastive_loss

        self.state.lm_loss = getattr(
            self.state, "lm_loss", torch.tensor(0.0).to(lm_loss.device)
        )
        self.state.cont_loss = getattr(
            self.state, "contrastive_loss", torch.tensor(0.0).to(cont_loss.device)
        )
        self.state.lm_loss += lm_loss.detach()
        self.state.cont_loss += cont_loss.detach()

        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)

            if hasattr(self.state, "lm_loss") and hasattr(self.state, "contrastive_loss"):
                tr_lm_loss_scalar = self._nested_gather(self.state.lm_loss).mean().item()
                tr_contrastive_loss_scalar = self._nested_gather(self.state.contrastive_loss).mean().item()
                logs["lm_loss"] = round(tr_lm_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
                logs["contrastive_loss"] = round(tr_contrastive_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)

                # reset lm_loss and contrastive_loss to zero
                self.state.lm_loss -= self.state.lm_loss
                self.state.contrastive_loss -= self.state.contrastive_loss

            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


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
