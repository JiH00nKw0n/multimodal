from typing import Callable, Dict, List, Optional, Tuple, Union, TypeVar

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model

from .registry import registry
import torch
from torch.utils.data import Dataset, RandomSampler
from transformers.trainer_utils import (
    has_length
)
from transformers.utils import (
    logging,
)
from transformers import (
    Trainer,
)

logger = logging.get_logger(__name__)

__all__ = ['BaseTrainer']


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

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        kwargs = {'return_loss': True, 'return_dict': True}
        inputs = dict(inputs, **kwargs)
        outputs = model(**inputs)

        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
