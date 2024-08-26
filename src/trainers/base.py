from typing import Optional
import torch
from torch.utils.data import Dataset, RandomSampler
from transformers.trainer_utils import has_length
from transformers.utils import logging
from transformers import Trainer
from src.common.registry import registry

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

