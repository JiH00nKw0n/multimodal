import logging
from dataclasses import dataclass

from builtins import list

from transformers import (
    PreTrainedTokenizerFast,
    PreTrainedTokenizerBase,
    BatchEncoding,
    ProcessorMixin,
)
from transformers.data.data_collator import (
    DataCollatorWithPadding,
)
from typing import Union, List, Dict, Any, Optional
from collections import defaultdict
import torch
import numpy as np
import os

logger = logging.getLogger(__name__)


@dataclass
class BaseCollator(DataCollatorWithPadding):
    """
    Customized collator inherited from DataCollatorWithPadding.
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        local_rank (`int`, *optional*):
            Local_rank attribute for tracking the number of trained dataset, by activating tracking when local_rank = 0.
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        seed (`int`, *optional*):
            Seed to use in choosing one of multiple positives or negatives.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """
    tokenizer: Optional[ProcessorMixin] = None
    local_rank: Optional[int] = None
    seed: Optional[int] = None

    def __post_init__(self):
        pass

    def __call__(self, inputs: List[Dict[str, Any]]) -> Dict:
        raise NotImplementedError

    def _process_inputs(self, inputs: List[Dict[str, List]]) -> defaultdict[str, list]:
        raise NotImplementedError

    def _process(self, input_texts: List[str]) -> Union[BatchEncoding | None]:
        raise NotImplementedError
