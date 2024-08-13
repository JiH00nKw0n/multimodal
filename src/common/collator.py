import logging
from dataclasses import dataclass

from builtins import list

import requests
from transformers import (
    BatchEncoding,
)
from transformers.data.data_collator import (
    pad_without_fast_tokenizer_warning,
)
from typing import Union, List, Dict, Any, Optional, TypeVar
from collections import defaultdict
from PIL import Image
from transformers.utils import PaddingStrategy

logger = logging.getLogger(__name__)

Processors = TypeVar("Processors", bound="ProcessorMixin")


@dataclass
class BaseCollator:
    """
    Customized collator inherited from DataCollatorWithPadding.
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        seed (`int`, *optional*):
            Seed to use in choosing one of multiple positives or negatives.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    processor: Processors
    padding: Union[bool, str, PaddingStrategy] = 'max_length'
    max_length: Optional[int] = 64
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __post_init__(self):
        pass

    def __call__(self, inputs: List[Dict[str, List]]) -> BatchEncoding:
        processed_dict = {key: list(map(lambda d: d[key], inputs)) for key in inputs[0].keys()}
        kwargs = {'return_tensors': self.return_tensors,
                  'padding': self.padding,
                  'pad_to_multiple_of': self.pad_to_multiple_of
                  }
        processor_input = dict(processed_dict, **kwargs)
        return self.processor(**processor_input)
