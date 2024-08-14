import logging
import numpy as np
import torch
from dataclasses import dataclass
from transformers import (
    BatchEncoding,
)
from typing import Union, List, Dict, Optional, TypeVar
from PIL import Image
from transformers.utils import PaddingStrategy

logger = logging.getLogger(__name__)

Processors = TypeVar("Processors", bound="ProcessorMixin")


def convert_to_rgb(image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
    # PIL.Image.Image 처리
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)

    # np.ndarray 처리
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.concatenate([image] * 3, axis=-1)
        image = torch.from_numpy(image)

    # torch.Tensor 처리
    if isinstance(image, torch.Tensor):
        if image.ndimension() == 3 and image.shape[0] == 1:
            image = image.expand(3, -1, -1)
        elif image.ndimension() == 2:
            image = image.unsqueeze(0).expand(3, -1, -1)

        if image.dtype != torch.uint8:
            image = image.clamp(0, 255).byte()

    return image


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
        processed_dict = {
            key: list(map(lambda d: convert_to_rgb(d[key]) if key == 'images' else d[key], inputs))
            for key in inputs[0].keys()
        }
        kwargs = {'return_tensors': self.return_tensors,
                  'padding': self.padding,
                  'pad_to_multiple_of': self.pad_to_multiple_of
                  }
        processor_input = dict(processed_dict, **kwargs)
        return self.processor(**processor_input)
