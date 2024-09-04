import logging
from collections import defaultdict
from src.common.registry import registry
import numpy as np
import torch
from dataclasses import dataclass
from transformers import (
    BatchEncoding,
)
from typing import Union, List, Dict, Optional, TypeVar, Any
from PIL import Image
from transformers.utils import PaddingStrategy
from transformers import ProcessorMixin
from tqdm import tqdm
from src.utils.utils import process_batch

logger = logging.getLogger(__name__)

ProcessorType = TypeVar("ProcessorType", bound=ProcessorMixin)


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

        elif image.ndimension() == 3 and image.shape[0] == 4:
            image = image[:3, :, :]

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
    processor: ProcessorType
    padding: Union[bool, str, PaddingStrategy] = 'max_length'
    truncation: bool = True
    max_length: Optional[int] = 64
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __post_init__(self):
        pass

    def __call__(self, inputs: List[Dict[str, Any]]) -> BatchEncoding:
        raise NotImplementedError


@dataclass
@registry.register_collator('Collator')
class Collator(BaseCollator):

    def __call__(self, inputs: List[Dict[str, Any]]) -> BatchEncoding:
        processed_dict = {
            key: list(map(lambda d: convert_to_rgb(d[key]) if key == 'images' else d[key], inputs))
            for key in inputs[0].keys()
        }
        kwargs = {
            'return_tensors': self.return_tensors,
            'padding': self.padding,
            'truncation': self.truncation,
            'pad_to_multiple_of': self.pad_to_multiple_of,
        }
        processor_input = dict(processed_dict, **kwargs)

        return self.processor(**processor_input)


@dataclass
@registry.register_collator('SequenceTextCollator')
class SequenceTextCollator(BaseCollator):

    def __call__(self, inputs: List[Dict[str, Any]]) -> BatchEncoding:
        processed_dict = defaultdict(list)

        for key in inputs[0].keys():
            values = [d[key] for d in inputs]

            if key == 'images':
                processed_dict[key].extend([convert_to_rgb(value) for value in values])
            else:
                for value in values:
                    if isinstance(value, str):
                        processed_dict[key].append(value)
                    elif isinstance(value, list):
                        processed_dict[key].extend(value)
                    else:
                        raise TypeError()

        kwargs = {
            'return_tensors': self.return_tensors,
            'padding': self.padding,
            'truncation': self.truncation,
            'pad_to_multiple_of': self.pad_to_multiple_of,
        }
        processor_input = dict(processed_dict, **kwargs)

        return self.processor(**processor_input)


@dataclass
@registry.register_collator('SequenceTextWithHNCollator')
class SequenceTextWithHNCollator(BaseCollator):
    seed: Optional[int] = 2024
    rng: Optional[np.random.Generator] = None

    def __post_init__(self):
        super().__post_init__()
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)

    def __call__(self, inputs: List[Dict[str, Any]]) -> BatchEncoding:
        processed_dict = defaultdict(list)

        # 모든 이미지 URL을 리스트에 담기
        all_images_urls = []
        all_hard_images_urls = []

        text_list = []
        hard_text_list = []
        neg_texts = []
        hard_neg_texts = []

        for _input in inputs:
            all_images_urls.append(_input['images'])

            text_idx = self.rng.integers(0, len(_input['text']))
            text_list.append(_input['text'][text_idx])

            hard_image_idx = self.rng.integers(0, len(_input['hard_images']))
            all_hard_images_urls.append(_input['hard_images'][hard_image_idx])

            hard_text_idx = self.rng.integers(0, len(_input['hard_texts'][hard_image_idx]))
            hard_text_list.append(_input['hard_texts'][hard_image_idx][hard_text_idx])

            neg_texts.append(self.rng.choice(_input['neg_texts'][text_idx]))
            hard_neg_texts.append(self.rng.choice(_input['hard_neg_texts'][hard_text_idx]))

        # 멀티프로세싱을 사용하여 이미지 로드 및 RGB 변환
        images_list = process_batch(all_images_urls)
        images_list = [convert_to_rgb(img) for img in images_list]

        hard_image_list = process_batch(all_hard_images_urls)
        hard_image_list = [convert_to_rgb(img) for img in hard_image_list]

        # 결과 저장
        processed_dict['images'] = [*images_list, *hard_image_list]
        processed_dict['text'] = [*text_list, *hard_text_list, *neg_texts, *hard_neg_texts]

        kwargs = {
            'return_tensors': self.return_tensors,
            'padding': self.padding,
            'truncation': self.truncation,
            'pad_to_multiple_of': self.pad_to_multiple_of,
        }
        processor_input = dict(processed_dict, **kwargs)

        return self.processor(**processor_input)
