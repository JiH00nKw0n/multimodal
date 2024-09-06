import logging
import asyncio
from dataclasses import dataclass
from typing import Union, List, Dict, Optional, Any

import numpy as np
from PIL import Image
from transformers import BatchEncoding
from transformers.utils import add_end_docstrings

from src.common.registry import registry
from src.utils.utils import process_batch_async
from .base import BaseCollator, BASE_COLLATOR_DOCSTRING

logger = logging.getLogger(__name__)

__all__ = [
    "ImageCollator",
    "ImageURLCollator",
    "HardNegImageAndTextWithImageURLCollator"
]


@add_end_docstrings(BASE_COLLATOR_DOCSTRING)
@dataclass
@registry.register_collator('ImageCollator')
class ImageCollator(BaseCollator):
    """
    A collator class for processing dictionaries containing image data. The 'images' key in the input
    dictionaries must hold `PIL.Image` objects, which are converted to RGB format. This class handles
    dynamic padding, truncation, and tensor conversion before passing the processed data to the processor.

    Raises:
        TypeError:
            If the 'images' key contains objects that are not `PIL.Image` instances.
    """

    def __call__(self, inputs: List[Dict[str, Any]]) -> BatchEncoding:
        """
        Processes a batch of input dictionaries containing image data. The 'images' key in each
        dictionary is expected to hold a `PIL.Image` object, which is converted to RGB format. The
        processed images are then passed to the processor for further encoding.

        Args:
            inputs (List[Dict[str, Any]]):
                A list of dictionaries where each dictionary contains an 'images' key that holds
                a `PIL.Image` object.

        Raises:
            TypeError:
                If any value in the 'images' key is not a valid `PIL.Image` object.

        Returns:
            BatchEncoding:
                A batch of encoded inputs, with padding, truncation, and other configurations ready
                for model consumption.
        """

        def _convert_image(img: Any, key: str) -> Union[Image.Image | Any]:
            # If the key is 'images', ensure the value is a PIL.Image and convert to RGB
            if key == 'images':
                if not isinstance(img, Image.Image):
                    raise TypeError(f"Expected PIL.Image.Image but got {type(img)} for key 'images'")
                return img.convert("RGB")
            return img

        # Convert 'images' to RGB format, raise TypeError if 'images' value is not a PIL.Image object
        processed_dict = {
            key: [
                _convert_image(img, key) for img in [d[key] for d in inputs]
            ]
            for key in inputs[0].keys()
        }

        # Create kwargs for processor, including padding, truncation, etc.
        kwargs = {
            'return_tensors': self.return_tensors,
            'padding': self.padding,
            'truncation': self.truncation,
            'pad_to_multiple_of': self.pad_to_multiple_of,
        }

        # Merge processed inputs with kwargs and pass to the processor
        processor_input = dict(processed_dict, **kwargs)

        return self.processor(**processor_input)


@add_end_docstrings(BASE_COLLATOR_DOCSTRING)
@dataclass
@registry.register_collator('ImageURLCollator')
class ImageURLCollator(BaseCollator):
    """
    A collator class for processing dictionaries containing image URLs. The 'image_url' key in the input
    dictionaries must hold image URLs, which are fetched asynchronously and converted into `PIL.Image`
    objects in RGB format. The collator then combines the processed data with padding and other configurations
    before passing it to the processor.

    Raises:
        TypeError:
            If the fetched image is not a valid `PIL.Image` object.
    """

    def __call__(self, inputs: List[Dict[str, Any]]) -> BatchEncoding:
        """
        Processes a batch of input dictionaries containing image URLs. The 'image_url' key in each
        dictionary is expected to hold a valid image URL. The images are fetched asynchronously and
        converted into `PIL.Image` objects in RGB format. The processed images are then passed to
        the processor for further encoding.

        Args:
            inputs (List[Dict[str, Any]]):
                A list of dictionaries where each dictionary contains an 'image_url' key that holds
                a URL to an image.

        Raises:
            TypeError:
                If any value fetched from the URLs is not a valid `PIL.Image` object.

        Returns:
            BatchEncoding:
                A batch of encoded inputs, with padding, truncation, and other configurations ready
                for model consumption.
        """
        # Extract all non-None image URLs from inputs
        all_image_urls = [d['image_url'] for d in inputs if d['image_url'] is not None]

        # Fetch images using the process_batch_async function if there are URLs
        images_list = asyncio.run(process_batch_async(all_image_urls)) if all_image_urls else []

        # Create a dictionary to store processed data
        processed_dict = {'images': images_list}

        # Process all other keys in inputs and store them in the processed_dict
        for key in inputs[0].keys():
            if key != 'image_url':  # Skip the 'image_url' key as it's already processed
                processed_dict[key] = [d[key] for d in inputs]

        # Create kwargs for processor, including padding, truncation, etc.
        kwargs = {
            'return_tensors': self.return_tensors,
            'padding': self.padding,
            'truncation': self.truncation,
            'pad_to_multiple_of': self.pad_to_multiple_of,
        }

        # Merge processed inputs with kwargs and pass to the processor
        processor_input = dict(processed_dict, **kwargs)

        return self.processor(**processor_input)


@add_end_docstrings(BASE_COLLATOR_DOCSTRING)
@dataclass
@registry.register_collator('HardNegImageAndTextWithImageURLCollator')
class HardNegImageAndTextWithImageURLCollator(ImageURLCollator):
    """
    A collator class for processing sequences of text and images, with support for hard negatives.
    This class handles text sampling from multiple text fields (`text`, `hard_texts`, `neg_texts`,
    `hard_neg_texts`) and image URLs (`images`, `hard_images`). It utilizes random sampling to select
    text and image fields and then passes the processed data to the processor.

    The class inherits from `ImageURLCollator`, meaning that all image URLs are fetched and converted
    to RGB `PIL.Image` objects, which are passed to the processor along with the text.

    Args:
        seed (`int`, *optional*, defaults to 2024):
            Random seed for text and image selection.
        rng (`np.random.Generator`, *optional*):
            Optional random number generator instance. If not provided, a new generator is created using
            the provided seed.
    """

    seed: Optional[int] = 2024
    rng: Optional[np.random.Generator] = None

    def __post_init__(self):
        # Initialize the random number generator if not provided
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)

    def __call__(self, inputs: List[Dict[str, Any]]) -> BatchEncoding:
        """
        Processes a batch of input dictionaries containing image URLs and text fields. This includes
        sampling from both the main and hard negative text fields, as well as from the image and hard
        image URLs.

        Args:
            inputs (List[Dict[str, Any]]):
                A list of dictionaries where each dictionary contains 'images', 'text', 'hard_images',
                'hard_texts', 'neg_texts', and 'hard_neg_texts' fields.

        Returns:
            BatchEncoding:
                A batch of encoded inputs, with padding, truncation, and other configurations ready
                for model consumption.
        """
        all_inputs = []
        all_images_urls = []
        text_list = []

        all_hard_images_urls = []
        hard_text_list = []

        neg_texts = []
        hard_neg_texts = []

        for _input in inputs:
            # Randomly sample from text fields
            text_idx = self.rng.integers(0, len(_input['text']))
            selected_text = _input['text'][text_idx]

            hard_image_idx = self.rng.integers(0, len(_input['hard_images']))
            selected_hard_image_url = _input['hard_images'][hard_image_idx]

            hard_text_idx = self.rng.integers(0, len(_input['hard_texts'][hard_image_idx]))
            selected_hard_text = _input['hard_texts'][hard_image_idx][hard_text_idx]

            selected_neg_text = self.rng.choice(_input['neg_texts'][text_idx])
            selected_hard_neg_text = self.rng.choice(_input['hard_neg_texts'][hard_text_idx])

            # Collect image URLs and corresponding text fields into lists
            all_images_urls.append(_input['images'])
            text_list.append(selected_text)

            all_hard_images_urls.append(selected_hard_image_url)
            hard_text_list.append(selected_hard_text)

            # No corresponding image for neg_texts and hard_neg_texts, so use None
            neg_texts.append({'image_url': None, 'text': selected_neg_text})
            hard_neg_texts.append({'image_url': None, 'text': selected_hard_neg_text})

        # Combine image URLs with their corresponding texts
        all_inputs.extend([{'image_url': url, 'text': text} for url, text in zip(all_images_urls, text_list)])
        all_inputs.extend([{'image_url': url, 'text': text} for url, text in zip(all_hard_images_urls, hard_text_list)])

        # Add neg_texts and hard_neg_texts, which do not have associated images (image_url is None)
        all_inputs.extend(neg_texts)
        all_inputs.extend(hard_neg_texts)

        # Pass the modified input list to the parent class (ImageURLCollator)
        return super().__call__(all_inputs)
