import asyncio
from dataclasses import dataclass
from typing import Union, List, Dict, Optional, Any
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from transformers import BatchEncoding
from transformers.utils import add_end_docstrings
from transformers.utils import logging

from src.common.registry import registry
from src.utils.utils import process_batch_async, load_metadata_from_tar_files, load_image_from_tar
from .base import BaseCollator, BASE_COLLATOR_DOCSTRING

logger = logging.get_logger(__name__)

__all__ = [
    "ImageCollator",
    "ImageCollatorForInstructor",
    "ImageURLCollator",
    "ImageURLCollatorForEvaluation",
    "NegCLIPWithImageURLCollator"
]


def is_url(url_or_filename: str) -> bool:
    """
    Checks if a given string is a valid URL.

    Args:
        url_or_filename (str): A string that may represent a URL.

    Returns:
        bool: True if the string is a valid URL, False otherwise.
    """
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


@add_end_docstrings(BASE_COLLATOR_DOCSTRING)
@dataclass
@registry.register_collator('ImageCollator')
class ImageCollator(BaseCollator):
    """
    A collator class for processing dictionaries containing image and text data. The 'images' key in the input
    dictionaries must hold `PIL.Image` objects, which are converted to RGB format, and the 'text' key must hold
    strings. This class handles dynamic padding, truncation, and tensor conversion before passing the processed
    data to the processor.

    Raises:
        TypeError:
            - If the 'images' key contains objects that are not `PIL.Image` instances.
            - If the 'text' key contains values that are not `str` instances.
        ValueError:
            - If the input dictionaries contain keys other than 'images' and 'text'.
    """

    def __call__(self, inputs: List[Dict[str, Any]]) -> BatchEncoding:
        """
        Processes a batch of input dictionaries containing image and text data. The 'images' key in each
        dictionary is expected to hold a `PIL.Image` object, which is converted to RGB format. The 'text'
        key must hold strings. The processed images and text are then passed to the processor for further encoding.

        Args:
            inputs (List[Dict[str, Any]]):
                A list of dictionaries where each dictionary contains an 'images' key that holds
                a `PIL.Image` object and a 'text' key that holds a string.

        Raises:
            TypeError:
                - If any value in the 'images' key is not a valid `PIL.Image` object.
                - If any value in the 'text' key is not a string.
            ValueError:
                - If any dictionary contains keys other than 'images' and 'text'.

        Returns:
            BatchEncoding:
                A batch of encoded inputs, with padding, truncation, and other configurations ready
                for model consumption.
        """

        def _check_type_and_convert_image(value: Any, key: str) -> Union[Image.Image, Any]:
            """
            Ensures the type of the value matches the expected type for the given key.

            Args:
                value (Any): The value to check and possibly convert.
                key (str): The key corresponding to the value (either 'images' or 'text').

            Returns:
                `PIL.Image`: If the key is 'images', returns the image converted to RGB.
                `str`: If the key is 'text', returns the string as-is.

            Raises:
                TypeError: If the value does not match the expected type for the given key.
            """
            if key == 'images':
                if value is None:
                    logger.warning_once(
                        "The 'image' key is None, which typically occurs when there is no corresponding image "
                        "for the text (e.g., a negative text) or when multiple texts correspond to a single image. "
                        "Please verify this scenario."
                    )
                    return None
                elif not isinstance(value, Image.Image):
                    raise TypeError(f"Expected `PIL.Image.Image` but got {type(value)} for key 'images'")
                return value.convert("RGB")
            elif key == 'text':
                if not isinstance(value, str):
                    raise TypeError(f"Expected `str` but got {type(value)} for key 'text'")
                return value
            return value

        # Check that all dictionaries contain only 'images' and 'text' as keys
        allowed_keys = {'images', 'text'}
        for input_dict in inputs:
            if set(input_dict.keys()) != allowed_keys:
                raise ValueError(
                    f"Input dictionaries must only contain the keys 'images' and 'text'. Found: {input_dict.keys()}")

        # Process 'images' and 'text', performing type checks and conversions
        processed_dict = {
            key: [
                _check_type_and_convert_image(value, key) for value in [d[key] for d in inputs if d[key] is not None]
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
@registry.register_collator('ImageCollatorForInstructor')
class ImageCollatorForInstructor(BaseCollator):
    """
    A collator class for processing dictionaries containing image and text data. The 'images' key in the input
    dictionaries must hold `PIL.Image` objects, which are converted to RGB format, and the 'text' key must hold
    strings. This class handles dynamic padding, truncation, and tensor conversion before passing the processed
    data to the processor.

    Raises:
        TypeError:
            - If the 'images' key contains objects that are not `PIL.Image` instances.
            - If the 'text' key contains values that are not `str` instances.
        ValueError:
            - If the input dictionaries contain keys other than 'images' and 'text'.
    """

    def __call__(self, inputs: List[Dict[str, Any]]) -> BatchEncoding:
        """
        Processes a batch of input dictionaries containing image and text data. The 'images' key in each
        dictionary is expected to hold a `PIL.Image` object, which is converted to RGB format. The 'text'
        key must hold strings. The processed images and text are then passed to the processor for further encoding.

        Args:
            inputs (List[Dict[str, Any]]):
                A list of dictionaries where each dictionary contains an 'images' key that holds
                a `PIL.Image` object and a 'text' key that holds a string.

        Raises:
            TypeError:
                - If any value in the 'images' key is not a valid `PIL.Image` object.
                - If any value in the 'text' key is not a string.
            ValueError:
                - If any dictionary contains keys other than 'images' and 'text'.

        Returns:
            BatchEncoding:
                A batch of encoded inputs, with padding, truncation, and other configurations ready
                for model consumption.
        """
        instruction = 'Represent the caption for retrieving matching images: '

        def _check_type_and_convert_image(value: Any, key: str) -> Union[Image.Image, Any]:
            """
            Ensures the type of the value matches the expected type for the given key.

            Args:
                value (Any): The value to check and possibly convert.
                key (str): The key corresponding to the value (either 'images' or 'text').

            Returns:
                `PIL.Image`: If the key is 'images', returns the image converted to RGB.
                `str`: If the key is 'text', returns the string as-is.

            Raises:
                TypeError: If the value does not match the expected type for the given key.
            """
            if key == 'images':
                if value is None:
                    logger.warning_once(
                        "The 'image' key is None, which typically occurs when there is no corresponding image "
                        "for the text (e.g., a negative text) or when multiple texts correspond to a single image. "
                        "Please verify this scenario."
                    )
                    return None
                elif not isinstance(value, Image.Image):
                    raise TypeError(f"Expected `PIL.Image.Image` but got {type(value)} for key 'images'")
                return value.convert("RGB")
            elif key == 'text':
                if not isinstance(value, str):
                    raise TypeError(f"Expected `str` but got {type(value)} for key 'text'")
                return instruction + value
            return value

        # Check that all dictionaries contain only 'images' and 'text' as keys
        allowed_keys = {'images', 'text'}
        for input_dict in inputs:
            if set(input_dict.keys()) != allowed_keys:
                raise ValueError(
                    f"Input dictionaries must only contain the keys 'images' and 'text'. Found: {input_dict.keys()}")

        # Process 'images' and 'text', performing type checks and conversions
        processed_dict = {
            key: [
                _check_type_and_convert_image(value, key) for value in [d[key] for d in inputs if d[key] is not None]
            ]
            for key in inputs[0].keys()
        }
        len_text = len(processed_dict['text'])
        # Create kwargs for processor, including padding, truncation, etc.
        kwargs = {
            'return_tensors': self.return_tensors,
            'padding': self.padding,
            'truncation': self.truncation,
            'pad_to_multiple_of': self.pad_to_multiple_of,
        }
        instruct_input = dict({'text': [instruction] * len_text}, **kwargs)
        processor_input = dict(processed_dict, **kwargs)

        output = self.processor(**processor_input)

        attention_mask_shape = output.data['attention_mask'].shape
        attention_mask_dtype = output.data['attention_mask'].dtype

        _instruct_mask = self.processor(**instruct_input).data['attention_mask']

        def _prepare_instruction_mask(
                instruct_mask: torch.Tensor,
                shape: torch.Size,
                dtype: torch.dtype,
        ) -> Tensor:
            # reducing the attention length by 1 in order to omit the attention corresponding to the end_token
            instruct_mask = instruct_mask[:, :-1]

            # creating instruction attention matrix equivalent to the size of the input attention matrix
            instruction_attention_mask = torch.zeros(
                shape, dtype=dtype
            )
            instruction_attention_mask[
            : instruct_mask.size(0), : instruct_mask.size(1)
            ] = instruct_mask

            # In the pooling layer we want to consider only the tokens corresponding to the input text
            # and not the instruction. This is achieved by inverting the
            # attention_mask corresponding to the instruction.
            instruction_attention_mask = 1 - instruction_attention_mask

            return instruction_attention_mask

        instruction_mask = _prepare_instruction_mask(
            instruct_mask=_instruct_mask,
            shape=attention_mask_shape,
            dtype=attention_mask_dtype
        )

        output.data['instruction_mask'] = instruction_mask

        return output


@add_end_docstrings(BASE_COLLATOR_DOCSTRING)
@dataclass
@registry.register_collator('ImageURLCollator')
class ImageURLCollator(BaseCollator):
    """
    A collator class for processing dictionaries containing image URLs and text. The 'images' key in the input
    dictionaries must hold image URLs, which are fetched asynchronously and converted into `PIL.Image` objects
    in RGB format. The 'text' key must hold strings. The collator then combines the processed data with padding
    and other configurations before passing it to the processor.

    Raises:
        TypeError:
            - If the 'text' key contains values that are not `str` instances.
        ValueError:
            - If the 'images' key contains values that are not valid URLs.
            - If the input dictionaries contain keys other than 'images' and 'text'.
    """

    def __call__(self, inputs: List[Dict[str, Any]]) -> BatchEncoding:
        """
        Processes a batch of input dictionaries containing image URLs and text.
        The 'images' key in each dictionary is expected to hold a valid image URL.
        The 'text' key must hold strings. The processed images and text are then passed to the processor.

        Args:
            inputs (List[Dict[str, Any]]):
                A list of dictionaries where each dictionary contains an 'images' key that holds
                a URL to an image and a 'text' key that holds a string.

        Returns:
            BatchEncoding:
                A batch of encoded inputs, with padding, truncation, and other configurations ready
                for model consumption.
        """

        # Extract valid image URLs and texts
        all_image_urls = []
        all_texts = []

        for input_dict in inputs:
            if 'images' in input_dict and 'text' in input_dict:
                image_url = input_dict['images']
                text = input_dict['text']

                # Exclude invalid image URLs and texts
                if image_url is not None and is_url(image_url) and isinstance(text, str):
                    all_image_urls.append(image_url)
                    all_texts.append(text)
                else:
                    # Log the exclusion if needed
                    logger.warning_once(f"Excluding invalid data: images={image_url}, text={text}")

        # Fetch images using the process_batch_async function if there are URLs
        images_list = asyncio.run(process_batch_async(all_image_urls)) if all_image_urls else []

        # Filter out False images and corresponding texts
        valid_images = []
        valid_texts = []
        for image, text in zip(images_list, all_texts):
            if image:  # Only include valid images and corresponding texts
                valid_images.append(image)
                valid_texts.append(text)

        # Create a dictionary to store processed data
        processed_dict = {'images': valid_images, 'text': valid_texts}
        print(len(valid_images))

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
@registry.register_collator('ImageURLCollatorForEvaluation')
class ImageURLCollatorForEvaluation(BaseCollator):
    """
    A collator class for processing dictionaries containing image URLs and text. The 'images' key in the input
    dictionaries must hold image URLs, which are fetched asynchronously and converted into `PIL.Image` objects
    in RGB format. The 'text' key must hold strings. The collator then combines the processed data with padding
    and other configurations before passing it to the processor.

    This version of the collator does not filter out any invalid URLs or texts but processes all lines.

    Raises:
        TypeError:
            - If the 'text' key contains values that are not `str` instances.
        ValueError:
            - If the 'images' key contains values that are not valid URLs.
    """

    def __call__(self, inputs: List[Dict[str, Any]]) -> BatchEncoding:
        """
        Processes a batch of input dictionaries containing image URLs and text.
        The 'images' key in each dictionary is expected to hold a valid image URL.
        The 'text' key must hold strings. The processed images and text are then passed to the processor.

        Args:
            inputs (List[Dict[str, Any]]):
                A list of dictionaries where each dictionary contains an 'images' key that holds
                a URL to an image and a 'text' key that holds a string.

        Returns:
            BatchEncoding:
                A batch of encoded inputs, with padding, truncation, and other configurations ready
                for model consumption.
        """

        # Initialize lists for image URLs and text data
        all_image_urls = []
        all_texts = []

        # Loop over each input dict, add image URL and text (even if URL is None)
        for input_dict in inputs:
            if 'images' in input_dict and 'text' in input_dict:
                image_url = input_dict['images']
                text = input_dict['text']

                # Append image URLs to the list (even if they are None)
                if image_url is not None:
                    all_image_urls.append(image_url)
                all_texts.append(text)

        # Fetch images using the process_batch_async function if there are URLs
        images_list = asyncio.run(process_batch_async(all_image_urls)) if all_image_urls else []
        if len(all_image_urls) != len(images_list):
            logger.warning("Some images are not included in images list. This will result in inaccurate result.")
        # No filtering based on image validity, maintain order as per inputs
        processed_dict = {'images': images_list, 'text': all_texts}

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
@registry.register_collator('ImageTarPathCollator')
class ImageTarPathCollator(BaseCollator):
    """
    A collator class for processing dictionaries containing image URLs and text. The 'images' key in the input
    dictionaries must hold image URLs, which are fetched asynchronously and converted into `PIL.Image` objects
    in RGB format. The 'text' key must hold strings. The collator then combines the processed data with padding
    and other configurations before passing it to the processor.
    """

    tar_directory: Optional[str] = None
    metadata: Dict[str, Dict[str, str]] = None  # To hold the metadata dictionary

    def __post_init__(self):
        # Load metadata from tar files only once at initialization
        if self.tar_directory is None:
            raise ValueError
        if self.metadata is None:
            self.metadata = load_metadata_from_tar_files(self.tar_directory)

    def __call__(self, inputs: List[Dict[str, Any]]) -> BatchEncoding:
        """
        Processes a batch of input dictionaries containing image URLs and text. The 'images' key in each
        dictionary is expected to hold a valid image URL. The images are fetched from tar files and converted
        into `PIL.Image` objects in RGB format. The 'text' key must hold strings. The processed images and
        text are then passed to the processor for further encoding.

        Args:
            inputs (List[Dict[str, Any]]):
                A list of dictionaries where each dictionary contains an 'images' key that holds
                a URL to an image and a 'text' key that holds a string.

        Returns:
            BatchEncoding:
                A batch of encoded inputs, with padding, truncation, and other configurations ready
                for model consumption.
        """
        # Check that all dictionaries contain only 'images' and 'text' as keys
        allowed_keys = {'images', 'text'}
        for input_dict in inputs:
            if set(input_dict.keys()) != allowed_keys:
                raise ValueError(
                    f"Input dictionaries must only contain the keys 'images' and 'text'. Found: {input_dict.keys()}")

        # Validate 'images' values and 'text' values
        for input_dict in inputs:
            if 'images' in input_dict:
                if input_dict['images'] is None:
                    logger.warning_once(
                        "The 'images' key is None, which typically occurs when there is no corresponding image "
                        "for the text (e.g., a negative text) or when multiple texts correspond to a single image. "
                        "Please verify this scenario."
                    )
            if 'text' in input_dict and not isinstance(input_dict['text'], str):
                raise TypeError(f"Expected a string for key 'text', but got: {type(input_dict['text'])}")

        # Extract all non-None image URLs from inputs and corresponding texts
        all_image_urls = []
        all_texts = []
        for d in inputs:
            if d['images'] is not None:
                all_image_urls.append(d['images'])
                all_texts.append(d['text'])

        # Fetch images corresponding to URLs
        images_list = []
        for url in all_image_urls:
            if url in self.metadata:
                tar_file = self.metadata[url]["tar_file"]
                key = self.metadata[url]["key"]
                image = load_image_from_tar(tar_file, key)
                if image:
                    images_list.append(image)

        # Filter out False images and corresponding texts
        valid_images = []
        valid_texts = []
        for image, text in zip(images_list, all_texts):
            if image:  # Only include valid images and corresponding texts
                valid_images.append(image)
                valid_texts.append(text)

        # Create a dictionary to store processed data
        processed_dict = {'images': valid_images, 'text': valid_texts}

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
@registry.register_collator('NegCLIPWithImageURLCollator')
class NegCLIPWithImageURLCollator(ImageURLCollator):
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
            neg_texts.append({'images': None, 'text': selected_neg_text})
            hard_neg_texts.append({'images': None, 'text': selected_hard_neg_text})

        # Combine image URLs with their corresponding texts
        all_inputs.extend([{'images': url, 'text': text} for url, text in zip(all_images_urls, text_list)])
        all_inputs.extend([{'images': url, 'text': text} for url, text in zip(all_hard_images_urls, hard_text_list)])

        # Add neg_texts and hard_neg_texts, which do not have associated images (image_url is None)
        all_inputs.extend(neg_texts)
        all_inputs.extend(hard_neg_texts)

        # Pass the modified input list to the parent class (ImageURLCollator)
        return super().__call__(all_inputs)
