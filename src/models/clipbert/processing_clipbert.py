"""
Image/Text processor class
"""
import os
from typing import List, Union

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

from transformers.image_utils import ImageInput
from transformers.processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
)
from transformers.tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput
from src.common import registry
from transformers import BertTokenizerFast, CLIPImageProcessor

__all__ = ["CLIPBertProcessor"]

IMG_PROCESSOR_CLASS = "CLIPImageProcessor"
TOKENIZER_CLASS = "BertTokenizerFast"


class CLIPBertProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    _defaults = {
        "text_kwargs": {
            "padding": "max_length",
            "max_length": 64,
        },
    }


@registry.register_processor("CLIPBertProcessor")
class CLIPBertProcessor(ProcessorMixin):

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = IMG_PROCESSOR_CLASS
    tokenizer_class = TOKENIZER_CLASS

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    @classmethod
    def from_text_vision_pretrained(
            cls,
            text_pretrained_model_name_or_path: Union[str, os.PathLike],
            vision_pretrained_model_name_or_path: Union[str, os.PathLike],
    ):
        tokenizer = BertTokenizerFast.from_pretrained(text_pretrained_model_name_or_path)
        image_processor = CLIPImageProcessor.from_pretrained(vision_pretrained_model_name_or_path)

        return cls(image_processor=image_processor, tokenizer=tokenizer)

    def __call__(
            self,
            text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
            images: ImageInput = None,
            audio=None,
            videos=None,
            **kwargs: Unpack[CLIPBertProcessorKwargs],
    ) -> BatchEncoding:

        if text is None and images is None:
            raise ValueError("You must specify either text or images.")
        output_kwargs = self._merge_kwargs(
            CLIPBertProcessor,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        # then, we can pass correct kwargs to each processor
        if text is not None:
            encoding = self.tokenizer(text, **output_kwargs["text_kwargs"])

        if images is not None:
            image_features = self.image_processor(images, **output_kwargs["images_kwargs"])

        # BC for explicit return_tensors
        if "return_tensors" in output_kwargs["common_kwargs"]:
            return_tensors = output_kwargs["common_kwargs"].pop("return_tensors", None)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values
            return encoding
        elif text is not None:
            return encoding
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
