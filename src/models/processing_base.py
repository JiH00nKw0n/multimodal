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
from transformers import AutoImageProcessor, AutoTokenizer

__all__ = ["BaseProcessor"]

IMG_PROCESSOR_CLASS = ("EfficientNetImageProcessor", "BitImageProcessor")
TOKENIZER_CLASS = (
    "XLMRobertaTokenizer",
    "XLMRobertaTokenizerFast",
    "DistilBertTokenizerFast",
    "BertTokenizerFast"
)


class BaseProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    _defaults = {
        "text_kwargs": {
            "padding": "max_length",
            "max_length": 64,
        },
    }


@registry.register_processor("BaseProcessor")
class BaseProcessor(ProcessorMixin):
    r"""
    Constructs a processor which wraps [`IMG_PROCESSOR_CLASS`] and
    [`TOKENIZER_CLASS`] into a single processor that interits both the image processor and
    tokenizer functionalities. See the [`~Processor.__call__`] and [`~Processor.decode`] for more
    information.
    The preferred way of passing kwargs is as a dictionary per modality, see usage example below.
        ```python
        processor(
            images=your_pil_image,
            text=["What is that?"],
            images_kwargs = {"crop_size": {"height": 224, "width": 224}},
            text_kwargs = {"padding": "do_not_pad"},
            common_kwargs = {"return_tensors": "pt"},
        )
        ```

    Args:
        image_processor (`IMG_PROCESSOR_CLASS`):
            The image processor is a required input.
        tokenizer (`TOKENIZER_CLASS`):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = IMG_PROCESSOR_CLASS
    tokenizer_class = TOKENIZER_CLASS

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    # ToDo: refactoring for more sophisticated method
    @classmethod
    def from_text_vision_pretrained(
            cls,
            text_pretrained_model_name_or_path: Union[str, os.PathLike],
            vision_pretrained_model_name_or_path: Union[str, os.PathLike],
    ):
        tokenizer = AutoTokenizer.from_pretrained(text_pretrained_model_name_or_path)
        image_processor = AutoImageProcessor.from_pretrained(vision_pretrained_model_name_or_path)

        return cls(image_processor=image_processor, tokenizer=tokenizer)

    def __call__(
            self,
            text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
            images: ImageInput = None,
            audio=None,
            videos=None,
            **kwargs: Unpack[BaseProcessorKwargs],
    ) -> BatchEncoding:
        """
        Main method to prepare text(s) and image(s) to be fed as input to the model. This method forwards the `text`
        arguments to BertTokenizerFast's [`~BertTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` arguments to
        EfficientNetImageProcessor's [`~EfficientNetImageProcessor.__call__`] if `images` is not `None`. Please refer
        to the doctsring of the above two methods for more information.

        Args:
            text (`str`, `List[str]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                    - `'tf'`: Return TensorFlow `tf.constant` objects.
                    - `'pt'`: Return PyTorch `torch.Tensor` objects.
                    - `'np'`: Return NumPy `np.ndarray` objects.
                    - `'jax'`: Return JAX `jnp.ndarray` objects.
        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if text is None and images is None:
            raise ValueError("You must specify either text or images.")
        output_kwargs = self._merge_kwargs(
            BaseProcessorKwargs,
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
