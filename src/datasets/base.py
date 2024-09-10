import logging
from typing import Optional, Any, Union
from pydantic import BaseModel, ConfigDict
from datasets import Sequence, Value, Image, IterableDataset, Dataset
from transformers import add_end_docstrings

logger = logging.getLogger(__name__)

__all__ = [
    "TextDatasetFeaturesWithImage",
    "TextDatasetFeaturesWithImageURL",
    "SequenceTextDatasetFeaturesWithImage",
    "SequenceTextDatasetFeaturesWithImageURL",
    "HardSequenceTextDatasetFeaturesWithImage",
    "HardSequenceTextDatasetFeaturesWithImageURL",
    "NegCLIPTextDatasetFeaturesWithImageURL",
    "NegCLIPSequenceTextDatasetFeaturesWithImageURL",
    "BaseBuilder",
]

# Common docstring for Dataset Features attributes
BASE_IMAGE_DATASET_FEATURES_ATTRIBUTES = """
Attributes:
    images (`Image`): 
        The image field of the dataset, represented by the `Image` feature type.
    text (`Value` or `Sequence`): 
        The text field of the dataset, either a single string (`Value`) or a sequence of strings (`Sequence`).
"""

BASE_IMAGE_URL_DATASET_FEATURES_ATTRIBUTES = """
Attributes:
    images (`str`): 
        The image field of the dataset, represented by the `str` image_url feature type.
    text (`Value` or `Sequence`): 
        The text field of the dataset, either a single string (`Value`) or a sequence of strings (`Sequence`).
"""

# Common docstring for Dataset Features with Hard Negatives attributes
NegCLIP_TEXT_ATTRIBUTES = """
Attributes:
    hard_images (`Sequence[Image]`): 
        A sequence of hard negative images.
    hard_texts (`Sequence[Value]`): 
        A sequence of hard negative texts.
    neg_texts (`Sequence[Value]`): 
        A sequence of negative texts.
    hard_neg_texts (`Sequence[Value]`): 
        A sequence of hard negative texts.
"""

# Common docstring for Sequence Dataset Features with Hard Negatives attributes
NegCLIP_SEQUENCE_TEXT_ATTRIBUTES = """
Attributes:
    hard_images (`Sequence[Image]`): 
        A sequence of hard negative images.
    hard_texts (`Sequence[Sequence[Value]]`): 
        A sequence of hard negative text sequences.
    neg_texts (`Sequence[Sequence[Value]]`): 
        A sequence of negative text sequences.
    hard_neg_texts (`Sequence[Sequence[Value]]`): 
        A sequence of hard negative text sequences.
"""


@add_end_docstrings(BASE_IMAGE_DATASET_FEATURES_ATTRIBUTES)
class TextDatasetFeaturesWithImage(BaseModel):
    """
    A dataset features class for handling basic image and text data.
    """
    images: Image = Image()
    text: Value = Value(dtype='string', id=None)

    model_config = ConfigDict(frozen=True, strict=True, validate_assignment=True)


@add_end_docstrings(BASE_IMAGE_URL_DATASET_FEATURES_ATTRIBUTES)
class TextDatasetFeaturesWithImageURL(BaseModel):
    """
    A dataset features class for handling basic image_url and text data.
    """
    images: Value = Value(dtype='string', id=None)
    text: Value = Value(dtype='string', id=None)

    model_config = ConfigDict(frozen=True, strict=True, validate_assignment=True)


@add_end_docstrings(BASE_IMAGE_DATASET_FEATURES_ATTRIBUTES)
class SequenceTextDatasetFeaturesWithImage(BaseModel):
    """
    A dataset features class for handling sequences of text and images.
    """
    images: Image = Image()
    text: Sequence = Sequence(Value(dtype='string', id=None))

    model_config = ConfigDict(frozen=True, strict=True, validate_assignment=True)


@add_end_docstrings(BASE_IMAGE_URL_DATASET_FEATURES_ATTRIBUTES)
class SequenceTextDatasetFeaturesWithImageURL(BaseModel):
    """
    A dataset features class for handling sequences of text and images.
    """
    images: Value = Value(dtype='string', id=None)
    text: Sequence = Sequence(Value(dtype='string', id=None))

    model_config = ConfigDict(frozen=True, strict=True, validate_assignment=True)


@add_end_docstrings(BASE_IMAGE_DATASET_FEATURES_ATTRIBUTES)
class HardSequenceTextDatasetFeaturesWithImage(SequenceTextDatasetFeaturesWithImage):
    """
    A dataset features class for handling sequences of text and images, with support for hard negative texts.
    It extends `SequenceTextDatasetFeaturesWithImage` by adding the `hard_texts` attribute.

    Attributes:
        hard_texts (Optional[Any]): A sequence of hard negative texts. If not provided, it will be initialized as a
                                    `Sequence` of strings during `model_post_init`.
        model_config (ConfigDict): Configuration settings for model validation, strictness, and frozen state.
                                   This allows for more flexibility with model validation and type checks.
    """
    hard_texts: Optional[Any] = None
    model_config = ConfigDict(frozen=False, strict=False, validate_assignment=False, arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes the `hard_texts` attribute if it is not already set. This method is called after the model is
        initialized, and it ensures that `hard_texts` is properly configured as a `Sequence` of strings.

        Args:
            __context (Any): Contextual information passed during initialization (unused in this method).
        """
        self.hard_texts = Sequence(Value(dtype='string', id=None)) if self.hard_texts is None else self.hard_texts


@add_end_docstrings(BASE_IMAGE_URL_DATASET_FEATURES_ATTRIBUTES)
class HardSequenceTextDatasetFeaturesWithImageURL(SequenceTextDatasetFeaturesWithImageURL):
    """
    A dataset features class for handling sequences of text and images with support for hard negatives.
    This class extends `SequenceTextDatasetFeaturesWithImageURL` by adding support for hard negatives in the text data.

    Attributes:
        hard_texts (`Optional[Any]`): A sequence of hard negative texts.
    """
    hard_texts: Optional[Any] = None
    model_config = ConfigDict(frozen=False, strict=False, validate_assignment=False, arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes the `hard_texts` field with a default sequence of values if not provided.

        Args:
            __context (Any): Context provided during initialization.
        """
        self.hard_texts = Sequence(Value(dtype='string', id=None)) if self.hard_texts is None else self.hard_texts


@add_end_docstrings(BASE_IMAGE_URL_DATASET_FEATURES_ATTRIBUTES)
@add_end_docstrings(NegCLIP_TEXT_ATTRIBUTES)
class NegCLIPTextDatasetFeaturesWithImageURL(TextDatasetFeaturesWithImageURL):
    """
    A dataset features class with support for hard negatives (HN) in NegCLIP.
    Attributes for hard negatives are added on top of the base dataset features.
    See https://github.com/vinid/neg_clip for more details.
    """
    model_config = ConfigDict(frozen=False, strict=False, validate_assignment=False, arbitrary_types_allowed=True)
    hard_images: Optional[Any] = None
    hard_texts: Optional[Any] = None
    neg_texts: Optional[Any] = None
    hard_neg_texts: Optional[Any] = None

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes the attributes related to hard negatives with default values if they are not provided.

        Args:
            __context (Any): Context provided during initialization.
        """
        self.hard_images = Sequence(Image()) if self.hard_images is None else self.hard_images
        self.hard_texts = Sequence(Value(dtype='string', id=None)) if self.hard_texts is None else self.hard_texts
        self.neg_texts = Sequence(Value(dtype='string', id=None)) if self.neg_texts is None else self.neg_texts
        self.hard_neg_texts = Sequence(
            Value(dtype='string', id=None)) if self.hard_neg_texts is None else self.hard_neg_texts


@add_end_docstrings(NegCLIP_SEQUENCE_TEXT_ATTRIBUTES)
@add_end_docstrings(BASE_IMAGE_URL_DATASET_FEATURES_ATTRIBUTES)
class NegCLIPSequenceTextDatasetFeaturesWithImageURL(SequenceTextDatasetFeaturesWithImageURL):
    """
    A dataset features class with support for hard negatives (HN) for sequences of text.
    Attributes for hard negatives are added on top of the sequence dataset features.
    """
    model_config = ConfigDict(frozen=False, strict=False, validate_assignment=False, arbitrary_types_allowed=True)
    hard_images: Optional[Any] = None
    hard_texts: Optional[Any] = None
    neg_texts: Optional[Any] = None
    hard_neg_texts: Optional[Any] = None

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes the attributes related to hard negatives for sequences of text.

        Args:
            __context (Any): Context provided during initialization.
        """
        self.hard_images = Sequence(Image()) if self.hard_images is None else self.hard_images
        self.hard_texts = Sequence(
            Sequence(Value(dtype='string', id=None))) if self.hard_texts is None else self.hard_texts
        self.neg_texts = Sequence(
            Sequence(Value(dtype='string', id=None))) if self.neg_texts is None else self.neg_texts
        self.hard_neg_texts = Sequence(
            Sequence(Value(dtype='string', id=None))) if self.hard_neg_texts is None else self.hard_neg_texts


class BaseBuilder(BaseModel):
    """
    A base class for building datasets. The `build_dataset` method must be implemented by subclasses
    to handle dataset creation.

    Attributes:
        features (`Optional[Any]`): The dataset's structure, which can be customized by subclasses.

    Raises:
        NotImplementedError: If the `build_dataset` method is not implemented in a subclass.
    """
    features: Optional[Any] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def build_dataset(self) -> Union[Dataset, IterableDataset]:
        """
        Builds the dataset. This method must be implemented by subclasses to define how the dataset
        should be constructed.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError
