import logging
from typing import Optional, Any, Union
from pydantic import BaseModel, ConfigDict
from datasets import Sequence, Value, Features, Image, IterableDataset, Dataset
from transformers import add_end_docstrings

logger = logging.getLogger(__name__)

# Common docstring for Dataset Features attributes
BASE_DATASET_FEATURES_ATTRIBUTES = """
Attributes:
    images (`Image`): 
        The image field of the dataset, represented by the `Image` feature type.
    text (`Value` or `Sequence`): 
        The text field of the dataset, either a single string (`Value`) or a sequence of strings (`Sequence`).
"""

# Common docstring for Dataset Features with Hard Negatives attributes
DATASET_FEATURES_WITH_HN_ATTRIBUTES = """
    hard_images (`Sequence[`Image`]`): 
        A sequence of hard negative images.
    hard_texts (`Sequence[`Value`]`): 
        A sequence of hard negative texts.
    neg_texts (`Sequence[`Value`]`): 
        A sequence of negative texts.
    hard_neg_texts (`Sequence[`Value`]`): 
        A sequence of hard negative texts.
"""

# Common docstring for Sequence Dataset Features with Hard Negatives attributes
SEQUENCE_DATASET_FEATURES_WITH_HN_ATTRIBUTES = """
    hard_images (`Sequence[`Image`]`): 
        A sequence of hard negative images.
    hard_texts (`Sequence[`Sequence[`Value`]`]`): 
        A sequence of hard negative text sequences.
    neg_texts (`Sequence[`Sequence[`Value`]`]`): 
        A sequence of negative text sequences.
    hard_neg_texts (`Sequence[`Sequence[`Value`]`]`): 
        A sequence of hard negative text sequences.
"""


@add_end_docstrings(BASE_DATASET_FEATURES_ATTRIBUTES)
class DatasetFeatures(BaseModel):
    """
    A dataset features class for handling basic image and text data.
    """
    images: Image = Image()
    text: Value = Value(dtype='string', id=None)

    model_config = ConfigDict(frozen=True, strict=True, validate_assignment=True)


@add_end_docstrings(BASE_DATASET_FEATURES_ATTRIBUTES)
class SequenceTextDatasetFeatures(BaseModel):
    """
    A dataset features class for handling sequences of text and images.
    """
    images: Image = Image()
    text: Sequence = Sequence(Value(dtype='string', id=None))

    model_config = ConfigDict(frozen=True, strict=True, validate_assignment=True)


@add_end_docstrings(DATASET_FEATURES_WITH_HN_ATTRIBUTES)
@add_end_docstrings(BASE_DATASET_FEATURES_ATTRIBUTES)
class DatasetFeaturesWithHN(DatasetFeatures):
    """
    A dataset features class with support for hard negatives (HN).

    Attributes for hard negatives are added on top of the base dataset features.
    """
    model_config = ConfigDict(frozen=False, strict=False, validate_assignment=False, arbitrary_types_allowed=True)
    hard_images: Optional[Any] = None
    hard_texts: Optional[Any] = None
    neg_texts: Optional[Any] = None
    hard_neg_texts: Optional[Any] = None

    def model_post_init(self, __context: Any) -> None:
        self.hard_images = Sequence(Image()) if self.hard_images is None else self.hard_images
        self.hard_texts = Sequence(Value(dtype='string', id=None)) if self.hard_texts is None else self.hard_texts
        self.neg_texts = Sequence(Value(dtype='string', id=None)) if self.neg_texts is None else self.neg_texts
        self.hard_neg_texts = Sequence(Value(dtype='string', id=None)) if self.hard_neg_texts is None else self.hard_neg_texts


@add_end_docstrings(SEQUENCE_DATASET_FEATURES_WITH_HN_ATTRIBUTES)
@add_end_docstrings(BASE_DATASET_FEATURES_ATTRIBUTES)
class SequenceTextDatasetFeaturesWithHN(SequenceTextDatasetFeatures):
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
        self.hard_images = Sequence(Image()) if self.hard_images is None else self.hard_images
        self.hard_texts = Sequence(Sequence(Value(dtype='string', id=None))) if self.hard_texts is None else self.hard_texts
        self.neg_texts = Sequence(Sequence(Value(dtype='string', id=None))) if self.neg_texts is None else self.neg_texts
        self.hard_neg_texts = Sequence(Sequence(Value(dtype='string', id=None))) if self.hard_neg_texts is None else self.hard_neg_texts


class BaseBuilder(BaseModel):
    """
    A base class for building datasets. The `build_dataset` method must be implemented by subclasses
    to handle dataset creation.

    Attributes:
        features (`Optional[`Any`]`): The dataset's structure, which can be customized by subclasses.

    Raises:
        NotImplementedError: If the `build_dataset` method is not implemented in a subclass.
    """

    features: Optional[Any] = None  # Added Optional[Any] for features
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def build_dataset(self) -> Union[Dataset, IterableDataset]:
        """
        Builds the dataset. This method must be implemented by subclasses to define how the dataset
        should be constructed.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError


class DatasetBuilder(BaseBuilder):
    """
    A builder class for creating a basic dataset. It extends the `BaseBuilder` class and uses
    `BaseDatasetFeatures` to define the dataset's structure. If no features are provided, it automatically
    sets `BaseDatasetFeatures`.

    Attributes:
        features (`Optional[`Features`]`):
            The dataset's structure and data types, which default to `BaseDatasetFeatures`.
    """

    def model_post_init(self, __context: Any) -> None:
        if not self.features:
            self.features = Features(DatasetFeatures())


class SequenceTextDatasetBuilder(BaseBuilder):
    """
    A builder class for creating a sequence dataset. It extends the `BaseBuilder` class and uses
    `SequenceTextDatasetFeatures` to define the dataset's structure. If no features are provided, it
    automatically sets `SequenceTextDatasetFeatures`.

    Attributes:
        features (`Optional`[`Features`]`):
            The dataset's structure and data types, which default to `SequenceTextDatasetFeatures`.
    """

    def model_post_init(self, __context: Any) -> None:
        if not self.features:
            self.features = Features(SequenceTextDatasetFeatures())


class SequenceTextDatasetWithHNBuilder(BaseBuilder):
    """
    A builder class for creating a sequence dataset with hard negatives (HN). It extends the `BaseBuilder`
    class and uses `SequenceTextDatasetFeaturesWithHN` to define the dataset's structure. If no features
    are provided, it automatically sets `SequenceTextDatasetFeaturesWithHN`.

    Attributes:
        features (`Optional[`Features`]`):
            The dataset's structure and data types, which default to `SequenceTextDatasetFeaturesWithHN`.
    """

    def model_post_init(self, __context: Any) -> None:
        if not self.features:
            self.features = Features(SequenceTextDatasetFeaturesWithHN())