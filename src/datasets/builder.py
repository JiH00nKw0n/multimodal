from typing import Any
from datasets import Features

from src.datasets.base import (
    BaseBuilder,
    HardSequenceTextDatasetFeaturesWithImageURL,
    HardSequenceTextDatasetFeaturesWithImage,
    NegCLIPSequenceTextDatasetFeaturesWithImageURL,
    NegCLIPTextDatasetFeaturesWithImageURL,
    SequenceTextDatasetFeaturesWithImage,
    SequenceTextDatasetFeaturesWithImageURL,
    TextDatasetFeaturesWithImage,
    TextDatasetFeaturesWithImageURL,
)

__all__ = [
    "TextDatasetFeaturesWithImageBuilder",
    "TextDatasetFeaturesWithImageURLBuilder",
    "SequenceTextDatasetFeaturesWithImageBuilder",
    "SequenceTextDatasetFeaturesWithImageURLBuilder",
    "HardSequenceTextDatasetWithImageBuilder",
    "HardSequenceTextDatasetFeaturesWithImageURLBuilder",
    "NegCLIPTextDatasetFeaturesWithImageURLBuilder",
    "NegCLIPSequenceTextDatasetFeaturesWithImageURLBuilder",
]


class TextDatasetFeaturesWithImageBuilder(BaseBuilder):
    """
    A builder class for creating a dataset with basic image and text data.
    It extends the `BaseBuilder` class and uses `TextDatasetFeaturesWithImage` to define the dataset's structure.
    If no features are provided, it automatically sets `TextDatasetFeaturesWithImage`.

    Attributes:
        features (`Optional[Features]`):
            The dataset's structure and data types, which default to `TextDatasetFeaturesWithImage`.
    """

    def model_post_init(self, __context: Any) -> None:
        if not self.features:
            self.features = Features(TextDatasetFeaturesWithImage())


class TextDatasetFeaturesWithImageURLBuilder(BaseBuilder):
    """
    A builder class for creating a dataset with image URLs and text data.
    It extends the `BaseBuilder` class and uses `TextDatasetFeaturesWithImageURL` to define the dataset's structure.
    If no features are provided, it automatically sets `TextDatasetFeaturesWithImageURL`.

    Attributes:
        features (`Optional[Features]`):
            The dataset's structure and data types, which default to `TextDatasetFeaturesWithImageURL`.
    """

    def model_post_init(self, __context: Any) -> None:
        if not self.features:
            self.features = Features(TextDatasetFeaturesWithImageURL())


class SequenceTextDatasetFeaturesWithImageBuilder(BaseBuilder):
    """
    A builder class for creating a dataset with sequences of text and images.
    It extends the `BaseBuilder` class and uses `SequenceTextDatasetFeaturesWithImage` to define the dataset's structure.
    If no features are provided, it automatically sets `SequenceTextDatasetFeaturesWithImage`.

    Attributes:
        features (`Optional[Features]`):
            The dataset's structure and data types, which default to `SequenceTextDatasetFeaturesWithImage`.
    """

    def model_post_init(self, __context: Any) -> None:
        if not self.features:
            self.features = Features(SequenceTextDatasetFeaturesWithImage())


class SequenceTextDatasetFeaturesWithImageURLBuilder(BaseBuilder):
    """
    A builder class for creating a dataset with sequences of text and image URLs.
    It extends the `BaseBuilder` class and uses `SequenceTextDatasetFeaturesWithImageURL` to define the dataset's structure.
    If no features are provided, it automatically sets `SequenceTextDatasetFeaturesWithImageURL`.

    Attributes:
        features (`Optional[Features]`):
            The dataset's structure and data types, which default to `SequenceTextDatasetFeaturesWithImageURL`.
    """

    def model_post_init(self, __context: Any) -> None:
        if not self.features:
            self.features = Features(SequenceTextDatasetFeaturesWithImageURL())


class HardSequenceTextDatasetWithImageBuilder(BaseBuilder):
    """
    A builder class for creating a dataset with sequences of text and images, along with hard negative texts.
    It extends `BaseBuilder` and uses `HardSequenceTextDatasetFeaturesWithImage` to define the dataset's structure.

    Attributes:
        features (Optional[Any]): The dataset's structure and features, which default to `HardSequenceTextDatasetFeaturesWithImage`.
    """

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization of the model. Ensures that the dataset features are properly set.

        Args:
            __context (Any): Contextual information passed during initialization (unused in this method).
        """
        if not self.features:
            self.features = HardSequenceTextDatasetFeaturesWithImage()


class HardSequenceTextDatasetFeaturesWithImageURLBuilder(BaseBuilder):
    """
    A builder class for creating a dataset with sequences of text, image URLs, and hard negative texts.
    It extends the `BaseBuilder` class and uses `HardSequenceTextDatasetFeaturesWithImageURL` to define the dataset's structure.
    If no features are provided, it automatically sets `HardSequenceTextDatasetFeaturesWithImageURL`.

    Attributes:
        features (`Optional[Features]`):
            The dataset's structure and data types, which default to `HardSequenceTextDatasetFeaturesWithImageURL`.
    """

    def model_post_init(self, __context: Any) -> None:
        if not self.features:
            self.features = Features(HardSequenceTextDatasetFeaturesWithImageURL())


class NegCLIPTextDatasetFeaturesWithImageURLBuilder(BaseBuilder):
    """
    A builder class for creating a dataset with image URLs, text data, and hard negatives.
    It extends the `BaseBuilder` class and uses `NegCLIPTextDatasetFeaturesWithImageURL` to define the dataset's structure.
    If no features are provided, it automatically sets `NegCLIPTextDatasetFeaturesWithImageURL`.

    Attributes:
        features (`Optional[Features]`):
            The dataset's structure and data types, which default to `NegCLIPTextDatasetFeaturesWithImageURL`.
    """

    def model_post_init(self, __context: Any) -> None:
        if not self.features:
            self.features = Features(NegCLIPTextDatasetFeaturesWithImageURL())


class NegCLIPSequenceTextDatasetFeaturesWithImageURLBuilder(BaseBuilder):
    """
    A builder class for creating a dataset with sequences of image URLs, text data, and hard negatives.
    It extends the `BaseBuilder` class and uses `NegCLIPSequenceTextDatasetFeaturesWithImageURL` to define the dataset's structure.
    If no features are provided, it automatically sets `NegCLIPSequenceTextDatasetFeaturesWithImageURL`.

    Attributes:
        features (`Optional[Features]`):
            The dataset's structure and data types, which default to `NegCLIPSequenceTextDatasetFeaturesWithImageURL`.
    """

    def model_post_init(self, __context: Any) -> None:
        if not self.features:
            self.features = Features(NegCLIPSequenceTextDatasetFeaturesWithImageURL())
