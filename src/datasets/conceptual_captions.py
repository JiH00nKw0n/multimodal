from typing import Optional

from datasets import Dataset, IterableDataset, load_dataset
from src.common import registry
from src.datasets.builder import TextDatasetFeaturesWithImageBuilder

__all__ = [
    "ConceptualCaptionsIterableDatasetBuilder",
    "ConceptualCaptionsDatasetBuilder"
]


@registry.register_builder('ConceptualCaptionsIterableDatasetBuilder')
class ConceptualCaptionsIterableDatasetBuilder(TextDatasetFeaturesWithImageBuilder):
    """
    A builder class for creating an iterable dataset for Conceptual Captions (CC3M).
    It extends `TextDatasetFeaturesWithImage` and uses the streaming mode to handle large datasets.

    Attributes:
        split (Optional[str]): The dataset split to load (default: 'train').
        name (Optional[str]): The name of the dataset (default: 'cc3m').
    """
    split: Optional[str] = 'train'
    name: Optional[str] = 'cc3m'

    def build_dataset(self) -> IterableDataset:
        """
        Builds and returns an iterable Conceptual Captions dataset.

        Returns:
            IterableDataset: The streaming iterable dataset with 'images' and 'text' columns.
        """
        dataset = load_dataset(
            "pixparse/cc3m-wds", trust_remote_code=True, streaming=True, split=self.split
        )
        dataset = dataset.rename_columns({"jpg": 'images', "txt": 'text'})
        dataset = dataset.select_columns(['images', 'text'])
        dataset = dataset.cast(self.features)

        return dataset


@registry.register_builder('ConceptualCaptionsDatasetBuilder')
class ConceptualCaptionsDatasetBuilder(TextDatasetFeaturesWithImageBuilder):
    """
    A builder class for creating a non-iterable dataset for Conceptual Captions (CC3M).
    It extends `TextDatasetFeaturesWithImage`.

    Attributes:
        split (Optional[str]): The dataset split to load (default: 'train').
        name (Optional[str]): The name of the dataset (default: 'cc3m').
    """
    split: Optional[str] = 'train'
    name: Optional[str] = 'cc3m'

    def build_dataset(self) -> Dataset:
        """
        Builds and returns a Conceptual Captions dataset.

        Returns:
            Dataset: The dataset with 'images' and 'text' columns.
        """
        dataset = load_dataset(
            "pixparse/cc3m-wds", trust_remote_code=True, split=self.split
        )
        dataset = dataset.rename_columns({"jpg": 'images', "txt": 'text'})
        dataset = dataset.select_columns(['images', 'text'])
        dataset = dataset.cast(self.features)

        return dataset
