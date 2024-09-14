from typing import List, Optional, Union

from datasets import Dataset, IterableDataset, load_dataset

from src.common import registry
from src.datasets.builder import SequenceTextDatasetFeaturesWithImageURLBuilder

__all__ = [
    "Laion400mIterableDatasetBuilder",
    "Laion400mDatasetBuilder",
    "LaionCOCOIterableDatasetBuilder"
]


@registry.register_builder('Laion400mIterableDatasetBuilder')
class Laion400mIterableDatasetBuilder(SequenceTextDatasetFeaturesWithImageURLBuilder):
    """
    A builder class for creating an iterable dataset for the Laion400m dataset.
    It extends `SequenceTextDatasetFeaturesWithImageURLBuilder`.

    Attributes:
        split (Union[str, List[str]]): The dataset split to load (default: 'train').
        name (Optional[str]): The name of the dataset (default: 'coco').
    """
    split: Union[str, List[str]] = 'train'
    name: Optional[str] = 'laion'

    def build_dataset(self) -> IterableDataset:
        """
        Builds and returns the Laion400m iterable dataset.

        Returns:
            IterableDataset: The Laion400m dataset as an iterable.
        """
        dataset = load_dataset(
            "laion/laion400m", trust_remote_code=True, split=self.split, streaming=True
        )
        dataset = dataset.rename_columns({"caption": 'text', "url": 'images'})
        dataset = dataset.select_columns(['images', 'text'])

        return dataset


@registry.register_builder('Laion400mDatasetBuilder')
class Laion400mDatasetBuilder(SequenceTextDatasetFeaturesWithImageURLBuilder):
    """
    A builder class for creating a non-iterable dataset for the Laion400m dataset.
    It extends `SequenceTextDatasetFeaturesWithImageURLBuilder`.

    Attributes:
        split (Union[str, List[str]]): The dataset split(s) to load (default: 'train').
        name (Optional[str]): The name of the dataset (default: 'coco').
    """
    split: Union[str, List[str]] = 'train'
    name: Optional[str] = 'laion'

    def build_dataset(self) -> Dataset:
        dataset = load_dataset(
            "laion/laion400m", trust_remote_code=True, split=self.split
        )
        dataset = dataset.rename_columns({"caption": 'text', "url": 'images'})
        dataset = dataset.select_columns(['images', 'text'])
        return dataset


@registry.register_builder('LaionCOCOIterableDatasetBuilder')
class LaionCOCOIterableDatasetBuilder(SequenceTextDatasetFeaturesWithImageURLBuilder):
    """
    A builder class for creating a non-iterable dataset for the Laion400m dataset.
    It extends `SequenceTextDatasetFeaturesWithImageURLBuilder`.

    Attributes:
        split (Union[str, List[str]]): The dataset split(s) to load (default: 'train').
        name (Optional[str]): The name of the dataset (default: 'coco').
    """
    split: Union[str, List[str]] = 'train'
    name: Optional[str] = 'laion'

    def build_dataset(self) -> Dataset:
        dataset = load_dataset(
            "kakaobrain/coyo-700m", trust_remote_code=True, split=self.split,
            streaming=True, token='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'
        )
        dataset = dataset.rename_columns({"url": 'images'})
        dataset = dataset.select_columns(['images', 'text'])
        return dataset
