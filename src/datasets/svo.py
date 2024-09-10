from typing import Optional

from datasets import load_dataset, Dataset

from src.common import registry
from src.datasets.base import BaseBuilder


@registry.register_builder('SVODatasetBuilder')
class SVODatasetBuilder(BaseBuilder):
    """
    A builder class for creating the SVO (Subject-Verb-Object) dataset.
    It extends `BaseBuilder` and loads the preformatted SVO dataset from the specified source.

    Attributes:
        split (Optional[str]): The dataset split to load (default: 'train').
        name (Optional[str]): The name of the dataset (default: 'winoground').
    """
    split: Optional[str] = 'train'
    name: Optional[str] = 'svo'

    def build_dataset(self) -> Dataset:
        """
        Builds and returns the SVO dataset.

        Returns:
            Dataset: The SVO dataset for the specified split.
        """
        dataset = load_dataset(
            "yjkimstats/SVO_fmt", trust_remote_code=True, split=self.split
        )

        return dataset