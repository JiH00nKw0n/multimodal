from typing import Optional

from datasets import load_dataset, Dataset

from src.common import registry
from src.datasets.base import BaseBuilder


@registry.register_builder('WinogroundDatasetBuilder')
class WinogroundDatasetBuilder(BaseBuilder):
    """
    A builder class for creating the Winoground dataset.
    It extends `BaseBuilder` to provide the structure for loading the Winoground dataset.

    Attributes:
        split (Optional[str]): The dataset split to load (default: 'test').
        name (Optional[str]): The name of the dataset (default: 'winoground').
    """
    split: Optional[str] = 'test'
    name: Optional[str] = 'winoground'

    def build_dataset(self) -> Dataset:
        """
        Builds and returns the Winoground dataset.

        Returns:
            Dataset: The Winoground dataset for the specified split.
        """
        dataset = load_dataset(
            "facebook/winoground", trust_remote_code=True, split=self.split
        )

        return dataset
