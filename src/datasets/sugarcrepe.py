from typing import Optional

from datasets import load_dataset, Dataset

from src.common import registry
from src.datasets.base import BaseBuilder


@registry.register_builder('SUGARCREPEDatasetBuilder')
class SUGARCREPEDatasetBuilder(BaseBuilder):
    """
    A builder class for creating the SUGARCREPE dataset.
    It extends `HardSequenceTextDatasetWithImageBuilder` to provide a structure for loading the SUGARCREPE dataset.

    Attributes:
        split (Optional[str]): The dataset split to load (default: 'train').
        name (Optional[str]): The name of the dataset (default: 'sugarcrepe').
    """
    split: Optional[str] = 'train'
    name: Optional[str] = 'sugarcrepe'

    def build_dataset(self) -> Dataset:
        """
        Builds and returns the SUGARCREPE dataset.

        Returns:
            Dataset: The SUGARCREPE dataset with sequences of text and images.
        """
        dataset = load_dataset(
            "yjkimstats/SUGARCREPE_fmt", trust_remote_code=True, split=self.split
        )
        dataset = dataset.map(
            lambda x: {'text': [x['text']]}
        )

        return dataset