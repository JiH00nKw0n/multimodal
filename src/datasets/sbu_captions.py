from typing import Optional

from datasets import load_dataset, IterableDataset

from src.common import registry
from src.datasets.builder import TextDatasetFeaturesWithImageURLBuilder


@registry.register_builder('SBUCaptionsDatasetBuilder')
class SBUCaptionsDatasetBuilder(TextDatasetFeaturesWithImageURLBuilder):
    """
    A builder class for creating an iterable dataset for SBU Captions.
    It extends `TextDatasetFeaturesWithImageURLBuilder` and uses the streaming mode to handle large datasets.

    Attributes:
        split (Optional[str]): The dataset split to load (default: 'train').
        name (Optional[str]): The name of the dataset (default: 'sbu_captions').
    """
    split: Optional[str] = 'train'
    name: Optional[str] = 'sbu_captions'

    def build_dataset(self) -> IterableDataset:
        """
        Builds and returns an iterable SBU Captions dataset with image URLs and text captions.

        Returns:
            IterableDataset: The streaming iterable dataset with 'images' and 'text' columns.
        """
        dataset = load_dataset(
            "sbu_captions", trust_remote_code=True, split=self.split
        )
        dataset = dataset.rename_columns({"caption": 'text'})
        dataset = dataset.select_columns(['images', 'text'])
        dataset = dataset.cast(self.features)

        return dataset
