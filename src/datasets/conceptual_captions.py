import logging

from typing import Optional

from datasets import Dataset, load_dataset

from src.common import registry
from src.datasets.base import BaseBuilder

logger = logging.getLogger(__name__)

__all__ = [
    "ConceptualCaptionsDatasetBuilder"
]


@registry.register_builder('ConceptualCaptionsDatasetBuilder')
class ConceptualCaptionsDatasetBuilder(BaseBuilder):
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
        dataset = load_dataset("pixparse/cc3m-wds", trust_remote_code=True, split=self.split)

        dataset = dataset.rename_columns({"jpg": 'images', "txt": 'text'})
        dataset = dataset.select_columns(['images', 'text'])
       
        return dataset

