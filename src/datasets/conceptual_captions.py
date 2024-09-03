from typing import Optional

from src.datasets.base import BaseDatasetBuilder
from src.common import registry
from datasets import load_dataset, Dataset
from datasets import IterableDataset


@registry.register_builder('ConceptualCaptionsIterableDatasetBuilder')
class ConceptualCaptionsIterableDatasetBuilder(BaseDatasetBuilder):
    split: Optional[str] = 'train'
    name: Optional[str] = 'cc3m'

    def build_dataset(self) -> IterableDataset:
        dataset = load_dataset(
            "pixparse/cc3m-wds", trust_remote_code=True, streaming=True, split=self.split
        )
        dataset = dataset.rename_columns({"jpg": 'images', "txt": 'text'})
        dataset = dataset.select_columns(['images', 'text'])
        dataset = dataset.cast(self.features)

        return dataset


@registry.register_builder('ConceptualCaptionsDatasetBuilder')
class ConceptualCaptionsDatasetBuilder(BaseDatasetBuilder):
    split: Optional[str] = 'train'
    name: Optional[str] = 'cc3m'

    def build_dataset(self) -> Dataset:
        dataset = load_dataset(
            "pixparse/cc3m-wds", trust_remote_code=True, split=self.split
        )
        dataset = dataset.rename_columns({"jpg": 'images', "txt": 'text'})
        dataset = dataset.select_columns(['images', 'text'])
        dataset = dataset.cast(self.features)

        return dataset
