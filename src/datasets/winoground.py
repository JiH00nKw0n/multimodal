from typing import Optional

from src.datasets.base import BaseDatasetBuilder
from src.common import registry
from datasets import load_dataset, Dataset


@registry.register_builder('WinogroundDatasetBuilder')
class WinogroundDatasetBuilder(BaseDatasetBuilder):
    split: Optional[str] = 'test'
    name: Optional[str] = 'winoground'

    def build_dataset(self) -> Dataset:
        dataset = load_dataset(
            "facebook/winoground", trust_remote_code=True, streaming=True, split=self.split
        )

        return dataset
