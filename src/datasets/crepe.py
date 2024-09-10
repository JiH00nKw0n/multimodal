from typing import Optional

from datasets import load_dataset, Dataset

from src.common import registry
from src.datasets.builder import HardSequenceTextDatasetWithImageBuilder


@registry.register_builder('CREPEDatasetBuilder')
class CREPEDatasetBuilder(HardSequenceTextDatasetWithImageBuilder):
    split: Optional[str] = 'train'
    name: Optional[str] = 'crepe'

    def build_dataset(self) -> Dataset:
        dataset = load_dataset(
            "yjkimstats/CREPE_productivity_fmt", trust_remote_code=True, split=self.split
        )
        dataset = dataset.map(
            lambda x: {'text': [x['text']]}
        )
        # dataset.cast(self.features)

        return dataset
