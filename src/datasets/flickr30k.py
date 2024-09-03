from typing import Optional

from src.datasets.base import SequenceTextDatasetBuilder
from src.common import registry
from datasets import load_dataset
from datasets import Dataset


@registry.register_builder('Flickr30kDatasetBuilder')
class Flickr30kDatasetBuilder(SequenceTextDatasetBuilder):
    split: Optional[str] = 'test'
    name: Optional[str] = 'flickr30k'

    def build_dataset(self) -> Dataset:
        dataset = load_dataset(
            "nlphuji/flickr30k", trust_remote_code=True, split=self.split,
            # streaming = True
        )
        split_dataset = dataset.filter(lambda sample: sample['split'] == 'test')
        split_dataset = split_dataset.rename_columns({"image": 'images', "caption": 'text'})
        split_dataset = split_dataset.select_columns(['images', 'text'])
        split_dataset = split_dataset.cast(self.features)

        return split_dataset
