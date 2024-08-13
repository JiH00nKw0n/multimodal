from typing import Union, List

from src.datasets.base import BaseDatasetBuilder
from src.common import registry
from datasets import load_dataset
from datasets import IterableDataset


@registry.register_builder('MSCOCOCaptionsDatasetBuilder')
class MSCOCOCaptionsDatasetBuilder(BaseDatasetBuilder):
    year: Union[str | List[str]] = '2014'
    split: str = 'train'

    def build_dataset(self) -> IterableDataset:
        dataset = load_dataset(
            "pixparse/cc3m-wds", trust_remote_code=True, streaming=True, split=self.split
        )
        dataset = dataset.rename_columns({"jpg": 'image', "txt": 'text'})
        dataset = dataset.select_columns(['image', 'text'])
        dataset = dataset.cast(self.features)

        return dataset

if __name__ == '__main__':
    dataset = load_dataset(
        "HuggingFaceM4/COCO",
        split='train',
        streaming=True, trust_remote_code=True
    )
    for example in dataset:
        print(example)
        break