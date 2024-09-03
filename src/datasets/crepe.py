import os
from typing import Optional, Union

from src.datasets.base import BaseDatasetBuilder
from src.common import registry
from datasets import load_dataset, Dataset


@registry.register_builder('CREPEDatasetBuilder')
class CREPEDatasetBuilder(BaseDatasetBuilder):
    split: Optional[str] = 'test'
    name: Optional[str] = 'crepe'
    download_dir: Optional[Union[str, os.PathLike]] = None

    def build_dataset(self) -> Dataset:
        raise NotImplementedError
