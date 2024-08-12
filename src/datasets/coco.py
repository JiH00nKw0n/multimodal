from .base import BaseDatasetBuilder
from src.common import registry
from typing import List, Optional


@registry.register_builder('COCODataset')
class COCODatasetBuilder(BaseDatasetBuilder):
    split: Optional[List] = ['train2014']

