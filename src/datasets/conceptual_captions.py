from .base import BaseDatasetBuilder
from src.common import registry
from typing import Union
from src.utils import download_url, load_tsv
from datasets import Dataset
import os


@registry.register_builder('ConceptualCaptionsDatasetBuilder')
class ConceptualCaptionsDatasetBuilder(BaseDatasetBuilder):
    url: Union[str, os.PathLike]
    output_dir: Union[str, os.PathLike]
    file_name: Union[str, os.PathLike]

    def _download_data(self):
        download_url(url=self.url, root=self.output_dir, filename=self.file_name)

    def build(self):
        reader = load_tsv(filename=os.path.join(self.output_dir, self.file_name))
        print(reader[0])
        return Dataset.from_list([{'text': r[0], 'img_url': r[1]} for r in reader])
