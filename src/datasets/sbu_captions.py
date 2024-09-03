import io
import urllib
from typing import Optional, Dict

import PIL.Image
from PIL.Image import Image
from datasets import load_dataset, IterableDataset
from datasets.utils.file_utils import get_datasets_user_agent

from src.datasets.base import BaseDatasetBuilder
from src.common import registry

USER_AGENT = get_datasets_user_agent()


def fetch_single_image(example: Dict, timeout: Optional[float] = None) -> Dict:
    try:
        request = urllib.request.Request(
            example['image_url'],
            data=None,
            headers={"user-agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=timeout) as req:
            image = PIL.Image.open(io.BytesIO(req.read()))
        example['images'] = image
    except Exception:
        example['images'] = None  # 이미지 로드 실패 시 image에 None을 할당

    return example


@registry.register_builder('SBUCaptionsDatasetBuilder')
class SBUCaptionsDatasetBuilder(BaseDatasetBuilder):
    split: Optional[str] = 'train'
    name: Optional[str] = 'sbu_captions'

    def build_dataset(self) -> IterableDataset:
        dataset = load_dataset(
            "sbu_captions", trust_remote_code=True, streaming=True, split=self.split
        )
        dataset = dataset.map(fetch_single_image)
        dataset = dataset.filter(lambda x: x['image'] is not None)  # image가 None인 데이터 필터링
        dataset = dataset.rename_columns({"caption": 'text'})
        dataset = dataset.select_columns(['images', 'text'])
        dataset = dataset.cast(self.features)

        return dataset
