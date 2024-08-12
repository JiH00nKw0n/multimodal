from pydantic import BaseModel, Field, ConfigDict
from datasets import Sequence, Value, Features, Image
from typing import Optional, Dict, Any
import torch.distributed as dist
import logging

from src.utils import is_main_process, is_dist_avail_and_initialized

logger = logging.getLogger(__name__)


class BaseDatasetFeatures(BaseModel):
    img: Sequence = Sequence(Image)
    text: Sequence = Sequence(Value(dtype='string', id=None))
    img_url: Sequence = Sequence(Value(dtype='string', id=None))

    model_config = ConfigDict(frozen=True, strict=True, validate_assignment=True)


class BaseDatasetBuilder(BaseModel):
    features: Optional[Features] = None
    dataset: Dict = Field(default_factory=dict, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        if self.features is None:
            self.features = Features(BaseDatasetFeatures())

    def build_dataset(self):
        if is_main_process():
            self._download_data()

        if is_dist_avail_and_initialized():
            dist.barrier()

            # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logger.info("Building datasets...")
        self.dataset = self.build()  # dataset['train'/'val'/'test']

        return self.dataset

    def _download_data(self):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError


if __name__ == '__main__':
    from datasets import Dataset, Features, Value

    features = Features({
        'img': Value(dtype='string'),
        'text': Value(dtype='string'),
        'img_url': Value(dtype='string')
    })

    dataset = Dataset.from_dict({
        'img': ['image1.png', 'image2.png'],
        'text': ['text1', 'text2'],
        'img_url': ['http://example.com/img1', 'http://example.com/img2']
    }, features=features)

    new_data = {
        'text': 'text3',
        'img': 'image3.png',
    }

    # add the new data to the dataset
    dataset = dataset.add_item(new_data)

    print(type(dataset[2]['img_url']))
