from pydantic import BaseModel, Field, ConfigDict
from datasets import Sequence, Value, Features, Image, IterableDataset, Dataset
from typing import Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


class BaseDatasetFeatures(BaseModel):
    images: Image = Image()
    text: Value = Value(dtype='string', id=None)

    model_config = ConfigDict(frozen=True, strict=True, validate_assignment=True)


class SequenceTextDatasetFeatures(BaseModel):
    images: Image = Image()
    text: Sequence = Sequence(Value(dtype='string', id=None))

    model_config = ConfigDict(frozen=True, strict=True, validate_assignment=True)


class BaseDatasetFeaturesWithHN(BaseDatasetFeatures):
    model_config = ConfigDict(frozen=False, strict=False, validate_assignment=False, arbitrary_types_allowed=True)
    hard_images: Optional[Any] = None
    hard_texts: Optional[Any] = None
    neg_texts: Optional[Any] = None
    hard_neg_texts: Optional[Any] = None

    def model_post_init(self, __context: Any) -> None:
        self.hard_images = Sequence(Image()) \
            if self.hard_images is None else self.hard_images
        self.hard_texts = Sequence(Value(dtype='string', id=None)) \
            if self.hard_texts is None else self.hard_texts
        self.neg_texts = Sequence(Value(dtype='string', id=None)) \
            if self.neg_texts is None else self.neg_texts
        self.hard_neg_texts = Sequence(Value(dtype='string', id=None)) \
            if self.hard_neg_texts is None else self.hard_neg_texts


class SequenceTextDatasetFeaturesWithHN(SequenceTextDatasetFeatures):
    model_config = ConfigDict(frozen=False, strict=False, validate_assignment=False, arbitrary_types_allowed=True)
    hard_images: Optional[Any] = None
    hard_texts: Optional[Any] = None
    neg_texts: Optional[Any] = None
    hard_neg_texts: Optional[Any] = None

    def model_post_init(self, __context: Any) -> None:
        self.hard_images = Sequence(Image()) \
            if self.hard_images is None else self.hard_images
        self.hard_texts = Sequence(Sequence(Value(dtype='string', id=None))) \
            if self.hard_texts is None else self.hard_texts
        self.neg_texts = Sequence(Sequence(Value(dtype='string', id=None))) \
            if self.neg_texts is None else self.neg_texts
        self.hard_neg_texts = Sequence(Sequence(Value(dtype='string', id=None))) \
            if self.hard_neg_texts is None else self.hard_neg_texts


class BaseBuilder(BaseModel):
    features: Optional[Features] = None
    dataset: Dict = Field(default_factory=dict, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def build_dataset(self) -> Union[Dataset, IterableDataset]:
        raise NotImplementedError


class BaseDatasetBuilder(BaseBuilder):
    features: Optional[Features] = None
    dataset: Dict = Field(default_factory=dict, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        if self.features is None:
            self.features = Features(BaseDatasetFeatures())


class SequenceTextDatasetBuilder(BaseBuilder):
    features: Optional[Features] = None
    dataset: Dict = Field(default_factory=dict, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        if self.features is None:
            self.features = Features(SequenceTextDatasetFeatures())


class SequenceTextDatasetWithHNBuilder(BaseBuilder):
    base_features: Optional[Features] = None
    features: Optional[Features] = None
    dataset: Dict = Field(default_factory=dict, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        if self.features is None:
            self.features = Features(SequenceTextDatasetFeaturesWithHN())


if __name__ == '__main__':
    print(SequenceTextDatasetWithHNBuilder().features)
