import logging
from pydantic import BaseModel, Extra
from typing import Any, Optional, Dict, TypeVar, Union
from src.common import registry
from transformers import PreTrainedModel
from datasets import IterableDataset, Dataset

ModelType = TypeVar("ModelType", bound=PreTrainedModel)

logger = logging.getLogger(__name__)
__all__ = ["BaseTask"]


class BaseTask(BaseModel, extra=Extra.forbid):
    config: Any

    @classmethod
    def setup_task(cls, config):
        return cls(config=config)

    def build_model(self, model_config: Optional[Dict] = None) -> ModelType:
        raise NotImplementedError

    def build_processor(self, processor_config: Optional[Dict] = None):
        processor_config = processor_config \
            if processor_config is not None else self.config.processor_config.copy()
        processor_cls = registry.get_processor_class(processor_config.processor_cls)

        assert processor_cls is not None, "Processor {} not properly registered.".format(processor_cls)

        return processor_cls.from_text_vision_pretrained(**processor_config.config)

    def build_datasets(self,
                       dataset_config: Optional[Dict] = None,
                       shuffle: Optional[bool] = False,
                       buffer_size: Optional[int] = 10000) -> Union[IterableDataset | Dataset]:
        raise NotImplementedError
