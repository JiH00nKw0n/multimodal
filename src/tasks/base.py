from pydantic import BaseModel, Extra
from typing import Any, Optional, Dict

__all__ = ["BaseTask"]


class BaseTask(BaseModel, extra=Extra.allow):
    config: Any

    @classmethod
    def setup_task(cls, config):
        return cls(config=config)

    def build_model(self, model_config: Optional[Dict] = None):
        model_config = model_config if model_config is not None else self.config.model_cfg

        raise NotImplementedError

        model = None
        return model

    def build_processor(self, processor_config: Optional[Dict] = None):
        processor_config = processor_config if processor_config is not None else self.config.processor_config

        raise NotImplementedError

        processor = None
        return processor

    def build_dataset(self, dataset_config: Optional[Dict] = None):
        dataset_config = dataset_config if dataset_config is not None else self.config.dataset_config

        raise NotImplementedError

        dataset = None
        return dataset

