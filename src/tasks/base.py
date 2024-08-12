from pydantic import BaseModel, Extra
from typing import Any, Optional, Dict
from src.common import registry

__all__ = ["BaseTask"]


class BaseTask(BaseModel, extra=Extra.allow):
    config: Any

    @classmethod
    def setup_task(cls, config):
        return cls(config=config)

    def build_model(self, model_config: Optional[Dict] = None):
        model_config = model_config \
            if model_config is not None else self.config.model_config.copy()

        model_cfg_cls = registry.get_model_config_class()
        model_cls = registry.get_model_class(model_config.architectures[0])

        model_cfg = model_cfg_cls(**model_config)
        model = model_cls(model_cfg)

        return model

    def build_processor(self, processor_config: Optional[Dict] = None):
        processor_config = processor_config \
            if processor_config is not None else self.config.processor_config.copy()
        processor_cls = registry.get_processor_class(processor_config.cls)
        del processor_config.architecture
        return processor_cls.from_config(processor_config)

    def build_dataset(self, dataset_config: Optional[Dict] = None):
        dataset_config = dataset_config if dataset_config is not None else self.config.dataset_config

        raise NotImplementedError

        dataset = None
        return dataset

