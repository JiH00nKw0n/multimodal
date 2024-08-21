import logging
from pydantic import BaseModel, Extra
from typing import Any, Optional, Dict
from src.common import registry
from transformers import AutoConfig
from datasets import IterableDataset, interleave_datasets

logger = logging.getLogger(__name__)
__all__ = ["BaseTask"]


class BaseTask(BaseModel, extra=Extra.forbid):
    config: Any

    @classmethod
    def setup_task(cls, config):
        return cls(config=config)

    def build_model(self, model_config: Optional[Dict] = None):
        model_config = model_config \
            if model_config is not None else self.config.model_config.copy()

        model_cfg_cls = registry.get_model_config_class(model_config.config_cls)
        model_cls = registry.get_model_class(model_config.model_cls)

        assert model_cls is not None, "Model {} not properly registered.".format(model_cls)
        assert model_cfg_cls is not None, "Model config {} not properly registered.".format(model_cfg_cls)

        model_cfg = model_cfg_cls(**model_config.config)
        model = model_cls(model_cfg)

        return model

    def build_processor(self, processor_config: Optional[Dict] = None):
        processor_config = processor_config \
            if processor_config is not None else self.config.processor_config.copy()
        processor_cls = registry.get_processor_class(processor_config.processor_cls)

        assert processor_cls is not None, "Processor {} not properly registered.".format(processor_cls)

        return processor_cls.from_text_vision_pretrained(**processor_config.config)

    def build_datasets(self,
                       dataset_config: Optional[Dict] = None,
                       shuffle: Optional[bool] = False,
                       buffer_size: Optional[int] = 10000) -> IterableDataset:
        dataset_config = dataset_config if dataset_config is not None else self.config.dataset_config

        datasets = list()

        assert len(dataset_config) > 0, "At least one dataset has to be specified."

        for builder_cls_name, config in dataset_config.items():
            builder = registry.get_builder_class(builder_cls_name)(**config)
            dataset = builder.build_dataset()
            if shuffle:
                dataset = dataset.shuffle(seed=self.config.run_config.seed, buffer_size=buffer_size)

            datasets.append(dataset)

        return interleave_datasets(datasets).with_format("torch")
