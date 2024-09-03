import os

from datasets import IterableDataset, Dataset, interleave_datasets, concatenate_datasets
from transformers import TrainingArguments, AutoModel, AutoProcessor
import logging
from src.common import TrainConfig, registry, BaseCollator
import yaml
from src.tasks.base import BaseTask
from typing import Optional, Dict, Union
from peft import (
    get_peft_model,
    LoraConfig,
)

__all__ = [
    "TrainTask",
    "PretrainedModelTrainTask",
    "CustomModelTrainTask",
    "DatasetTrainTask",
    "IterableDatasetTrainTask",
    "DatasetPretrainedModelTrainTask"
]

logger = logging.getLogger(__name__)


class TrainTask(BaseTask):
    config: TrainConfig

    def build_trainer(self, trainer_config: Optional[Dict] = None):
        assert "runner" in self.config.run_config, "Trainer name must be provided."

        trainer_name = self.config.run_config.runner
        trainer_cls = registry.get_trainer_class(trainer_name)
        assert trainer_cls is not None, "Trainer {} not properly registered.".format(trainer_name)

        trainer_config = trainer_config if trainer_config is not None else self.config.trainer_config

        collator_name = self.config.run_config.collator
        collator_cls = registry.get_collator_class(collator_name)

        assert collator_cls is not None, "Collator {} not properly registered.".format(collator_name)

        collator = collator_cls(
            processor=self.build_processor(), max_length=self.config.run_config.max_seq_length
        )

        train_dataset = self.build_datasets()

        return trainer_cls(
            model=self.build_model(),
            args=TrainingArguments(**trainer_config),
            train_dataset=train_dataset,
            data_collator=collator,
        )


class PretrainedModelTrainTask(TrainTask):
    config: TrainConfig

    def build_model(self, model_config: Optional[Dict] = None):
        model_config = model_config \
            if model_config is not None else self.config.model_config.copy()

        model = AutoModel.from_pretrained(**model_config.config)
        if model_config.lora is not None:
            if isinstance(model_config.lora, str):
                lora_config = yaml.safe_load(model_config.lora)
            elif isinstance(model_config.lora, os.PathLike):
                lora_config = yaml.safe_load(os.fspath(model_config.lora))
            else:
                raise TypeError
            peft_config = LoraConfig(**lora_config)
            model = get_peft_model(model, peft_config)

        return model

    def build_processor(self, processor_config: Optional[Dict] = None):
        processor_config = processor_config \
            if processor_config is not None else self.config.processor_config.copy()

        return AutoProcessor.from_pretrained(**processor_config.config)


class CustomModelTrainTask(TrainTask):
    config: TrainConfig

    def build_model(self, model_config: Optional[Dict] = None):
        model_config = model_config \
            if model_config is not None else self.config.model_config.copy()

        model_cfg_cls = registry.get_model_config_class(model_config.config_cls)
        model_cls = registry.get_model_class(model_config.model_cls)

        assert model_cls is not None, "Model {} not properly registered.".format(model_cls)
        assert model_cfg_cls is not None, "Model config {} not properly registered.".format(model_cfg_cls)

        model_cfg = model_cfg_cls(**model_config.config)
        model = model_cls(model_cfg)

        if model_config.lora is not None:
            if isinstance(model_config.lora, Dict):
                text_model_config_path = model_config.lora.pop('text_model', None)
                image_model_config_path = model_config.lora.pop('text_model', None)
            else:
                raise TypeError
            if text_model_config_path is not None:
                text_model_lora_config = yaml.safe_load(text_model_config_path)
                text_model_peft_config = LoraConfig(**text_model_lora_config)
                model.text_model = get_peft_model(model.text_model, text_model_peft_config)
            if image_model_config_path is not None:
                image_model_lora_config = yaml.safe_load(image_model_config_path)
                image_model_peft_config = LoraConfig(**image_model_lora_config)
                model.image_model = get_peft_model(model.image_model, image_model_peft_config)

        return model


class DatasetTrainTask(TrainTask):
    config: TrainConfig

    def build_datasets(self,
                       dataset_config: Optional[Dict] = None,
                       shuffle: Optional[bool] = False,
                       buffer_size: Optional[int] = 10000) -> Dataset:
        dataset_config = dataset_config if dataset_config is not None else self.config.dataset_config

        datasets = list()

        assert len(dataset_config) > 0, "At least one dataset has to be specified."

        for builder_cls_name, config in dataset_config.items():
            builder = registry.get_builder_class(builder_cls_name)(**config)
            dataset = builder.build_dataset()
            if not isinstance(dataset, Dataset):
                raise TypeError("DatasetTrainTask must build dataset with `Dataset` type.")
            if shuffle:
                dataset = dataset.shuffle(seed=self.config.run_config.seed, buffer_size=buffer_size)

            datasets.append(dataset)

        return concatenate_datasets(datasets)


class IterableDatasetTrainTask(TrainTask):
    config: TrainConfig

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
            if not isinstance(dataset, IterableDataset):
                raise TypeError("DatasetTrainTask must build dataset with `IterableDataset` type.")
            if shuffle:
                dataset = dataset.shuffle(seed=self.config.run_config.seed, buffer_size=buffer_size)

            datasets.append(dataset)

        return interleave_datasets(datasets).with_format("torch")


@registry.register_task("DatasetPretrainedModelTrainTask")
class DatasetPretrainedModelTrainTask(PretrainedModelTrainTask, DatasetTrainTask):
    config: TrainConfig
