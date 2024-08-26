from datasets import IterableDataset, Dataset, interleave_datasets, concatenate_datasets
from transformers import TrainingArguments
import logging
from src.common import TrainConfig, registry, BaseCollator
from src.tasks.base import BaseTask
from typing import Optional, Dict, Union

__all__ = ["IterableDatasetTrainTask"]

logger = logging.getLogger(__name__)


class TrainTask(BaseTask):
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

        return model

    def build_trainer(self, trainer_config: Optional[Dict] = None):
        assert "runner" in self.config.run_config, "Trainer name must be provided."

        trainer_name = self.config.run_config.runner
        trainer = registry.get_trainer_class(trainer_name)
        assert trainer is not None, "Task {} not properly registered.".format(trainer_name)

        trainer_config = trainer_config if trainer_config is not None else self.config.trainer_config
        collator = BaseCollator(
            processor=self.build_processor(), max_length=self.config.run_config.max_seq_length
        )
        train_dataset = self.build_datasets()

        return trainer(
            model=self.build_model(),
            args=TrainingArguments(**trainer_config),
            train_dataset=train_dataset,
            data_collator=collator,
        )


@registry.register_task("DatasetTrainTask")
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


@registry.register_task("IterableDatasetTrainTask")
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
