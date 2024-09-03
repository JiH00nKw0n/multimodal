import json
import logging
from src.common import EvaluateConfig, registry, SequenceTextCollator
from src.utils import load_json
from src.tasks.base import BaseTask
from datasets import DatasetDict
from typing import Optional, Dict

__all__ = [
    "EvaluateTask",
    "CustomModelEvaluateTask"
]

logger = logging.getLogger(__name__)


class EvaluateTask(BaseTask):
    config: EvaluateConfig

    def build_datasets(self,
                       dataset_config: Optional[Dict] = None,
                       shuffle: Optional[bool] = False,
                       buffer_size: Optional[int] = 10000) -> DatasetDict:
        dataset_config = dataset_config if dataset_config is not None else self.config.dataset_config

        dataset_dict = DatasetDict()

        assert len(dataset_config) > 0, "At least one dataset has to be specified."

        for builder_cls_name, config in dataset_config.items():
            builder = registry.get_builder_class(builder_cls_name)(**config)
            dataset = builder.build_dataset()
            if shuffle:
                dataset = dataset.shuffle(seed=self.config.run_config.seed, buffer_size=buffer_size)

            dataset_dict[builder.name] = dataset

        return dataset_dict

    def build_evaluator(self, evaluator_config: Optional[Dict] = None):
        assert "runner" in self.config.run_config, "Evaluator name must be provided."

        evaluator_name = self.config.run_config.runner
        evaluator = registry.get_evaluator_class(evaluator_name)
        assert evaluator is not None, "Task {} not properly registered.".format(evaluator_name)

        evaluator_config = evaluator_config if evaluator_config is not None else self.config.evaluator_config
        collator = SequenceTextCollator(
            processor=self.build_processor(), max_length=self.config.run_config.max_seq_length
        )
        evaluate_dataset = self.build_datasets()

        return evaluator(
            model=self.build_model(),
            evaluate_dataset=evaluate_dataset,
            data_collator=collator,
            **evaluator_config
        )


@registry.register_task("CustomModelEvaluateTask")
class CustomModelEvaluateTask(BaseTask):
    config: EvaluateConfig

    def build_model(self, model_config: Optional[Dict] = None):
        model_config = model_config \
            if model_config is not None else self.config.model_config.copy()

        model_cls = registry.get_model_class(model_config.model_cls)
        model_cfg_cls = registry.get_model_config_class(model_config.config_cls)

        assert model_cls is not None, "Model {} not properly registered.".format(model_cls)
        assert model_cfg_cls is not None, "Model config {} not properly registered.".format(model_cfg_cls)

        config_dict = load_json(model_config.config_path)
        model_cfg = model_cfg_cls(**config_dict)

        model = model_cls.from_pretrained(**dict(model_config.config, **{"config": model_cfg}))

        return model.eval()

