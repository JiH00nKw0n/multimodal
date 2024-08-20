import logging
from src.common import EvaluateConfig, registry, SequenceTextCollator
from src.tasks.base import BaseTask
from typing import Optional, Dict

__all__ = ["EvaluateTask"]

logger = logging.getLogger(__name__)


@registry.register_task("EvalFlickrTask")
class EvaluateTask(BaseTask):
    config: EvaluateConfig

    def build_model(self, model_config: Optional[Dict] = None):
        model_config = model_config \
            if model_config is not None else self.config.model_config.copy()

        model_cls = registry.get_model_class(model_config.model_cls)
        assert model_cls is not None, "Model {} not properly registered.".format(model_cls)

        model = model_cls.from_pretrained(**model_config.config)

        return model

    def build_evaluator(self, evaluator_config: Optional[Dict] = None):
        assert "runner" in self.config.run_config, "Trainer name must be provided."

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
