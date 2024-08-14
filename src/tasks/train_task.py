from transformers import TrainingArguments
import logging
from src.common import TrainConfig, registry, BaseCollator
from src.tasks.base import BaseTask
from typing import Optional, Dict

__all__ = ["TrainTask"]

logger = logging.getLogger(__name__)


@registry.register_task("TrainTask")
class TrainTask(BaseTask):
    config: TrainConfig

    def build_trainer(self, trainer_config: Optional[Dict] = None):
        assert "trainer" in self.config.run_config, "Trainer name must be provided."

        trainer_name = self.config.run_config.trainer
        trainer = registry.get_trainer_class(trainer_name)
        assert trainer is not None, "Task {} not properly registered.".format(trainer_name)

        trainer_config = trainer_config if trainer_config is not None else self.config.trainer_config
        collator = BaseCollator(processor=self.build_processor())
        train_dataset = self.build_datasets()

        return trainer(
            model=self.build_model(),
            args=TrainingArguments(**trainer_config),
            train_dataset=train_dataset,
            data_collator=collator,
        )
