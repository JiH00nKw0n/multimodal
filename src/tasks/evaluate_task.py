import logging
from typing import Optional, Dict, Type, List

import omegaconf
from pydantic import BaseModel, ConfigDict, Field
from transformers import add_end_docstrings

from src.common import EvaluateConfig, registry, experimental
from src.datasets import BUILDER_EVALUATOR_MAPPING, BaseBuilder
from src.evaluators import BaseEvaluator, EVALUATOR_COLLATOR_MAPPING
from src.tasks.base import BaseTask, TaskWithPretrainedModel, TaskWithCustomModel

__all__ = [
    "EvaluateTask",
    "EvaluateTaskWithPretrainedModel",
    "EvaluateTaskWithCustomModel"
]

EvaluatorType = Type[BaseEvaluator]
BuilderType = Type[BaseBuilder]

logger = logging.getLogger(__name__)

# 공통 docstring 선언
TASK_DOCSTRING = """
    A task class for evaluating models using predefined or custom evaluators and datasets. 
    The `EvaluateTask` class provides methods for building datasets and evaluators, and for adding 
    evaluators to the container. 

    Methods:
        build_datasets(dataset_config, shuffle, buffer_size):
            Builds and returns datasets using the provided dataset configuration.

        build_evaluator(evaluator_config):
            Builds and returns evaluators using the provided evaluator configuration.
"""


@experimental
class EvaluatorContainer(BaseModel):
    """
    A container for holding multiple evaluators. This class allows for adding evaluators
    and evaluating them in a batch.

    Attributes:
        container (Optional[List[EvaluatorType]]): A list of evaluators to be added to the container.

    Methods:
        evaluate(batch_size):
            Runs the evaluation for all the evaluators in the container.

        add(evaluator):
            Adds a new evaluator to the container.
    """

    container: Optional[List[EvaluatorType]] = Field(default_factory=list)

    model_config = ConfigDict(frozen=False, strict=False, validate_assignment=False)

    def evaluate(self, batch_size: int = 128):
        """
        Runs the evaluation process for each evaluator in the container.

        Args:
            batch_size (int, optional): The batch size for evaluation. Defaults to 128.
        """
        for evaluator in self.container:
            if issubclass(evaluator, BaseEvaluator):
                evaluator.evaluate(batch_size=batch_size)

    def add(self, evaluator: EvaluatorType):
        """
        Adds an evaluator to the container.

        Args:
            evaluator (EvaluatorType): The evaluator to add to the container.
        """
        self.container.append(evaluator)


@add_end_docstrings(TASK_DOCSTRING)
class EvaluateTask(BaseTask):
    """
    A base class for tasks that evaluate models using evaluators and datasets.
    It provides methods for building datasets and evaluators.

    Attributes:
        config (EvaluateConfig): The configuration for the evaluation task.
    """

    config: EvaluateConfig

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ) -> Dict[str, BuilderType]:
        """
        Builds the datasets based on the dataset configuration provided.

        Args:
            dataset_config (Optional[Dict]): The dataset configuration. Defaults to `self.config.dataset_config`.
            shuffle (Optional[bool]): Whether to shuffle the dataset. Defaults to `False`.
            buffer_size (Optional[int]): The buffer size for shuffling. Defaults to `10000`.

        Returns:
            Dict[str, BuilderType]: A dictionary of dataset builders keyed by dataset name.
        """
        dataset_config = dataset_config if dataset_config is not None else self.config.dataset_config

        builder_dict = {}
        assert len(dataset_config) > 0, "At least one dataset has to be specified."

        for builder_cls_name, config in dataset_config.items():
            builder = registry.get_builder_class(builder_cls_name)(**config)
            builder_dict[builder_cls_name] = builder

        return builder_dict

    def build_evaluator(self, evaluator_config: Optional[omegaconf.DictConfig] = None):
        """
        Builds the evaluators using the provided evaluator configuration. Evaluators are
        added to the container and evaluated on the datasets.

        Args:
            evaluator_config (Optional[omegaconf.DictConfig]): The evaluator configuration. Defaults to `self.config.evaluator_config`.
        """
        evaluator_config = evaluator_config if evaluator_config is not None else self.config.evaluator_config
        dataset_dict = self.build_datasets()

        container = EvaluatorContainer()

        for (evaluator_cls_name, config), (builder_cls_name, builder) in zip(
                evaluator_config.items(),
                dataset_dict.items()
        ):
            if evaluator_cls_name not in BUILDER_EVALUATOR_MAPPING[builder_cls_name]:
                raise TypeError(f"Evaluator {evaluator_cls_name} not valid for builder {builder_cls_name}.")

            evaluator_cls = registry.get_evaluator_class(evaluator_cls_name)
            assert evaluator_cls is not None, f"Evaluator {evaluator_cls_name} not properly registered."

            collator_cls_name = EVALUATOR_COLLATOR_MAPPING[evaluator_cls_name]
            collator_cls = registry.get_collator_class(collator_cls_name)
            assert collator_cls is not None, f"Collator {collator_cls_name} not properly registered."

            collator = collator_cls(
                processor=self.build_processor(),
                **self.config.collator_config
            )

            container.add(
                evaluator=evaluator_cls(
                    model=self.build_model(),
                    evaluate_dataset=builder.build_datasets(),
                    data_collator=collator,
                    **config
                )
            )


@add_end_docstrings(TASK_DOCSTRING)
@registry.register_task("EvaluateTaskWithPretrainedModel")
class EvaluateTaskWithPretrainedModel(EvaluateTask, TaskWithPretrainedModel):
    """
    An evaluation task for pretrained models. Inherits from `EvaluateTask` and `TaskWithPretrainedModel`.
    """

    config: EvaluateConfig

    def build_model(self, model_config: Optional[Dict] = None):
        """
        Builds and returns the pretrained model in evaluation mode.

        Args:
            model_config (Optional[Dict]): The model configuration. Defaults to `None`.

        Returns:
            PreTrainedModel: The pretrained model in evaluation mode.
        """
        return super().build_model(model_config=model_config).eval()


@add_end_docstrings(TASK_DOCSTRING)
@registry.register_task("EvaluateTaskWithCustomModel")
class EvaluateTaskWithCustomModel(EvaluateTask, TaskWithCustomModel):
    """
    An evaluation task for custom models. Inherits from `EvaluateTask` and `TaskWithCustomModel`.
    """

    config: EvaluateConfig

    def build_model(self, model_config: Optional[Dict] = None):
        """
        Builds and returns the custom model in evaluation mode.

        Args:
            model_config (Optional[Dict]): The model configuration. Defaults to `None`.

        Returns:
            PreTrainedModel: The custom model in evaluation mode.
        """
        return super().build_model(model_config=model_config).eval()
