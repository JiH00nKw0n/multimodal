import logging
import os
from typing import Any, Optional, Dict, Type, Union, TypeVar

from datasets import Dataset, IterableDataset
from omegaconf import DictConfig
from peft import get_peft_model, LoraConfig
from pydantic import BaseModel, ConfigDict, Extra
from transformers import AutoModel, AutoProcessor, PreTrainedModel, ProcessorMixin, add_end_docstrings

from src.utils import load_yml

ModelType = Type[PreTrainedModel]
ProcessorType = Type[ProcessorMixin]
DatasetType = TypeVar("DatasetType", Dataset, IterableDataset)

__all__ = ["BaseTask", "TaskWithPretrainedModel", "TaskWithCustomModel"]

logger = logging.getLogger(__name__)

TASK_DOCSTRING = """
    A task class that defines the fundamental structure and workflow for different tasks involving models, processors, 
    and datasets. This class serves as the base class for task-specific implementations, providing an interface for 
    building models, processors, and datasets, as well as setting up tasks.

    The `BaseTask` class acts as a blueprint for other task-specific classes, ensuring a consistent interface across 
    different tasks. Each task class that inherits from `BaseTask` must override the `build_model`, `build_processor`, 
    and `build_datasets` methods.

    Args:
        config (`Any`):
            The configuration object for setting up the task, including model, processor, and dataset configurations.

    Methods:
        setup_task(config):
            A class method to initialize the task with the provided configuration.

        build_model(model_config: `Optional[Dict]`):
            Abstract method to build and return a model instance. Must be overridden by subclasses.

        build_processor(processor_config: `Optional[Dict]`):
            Abstract method to build and return a processor instance. Must be overridden by subclasses.

        build_datasets(dataset_config: `Optional[Dict]`, shuffle: `Optional[bool]`, buffer_size: `Optional[int]`):
            Abstract method to build and return datasets for the task. Must be overridden by subclasses.
"""


@add_end_docstrings(TASK_DOCSTRING)
class BaseTask(BaseModel):
    """
    The `BaseTask` is an abstract class that provides a template for task-specific classes. It defines methods for
    building models, processors, and datasets, which must be implemented by any class that inherits from `BaseTask`.

    This base class enforces consistency across tasks by requiring the implementation of core methods that set up
    the necessary components for task execution.
    """

    config: Any

    model_config = ConfigDict(extra=Extra.forbid)

    @classmethod
    def setup_task(cls, config):
        return cls(config=config)

    @add_end_docstrings("This method must be overridden by subclasses.")
    def build_model(
            self,
            model_config: Optional[Dict] = None
    ) -> PreTrainedModel:
        raise NotImplementedError

    @add_end_docstrings("This method must be overridden by subclasses.")
    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ) -> ProcessorMixin:
        raise NotImplementedError

    @add_end_docstrings("This method must be overridden by subclasses.")
    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ) -> Union[IterableDataset, Dataset]:
        raise NotImplementedError


@add_end_docstrings(TASK_DOCSTRING)
class TaskWithPretrainedModel(BaseTask):
    """
    A task class for handling pretrained models with optional LoRA configuration for parameter-efficient fine-tuning.
    This class extends the `BaseTask` and provides implementations for building a model and processor.

    Methods:
        build_model(model_config: `Optional[Dict]`):
            Builds and returns a pretrained model. If a LoRA configuration is provided, it applies LoRA to the model.

        build_processor(processor_config: `Optional[Dict]`):
            Builds and returns a processor for the task.
    """

    def build_model(
            self,
            model_config: Optional[Dict] = None
    ) -> ModelType:
        """
        Builds a pretrained model using the provided configuration. If a LoRA configuration is specified,
        it applies the LoRA configuration to the model for parameter-efficient fine-tuning.

        Args:
            model_config (`Optional[Dict]`, *optional*):
                The model configuration dictionary. If not provided, uses the configuration from `self.config`.

        Returns:
            `ModelType`: The model instance loaded with optional LoRA fine-tuning.

        Raises:
            TypeError: If `model_config.lora` is neither a string nor a valid path.
        """
        model_config = model_config if model_config is not None else self.config.model_config.copy()

        model = AutoModel.from_pretrained(**model_config.config)

        if model_config.lora is not None:
            if isinstance(model_config.lora, str):
                lora_config = load_yml(model_config.lora)
            elif isinstance(model_config.lora, os.PathLike):
                lora_config = load_yml(os.fspath(model_config.lora))
            else:
                raise TypeError("`lora` configuration must be either a string or a valid path.")

            peft_config = LoraConfig(**lora_config)
            model = get_peft_model(model, peft_config)

            logger.info(f"{repr(model)}")
            trainable_params, all_param = model.get_nb_trainable_parameters()
            logger.info(
                f'ALL PARAM: {all_param} / TRAINABLE PARAM: {trainable_params} / '
                f'RATIO: {trainable_params / all_param * 100}%'
            )

        return model.to('cuda')

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ) -> ProcessorType:
        """
        Builds a processor using the provided configuration. If no configuration is provided, it uses
        the processor configuration from `self.config`.

        Args:
            processor_config (`Optional[Dict]`, *optional*):
                The processor configuration dictionary. If not provided, uses the configuration from `self.config`.

        Returns:
            `ProcessorType`: The processor instance.
        """
        processor_config = processor_config if processor_config is not None else self.config.processor_config.copy()
        return AutoProcessor.from_pretrained(**processor_config.config)


@add_end_docstrings(TASK_DOCSTRING)
class TaskWithCustomModel(BaseTask):
    """
    A task class for building a custom model and processor using configurations from a registry.
    This class supports models and processors that require custom configurations, and optionally
    supports LoRA-based parameter-efficient fine-tuning for text and vision models.

    Methods:
        build_model(model_config: `Optional[Dict]`):
            Builds and returns a custom model based on configurations from the registry.
            Supports LoRA fine-tuning for text and vision models.

        build_processor(processor_config: `Optional[Dict]`):
            Builds and returns a custom processor for text and vision tasks.
    """

    def build_model(
            self,
            model_config: Optional[Dict] = None
    ) -> ModelType:
        """
        Builds a custom model using the provided configuration from the registry. If a LoRA configuration
        is provided for either text or vision models, it applies the configuration for parameter-efficient fine-tuning.

        Args:
            model_config (`Optional[Dict]`, *optional*):
                The model configuration dictionary. If not provided, uses the configuration from `self.config`.

        Returns:
            `ModelType`: The custom model instance with optional LoRA fine-tuning.

        Raises:
            TypeError: If `model_config.lora` is not a valid `DictConfig` object.
        """
        from src.common.registry import registry

        model_config = model_config if model_config is not None else self.config.model_config.copy()

        # Get the model configuration and model class from the registry
        model_cfg_cls = registry.get_model_config_class(model_config.config_cls)
        model_cls = registry.get_model_class(model_config.model_cls)

        assert model_cls is not None, "Model {} not properly registered.".format(model_cls)
        assert model_cfg_cls is not None, "Model config {} not properly registered.".format(model_cfg_cls)

        # Initialize the model configuration and model
        model_cfg = model_cfg_cls(**model_config.config)
        model = model_cls(model_cfg)

        # Apply LoRA configurations if provided
        if model_config.lora is not None:
            if isinstance(model_config.lora, DictConfig):
                text_model_config_path = model_config.lora.pop('text_model', None)
                vision_model_config_path = model_config.lora.pop('vision_model', None)
            else:
                raise TypeError("LoRA configuration must be a valid `DictConfig` object.")

            # Apply LoRA configuration to text model
            if text_model_config_path is not None:
                text_model_lora_config = load_yml(text_model_config_path)
                text_model_peft_config = LoraConfig(**text_model_lora_config)
                model.text_model = get_peft_model(model.text_model, text_model_peft_config)

            # Apply LoRA configuration to vision model
            if vision_model_config_path is not None:
                vision_model_lora_config = load_yml(vision_model_config_path)
                vision_model_peft_config = LoraConfig(**vision_model_lora_config)
                model.vision_model = get_peft_model(model.vision_model, vision_model_peft_config)

        return model.to('cuda')

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ) -> ProcessorType:
        """
        Builds a custom processor using the provided configuration from the registry.

        Args:
            processor_config (`Optional[Dict]`, *optional*):
                The processor configuration dictionary. If not provided, uses the configuration from `self.config`.

        Returns:
            `ProcessorType`: The processor instance.

        Raises:
            AssertionError: If the processor class is not properly registered in the registry.
        """
        from src.common.registry import registry

        processor_config = processor_config if processor_config is not None else self.config.processor_config.copy()

        # Get the processor class from the registry
        processor_cls = registry.get_processor_class(processor_config.processor_cls)

        assert processor_cls is not None, "Processor {} not properly registered.".format(processor_cls)

        return processor_cls.from_text_vision_pretrained(**processor_config.config)
