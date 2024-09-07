from typing import Dict
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, Extra
from omegaconf import OmegaConf, DictConfig

__all__ = ["TrainConfig", "EvaluateConfig"]

@dataclass(config=ConfigDict(
    extra=Extra.ignore, frozen=True, strict=True, validate_assignment=True
))
class BaseConfig:
    """
    Base configuration class that contains settings for model, processor, dataset, and run configurations.

    Args:
        model (`Dict`):
            Dictionary containing model configuration settings.
        processor (`Dict`):
            Dictionary containing processor configuration settings.
        dataset (`Dict`):
            Dictionary containing dataset configuration settings.
        collator (`Dict`):
            Dictionary containing collator configuration settings.
        run (`Dict`):
            Dictionary containing run settings.

    Properties:
        model_config (`DictConfig`):
            Returns the model configuration as an `OmegaConf` object.
        processor_config (`DictConfig`):
            Returns the processor configuration as an `OmegaConf` object.
        dataset_config (`Dict`):
            Returns the dataset configuration.
        collator_config (`Dict`):
            Returns the collator configuration.
        run_config (`DictConfig`):
            Returns the run configuration as an `OmegaConf` object.
    """
    model: Dict
    processor: Dict
    dataset: Dict
    collator: Dict
    run: Dict

    @property
    def model_config(self) -> DictConfig:
        return OmegaConf.create(self.model)

    @property
    def processor_config(self) -> DictConfig:
        return OmegaConf.create(self.processor)

    @property
    def dataset_config(self) -> Dict:
        return self.dataset

    @property
    def collator_config(self) -> Dict:
        return self.collator

    @property
    def run_config(self) -> DictConfig:
        return OmegaConf.create(self.run)


@dataclass(config=ConfigDict(
    extra=Extra.ignore, frozen=True, strict=True, validate_assignment=True
))
class TrainConfig(BaseConfig):
    """
    Configuration class for training, which extends `BaseConfig` by adding trainer-specific settings.

    Args:
        trainer (`Dict`):
            Dictionary containing trainer configuration settings.

    Properties:
        trainer_config (`Dict`):
            Returns the trainer configuration dictionary.
    """
    trainer: Dict

    @property
    def trainer_config(self) -> Dict:
        return self.trainer


@dataclass(config=ConfigDict(
    extra=Extra.ignore, frozen=True, strict=True, validate_assignment=True
))
class EvaluateConfig(BaseConfig):
    """
    Configuration class for evaluation, which extends `BaseConfig` by adding evaluator-specific settings.

    Args:
        evaluator (`Dict`):
            Dictionary containing evaluator configuration settings.

    Properties:
        evaluator_config (`Dict`):
            Returns the evaluator configuration dictionary.
    """
    evaluator: Dict

    @property
    def evaluator_config(self) -> Dict:
        return self.evaluator
