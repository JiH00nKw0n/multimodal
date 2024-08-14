from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, Extra
from typing import Dict
from omegaconf import OmegaConf, DictConfig

__all__ = ["TrainConfig"]


@dataclass(config=ConfigDict(
    extra=Extra.ignore, frozen=True, strict=True, validate_assignment=True
))
class TrainConfig:
    """
    Config that contains each config's file path.
    """
    model: Dict
    processor: Dict
    dataset: Dict
    trainer: Dict
    run: Dict

    @property
    def model_config(self) -> DictConfig:
        return OmegaConf.create(self.model)

    @property
    def dataset_config(self) -> Dict:
        return self.dataset

    @property
    def trainer_config(self) -> Dict:
        return self.trainer

    @property
    def processor_config(self) -> DictConfig:
        return OmegaConf.create(self.processor)

    @property
    def run_config(self) -> DictConfig:
        return OmegaConf.create(self.run)
