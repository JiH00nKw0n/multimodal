from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from typing import Dict
import yaml

__all__ = ["TrainConfig"]


def load_yml(path: str) -> Dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


@dataclass(config=ConfigDict(
    extra='ignore', frozen=True, strict=True, validate_assignment=True
))
class TrainConfig:
    """
    Config that contains each config's file path.
    """
    model: str
    processor: str
    data: str
    trainer: str

    @property
    def model_config(self) -> Dict:
        return load_yml(self.model)

    @property
    def data_config(self) -> Dict:
        return load_yml(self.data)

    @property
    def trainer_config(self) -> Dict:
        return load_yml(self.trainer)

    @property
    def processor_config(self) -> Dict:
        return load_yml(self.processor)