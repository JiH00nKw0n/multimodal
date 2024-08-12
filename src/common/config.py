from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, Extra
from typing import Dict
import yaml

__all__ = ["TrainConfig"]


def load_yml(path: str) -> Dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


@dataclass(config=ConfigDict(
    extra=Extra.ignore, frozen=True, strict=True, validate_assignment=True
))
class TrainConfig:
    """
    Config that contains each config's file path.
    """
    model: str
    processor: str
    data: str
    trainer: str
    task: str

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

    @property
    def task_config(self) -> Dict:
        return load_yml(self.processor)

    def pretty_print(self):
        logging.info("\n=====  Running Parameters    =====")
        logging.info(self._convert_node_to_json(self.config.run))

        logging.info("\n======  Dataset Attributes  ======")
        datasets = self.config.datasets

        for dataset in datasets:
            if dataset in self.config.datasets:
                logging.info(f"\n======== {dataset} =======")
                dataset_config = self.config.datasets[dataset]
                logging.info(self._convert_node_to_json(dataset_config))
            else:
                logging.warning(f"No dataset named '{dataset}' in config. Skipping")

        logging.info(f"\n======  Model Attributes  ======")
        logging.info(self._convert_node_to_json(self.config.model))

