import json
import os
from typing import Any, Dict, Optional, Type, Union, TypeVar

from datasets import Dataset
from pydantic import BaseModel, ConfigDict
from transformers import PreTrainedModel, Trainer
from transformers.utils import logging

# from src.collators import BaseCollator

logger = logging.get_logger(__name__)

# CollatorType = Type[BaseCollator]
CollatorType = TypeVar("CollatorType", bound="BaseCollator")

__all__ = ["BaseTrainer", "BaseEvaluator"]


class BaseTrainer(Trainer):
    """
    A subclass of the Hugging Face `Trainer` class, designed to be extended with additional
    training logic and customized behavior.
    """
    pass


class BaseEvaluator(BaseModel):
    """
    A base class for model evaluation. This class provides a structure for evaluating models 
    using various datasets, collators, and model configurations. 

    Attributes:
        model (`Optional[PreTrainedModel]`): The model to be evaluated.
        data_collator (`Optional[CollatorType]`): A custom collator for preparing the data during evaluation.
        dataset_name (`Optional[str]`): Name of the dataset used for evaluation.
        evaluate_dataset (`Optional[Dataset]`): The dataset used for evaluation.
        output_dir (`Optional[Union[str, os.PathLike]]`): Directory where evaluation results will be saved.
    """

    model: Optional[PreTrainedModel] = None
    data_collator: Optional[CollatorType] = None
    dataset_name: Optional[str] = None
    evaluate_dataset: Optional[Dataset] = None
    output_dir: Optional[Union[str, os.PathLike]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization method to set the model in evaluation mode and format the dataset name.

        Args:
            __context (`Any`): Context or additional information required for post-initialization.
        """
        self.model.eval()
        self.dataset_name = self.dataset_name.upper()

    def _encode_dataset(self, batch_size: int = 128):
        """
        Encodes the dataset for evaluation. This method must be implemented by subclasses.

        Args:
            batch_size (`int`, *optional*, defaults to 128): The batch size for encoding the dataset.

        Raises:
            `NotImplementedError`: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    def evaluate(self, batch_size: int = 128):
        """
        Evaluates the model on the provided dataset. This method must be implemented by subclasses.

        Args:
            batch_size (`int`, *optional*, defaults to 128): The batch size for evaluation.

        Raises:
            `NotImplementedError`: If the method is not implemented in the subclass.

        Note:
            This method overrides the `__call__` special method.
            Although the forward pass is defined here, you should call the [`Module`] instance itself instead 
            of this method directly as the former ensures pre- and post-processing.
        """
        raise NotImplementedError

    def _save_result(self, result: Dict):
        """
        Saves the evaluation results to a JSON file in the specified output directory.

        Args:
            result (`Dict`): A dictionary containing the evaluation results.

        Notes:
            - Creates the output directory if it doesn't exist.
            - The results are saved under the filename `{dataset_name}.json`.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        if self.output_dir is not None:
            with open(os.path.join(self.output_dir, f'{self.dataset_name}.json'), "w") as f:
                json.dump(result, f, indent=2)
