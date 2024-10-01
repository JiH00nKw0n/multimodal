import random
from typing import List, Optional, Union

from datasets import Dataset, IterableDataset, load_dataset

from src.utils import load_metadata_from_tar_files, download_images_with_img2dataset
from src.common import registry
from src.datasets.builder import SequenceTextDatasetFeaturesWithImageURLBuilder

__all__ = [
    "Laion400mIterableDatasetBuilder",
    "Laion400mDatasetBuilder",
    "Laion400mTarPathDatasetBuilder",
]


@registry.register_builder('Laion400mIterableDatasetBuilder')
class Laion400mIterableDatasetBuilder(SequenceTextDatasetFeaturesWithImageURLBuilder):
    """
    A builder class for creating an iterable dataset for the Laion400m dataset.
    It extends `SequenceTextDatasetFeaturesWithImageURLBuilder`.

    Attributes:
        split (Union[str, List[str]]): The dataset split to load (default: 'train').
        name (Optional[str]): The name of the dataset (default: 'laion').
    """
    split: Union[str, List[str]] = 'train'
    name: Optional[str] = 'laion'

    def build_dataset(self) -> IterableDataset:
        """
        Builds and returns the Laion400m iterable dataset.

        Returns:
            IterableDataset: The Laion400m dataset as an iterable.
        """
        dataset = load_dataset(
            "laion/laion400m", trust_remote_code=True, split=self.split, streaming=True
        )
        dataset = dataset.rename_columns({"caption": 'text', "url": 'images'})
        dataset = dataset.select_columns(['images', 'text'])

        return dataset


@registry.register_builder('Laion400mDatasetBuilder')
class Laion400mDatasetBuilder(SequenceTextDatasetFeaturesWithImageURLBuilder):
    """
    A builder class for creating a non-iterable dataset for the Laion400m dataset.
    It extends `SequenceTextDatasetFeaturesWithImageURLBuilder`.

    Attributes:
        split (Union[str, List[str]]): The dataset split(s) to load (default: 'train').
        name (Optional[str]): The name of the dataset (default: 'laion').
    """
    split: Union[str, List[str]] = 'train'
    name: Optional[str] = 'laion'
    num_sample: Optional[int] = None
    seed: Optional[int] = 2024

    def build_dataset(self) -> Dataset:
        dataset = load_dataset(
            "laion/laion400m", trust_remote_code=True, split=self.split
        )
        dataset = dataset.rename_columns({"caption": 'text', "url": 'images'})
        dataset = dataset.select_columns(['images', 'text'])
        if self.num_sample is not None:
            random_indices = random.sample(range(len(dataset)), self.num_sample)
            dataset = dataset.select(random_indices)

        return dataset


@registry.register_builder('Laion400mTarPathDatasetBuilder')
class Laion400mTarPathDatasetBuilder(SequenceTextDatasetFeaturesWithImageURLBuilder):
    """
    A builder class for creating a non-iterable dataset for the Laion400m dataset.
    It extends `SequenceTextDatasetFeaturesWithImageURLBuilder`.
    """
    split: Union[str, List[str]] = 'train'
    name: Optional[str] = 'laion'
    num_sample: Optional[int] = None
    output_dir: Optional[str] = None
    output_format: str = "webdataset"  # Default to 'files'

    def build_dataset(self) -> Dataset:
        dataset = load_dataset(
            "laion/laion400m", trust_remote_code=True, split=self.split, token='your_huggingface_token'
        )
        dataset = dataset.rename_columns({"caption": 'text', "url": 'images'})
        dataset = dataset.select_columns(['images', 'text'])

        # Sample from dataset if num_sample is specified
        if self.num_sample is not None:
            random.seed(self.seed)
            random_indices = random.sample(range(len(dataset)), self.num_sample)
            dataset = dataset.select(random_indices)

        # Extract URLs for downloading images
        print('url loading')
        image_urls = dataset["images"]

        # Download images using img2dataset
        download_images_with_img2dataset(image_urls, self.output_dir, self.output_format)
        print('load metadata')
        metadata = load_metadata_from_tar_files(self.output_dir)
        # Filter out examples where images are None
        print('filtering missing images')
        dataset = dataset.filter(lambda example: example['images'] in metadata)

        return dataset
