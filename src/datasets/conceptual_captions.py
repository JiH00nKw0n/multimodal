import os
import json
import torch
import pickle
import logging
from glob import glob

import spacy
import numpy as np
from tqdm import tqdm
from spacy.tokens import Doc, Token, Span
from typing import Union, List, Dict, Optional, Any, Callable

from datasets import Dataset, IterableDataset, load_dataset, concatenate_datasets

from src.common import registry, ImageSimilarityCalculator
from src.datasets.builder import TextDatasetFeaturesWithImageBuilder
from .utils import swap_spans

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler(f'./logging/{__name__}.log', 'w'))

__all__ = [
    "ConceptualCaptionsIterableDatasetBuilder",
    "ConceptualCaptionsDatasetBuilder"
]


# TODO; hard coding! need to migrate to .utils by integrating similar functions from coco_captions
def process_example(example: Dict, nlp: spacy.Language, generate_negative_captions: Callable) -> Dict:
    original_texts = example['text']
    neg_texts = []
    has_zero_negative = False

    for original_text in original_texts:
        logger.debug(f'original_text : {original_text}')
        doc = nlp(original_text)
        negative_captions = generate_negative_captions(doc)
        if not negative_captions:
            has_zero_negative = True
            break
        else:
            neg_texts.append(negative_captions)

    if has_zero_negative:
        return {
            'images': example['images'],
            'text': example['text'],
            'hard_texts': example['hard_texts'],
            'hard_images': example['hard_images'],
            'neg_texts': [],
            'hard_neg_texts': []
        }

    hard_texts = example['hard_texts']
    hard_neg_texts = []

    # Why nested loop?
    for hard_text in hard_texts:
        # for hard_caption in hard_text:
        logger.debug(f'hard_text : {hard_text}')
        doc = nlp(hard_text)
        hard_negative_captions = generate_negative_captions(doc)
        if not hard_negative_captions:
            has_zero_negative = True
            break
        else:
            hard_neg_texts.append(hard_negative_captions)
        if has_zero_negative:
            return {
                'images': example['images'],
                'text': example['text'],
                'hard_texts': example['hard_texts'],
                'hard_images': example['hard_images'],
                'neg_texts': [],
                'hard_neg_texts': []
            }

    return {
        'images': example['images'],
        'text': example['text'],
        'hard_texts': hard_texts,
        'hard_images': example['hard_images'],
        'neg_texts': neg_texts,
        'hard_neg_texts': hard_neg_texts
    }


@registry.register_builder('ConceptualCaptionsIterableDatasetBuilder')
class ConceptualCaptionsIterableDatasetBuilder(TextDatasetFeaturesWithImageBuilder):
    """
    A builder class for creating an iterable dataset for Conceptual Captions (CC3M).
    It extends `TextDatasetFeaturesWithImage` and uses the streaming mode to handle large datasets.

    Attributes:
        split (Optional[str]): The dataset split to load (default: 'train').
        name (Optional[str]): The name of the dataset (default: 'cc3m').
    """
    split: Optional[str] = 'train'
    name: Optional[str] = 'cc3m'

    def build_dataset(self) -> IterableDataset:
        """
        Builds and returns an iterable Conceptual Captions dataset.

        Returns:
            IterableDataset: The streaming iterable dataset with 'images' and 'text' columns.
        """
        dataset = load_dataset(
            "pixparse/cc3m-wds", trust_remote_code=True, streaming=True, split=self.split
        )
        dataset = dataset.rename_columns({"jpg": 'images', "txt": 'text'})
        dataset = dataset.select_columns(['images', 'text'])
        dataset = dataset.cast(self.features)

        return dataset


@registry.register_builder('ConceptualCaptionsDatasetBuilder')
class ConceptualCaptionsDatasetBuilder(TextDatasetFeaturesWithImageBuilder):
    """
    A builder class for creating a non-iterable dataset for Conceptual Captions (CC3M).
    It extends `TextDatasetFeaturesWithImage`.

    Attributes:
        split (Optional[str]): The dataset split to load (default: 'train').
        name (Optional[str]): The name of the dataset (default: 'cc3m').
        spacy_model_name (Optional[str]): The name of the spaCy language model.
        similarity_model_name_or_path (Optional[Union[str, os.PathLike]]): The path or name of the similarity model.
        batch_size (Optional[int]): Batch size for similarity calculation.
        image_top_k (Optional[int]): Top K similar images to select.
        max_num_texts (Optional[int]): Maximum number of negative captions to generate.
        seed (Optional[int]): Random seed for reproducibility.
        rng (Optional[np.random.Generator]): A random generator for sampling.

    """
    split: Optional[str] = 'train'
    name: Optional[str] = 'cc3m'
    spacy_model_name: Optional[str] = "en_core_web_sm"
    similarity_model_name_or_path: Optional[Union[str]] = 'openai/clip-vit-large-patch14'
    batch_size: Optional[int] = 128
    image_top_k: Optional[int] = 3
    max_num_texts: Optional[int] = 5
    seed: Optional[int] = 2024
    rng: Optional[int] = None

    def model_post_init(self, __context: Any) -> None:
        # numpy RandomState 초기화
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)
        super().model_post_init(None)

    def build_dataset(self) -> Dataset:        
        """
        Builds and returns a Conceptual Captions dataset.

        Returns:
            Dataset: The dataset with 'images' and 'text' columns.
        """
        def map_key_to_image(example):
            def get_img_from_key(img_key):
                return raw_dataset[raw_dataset_keys.index(img_key)]['jpg']
            example['images'] = get_img_from_key(example['images'])
            example['hard_images'] = list(map(lambda img_key: get_img_from_key(img_key), example['hard_images']))
            return example
        # NOTE; how to merge with published mined dataset?
        # Step 1 : Load the mined dataset and the raw dataset
        # Step 2 : Map key (from mined dataset) to images (from the raw dataset)
        
        raw_dataset = load_dataset("pixparse/cc3m-wds", trust_remote_code=True, split=self.split)
        logger.info(f'raw_dataset:\n{raw_dataset}')
        raw_dataset_keys = raw_dataset['__key__']
        
        mined_dataset = load_dataset("yjkimstats/CC3M_500k_mined", trust_remote_code=True, split=self.split)
        logger.info(f'mined_dataset:\n{mined_dataset}')
        mined_dataset = mined_dataset.map(map_key_to_image)
       
        return dataset

