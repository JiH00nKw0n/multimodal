import os
from glob import glob
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import spacy
import torch
from datasets import concatenate_datasets, Dataset, IterableDataset, load_dataset
from spacy.tokens import Doc, Span, Token
from tqdm import tqdm

from src.common import registry, ImageSimilarityCalculator
from src.datasets.builder import (
    SequenceTextDatasetFeaturesWithImageURLBuilder,
    NegCLIPSequenceTextDatasetFeaturesWithImageURLBuilder
)

__all__ = [
    "COCOCaptionsIterableDatasetBuilder",
    "COCOCaptionsDatasetBuilder",
    "COCOCaptionsDatasetBuilderWithMinedNegCLIP",
    "COCOCaptionsDatasetBuilderWithNegCLIPMining",
]


@registry.register_builder('COCOCaptionsIterableDatasetBuilder')
class COCOCaptionsIterableDatasetBuilder(SequenceTextDatasetFeaturesWithImageURLBuilder):
    """
    A builder class for creating an iterable dataset for COCO captions.
    It extends `SequenceTextDatasetFeaturesWithImageURL`.

    Attributes:
        split (str): The dataset split to load (e.g., 'train').
        name (Optional[str]): The name of the dataset.
    """
    split: str = 'train'
    name: Optional[str] = 'coco'

    def build_dataset(self) -> IterableDataset:
        """
        This method should implement the logic for building the dataset.
        Since it's not implemented here, it raises `NotImplementedError`.

        Returns:
            IterableDataset: The built dataset.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError


@registry.register_builder('COCOCaptionsDatasetBuilder')
class COCOCaptionsDatasetBuilder(SequenceTextDatasetFeaturesWithImageURLBuilder):
    """
    A builder class for creating a dataset for COCO captions with non-iterable format.
    It extends `SequenceTextDatasetFeaturesWithImageURL`.

    Attributes:
        split (Union[str, List[str]]): The dataset split(s) to load.
        name (Optional[str]): The name of the dataset.
    """
    split: Union[str, List[str]] = ['train', 'restval']
    name: Optional[str] = 'coco'

    def build_dataset(self) -> Dataset:
        """
        Builds and returns the COCO captions dataset.

        Returns:
            Dataset: The COCO captions dataset.
        """
        if isinstance(self.split, list):
            dataset = concatenate_datasets(load_dataset(
                "yerevann/coco-karpathy", trust_remote_code=True, split=self.split
            ))
        else:
            dataset = load_dataset(
                "yerevann/coco-karpathy", trust_remote_code=True, split=self.split
            )
        dataset = dataset.rename_columns({"sentences": 'text', "url": 'images'})
        dataset = dataset.select_columns(['images', 'text'])
        # dataset = dataset.cast(self.features)

        return dataset


@registry.register_builder('COCOCaptionsDatasetBuilderWithMinedNegCLIP')
class COCOCaptionsDatasetBuilderWithMinedNegCLIP(NegCLIPSequenceTextDatasetFeaturesWithImageURLBuilder):
    """
    A builder class for creating a COCO captions dataset with mined negative samples for NegCLIP.
    It extends `NegCLIPSequenceTextDatasetFeaturesWithImageURL`.

    Attributes:
        split (Union[str, List[str]]): The dataset split(s) to load.
        name (Optional[str]): The name of the dataset.
    """
    split: Union[str, List[str]] = 'train'
    name: Optional[str] = 'coco'

    def build_dataset(self) -> Dataset:
        """
        Builds and returns the COCO captions dataset with mined negative samples.

        Returns:
            Dataset: The COCO captions dataset with negative samples.
        """
        dataset = load_dataset(
            "yjkimstats/COCOCaption_mined", trust_remote_code=True, split=self.split
        )
        # dataset = dataset.cast(self.features)

        return dataset


def swap_spans(tokens: List[Token], span1: Span, span2: Span) -> List[Token]:
    """
    Swap the positions of two noun phrases (spans) in the list of tokens.

    Args:
        tokens (List[Token]): The list of tokens.
        span1 (Span): The first noun phrase span.
        span2 (Span): The second noun phrase span.

    Returns:
        List[Token]: A new list of tokens with the positions of `span1` and `span2` swapped.
    """
    start1, end1 = span1.start, span1.end
    start2, end2 = span2.start, span2.end

    if start1 < start2:
        return tokens[:start1] + tokens[start2:end2] + tokens[end1:start2] + tokens[start1:end1] + tokens[end2:]
    else:
        return tokens[:start2] + tokens[start1:end1] + tokens[end2:start1] + tokens[start2:end2] + tokens[end1:]


def process_example(example: Dict, nlp: spacy.Language, generate_negative_captions: Callable) -> Dict:
    """
    Processes an example from the dataset by generating negative captions based on the original captions.

    Args:
        example (Dict): The example from the dataset.
        nlp (spacy.Language): The spaCy language model used to process the text.
        generate_negative_captions (Callable): A function to generate negative captions.

    Returns:
        Dict: A dictionary containing original and generated negative captions and images.
    """
    original_texts = example['text']
    neg_texts = []
    has_zero_negative = False

    for original_text in original_texts:
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

    for hard_text in hard_texts:
        for hard_caption in hard_text:
            doc = nlp(hard_caption)
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


@registry.register_builder('COCOCaptionsDatasetBuilderWithNegCLIPMining')
class COCOCaptionsDatasetBuilderWithNegCLIPMining(NegCLIPSequenceTextDatasetFeaturesWithImageURLBuilder):
    """
    A builder class for creating a COCO captions dataset with NegCLIP and mined negative samples.
    It extends `NegCLIPSequenceTextDatasetFeaturesWithImageURL`.

    Attributes:
        split (Union[str, List[str]]): The dataset split(s) to load.
        name (Optional[str]): The name of the dataset.
        spacy_model_name (Optional[str]): The name of the spaCy language model.
        similarity_model_name_or_path (Optional[Union[str, os.PathLike]]): The path or name of the similarity model.
        batch_size (Optional[int]): Batch size for similarity calculation.
        image_top_k (Optional[int]): Top K similar images to select.
        max_num_texts (Optional[int]): Maximum number of negative captions to generate.
        seed (Optional[int]): Random seed for reproducibility.
        rng (Optional[np.random.Generator]): A random generator for sampling.
    """
    split: Union[str, List[str]] = ['train', 'restval']
    name: Optional[str] = 'coco'
    spacy_model_name: Optional[str] = "en_core_web_sm"
    similarity_model_name_or_path: Optional[Union[str, os.PathLike]] = 'openai/clip-vit-large-patch14'
    batch_size: Optional[int] = 128
    image_top_k: Optional[int] = 3
    max_num_texts: Optional[int] = 5
    seed: Optional[int] = 2024
    rng: Optional[np.random.Generator] = None

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes the random generator if it has not been set.

        Args:
            __context (Any): The context passed during model initialization.
        """
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)
        super().model_post_init(None)

    def build_dataset(self) -> Dataset:
        """
        Builds and returns the COCO captions dataset with negative samples using NegCLIP.

        Returns:
            Dataset: The COCO captions dataset with negative samples.
        """
        cache_list = list(map(lambda x: x.split('/')[-1], glob(f'{os.getenv("DATA_DIR")}/*.parquet')))

        if f'{self.cache_file_name}.parquet' in cache_list:
            dataset = load_dataset("parquet", data_files={'train': f'{self.cache_file_name}.parquet'})
        else:
            if isinstance(self.split, list):
                dataset = concatenate_datasets(load_dataset(
                    "yerevann/coco-karpathy", trust_remote_code=True, split=self.split
                ))
            else:
                dataset = load_dataset(
                    "yerevann/coco-karpathy", trust_remote_code=True, split=self.split
                )
            dataset = dataset.rename_columns({"sentences": 'text', "url": 'images'})
            dataset = dataset.select_columns(['images', 'text'])

            # Perform image and text mining
            dataset = self.negative_image_mining(dataset)
            dataset = self.negative_text_mining(dataset)

            # Cast to required features
            dataset = dataset.cast(self.features)

        return dataset

    def negative_image_mining(self, dataset: Dataset) -> Dataset:
        """
        Performs negative image mining by finding top K similar images for each example in the dataset.

        Args:
            dataset (Dataset): The input dataset.

        Returns:
            Dataset: The dataset with added hard negative images and captions.
        """
        new_examples: List[Dict[str, str]] = []
        image_similarity_calculator = ImageSimilarityCalculator(
            similarity_model_name_or_path=self.similarity_model_name_or_path,
            batch_size=self.batch_size,
            top_k=self.image_top_k,
        )
        torch.cuda.empty_cache()
        similarity_dict = image_similarity_calculator.compute_image_similarity(dataset=dataset)

        for idx, example in tqdm(enumerate(dataset)):
            similar_indices = similarity_dict[idx]
            hard_images = [dataset[idx]['images'] for idx in similar_indices]
            hard_texts = [dataset[idx]['text'] for idx in similar_indices]
            new_examples.append({
                'images': example['images'],
                'text': example["text"],
                'hard_images': hard_images,
                'hard_texts': hard_texts
            })

        return Dataset.from_list(new_examples)

    def negative_text_mining(self, dataset: Dataset) -> Dataset:
        """
        Performs negative text mining by generating negative captions for each example in the dataset.

        Args:
            dataset (Dataset): The input dataset.

        Returns:
            Dataset: The dataset with added hard negative texts.
        """
        nlp = spacy.load(self.spacy_model_name)

        # 시스템의 CPU 코어 수에 따라 num_proc 설정
        max_num_proc = os.cpu_count()

        # Dataset의 각 요소에 대해 process_example 함수를 적용
        dataset_with_neg_text = dataset.map(
            lambda example: process_example(example, nlp, self.generate_negative_captions),
            remove_columns=dataset.column_names,
            num_proc=max_num_proc,  # 병렬로 처리할 프로세스 개수 설정
            desc="Negative Text Mining"
        )

        # 필터링: 'neg_texts'와 'hard_neg_texts'가 빈 리스트가 아닌 경우만 유지
        dataset_with_neg_text = dataset_with_neg_text.filter(
            lambda x: len(x['neg_texts']) > 0 and len(x['hard_neg_texts']) > 0)

        return dataset_with_neg_text

    def generate_negative_captions(self, doc: Doc) -> List[str]:
        """
        Generates negative captions by swapping noun phrases, adjectives, verbs, and adverbs in the text.

        Args:
            doc (Doc): The spaCy Doc object representing the parsed text.

        Returns:
            List[str]: A list of generated negative captions.
        """
        # 명사구(Noun Phrases)는 3개 이상의 토큰으로 구성된 경우만 포함
        noun_phrases = [np for np in doc.noun_chunks if len(np) >= 3]

        if len(noun_phrases):
            swap_groups = {
                'NOUN': [token for token in doc if token.pos_ == 'NOUN'],
                'ADJ': [token for token in doc if token.pos_ == 'ADJ'],
                'VERB': [token for token in doc if token.pos_ == 'VERB'],
                'ADV': [token for token in doc if token.pos_ == 'ADV'],
                'NP': noun_phrases,  # 명사구는 따로 그룹화
            }
        else:
            swap_groups = {
                'NOUN': [token for token in doc if token.pos_ == 'NOUN'],
                'ADJ': [token for token in doc if token.pos_ == 'ADJ'],
                'VERB': [token for token in doc if token.pos_ == 'VERB'],
                'ADV': [token for token in doc if token.pos_ == 'ADV'],
            }

            negative_captions: List[str] = []

            for _ in range(self.max_num_texts):  # 최대 self.max_num_texts 개의 부정적인 캡션 생성
                # 스왑할 수 있는 그룹 중에서 무작위로 하나 선택
                eligible_groups = [group for group in swap_groups.values() if len(group) >= 2]
                if not eligible_groups:
                    break  # 스왑할 요소가 부족한 경우 중단

                group_index = self.rng.choice(len(eligible_groups))
                group = eligible_groups[group_index]
                indices = self.rng.choice(len(group), 2, replace=False)
                a, b = group[indices[0]], group[indices[1]]
                new_tokens = list(doc)

                if isinstance(a, Span) and isinstance(b, Span):  # 둘 다 명사구인 경우
                    # 명사구의 위치를 전체적으로 교환
                    new_tokens = swap_spans(new_tokens, a, b)
                else:  # 단일 토큰끼리 또는 명사구와 명사가 아닌 토큰
                    new_tokens[a.i], new_tokens[b.i] = new_tokens[b.i], new_tokens[a.i]

                new_caption = " ".join([token.text for token in new_tokens])
                if new_caption not in negative_captions and new_caption != doc.text:
                    negative_captions.append(new_caption)

            return negative_captions
