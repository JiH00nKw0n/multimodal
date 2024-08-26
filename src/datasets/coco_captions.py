import os
from typing import Union, List, Dict, Optional, Any

import numpy as np
import spacy
from spacy.tokens import Doc, Span, Token
from tqdm import tqdm

from src.datasets.base import SequenceTextDatasetBuilder, SequenceTextDatasetWithHNBuilder
from src.common import registry, ImageSimilarityCalculator
from datasets import concatenate_datasets, load_dataset, Dataset, IterableDataset, Image


@registry.register_builder('COCOCaptionsIterableDatasetBuilder')
class COCOCaptionsIterableDatasetBuilder(SequenceTextDatasetBuilder):
    split: str = 'train'
    name: Optional[str] = 'coco'

    def build_dataset(self) -> IterableDataset:
        raise NotImplementedError


@registry.register_builder('COCOCaptionsDatasetBuilder')
class COCOCaptionsDatasetBuilder(SequenceTextDatasetBuilder):
    split: Union[str | List[str]] = ['train', 'restval']
    name: Optional[str] = 'coco'

    def build_dataset(self) -> Dataset:
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
        dataset = dataset.cast(self.features)

        return dataset


def swap_spans(tokens: List[Token], span1: Span, span2: Span) -> List[Token]:
    """
    두 명사구의 위치를 교환합니다.

    Parameters:
    - tokens (List[Token]): Token 객체 리스트
    - span1 (Span): 첫 번째 명사구
    - span2 (Span): 두 번째 명사구

    Returns:
    - List[Token]: 위치가 교환된 새로운 Token 리스트
    """
    start1, end1 = span1.start, span1.end
    start2, end2 = span2.start, span2.end

    if start1 < start2:
        return tokens[:start1] + tokens[start2:end2] + tokens[end1:start2] + tokens[start1:end1] + tokens[end2:]
    else:
        return tokens[:start2] + tokens[start1:end1] + tokens[end2:start1] + tokens[start2:end2] + tokens[end1:]


@registry.register_builder('COCOCaptionsWithNegCLIPHNDatasetBuilder')
class COCOCaptionsWithNegCLIPHNDatasetBuilder(SequenceTextDatasetWithHNBuilder):
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
        # numpy RandomState 초기화
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)

    def build_dataset(self) -> Dataset:
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
        # dataset = dataset.cast_column(column='images', feature=Image())

        dataset = self.negative_image_mining(dataset)
        dataset = self.negative_text_mining(dataset)
        dataset = dataset.cast(self.features)

        return dataset

    def negative_image_mining(self, dataset: Dataset) -> Dataset:
        new_examples: List[Dict[str, str]] = []

        image_similarity_calculator = ImageSimilarityCalculator(
            similarity_model_name_or_path=self.similarity_model_name_or_path,
            batch_size=self.batch_size,
            top_k=self.image_top_k,
        )
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
        nlp = spacy.load(self.spacy_model_name)
        new_examples: List[Dict[str, str]] = []

        for example in tqdm(dataset):
            original_image = example['images']
            original_texts = example['text']
            has_zero_negative = False
            neg_texts = []
            for original_text in original_texts:
                doc = nlp(original_text)
                negative_captions = self.generate_negative_captions(doc)

                if not negative_captions:
                    has_zero_negative = True
                    break
                else:
                    neg_texts.append(negative_captions)
            if has_zero_negative:
                continue
            hard_texts = example['hard_texts']
            hard_images = example['hard_images']
            hard_neg_texts = []
            for hard_text in hard_texts:
                for hard_caption in hard_text:
                    doc = nlp(hard_caption)
                    hard_negative_captions = self.generate_negative_captions(doc)

                    if not hard_negative_captions:
                        has_zero_negative = True
                        break
                    else:
                        hard_neg_texts.append(hard_negative_captions)
                if has_zero_negative:
                    break
            if has_zero_negative:
                continue
            else:
                new_examples.append({
                    'images': original_image,
                    'text': original_texts,
                    'hard_texts': hard_texts,
                    'hard_images': hard_images,
                    'neg_texts': neg_texts,
                    'hard_neg_texts': hard_neg_texts
                })
        return Dataset.from_list(new_examples)

    def generate_negative_captions(self, doc: Doc) -> List[str]:
        # 명사구(Noun Phrases)는 3개 이상의 토큰으로 구성된 경우만 포함
        noun_phrases = [np for np in doc.noun_chunks if len(np) >= 3]

        # 개별 명사, 형용사, 동사, 부사를 그룹으로 설정
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

        for _ in range(self.max_num_texts):  # 최대 5개의 부정적인 캡션 생성
            # 스왑할 수 있는 그룹 중에서 무작위로 하나 선택
            eligible_groups = [group for group in swap_groups.values() if len(group) >= 2]
            if not eligible_groups:
                break  # 스왑할 요소가 부족한 경우 중단

            group_index = self.rng.choice(len(eligible_groups))
            group = eligible_groups[group_index]
            a, b = self.rng.choice(group, 2, replace=False)
            new_tokens = list(doc)

            if isinstance(a, Span) and isinstance(b, Span):  # 둘 다 명사구인 경우
                # 명사구의 위치를 전체적으로 교환
                new_tokens = self.swap_spans(new_tokens, a, b)
            else:  # 단일 토큰끼리 또는 명사구와 명사가 아닌 토큰
                new_tokens[a.i], new_tokens[b.i] = new_tokens[b.i], new_tokens[a.i]

            new_caption = " ".join([token.text for token in new_tokens])
            if new_caption not in negative_captions and new_caption != doc.text:
                negative_captions.append(new_caption)

        return negative_captions
