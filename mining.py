import os
import spacy
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from spacy.tokens import Doc, Token, Span
from typing import Union, List, Dict, Optional, Any, Callable

from src.common import registry, ImageSimilarityCalculator
from src.datasets.base import SequenceTextDatasetBuilder, SequenceTextDatasetWithHNBuilder, BaseDatasetBuilder

from datasets import concatenate_datasets, load_dataset, Dataset, IterableDataset

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action=argparse.BooleanOptionalAction)
parser.add_argument('--raw_dataset_url_or_path', required=True, type=str)
parser.add_argument('--subset_size', type=int)
parser.add_argument('--seed', default=2024, type=int)

parser.add_argument('--spacy_model_name', default='en_core_web_sm')

# Image mining option
parser.add_argument('--similarity_model_name_or_path', default='openai/clip-vit-large-patch14')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--image_top_k', default=8, type=int)
parser.add_argument('--max_num_texts', default=5, type=int)
parser.add_argument('--rng', default=None)

# Export option
parser.add_argument('--export_fname', required=True, type=str)


def setup_seeds(seed: int) -> None:
    seed = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # NOTE : no need to set up cudnn 
    # cudnn.benchmark = False
    # cudnn.deterministic = True

def swap_spans(tokens: List[Token], span1: Span, span2: Span) -> List[Token]:
    """
    두 명사구의 위치를 교환합니다.

    Returns:
    - List[Token]: 위치가 교환된 새로운 Token 리스트
    """

    start1, end1 = span1.start, span1.end
    start2, end2 = span2.start, span2.end

    if start1 < start2:
        return tokens[:start1] + tokens[start2:end2] + tokens[end1:start2] + tokens[start1:end1] + tokens[end2:]
    else:
        return tokens[:start2] + tokens[start1:end1] + tokens[end2:start1] + tokens[start2:end2] + tokens[end1:]


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

    # TODO : need to test
    # Why nested loop?
    for hard_text in hard_texts:
        # for hard_caption in hard_text:
        logger.debug(f'hard_text : {hard_text}')
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

def negative_image_mining(dataset: Dataset) -> Dataset:   
    if os.path.isfile(similarity_dict_cahe_file_name):
        with open(similarity_dict_cahe_file_name, 'rb') as f:
            similarity_dict = pickle.load(f)
        logger.debug('Similarity dictionary loaded')
    else:
        image_similarity_calculator = ImageSimilarityCalculator(
            similarity_model_name_or_path=args.similarity_model_name_or_path,
            batch_size=args.batch_size,
            top_k=args.image_top_k,
        )
        torch.cuda.empty_cache()
        similarity_dict = image_similarity_calculator.compute_image_similarity(dataset=dataset)
        with open(similarity_dict_cahe_file_name, 'wb') as f:
            pickle.dump(similarity_dict, f)
        logger.debug('Similarity dictionary computed')
    
    new_examples: List[Dict[str, str]] = []
    for idx, example in tqdm(enumerate(dataset), desc='Add hard text/images'):

        similar_indices = similarity_dict[idx]
        hard_images = [dataset[idx]['images'] for idx in similar_indices]
        hard_texts = [dataset[idx]['text'] for idx in similar_indices]

        # NOTE : Dataset format
        # images : need to be url or path of string
        # "text" should be list of string, that is , List[str] for process_example function
        # "hard_texts" should be list of string
        new_examples.append({
            'images': example['images'],
            'text': [example["text"]],
            'hard_images': hard_images,
            'hard_texts': hard_texts
        })

    return Dataset.from_list(new_examples)


def negative_text_mining(self, dataset: Dataset) -> Dataset:
    nlp = spacy.load(args.spacy_model_name)

    # 시스템의 CPU 코어 수에 따라 num_proc를 설정
    max_num_proc = int(os.cpu_count() / 2)

    # Dataset의 각 요소에 대해 process_example 함수를 적용
    dataset_with_neg_text = dataset.map(
        lambda example: process_example(example, nlp, generate_negative_captions),
        remove_columns=dataset.column_names,
        num_proc=max_num_proc,  # 병렬로 처리할 프로세스 개수 설정
        desc="Negative Text Mining"
    )

    # 필터링: 'neg_texts'와 'hard_neg_texts'가 빈 리스트가 아닌 경우만 유지
    dataset_with_neg_text = dataset_with_neg_text.filter(
        lambda x: len(x['neg_texts']) > 0 and len(x['hard_neg_texts']) > 0)

    return dataset_with_neg_text

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

    for _ in range(args.max_num_texts):  # 최대 5개의 부정적인 캡션 생성
        # 스왑할 수 있는 그룹 중에서 무작위로 하나 선택
        eligible_groups = [group for group in swap_groups.values() if len(group) >= 2]
        if not eligible_groups:
            break  # 스왑할 요소가 부족한 경우 중단

        group_index = args.rng.choice(len(eligible_groups))
        group = eligible_groups[group_index]
        indices = args.rng.choice(len(group), 2, replace=False)
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
    logger.debug(f'negative_captions : {negative_captions}')
    return negative_captions


if __name__ == '__main__':
    args = parser.parse_args()
    if args.rng is None:
        args.rng = np.random.default_rng(args.seed)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    assert os.getenv('DATA_ROOT_DIR')
    assert os.getnev('TARGET_DATASET')
    assert os.getnev('SIMILARITY_DICT_FILE')
    
    os.makedirs(f'{os.getenv("DATA_ROOT_DIR")}/{os.getenv("TARGET_DATASET")}', exist_ok=True)
    similarity_dict_cahe_file_name = f'{os.getenv("DATA_ROOT_DIR")}/{os.getenv("TARGET_DATASET")}/similarity_dict/{os.getnev("SIMILARITY_DICT_FILE")}.pickle'

    setup_seeds(args.seed)

    # dataset = load_dataset("pixparse/cc3m-wds", trust_remote_code=True, split='train')
    dataset = load_dataset(args.raw_dataset_url_or_path, trust_remote_code=True, split='train')
    logger.debug(f'Raw dataset:\n{dataset}')
    
    # For CC3M
    # __key__ : local file path
    # jpg : PIL images which are heavy! We cannot use them at all.
    dataset = dataset.rename_columns({"__key__": 'images', "txt": 'text'})
    dataset = dataset.select_columns(['images', 'text'])
    logger.debug(f'Renamed dataset:\n{dataset}')

    logger.debug('Image mining start')
    dataset = negative_image_mining(dataset)
    logger.debug(f'Negative image minded dataset :\n{dataset}')

    if args.subset_size:
        logger.debug('subsetting start')
        indices = np.random.choice(len(dataset), 500000, replace=False)
        dataset = dataset.select(indices)
        logger.debug(f'Sampled dataset:\n{dataset}')
    else:
        indices = None
    
    logger.debug(f'Negative text mining begin')
    dataset = negative_text_mining(dataset)
    logger.debug(f'Negative text mined dataset:\n{dataset}')

    dataset.to_parquet(args.export_fname)