"""
    TODO : Add image path/url 
    TODO : Add formatting into common architecture

"""

import os
import json
import argparse
import pandas as pd
from datasets import Dataset

import logging
from PIL import Image


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler(f'{os.getenv("LOG_DIR")}/prepare.log', 'w'))

parser = argparse.ArgumentParser()

parser.add_argument('--is_crepe', action=argparse.BooleanOptionalAction)
parser.add_argument('--crepe_negative_type', type=str, choices=['swap', 'negate', 'atom'])
parser.add_argument('--crepe_complexity', type=int, choices=[4, 5, 6, 7, 8, 9, 10, 11, 12])

parser.add_argument('--is_sugarCrepe', action=argparse.BooleanOptionalAction)
parser.add_argument('--sugarCrepe_negative_type', type=str)

parser.add_argument('--is_svo', action=argparse.BooleanOptionalAction)

parser.add_argument('--is_aro', action=argparse.BooleanOptionalAction)


EXPORT_DIR = f'{os.getenv("DATA_DIR")}/hf_datasets'
DATA_DIR_PREFIX = '{data_dir}/'
IMAGE_URL_OR_PATH_KEY = 'image'
POSITIVE_CAPTION_KEY = 'text'
HARD_NEGATIVE_CAPTIONS_KEY ='hard_texts'


def cast_type(_instance, _target_type):
    if isinstance(_instance, _target_type):
        return _instance
    else:
        if _target_type == list:
            return [_instance]

def format_logging(*args, **kwargs):
    logger.info('Before')
    logger.info(kwargs['dataset'])
    kwargs['dataset'] = kwargs['dataset'].map(kwargs['_formatting'], batched=False, remove_columns=kwargs['dataset'].column_names)        
    logger.info('After')
    logger.info(kwargs['dataset'])
    logger.info(kwargs['dataset'][0][IMAGE_URL_OR_PATH_KEY])    
    return kwargs['dataset']


def convert_to_rgb(image):
    # PIL.Image.Image 처리
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)

    # np.ndarray 처리
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.concatenate([image] * 3, axis=-1)
        image = torch.from_numpy(image)

    # torch.Tensor 처리
    if isinstance(image, torch.Tensor):
        if image.ndimension() == 3 and image.shape[0] == 1:
            image = image.expand(3, -1, -1)

        elif image.ndimension() == 3 and image.shape[0] == 4:
            image = image[:3, :, :]

        elif image.ndimension() == 2:
            image = image.unsqueeze(0).expand(3, -1, -1)

        if image.dtype != torch.uint8:
            image = image.clamp(0, 255).byte()

    return image


def load_image(url_or_path, is_url=False, is_local=False) -> Image:
    if is_url:
        response = requests.get(url_or_path)
        response.raise_for_status()  # HTTP 오류 상태일 경우 예외 발생
        raw = BytesIO(response.content)
    elif is_local:
        raw = url_or_path
    else:
        raise ValueError()
    img = Image.open(raw).convert("RGB")
    return img


def process_crepe(negative_type: str, complexity: int):
    """
    negative_type : swap, negate, atom
    complexity : 4, 5, 6, 7, 8, 9, 10, 11, 12
    """

    def _formatting(examples):
        # TODO : 이미지 경로 처리
        examples[IMAGE_URL_OR_PATH_KEY] = DATA_DIR_PREFIX + f'{examples[csv_img_key]}.png'
        examples[POSITIVE_CAPTION_KEY] = examples['caption']
        examples[HARD_NEGATIVE_CAPTIONS_KEY] = cast_type(examples['hard_negs'], list)
        return examples
    logger.info(f'='*100)
    logger.info(f'CREPE process')
    _CREPE_DATA_DIR = f'{os.getenv("DATA_DIR")}/crepe/prod_hard_negatives'
    csv_caption_key = 'caption'
    csv_img_key = 'image_id'
    df = pd.read_csv(f'{_CREPE_DATA_DIR}/{negative_type}/prod_vg_hard_negs_{negative_type}_complexity_{complexity}.csv')
    logger.debug(df.info())
    logger.debug(df.head())

    images = df[csv_img_key].tolist()
    captions = df[csv_caption_key].tolist()

    dataset = Dataset.from_pandas(df)

    dataset = format_logging(dataset=dataset, _formatting=_formatting)
    # logger.info('Before')
    # logger.info(dataset)
    # dataset = dataset.map(_formatting, batched=False, remove_colums=dataset.column_names)

    # logger.info('After')
    # logger.info(dataset)
    # logger.info(dataset[0][IMAGE_URL_OR_PATH_KEY])

    # dataset.to_parquet(f'{EXPORT_DIR}/CREPE_negs_{negative_type}_complexity_{complexity}.parquet')


def process_sugarCrepe(negative_type: str):
    """
    negative_type : add_att, add_obj, replace_att, replace_obj, replace_rel, swap_att, swap_obj
    """

    def _formatting(examples):
        # TODO : 이미지 경로 처리
        examples[IMAGE_URL_OR_PATH_KEY] = DATA_DIR_PREFIX + examples['filename']
        examples[IMAGE_URL_OR_PATH_KEY] = load_image(examples[IMAGE_URL_OR_PATH_KEY].format(data_dir=f'{os.getenv("DATA_DIR")}/sugar_crepe/val2017'), is_local=True)
        examples[POSITIVE_CAPTION_KEY] = examples['caption']
        examples[HARD_NEGATIVE_CAPTIONS_KEY] = cast_type(examples['negative_caption'], list)
        return examples
    logger.info(f'='*100)
    logger.info(f'Sugar CREPE process')

    _SUGARCREPE_DATA_DIR = f'{os.getenv("DATA_DIR")}/sugar_crepe'
    data_path = f'{_SUGARCREPE_DATA_DIR}/{negative_type}.json'
    df = json.load(open(data_path, 'r', encoding='utf-8'))
    df = pd.DataFrame(list(df.values()))

    logger.debug(df.info())
    logger.debug(df.head())

    dataset = Dataset.from_pandas(df)

    dataset = format_logging(dataset=dataset, _formatting=_formatting)

    dataset.to_parquet(f'{EXPORT_DIR}/SUGARCREPE_negs_{negative_type}.parquet')
    logger.info(f'saved at {EXPORT_DIR}/SUGARCREPE_negs_{negative_type}.parquet')


def process_svo():
    def _formatting(examples):
        # TODO : 이미지 경로 처리 및 어떻게 해야할까....
        # Positive image와 negative image를 어떻게 처리할 것인가?
        examples[IMAGE_URL_OR_PATH_KEY] = examples['pos_url']
        # examples[IMAGE_URL_OR_PATH_KEY] = examples['neg_url']        
        return examples
    logger.info(f'='*100)
    logger.info(f'SVO process')

    fpath = f'{os.getenv("DATA_DIR")}/svo/svo_probes.csv'
    df = pd.read_csv(fpath)

    logger.debug(df.info())
    logger.debug(df.head())

    dataset = Dataset.from_pandas(df)

    dataset = format_logging(dataset=dataset, _formatting=_formatting)
    # logger.info('Before')
    # logger.info(dataset)
    # dataset = dataset.map(_formatting, batched=False, remove_colums=dataset.column_names)

    # logger.info('After')
    # logger.info(dataset)
    # logger.info(dataset[0][IMAGE_URL_OR_PATH_KEY])

    # dataset.to_parquet(f'{EXPORT_DIR}/SVO.parquet')


def process_aro():
    def _formatting(examples):
        # TODO : 이미지 경로 처리
        examples[IMAGE_URL_OR_PATH_KEY] = DATA_DIR_PREFIX + f'{examples["image_id"]}.jpg'
        examples[POSITIVE_CAPTION_KEY] = examples['true_caption']
        examples[HARD_NEGATIVE_CAPTIONS_KEY] = cast_type(examples['false_caption'], list)
        return examples
    logger.info(f'='*100)
    logger.info(f'ARO process')

    fpath = f'{os.getenv("DATA_DIR")}/aro/visual_genome_relation.json'
    with open(fpath, "r") as f:
        df = pd.DataFrame(json.load(f))
    logger.debug(df.info())
    logger.debug(df.head())

    dataset = Dataset.from_pandas(df)
    
    dataset = format_logging(dataset=dataset, _formatting=_formatting)
    # logger.info('Before')
    # logger.info(dataset)
    # dataset = dataset.map(_formatting, batched=False, remove_colums=dataset.column_names)

    # logger.info('After')
    # logger.info(dataset)
    # logger.info(dataset[0][IMAGE_URL_OR_PATH_KEY])

    # dataset.to_parquet(f'{EXPORT_DIR}/SVO.parquet')


if __name__ == '__main__':
    args = parser.parse_args()

    if args.is_crepe:
        process_crepe(negative_type=args.crepe_negative_type, complexity=args.crepe_complexity)

    if args.is_sugarCrepe:
        process_sugarCrepe(negative_type=args.sugarCrepe_negative_type)

    if args.is_aro:
        process_aro()

    if args.is_svo:
        process_svo()
