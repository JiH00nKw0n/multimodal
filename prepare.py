"""
    TODO : Add image path/url 
    TODO : Add formatting into common architecture

"""

import json
import argparse
import pandas as pd
from datasets import Dataset


parser = argparse.ArgumentParser()

parser.add_argument('--is_crepe', action=argparse.BooleanOptionalAction)
parser.add_argument('--crepe_negative_type', type=str, choices=['swap', 'negate', 'atom'])
parser.add_argument('--crepe_complexity', type=int, choices=[4, 5, 6, 7, 8, 9, 10, 11, 12])

parser.add_argument('--is_sugarCrepe', action=argparse.BooleanOptionalAction)
parser.add_argument('--sugarCrepe_negative_type', type=str)

parser.add_argument('--is_svo', action=argparse.BooleanOptionalAction)

parser.add_argument('--is_aro', action=argparse.BooleanOptionalAction)


EXPORT_DIR = '/data/yjkim/mm/local_dataset/hf_datasets'
DATA_DIR_PREFIX = '{data_dir}/'
IMAGE_URL_OR_PATH_KEY = 'image_url_or_path'

def process_crepe(negative_type: str, complexity: int):
    """
    negative_type : swap, negate, atom
    complexity : 4, 5, 6, 7, 8, 9, 10, 11, 12
    """
    
    def _formatting(examples):
        # TODO : 이미지 경로 처리
        examples[IMAGE_URL_OR_PATH_KEY] = DATA_DIR_PREFIX + f'{examples[csv_img_key]}.png'
        return examples
    _CREPE_DATA_DIR = '/data/yjkim/mm/local_dataset/crepe/prod_hard_negatives'    
    csv_caption_key = 'caption'
    csv_img_key = 'image_id'
    df = pd.read_csv(f'{_CREPE_DATA_DIR}/{negative_type}/prod_vg_hard_negs_{negative_type}_complexity_{complexity}.csv')
    print(df.info())
    print(df.head())

    images = df[csv_img_key].tolist()
    captions = df[csv_caption_key].tolist()        

    print(images[:3])
    print(captions[:3])

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(_formatting, batched=False)

    print(dataset)
    print(dataset[0])
    
    dataset.to_parquet(f'{EXPORT_DIR}/CREPE_negs_{negative_type}_complexity_{complexity}.parquet')


def process_sugarCrepe(negative_type: str):
    """
    negative_type : add_att, add_obj, replace_att, replace_obj, replace_rel, swap_att, swap_obj
    """
    def _formatting(examples):
        # TODO : 이미지 경로 처리
        examples[IMAGE_URL_OR_PATH_KEY] = DATA_DIR_PREFIX + examples['filename']
        return examples

    _SUGARCREPE_DATA_DIR = '/home/yjkim/multimodal/data/sugar_crepe'  
    data_path = f'{_SUGARCREPE_DATA_DIR}/{negative_type}.json'  
    df = json.load(open(data_path, 'r', encoding='utf-8'))
    df = pd.DataFrame(list(df.values()))
    
    print(df.info())
    print(df.head())

    dataset = Dataset.from_pandas(df)

    dataset = dataset.map(_formatting, batched=False)

    print(dataset)
    print(dataset[0])

    dataset.to_parquet(f'{EXPORT_DIR}/SUGARCREPE_negs_{negative_type}.parquet')


def process_svo():
    def _formatting(examples):
        # TODO : 이미지 경로 처리
        # Positive image와 negative image를 어떻게 처리할 것인가?
        examples[IMAGE_URL_OR_PATH_KEY] = examples['pos_url']
        # examples[IMAGE_URL_OR_PATH_KEY] = examples['neg_url']
        return examples

    fpath = '/home/yjkim/multimodal/data/svo_probes.csv'
    df = pd.read_csv(fpath)

    print(df.info())
    print(df.head())

    dataset = Dataset.from_pandas(df)

    dataset = dataset.map(_formatting, batched=False)

    print(dataset)
    print(dataset[0])

    dataset.to_parquet(f'{EXPORT_DIR}/SVO.parquet')


def process_aro():
    def _formatting(examples):
        # TODO : 이미지 경로 처리
        examples[IMAGE_URL_OR_PATH_KEY] = DATA_DIR_PREFIX + f'{examples["image_id"]}.jpg'
        return examples

    fpath = '/data/yjkim/mm/local_dataset/aro/visual_genome_relation.json'
    with open(fpath, "r") as f:
        df = pd.DataFrame(json.load(f))
    print(df.info())
    print(df.head())

    dataset = Dataset.from_pandas(df)

    dataset = dataset.map(_formatting, batched=False)

    print(dataset)
    print(dataset[0])

    dataset.to_parquet(f'{EXPORT_DIR}/SVO.parquet')


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