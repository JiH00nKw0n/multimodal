import os
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb
import logging
import yaml

from src.common import TrainConfig, setup_logger
import src.tasks as tasks

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Negative Mining")

    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--export_fname", required=True)
    parser.add_argument("--cfg-path", required=False, help="path to configuration file.")

    args = parser.parse_args()

    return args


def setup_seeds(seed: int) -> None:
    seed = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main() -> None:
    args = parse_args()
    
    # Export 공간 준비
    os.makedirs(args.root_dir, exist_ok=True)

    # 파일 핸들러 생성
    file_handler = logging.FileHandler(f'{args.root_dir}/negative_mining.log')
    file_handler.setLevel(logging.DEBUG)

    # 로그 메시지 포맷 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 핸들러를 로거에 추가
    logger.addHandler(file_handler)
    
    
    with open(args.cfg_path, "r") as f:
        config = yaml.safe_load(f)
    train_cfg = TrainConfig(**config)
    setup_seeds(train_cfg.run_config.seed)
    setup_logger()

    task = tasks.setup_task(train_cfg)

    datasets = task.build_datasets()
    logger.info(f'Datasets\n{datasets}')
    datasets.to_parquet(f'{args.root_dir}/{args.export_fname}.parquet')


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
