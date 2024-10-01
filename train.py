import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

import src.tasks as tasks
from src.common import TrainConfig, setup_logger, CustomWandbCallback, Logger
from src.utils import get_rank, init_distributed_mode, now, load_yml

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=False, help="path to configuration file.")
    parser.add_argument('--wandb-key', type=str, required=False, help="weights & biases key.")
    parser.add_argument('--resume-from-checkpoint', type=str, required=False, default=None)

    args = parser.parse_args()

    return args


def setup_seeds(seed: int) -> None:
    seed = seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main() -> None:
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    setup_logger()

    args = parse_args()
    file_name = args.cfg_path.split('/')[-1].replace('.yml', '')
    file_handler = logging.FileHandler(f'{os.getenv("LOG_DIR")}/{file_name}_{job_id}.log/')
    file_handler.setLevel(logging.DEBUG)

    # 로그 메시지 포맷 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 핸들러를 로거에 추가
    logger.addHandler(file_handler)

    args = parse_args()
    if args.debug:
        logger.basicConfig(level=logging.DEBUG)
    else:
        logger.basicConfig(level=logging.INFO)

    train_cfg = TrainConfig(**load_yml(args.cfg_path))

    init_distributed_mode(args)
    setup_seeds(train_cfg.run_config.seed)

    wandb.login(key=args.wandb_key)

    task = tasks.setup_task(train_cfg)

    trainer = task.build_trainer()
    trainer.add_callback(CustomWandbCallback())
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    wandb.finish()
    trainer.save_model()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
