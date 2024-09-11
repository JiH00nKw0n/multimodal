import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import logging
import os

from src.utils import get_rank, init_distributed_mode, now, load_yml
from src.common import EvaluateConfig, setup_logger, CustomWandbCallback
import src.tasks as tasks

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluating")

    parser.add_argument("--cfg-path", required=False, help="path to configuration file.")

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

    args = parse_args()

    # 파일 핸들러 생성
    file_handler = logging.FileHandler(f'/{os.getenv("LOG_DIR")}/clip_{job_id}.log/')
    file_handler.setLevel(logging.DEBUG)

    # 로그 메시지 포맷 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 핸들러를 로거에 추가
    logger.addHandler(file_handler)

    evaluate_cfg = EvaluateConfig(**load_yml(args.cfg_path))
    init_distributed_mode(args)
    setup_seeds(evaluate_cfg.run_config.seed)
    setup_logger()

    task = tasks.setup_task(evaluate_cfg)

    evaluator = task.build_evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)