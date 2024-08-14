import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb
import logging

from src.utils import get_rank, init_distributed_mode, now, load_yml
from src.common import TrainConfig, setup_logger, CustomWandbCallback
import src.tasks as tasks

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    args = parse_args()

    train_cfg = TrainConfig(**load_yml(args.cfg_path))
    init_distributed_mode(args)
    setup_seeds(train_cfg.run_config.seed)
    setup_logger()

    wandb.login(key=args.wandb_key)

    task = tasks.setup_task(train_cfg)

    trainer = task.build_trainer()
    trainer.add_callback(CustomWandbCallback())
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    wandb.finish()
    trainer.save_model()


if __name__ == "__main__":
    main()
