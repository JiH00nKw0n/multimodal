import os
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import TrainingArguments

from src.utils import get_rank, init_distributed_mode
from src.common import BaseTrainer, TrainConfig
import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument('--resume-from-checkpoint', type=str, required=False, default=None)

    args = parser.parse_args()

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main() -> None:

    args = parse_args()

    init_distributed_mode(args)

    setup_seeds(args.seed)

    wandb.login(key=args.wandb_key)

    train_cfg = TrainConfig(**args)

    model = None
    processor = None
    train_dataset = None
    data_collator = None
    trainer = BaseTrainer(
        model=model,
        args=TrainingArguments(train_cfg.trainer_config),
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    wandb.finish()

    trainer.save_model()

    return None
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"
    #
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    # job_id = now()
    #
    # cfg = Config(parse_args())
    #
    # init_distributed_mode(cfg.run_cfg)
    #
    # setup_seeds(cfg)
    #
    # # set after init_distributed_mode() to only log on master.
    # setup_logger()
    #
    # cfg.pretty_print()
    #
    # # append job_id to cfg
    # cfg.job_id = job_id
    #
    # task = tasks.setup_task(cfg)
    # datasets = task.build_datasets(cfg)
    # model = task.build_model(cfg)
    #
    # runner = get_runner_class(cfg)(
    #     cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    # )
    # runner.train()


if __name__ == "__main__":
    main()
