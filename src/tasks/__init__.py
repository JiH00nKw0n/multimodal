from .base import BaseTask
from .train_task import TrainTask
from src.common.registry import registry

__all__ = ["BaseTask", "TrainTask"]


def setup_task(config):
    assert "task" in config.task_config, "Task name must be provided."

    task_name = config.task_config.task
    task = registry.get_task_class(task_name).setup_task(config=config)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task
