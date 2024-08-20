from transformers.integrations import WandbCallback
import logging

logger = logging.getLogger(__name__)

__all__ = ["CustomWandbCallback"]


class CustomWandbCallback(WandbCallback):
    def __init__(self):
        super().__init__()

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        super().on_log(args, state, control, model, logs, **kwargs)