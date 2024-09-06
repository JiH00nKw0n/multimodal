from transformers.integrations import WandbCallback
import logging

logger = logging.getLogger(__name__)

__all__ = ["CustomWandbCallback"]


class CustomWandbCallback(WandbCallback):
    """
    A custom callback class that extends the functionality of `WandbCallback` from Hugging Face's Transformers library.
    This class integrates Weights and Biases (Wandb) logging into the training process, while allowing for custom behavior
    during the logging of metrics.

    Methods:
        on_log(args, state, control, model=None, logs=None, **kwargs):
            Custom behavior that occurs when logging metrics during training.
    """

    def __init__(self):
        """
        Initializes the `CustomWandbCallback` class by calling the parent `WandbCallback` class's constructor.
        Ensures Wandb integration is set up properly.
        """
        super().__init__()

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """
        Overrides the `on_log` method from `WandbCallback` to implement custom behavior when logging metrics
        during training. This method can be further extended to log additional information or perform
        actions based on the logged data.

        Args:
            args:
                The training arguments.
            state:
                The current training state, including information like the current step and number of completed epochs.
            control:
                Control flow object that can modify the behavior of training.
            model (Optional):
                The model being trained (if any).
            logs (Optional):
                A dictionary of metrics and logs that are being logged at the current step.
            **kwargs:
                Additional keyword arguments passed to the function.

        This method calls the parent class's `on_log` method to ensure Wandb logging functionality remains intact.
        """
        super().on_log(args, state, control, model, logs, **kwargs)