import logging
import src.utils.dist_utils as dist_utils

__all__ = ["setup_logger"]


def setup_logger():
    logging.basicConfig(
        level=logging.INFO if dist_utils.is_main_process() else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
