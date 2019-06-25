import logging
import os
from typing import List, Optional

DIFF_MODEL = "diff"
HALL_MODEL = "hall"


def create_bow(
    input_path: str,
    output_dir: str,
    dataset_name: str,
    langs: Optional[List[str]],
    exclude_langs: Optional[List[str]],
    features: List[str],
    topic_model: str,
    thresh: Optional[float],
    tfidf: Optional[bool],
    log_level: str,
) -> None:
    logger = logging.getLogger("create_bow")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(log_level)

    if not os.path.exists(input_path):
        raise RuntimeError("File {} does not exists, aborting.".format(input_path))
    if not (output_dir == "" or os.path.exists(output_dir)):
        logger.warn("Creating directory {}.".format(output_dir))
        os.makedirs(output_dir)
