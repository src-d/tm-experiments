from collections import Counter, defaultdict
import logging
import os
import pickle
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

import numpy as np

from .gitbase_constants import SUPPORTED_LANGUAGES

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
    if dataset_name == "":
        dataset_name == topic_model
    vocab_output_path = os.path.join(output_dir, "vocab." + dataset_name + ".txt")
    if os.path.exists(vocab_output_path):
        raise RuntimeError(
            "File {} already exists, aborting.".format(vocab_output_path)
        )
    docword_output_path = os.path.join(output_dir, "docword." + dataset_name + ".txt")
    if os.path.exists(docword_output_path):
        raise RuntimeError(
            "File {} already exists, aborting.".format(docword_output_path)
        )
    docs_output_path = os.path.join(output_dir, "docs." + dataset_name + ".txt")
    if os.path.exists(docs_output_path):
        raise RuntimeError("File {} already exists, aborting.".format(docs_output_path))
