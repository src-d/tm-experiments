import os
import pickle
from typing import Any, Dict, List

from .io_constants import DATASET_DIR
from .utils import check_file_exists, check_remove, create_logger


def merge_datasets(
    input_datasets: List[str], output_dataset: str, force: bool, log_level: str
) -> None:

    if len(input_datasets) < 2:
        raise RuntimeError("Less then 2 datasets were given, aborting.")

    logger = create_logger(log_level, __name__)

    output_path = os.path.join(DATASET_DIR, output_dataset + ".pkl")
    check_remove(output_path, logger, force)

    input_paths = [
        os.path.join(DATASET_DIR, input_dataset + ".pkl")
        for input_dataset in input_datasets
    ]
    for input_path in input_paths:
        check_file_exists(input_path)

    logger.info("Merging datasets ...")
    output_dict: Dict[str, Any] = {}
    for input_path in input_paths:
        with open(input_path, "rb") as fin:
            input_dict = pickle.load(fin)
        if "refs" not in output_dict:
            output_dict.update(input_dict)
        else:
            output_dict["files_info"].update(input_dict["files_info"])
            output_dict["files_content"].update(input_dict["files_content"])
    logger.info("Merged %d datasets." % len(input_paths))

    logger.info("Saving merged dataset ...")
    with open(output_path, "wb") as fout:
        pickle.dump(output_dict, fout)
    logger.info("Saved merged dataset in '%s'." % output_path)
