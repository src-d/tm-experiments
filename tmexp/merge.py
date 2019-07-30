from argparse import ArgumentParser
import os
import pickle
from typing import Any, Dict, List

from .cli import CLIBuilder, register_command
from .io_constants import DATASET_DIR
from .utils import check_file_exists, check_remove, create_logger, recursive_update


def _define_parser(parser: ArgumentParser) -> None:
    cli_builder = CLIBuilder(parser)
    cli_builder.add_force_arg()
    parser.add_argument(
        "-i", "--input-datasets", help="Datasets to merge.", nargs="*", required=True
    )
    parser.add_argument(
        "-o", "--output-dataset", help="Name of the output dataset.", required=True
    )


@register_command(parser_definer=_define_parser)
def merge(
    input_datasets: List[str], output_dataset: str, force: bool, log_level: str
) -> None:
    """Merge multiple datasets."""
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
        for repo, refs in input_dict["refs"].items():
            if len(refs) > 1:
                logger.warning(
                    "Found %d references for repository %s. Please make sure you "
                    "intended to merge several revisions.",
                    len(refs),
                    repo,
                )
        recursive_update(output_dict, input_dict)
    logger.info("Merged %d datasets." % len(input_paths))

    logger.info("Saving merged dataset ...")
    with open(output_path, "wb") as fout:
        pickle.dump(output_dict, fout)
    logger.info("Saved merged dataset in '%s'." % output_path)
