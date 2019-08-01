from argparse import ArgumentParser
import os
import pickle
from typing import List

from .cli import CLIBuilder, register_command
from .io_constants import Dataset, DATASET_DIR
from .utils import check_file_exists, check_remove, create_logger, recursive_update


def _define_parser(parser: ArgumentParser) -> None:
    cli_builder = CLIBuilder(parser)
    cli_builder.add_force_arg()
    cli_builder.add_dataset_arg(required=False)
    parser.add_argument(
        "-i", "--input-datasets", help="Datasets to merge.", nargs="*", required=True
    )


@register_command(parser_definer=_define_parser)
def merge(
    input_datasets: List[str], dataset_name: str, force: bool, log_level: str
) -> None:
    """Merge multiple datasets."""
    if len(input_datasets) < 2:
        raise RuntimeError("Less then 2 datasets were given, aborting.")

    logger = create_logger(log_level, __name__)

    output_path = os.path.join(DATASET_DIR, dataset_name + ".pkl")
    check_remove(output_path, logger, force)

    input_paths = [
        os.path.join(DATASET_DIR, input_dataset + ".pkl")
        for input_dataset in input_datasets
    ]
    for input_path in input_paths:
        check_file_exists(input_path)

    logger.info("Merging datasets ...")
    output_dataset: Dataset = Dataset()
    for dataset_name, input_path in zip(input_datasets, input_paths):
        with open(input_path, "rb") as fin:
            input_dataset: Dataset = pickle.load(fin)
        recursive_update(output_dataset.files_info, input_dataset.files_info)
        recursive_update(output_dataset.files_content, input_dataset.files_content)

        for repo, refs in input_dataset.refs.items():
            if len(refs) > 1:
                logger.warning(
                    "Found %d references for repository %s. Please make sure you "
                    "intended to merge several revisions.",
                    len(refs),
                    repo,
                )
            if repo not in output_dataset.refs:
                logger.info("Adding new repository '%s' from '%s'.", repo, dataset_name)
                output_dataset.refs[repo] = input_dataset.refs[repo]
            else:
                if output_dataset.refs[repo] != input_dataset.refs[repo]:
                    logger.error(
                        "Discrepancy between references for repository '%s', data from "
                        "'%s' will not be added.",
                        repo,
                        dataset_name,
                    )
                    output_dataset.files_info.pop(repo)
                    output_dataset.files_content.pop(repo)
                    continue
                logger.info(
                    "Adding data to repository '%s' from '%s'.", repo, input_path
                )

    logger.info("Merged %d datasets." % len(input_paths))

    logger.info("Saving merged dataset ...")
    with open(output_path, "wb") as fout:
        pickle.dump(output_dataset, fout)
    logger.info("Saved merged dataset in '%s'." % output_path)
