import argparse
import logging
from typing import Any

from .create_bow import create_bow, DIFF_MODEL, HALL_MODEL
from .preprocess import COMMENTS, IDENTIFIERS, LITERALS, preprocess
from .train_hdp import train_hdp
from .utils import SUPPORTED_LANGUAGES


def add_lang_args(cmd_parser: argparse.ArgumentParser) -> None:
    lang_group = cmd_parser.add_mutually_exclusive_group()
    lang_group.add_argument(
        "--select-langs",
        help="To select a perticular set of languages, defaults to all.",
        nargs="*",
        dest="langs",
        choices=SUPPORTED_LANGUAGES,
    )
    lang_group.add_argument(
        "--exclude-langs",
        help="To exclude a perticular set of languages, defaults to none.",
        nargs="*",
        choices=SUPPORTED_LANGUAGES,
    )


def add_feature_arg(cmd_parser: argparse.ArgumentParser) -> None:
    cmd_parser.add_argument(
        "--features",
        help="To select which tokens to use as words, defaults to all.",
        nargs="*",
        choices=[COMMENTS, IDENTIFIERS, LITERALS],
        default=[COMMENTS, IDENTIFIERS, LITERALS],
    )


def add_force_arg(cmd_parser: argparse.ArgumentParser) -> None:
    cmd_parser.add_argument(
        "-f",
        "--force",
        help="Delete and replace existing output(s).",
        action="store_true",
    )


def add_dataset_arg(cmd_parser: argparse.ArgumentParser) -> None:
    cmd_parser.add_argument(
        "--dataset-name",
        help="Name of the dataset created by `preprocess`, defaults"
        " to 'MM-DD-HH:MM-dataset'.",
        default=None,
        type=str,
    )


def add_bow_arg(cmd_parser: argparse.ArgumentParser) -> None:
    cmd_parser.add_argument(
        "--bow-name",
        help="Name of the BoW created by `create_bow`, defaults"
        " to 'MM-DD-HH:MM-bow'.",
        default=None,
        type=str,
    )


def add_experiment_arg(cmd_parser: argparse.ArgumentParser) -> None:
    cmd_parser.add_argument(
        "--exp-name",
        help="Name of the experiment created by `train_$`, defaults"
        " to 'MM-DD-HH:MM-experiment'.",
        default=None,
        type=str,
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=logging._nameToLevel,
        help="Logging verbosity.",
    )
    # Create and construct subparsers

    subparsers = parser.add_subparsers(help="Commands")

    # ------------------------------------------------------------------------

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Extract the raw feature count per file from all tagged refs of a"
        "repository and store them as a pickled dict.",
    )
    preprocess_parser.set_defaults(handler=preprocess)

    add_dataset_arg(preprocess_parser)
    add_feature_arg(preprocess_parser)
    add_force_arg(preprocess_parser)
    add_lang_args(preprocess_parser)

    preprocess_parser.add_argument(
        "-r", "--repo", help="Name of the repo to preprocess.", type=str, required=True
    )
    preprocess_parser.add_argument(
        "--exclude-refs",
        help="All refs containing one of these keywords will be excluded "
        "(e.g. all refs with `alpha`).",
        nargs="*",
        default=[],
    )
    preprocess_parser.add_argument(
        "--only-by-date",
        help="To sort the references only by date (may cause errors).",
        action="store_true",
    )
    preprocess_parser.add_argument(
        "--version-sep",
        help="If sorting by version, provide the seperator between major and minor.",
        type=str,
        default=".",
    )
    preprocess_parser.add_argument(
        "--no-tokenize",
        help="To skip tokenization.",
        dest="tokenize",
        action="store_false",
    )
    preprocess_parser.add_argument(
        "--no-stem", help="To skip stemming.", dest="stem", action="store_false"
    )
    preprocess_parser.add_argument(
        "--bblfsh-timeout",
        help="Timeout for parse requests made to Babelfish.",
        type=float,
        default=10.0,
    )
    # ------------------------------------------------------------------------

    create_bow_parser = subparsers.add_parser(
        "create_bow", help="Create the BoW dataset from a pickled dict, in UCI format."
    )
    create_bow_parser.set_defaults(handler=create_bow)

    add_bow_arg(create_bow_parser)
    add_dataset_arg(create_bow_parser)
    add_feature_arg(create_bow_parser)
    add_force_arg(create_bow_parser)
    add_lang_args(create_bow_parser)

    create_bow_parser.add_argument(
        "--topic-model",
        help="Topic evolution model to use.",
        required=True,
        type=str,
        choices=[DIFF_MODEL, HALL_MODEL],
    )
    create_bow_parser.add_argument(
        "--min-word-frac",
        help="Words occuring in less then this draction of all documents are removed,"
        " defaults to %(default)s.",
        type=float,
        default=0.02,
    )
    create_bow_parser.add_argument(
        "--max-word-frac",
        help="Words occuring in more then this fraction of all documents are removed,"
        " defaults to %(default)s.",
        type=float,
        default=0.8,
    )
    # ------------------------------------------------------------------------

    train_hdp_parser = subparsers.add_parser(
        "train_hdp", help="Train an HDP model from the input BoW."
    )
    train_hdp_parser.set_defaults(handler=train_hdp)

    add_bow_arg(train_hdp_parser)
    add_experiment_arg(train_hdp_parser)
    add_force_arg(train_hdp_parser)

    train_hdp_parser.add_argument(
        "--chunk-size", help="Number of documents in one chunk.", default=256, type=int
    )
    train_hdp_parser.add_argument(
        "--kappa",
        help="Learning parameter which acts as exponential decay factor to influence "
        "extent of learning from each batch.",
        default=1.0,
        type=float,
    )
    train_hdp_parser.add_argument(
        "--tau",
        help="Learning parameter which down-weights early iterations of documents.",
        default=64.0,
        type=float,
    )
    train_hdp_parser.add_argument(
        "--K", help="Document level truncation level.", default=15, type=int
    )
    train_hdp_parser.add_argument(
        "--T", help="Topic level truncation level.", default=150, type=int
    )
    train_hdp_parser.add_argument(
        "--alpha", help="Document level concentration.", default=1, type=int
    )
    train_hdp_parser.add_argument(
        "--gamma", help="Topic level concentration.", default=1, type=int
    )
    train_hdp_parser.add_argument(
        "--eta", help="Topic Dirichlet.", default=0.01, type=float
    )
    train_hdp_parser.add_argument(
        "--scale",
        help="Weights information from the mini-chunk of corpus to calculate rhot.",
        default=1.0,
        type=float,
    )
    train_hdp_parser.add_argument(
        "--var-converge",
        help="Lower bound on the right side of convergence.",
        default=0.0001,
        type=float,
    )
    train_hdp_parser.add_argument(
        "--min-proba",
        help="Lower bound on the probability a topic is affected to a document, "
        "defaults to %(default)s.",
        default=0.01,
        type=float,
    )

    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    args.log_level = logging._nameToLevel[args.log_level]
    try:
        handler = args.handler
        delattr(args, "handler")
    except AttributeError:

        def print_usage(_: Any) -> None:
            parser.print_usage()

        handler = print_usage
    return handler(**vars(args))


if __name__ == "__main__":
    main()
