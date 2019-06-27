import argparse
import logging
from typing import Any

from .create_bow import create_bow, DIFF_MODEL, HALL_MODEL
from .gitbase_constants import COMMENTS, IDENTIFIERS, LITERALS, SUPPORTED_LANGUAGES
from .preprocess import preprocess


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
    add_lang_args(preprocess_parser)
    add_feature_arg(preprocess_parser)
    add_force_arg(preprocess_parser)
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
        "-o",
        "--output-path",
        help="Output path for the pickled dict.",
        required=True,
        type=str,
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
        "--gitbase-host", help="Gitbase hostname.", type=str, default="0.0.0.0"
    )
    preprocess_parser.add_argument(
        "--gitbase-port", help="Gitbase port.", type=int, default=3306
    )
    preprocess_parser.add_argument(
        "--gitbase-user", help="Gitbase user.", type=str, default="root"
    )
    preprocess_parser.add_argument(
        "--gitbase-pass", help="Gitbase password.", type=str, default=""
    )
    preprocess_parser.add_argument(
        "--bblfsh-host", help="Babelfish hostname.", type=str, default="0.0.0.0"
    )
    preprocess_parser.add_argument(
        "--bblfsh-port", help="Babelfish port.", type=int, default=9432
    )

    # ------------------------------------------------------------------------

    create_bow_parser = subparsers.add_parser(
        "create_bow", help="Create the BoW dataset from a pickled dict, in UCI format."
    )
    create_bow_parser.set_defaults(handler=create_bow)
    add_lang_args(create_bow_parser)
    add_feature_arg(create_bow_parser)
    add_force_arg(create_bow_parser)
    create_bow_parser.add_argument(
        "-i",
        "--input-path",
        help="Input path for the pickled dict.",
        required=True,
        type=str,
    )
    create_bow_parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory for the BoW files.",
        required=True,
        type=str,
    )
    create_bow_parser.add_argument(
        "--dataset-name",
        help="Name of the dataset, used for filenames, defaults to chosen topic-model.",
        type=str,
        default=None,
    )
    create_bow_parser.add_argument(
        "--topic-model",
        help="Topic evolution model to use.",
        required=True,
        type=str,
        choices=[DIFF_MODEL, HALL_MODEL],
    )
    create_bow_parser.add_argument(
        "--min-word-frac",
        help="Words occuring in less then this percentage of all documents are removed,"
        " default to 2 %.",
        type=float,
        default=0.02,
    )
    create_bow_parser.add_argument(
        "--max-word-frac",
        help="Words occuring in more then this percentage of all documents are removed,"
        " default to 80 %.",
        type=float,
        default=0.8,
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
