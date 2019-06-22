import argparse
import logging
from typing import Any

from .gitbase_constants import COMMENTS, IDENTIFIERS, LITERALS, SUPPORTED_LANGUAGES
from .preprocess import preprocess


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
        help="Extract the feature count per document and store them as a pickled dict.",
    )
    preprocess_parser.set_defaults(handler=preprocess)
    preprocess_parser.add_argument(
        "-r", "--repo", help="Name of the repo to preprocess.", type=str, required=True
    )
    preprocess_parser.add_argument(
        "--batch-size",
        help="Select a batch size for processing files, no batching by default.",
        type=int,
        default=0,
    )
    preprocess_parser.add_argument(
        "--batch-dir",
        help="Temporary directory for saving batches, no saving by default.",
        type=str,
        default="",
    )
    preprocess_parser.add_argument(
        "-o",
        "--output",
        help="Output path for the pickled dict.",
        required=True,
        type=str,
    )
    lang_group = preprocess_parser.add_mutually_exclusive_group()
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
    preprocess_parser.add_argument(
        "--features",
        help="To select which tokens to use as words, defaults to all.",
        nargs="*",
        choices=[COMMENTS, IDENTIFIERS, LITERALS],
        default=[COMMENTS, IDENTIFIERS, LITERALS],
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
    # tfidf_group = preprocess_parser.add_mutually_exclusive_group()
    # tfidf_group.add_argument("--thresh", help="To set the threshold for TF-IDF.",
    #                          default=0., type=float)
    # tfidf_group.add_argument("--no-tfidf", help="To skip TF-IDF.", dest="tfidf",
    #                          action="store_false")
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
