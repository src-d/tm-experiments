import sys
import logging
import argparse

from preprocess import preprocess


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--log-level", default="INFO", choices=logging._nameToLevel,
                        help="Logging verbosity.")
    # Create and construct subparsers

    subparsers = parser.add_subparsers(help="Commands", dest="command")

    # ------------------------------------------------------------------------

    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Extract the feature count per document and store them as a pickled dict.")
    preprocess_parser.set_defaults(handler=preprocess)
    input_group = preprocess_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-r", "--repo", help="Name of a repo to preprocess.", type=str, default=None)
    input_group.add_argument("--repo-list", help="Filename containing a list of repos to preprocess, one per line.",
                             type=str, default=None)
    preprocess_parser.add_argument("-o", "--output", help="Filepath for the output.", default="features.pkl", type=str)
    preprocess_parser.add_argument("-f", "--force", help="Replace output file if it already exists.",
                                   action="store_true")
    preprocess_parser.add_argument("--no-comments", help="To exclude comments from features.", dest="comments",
                                   action="store_false")
    preprocess_parser.add_argument("--no-literals", help="To exclude literals from features.", dest="literals",
                                   action="store_false")
    preprocess_parser.add_argument("--no-tokenize", help="To skip tokenization.", dest="tokenize", action="store_false")
    preprocess_parser.add_argument("--no-stem", help="To skip stemming.", dest="stem", action="store_false")
    # tfidf_group = preprocess_parser.add_mutually_exclusive_group()
    # tfidf_group.add_argument("--thresh", help="To set the threshold for TF-IDF.", default=0., type=float)
    # tfidf_group.add_argument("--no-tfidf", help="To skip TF-IDF.", dest="tfidf", action="store_false")
    preprocess_parser.add_argument("--gitbase-host", help="Gitbase hostname.", type=str, default="127.0.0.1")
    preprocess_parser.add_argument("--gitbase-port", help="Gitbase port.", type=int, default=3306)
    preprocess_parser.add_argument("--gitbase-user", help="Gitbase user.", type=str, default="root")
    preprocess_parser.add_argument("--gitbase-pass", help="Gitbase password.", type=str, default="")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.log_level = logging._nameToLevel[args.log_level]
    try:
        handler = args.handler
    except AttributeError:
        def print_usage(_):
            parser.print_usage()
        handler = print_usage
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
