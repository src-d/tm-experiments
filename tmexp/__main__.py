import argparse
import logging
from typing import Any


from .create_bow import create_bow, DIFF_MODEL, HALL_MODEL
from .label import label_topics
from .merge import merge_datasets
from .metrics import compute_metrics
from .postprocess import postprocess
from .preprocess import COMMENTS, IDENTIFIERS, LITERALS, preprocess
from .train_artm import train_artm
from .train_hdp import train_hdp
from .utils import check_create_default, SUPPORTED_LANGUAGES
from .visualize import visualize


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


def add_required(cmd_parser: argparse.ArgumentParser, flag: str, help: str) -> None:
    cmd_parser.add_argument(flag, help=help, required=True)


def add_default(
    cmd_parser: argparse.ArgumentParser, flag: str, help: str, out_type: str
) -> None:
    cmd_parser.add_argument(flag, help=help, default=check_create_default(out_type))


def add_dataset_arg(cmd_parser: argparse.ArgumentParser, required: bool) -> None:
    help = "Name of the dataset created by `preprocess`%s."
    if required:
        add_required(cmd_parser, "--dataset-name", help % "")
    else:
        add_default(
            cmd_parser,
            "--dataset-name",
            help % ", defaults to '%(default)s'",
            "dataset",
        )


def add_bow_arg(cmd_parser: argparse.ArgumentParser, required: bool) -> None:
    help = "Name of the BoW created by `create_bow`%s."
    if required:
        add_required(cmd_parser, "--bow-name", help % "")
    else:
        add_default(
            cmd_parser, "--bow-name", help % ", defaults to '%(default)s'", "bow"
        )


def add_experiment_arg(cmd_parser: argparse.ArgumentParser, required: bool) -> None:
    help = "Name of the experiment created by `train_$`%s."
    if required:
        add_required(cmd_parser, "--exp-name", help % "")
    else:
        add_default(
            cmd_parser, "--exp-name", help % ", defaults to '%(default)s'", "experiment"
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

    add_dataset_arg(preprocess_parser, required=False)
    add_feature_arg(preprocess_parser)
    add_force_arg(preprocess_parser)
    add_lang_args(preprocess_parser)

    preprocess_parser.add_argument(
        "-r", "--repo", help="Name of the repo to preprocess.", required=True
    )
    preprocess_parser.add_argument(
        "--exclude-refs",
        help="All refs containing one of these keywords will be excluded "
        "(e.g. all refs with `alpha`).",
        nargs="*",
        default=[],
    )
    preprocess_parser.add_argument(
        "--keep-vendors", help="Keep vendors in processed files.", action="store_true"
    )
    preprocess_parser.add_argument(
        "--only-head", help="Preprocess only the head revision.", action="store_true"
    )
    preprocess_parser.add_argument(
        "--only-by-date",
        help="To sort the references only by date (may cause errors).",
        action="store_true",
    )
    preprocess_parser.add_argument(
        "--version-sep",
        help="If sorting by version, provide the seperator between major and minor.",
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

    merge_parser = subparsers.add_parser(
        "merge",
        help="Merges multiple datasets (it is assumed they stem from the same repo and "
        "contain distinct files).",
    )
    merge_parser.set_defaults(handler=merge_datasets)
    add_force_arg(merge_parser)
    merge_parser.add_argument(
        "-i", "--input-datasets", help="Datasets to merge.", nargs="*", required=True
    )
    merge_parser.add_argument(
        "-o", "--output-dataset", help="Name of the output dataset.", required=True
    )

    # ------------------------------------------------------------------------

    create_bow_parser = subparsers.add_parser(
        "create_bow", help="Create the BoW dataset from a pickled dict, in UCI format."
    )
    create_bow_parser.set_defaults(handler=create_bow)

    add_bow_arg(create_bow_parser, required=False)
    add_dataset_arg(create_bow_parser, required=True)
    add_feature_arg(create_bow_parser)
    add_force_arg(create_bow_parser)
    add_lang_args(create_bow_parser)

    create_bow_parser.add_argument(
        "--topic-model",
        help="Topic evolution model to use.",
        required=True,
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

    add_bow_arg(train_hdp_parser, required=True)
    add_experiment_arg(train_hdp_parser, required=False)
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
    # ------------------------------------------------------------------------

    train_artm_parser = subparsers.add_parser(
        "train_artm", help="Train ARTM model from the input BoW."
    )
    train_artm_parser.set_defaults(handler=train_artm)

    add_bow_arg(train_artm_parser, required=True)
    add_experiment_arg(train_artm_parser, required=False)
    add_force_arg(train_artm_parser)
    train_artm_parser.add_argument(
        "--batch-size",
        help="Number of documents to be stored in each batch, defaults to %(default)s.",
        default=1000,
        type=int,
    )
    train_artm_parser.add_argument(
        "--max-topic",
        help="Maximum number of topics, used as initial value.",
        default=200,
        type=int,
    )
    train_artm_parser.add_argument(
        "--converge-thresh",
        help="When the selection metric does not improve more then this between passes "
        "we assume convergence, defaults to %(default)s.",
        default=0.001,
        type=float,
    )
    train_artm_parser.add_argument(
        "--converge-metric",
        help="Selection metric to use.",
        choices=["perplexity", "distinctness"],
        default="distinctness",
    )
    train_artm_parser.add_argument(
        "--sparse-word-coeff",
        help="Coefficient used by the sparsity inducing regularizer for the word topic "
        "distribution (phi) defaults to %(default)s.",
        default=0.5,
        type=float,
    )
    train_artm_parser.add_argument(
        "--sparse-doc-coeff",
        help="Coefficient used by the sparsity inducing regularizer for the doc topic "
        "distribution (theta), defaults to %(default)s.",
        default=0.5,
        type=float,
    )
    train_artm_parser.add_argument(
        "--decor-coeff",
        help="Coefficient used by the topic decorrelation regularizer, defaults to "
        "%(default)s.",
        default=1e5,
        type=float,
    )
    train_artm_parser.add_argument(
        "--select-coeff",
        help="Coefficient used by the topic selection regularizer, defaults to "
        "%(default)s.",
        default=0.1,
        type=float,
    )
    train_artm_parser.add_argument(
        "--doctopic-eps",
        help="Minimum document topic probability, used as tolerance when computing "
        "sparsity of the document topic matrix, defaults to %(default)s.",
        default=0.05,
        type=float,
    )
    train_artm_parser.add_argument(
        "--wordtopic-eps",
        help="Minimum word topic probability, used as tolerance when computing "
        "sparsity of the word topic matrix, defaults to %(default)s.",
        default=1e-4,
        type=float,
    )
    train_artm_parser.add_argument(
        "--min-prob",
        help="Topics that do not have min-docs documents with at least this"
        "probability will be removed, defaults to %(default)s.",
        default=0.5,
        type=float,
    )
    min_doc_group = train_artm_parser.add_mutually_exclusive_group(required=True)
    min_doc_group.add_argument(
        "--min-docs-abs",
        help="Topics that do not have this amount of docs with at least min-prob"
        "probability will be removed=.",
        type=int,
    )
    min_doc_group.add_argument(
        "--min-docs-rel",
        help="Topics that do not have this proportion of all docs with at least "
        "min-prob probability will be removed.",
        type=float,
    )
    train_artm_parser.add_argument(
        "-q",
        "--quiet",
        help="To only output scores of first and last iteration during each training "
        "phases",
        action="store_true",
    )
    # ------------------------------------------------------------------------

    postprocess_parser = subparsers.add_parser(
        "postprocess",
        help="Compute document word count and membership given a topic model.",
    )
    postprocess_parser.set_defaults(handler=postprocess)
    # TODO(https://github.com/src-d/tm-experiments/issues/21)
    postprocess_parser.add_argument(
        "--original-document-index",
        action="store_true",
        help="Use the original document index instead of the workaround computed "
        "during ARTM training.",
    )
    add_bow_arg(postprocess_parser, required=True)
    add_experiment_arg(postprocess_parser, required=True)
    add_force_arg(postprocess_parser)
    # ------------------------------------------------------------------------

    compute_metrics_parser = subparsers.add_parser(
        "compute_metrics",
        help="Compute metrics given topic distributions over each version.",
    )
    compute_metrics_parser.set_defaults(handler=compute_metrics)
    add_bow_arg(compute_metrics_parser, required=True)
    add_experiment_arg(compute_metrics_parser, required=True)
    add_force_arg(compute_metrics_parser)
    # ------------------------------------------------------------------------

    visualize_parser = subparsers.add_parser(
        "visualize", help="Create visualizations for precomputed metrics."
    )
    visualize_parser.set_defaults(handler=visualize)
    add_bow_arg(visualize_parser, required=True)
    add_experiment_arg(visualize_parser, required=True)
    add_force_arg(visualize_parser)
    visualize_parser.add_argument(
        "--max-topics",
        help="Limit to this amount the number of topics displayed on visualizations "
        "simultaniously (will select most interesting), defaults to %(default)s.",
        default=10,
        type=int,
    )
    # ------------------------------------------------------------------------

    label_topics_parser = subparsers.add_parser(
        "label_topics",
        help="Given a topic model, automatically infer labels for each topic.",
    )
    label_topics_parser.set_defaults(handler=label_topics)
    add_bow_arg(label_topics_parser, required=True)
    add_experiment_arg(label_topics_parser, required=True)
    add_force_arg(label_topics_parser)
    label_topics_parser.add_argument(
        "--mu",
        help="Weights how discriminative we want the label to be relative to other"
        " topics , defaults to %(default)s.",
        default=1.0,
        type=float,
    )
    label_topics_parser.add_argument(
        "--label-size",
        help="Number of words in a label, defaults to %(default)s.",
        default=2,
        type=int,
    )
    label_topics_parser.add_argument(
        "--min-prob",
        help="Admissible words for a topic label must have a topic probability over "
        "this value, defaults to %(default)s.",
        default=0.001,
        type=float,
    )
    label_topics_parser.add_argument(
        "--max-topics",
        help="Admissible words for a topic label must be admissible for less then this"
        " amount of topics, defaults to %(default)s.",
        default=10,
        type=int,
    )
    label_topics_parser.add_argument(
        "--no-smoothing",
        help="To ignore words that don't cooccur with a given label rather then use "
        "Laplacian smoothing on the joint word/label probabilty.",
        dest="smoothing",
        action="store_false",
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
