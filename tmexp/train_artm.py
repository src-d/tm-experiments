from argparse import ArgumentParser
import logging
import os
from typing import Dict, Optional
import warnings

from artm import (
    ARTM,
    BatchVectorizer,
    DecorrelatorPhiRegularizer,
    PerplexityScore,
    SmoothSparsePhiRegularizer,
    SmoothSparseThetaRegularizer,
    SparsityPhiScore,
    SparsityThetaScore,
    TopicSelectionThetaRegularizer,
)
import numpy as np

from .cli import CLIBuilder, register_command
from .io_constants import (
    BOW_DIR,
    DOC_ARTM_FILENAME,
    DOC_FILENAME,
    DOCTOPIC_FILENAME,
    DOCWORD_FILENAME,
    TOPICS_DIR,
    VOCAB_FILENAME,
    WORDTOPIC_FILENAME,
)
from .metrics import compute_distinctness
from .utils import (
    check_file_exists,
    check_range,
    check_remove,
    create_directory,
    create_logger,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _define_parser(parser: ArgumentParser) -> None:
    cli_builder = CLIBuilder(parser)
    cli_builder.add_bow_arg(required=True)
    cli_builder.add_experiment_arg(required=False)
    cli_builder.add_force_arg()
    parser.add_argument(
        "--batch-size",
        help="Number of documents to be stored in each batch, defaults to %(default)s.",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--max-topic",
        help="Maximum number of topics, used as initial value.",
        default=200,
        type=int,
    )
    parser.add_argument(
        "--converge-thresh",
        help="When the selection metric does not improve more then this between passes "
        "we assume convergence, defaults to %(default)s.",
        default=0.001,
        type=float,
    )
    parser.add_argument(
        "--converge-metric",
        help="Selection metric to use.",
        choices=["perplexity", "distinctness"],
        default="distinctness",
    )
    parser.add_argument(
        "--sparse-word-coeff",
        help="Coefficient used by the sparsity inducing regularizer for the word topic "
        "distribution (phi) defaults to %(default)s.",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--sparse-doc-coeff",
        help="Coefficient used by the sparsity inducing regularizer for the doc topic "
        "distribution (theta), defaults to %(default)s.",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--decor-coeff",
        help="Coefficient used by the topic decorrelation regularizer, defaults to "
        "%(default)s.",
        default=1e5,
        type=float,
    )
    parser.add_argument(
        "--select-coeff",
        help="Coefficient used by the topic selection regularizer, defaults to "
        "%(default)s.",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--doctopic-eps",
        help="Minimum document topic probability, used as tolerance when computing "
        "sparsity of the document topic matrix, defaults to %(default)s.",
        default=0.05,
        type=float,
    )
    parser.add_argument(
        "--wordtopic-eps",
        help="Minimum word topic probability, used as tolerance when computing "
        "sparsity of the word topic matrix, defaults to %(default)s.",
        default=1e-4,
        type=float,
    )
    parser.add_argument(
        "--min-prob",
        help="Topics that do not have min-docs documents with at least this"
        "probability will be removed, defaults to %(default)s.",
        default=0.5,
        type=float,
    )
    min_doc_group = parser.add_mutually_exclusive_group(required=True)
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
    parser.add_argument(
        "-q",
        "--quiet",
        help="To only output scores of first and last iteration during each training "
        "phases",
        action="store_true",
    )


def print_scores(
    logger: logging.Logger, num_iter: int, scores: Dict[str, float]
) -> None:
    logger.info("\tIteration %d", num_iter)
    for label, score in scores.items():
        logger.info("\t\t%s : %.4f", label, score)


def compute_scores(model: ARTM) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for label in ["Topic Sparsity", "Doc Sparsity", "Perplexity"]:
        scores[label.lower()] = model.score_tracker[label].last_value
    wordtopic, words, topics = model.get_phi_dense()
    scores["distinctness"] = np.mean(
        np.sum(compute_distinctness(wordtopic.T, len(topics), len(words)), axis=1)
        / (len(topics) - 1)
    )
    return scores


def loop_until_convergence(
    logger: logging.Logger,
    batch_vectorizer: BatchVectorizer,
    model_artm: ARTM,
    converge_metric: str,
    converge_thresh: float,
    quiet: bool,
) -> ARTM:
    converged = False
    num_iter = 0
    prev_score = np.inf
    while not converged:
        num_iter += 1
        model_artm.fit_offline(
            batch_vectorizer=batch_vectorizer, num_collection_passes=1, reset_nwt=False
        )
        scores = compute_scores(model_artm)
        score = scores[converge_metric]
        converged = abs(score - prev_score) / prev_score < converge_thresh
        if converged or not quiet or num_iter == 1:
            print_scores(logger, num_iter, scores)
        prev_score = score
    return model_artm


@register_command(parser_definer=_define_parser)
def train_artm(
    bow_name: str,
    exp_name: str,
    force: bool,
    batch_size: int,
    max_topic: int,
    converge_metric: str,
    converge_thresh: float,
    sparse_word_coeff: float,
    sparse_doc_coeff: float,
    decor_coeff: float,
    select_coeff: float,
    doctopic_eps: float,
    wordtopic_eps: float,
    min_prob: float,
    min_docs_abs: Optional[int],
    min_docs_rel: Optional[float],
    quiet: bool,
    log_level: str,
) -> None:
    """Train ARTM model from the input BoW."""
    logger = create_logger(log_level, __name__)

    input_dir = os.path.join(BOW_DIR, bow_name)
    check_file_exists(os.path.join(input_dir, VOCAB_FILENAME))
    docword_input_path = os.path.join(input_dir, DOCWORD_FILENAME)
    check_file_exists(docword_input_path)
    doc_input_path = os.path.join(input_dir, DOC_FILENAME)
    check_file_exists(doc_input_path)

    output_dir = os.path.join(TOPICS_DIR, bow_name, exp_name)
    doc_output_path = os.path.join(output_dir, DOC_ARTM_FILENAME)
    check_remove(doc_output_path, logger, force)
    doctopic_output_path = os.path.join(output_dir, DOCTOPIC_FILENAME)
    check_remove(doctopic_output_path, logger, force)
    wordtopic_output_path = os.path.join(output_dir, WORDTOPIC_FILENAME)
    check_remove(wordtopic_output_path, logger, force)
    create_directory(output_dir, logger)

    logger.info("Loading bags of words ...")

    batch_vectorizer = BatchVectorizer(
        collection_name="bow_tm",
        data_path=input_dir,
        data_format="bow_uci",
        batch_size=batch_size,
    )
    with open(docword_input_path, "r", encoding="utf-8") as fin:
        num_docs = int(fin.readline())
        logger.info("Number of documents: %d", num_docs)
        num_words = int(fin.readline())
        logger.info("Number of words: %d", num_words)
        num_rows = int(fin.readline())
        logger.info("Number of document/word pairs: %d", num_rows)

    with open(doc_input_path, "r", encoding="utf8") as fin:
        doc_names = fin.read().splitlines()

    logger.info(
        "Loaded bags of words, created %d batches of up to %d documents.",
        batch_vectorizer.num_batches,
        batch_size,
    )

    if min_docs_rel is None:
        min_docs = min_docs_abs
    else:
        check_range(min_docs_rel, "min-docs-rel")
        min_docs = int(num_docs * min_docs_rel)

    model_artm = ARTM(
        cache_theta=True,
        dictionary=batch_vectorizer.dictionary,
        num_document_passes=1,
        num_topics=max_topic,
        scores=[
            PerplexityScore(name="Perplexity", dictionary=batch_vectorizer.dictionary),
            SparsityPhiScore(name="Topic Sparsity", eps=wordtopic_eps),
            SparsityThetaScore(name="Doc Sparsity", eps=doctopic_eps),
        ],
        regularizers=[
            SmoothSparsePhiRegularizer(name="Sparse Topic", tau=0),
            SmoothSparseThetaRegularizer(name="Sparse Doc", tau=0),
            DecorrelatorPhiRegularizer(name="Decorrelator", tau=decor_coeff),
            TopicSelectionThetaRegularizer(name="Selector", tau=0),
        ],
    )
    logger.info("Starting training ...")

    logger.info("Decorrelating topics ...")
    model_artm = loop_until_convergence(
        logger, batch_vectorizer, model_artm, converge_metric, converge_thresh, quiet
    )

    logger.info("Applying selection regularization on topics ...")
    model_artm.regularizers["Sparse Topic"].tau = 0
    model_artm.regularizers["Sparse Doc"].tau = 0
    model_artm.regularizers["Decorrelator"].tau = 0
    model_artm.regularizers["Selector"].tau = select_coeff
    model_artm = loop_until_convergence(
        logger, batch_vectorizer, model_artm, converge_metric, converge_thresh, quiet
    )

    logger.info(
        "Removing topics with less than %d documents with probability over %.2f.",
        min_docs,
        min_prob,
    )
    doctopic, _, _ = model_artm.get_theta_sparse(eps=doctopic_eps)
    doctopic = doctopic.todense()
    topics = np.argwhere(np.sum(doctopic > min_prob, axis=1) > min_docs).flatten()
    topic_names = [t for i, t in enumerate(model_artm.topic_names) if i in topics]
    model_artm.reshape(topic_names)
    if len(topics):
        logger.info("New number of topics: %d", len(topic_names))
    else:
        raise RuntimeError(
            "Removed all topics, please soften your selection criteria (aborting)."
        )

    logger.info("Inducing sparsity ...")
    model_artm.regularizers["Selector"].tau = 0
    model_artm.regularizers["Decorrelator"].tau = decor_coeff
    model_artm.regularizers["Sparse Topic"].tau = -sparse_word_coeff
    model_artm.regularizers["Sparse Doc"].tau = -sparse_doc_coeff
    model_artm = loop_until_convergence(
        logger, batch_vectorizer, model_artm, converge_metric, converge_thresh, quiet
    )

    logger.info("Finished training.")
    # TODO(https://github.com/src-d/tm-experiments/issues/21)
    doctopic, _, doc_indexes = model_artm.get_theta_sparse()
    doctopic = doctopic.todense()
    with open(doc_output_path, "w", encoding="utf8") as fout:
        fout.write(
            "%s\n" % "\n".join(doc_names[doc_index] for doc_index in doc_indexes)
        )
    logger.info("Saving topics per document ...")
    np.save(doctopic_output_path, doctopic.T)
    logger.info("Saved topics per document in '%s'.", doctopic_output_path)
    wordtopic, _, _ = model_artm.get_phi_dense()
    logger.info("Saving word/topic distribution ...")
    np.save(wordtopic_output_path, wordtopic.T)
    logger.info("Saved word/topic distribution in '%s'.", wordtopic_output_path)
