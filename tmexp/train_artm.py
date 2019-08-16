from argparse import ArgumentParser
import logging
import os
from typing import Dict, Optional, Tuple
import warnings

from artm import (
    ARTM,
    BatchVectorizer,
    DecorrelatorPhiRegularizer,
    SmoothSparsePhiRegularizer,
    SmoothSparseThetaRegularizer,
    SparsityPhiScore,
    TopicSelectionThetaRegularizer,
)
import numpy as np

from .cli import CLIBuilder, register_command
from .io_constants import (
    BOW_DIR,
    DOCTOPIC_FILENAME,
    DOCWORD_CONCAT_FILENAME,
    DOCWORD_FILENAME,
    TOPICS_DIR,
    VOCAB_CONCAT_FILENAME,
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
    cli_builder.add_consolidate_arg()
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
        "--max-iter",
        help="Maximum number of iterations for each training phase.",
        default=1000,
        type=int,
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


def check_doctopic(logger: logging.Logger, doctopic: np.ndarray, num_docs: int) -> bool:
    if num_docs != doctopic.shape[0]:
        logger.debug("Missing %d documents in matrix.", num_docs - doctopic.shape[0])
        return False
    return True


def create_artm_batch_vectorizer(
    collection_name: str,
    input_dir: str,
    batch_size: int,
    input_path: str,
    logger: logging.Logger,
) -> Tuple[BatchVectorizer, int]:
    batch_vectorizer = BatchVectorizer(
        collection_name=collection_name,
        data_path=input_dir,
        data_format="bow_uci",
        batch_size=batch_size,
    )
    with open(input_path, "r", encoding="utf-8") as fin:
        num_docs = int(fin.readline())
        logger.info("\tNumber of documents: %d", num_docs)
        num_words = int(fin.readline())
        logger.info("\tNumber of words: %d", num_words)
        num_rows = int(fin.readline())
        logger.info("\tNumber of document/word pairs: %d", num_rows)
    logger.info(
        "Created vectorizer contains %d batches of up to %d documents.\n",
        batch_vectorizer.num_batches,
        batch_size,
    )
    return batch_vectorizer, num_docs


def print_scores(logger: logging.Logger, scores: Dict[str, float], header: str) -> None:
    msg = "%s\n" % header
    num_tab = 1 + int("\t" in header)
    for label, score in scores.items():
        msg += "%s%s : %.4f\n" % (num_tab * "\t", label, score)
    logger.info(msg)


def compute_scores(
    model: ARTM, batch_vectorizer: BatchVectorizer, doctopic_eps: float
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    scores["topic sparsity"] = model.score_tracker["topic sparsity"].last_value
    wordtopic, words, topics = model.get_phi_dense()
    doctopic = model.transform_sparse(batch_vectorizer)[0].todense().T
    scores["doc sparsity"] = np.sum(doctopic < doctopic_eps) / (
        doctopic.shape[0] * doctopic.shape[1]
    )
    scores["distinctness"] = np.mean(
        np.sum(compute_distinctness(wordtopic.T, len(topics), len(words)), axis=1)
        / (len(topics) - 1)
    )
    return scores


def _loop_until_convergence(
    model: ARTM,
    start_iter: int,
    logger: logging.Logger,
    batch_vectorizer: BatchVectorizer,
    converge_thresh: float,
    max_iter: int,
    doctopic_eps: float,
    quiet: bool,
) -> Tuple[ARTM, float, int]:
    converged = False
    num_iter = 0
    prev_score = np.inf
    while not converged and num_iter < max_iter:
        num_iter += 1
        model.fit_offline(
            batch_vectorizer=batch_vectorizer, num_collection_passes=1, reset_nwt=False
        )
        scores = compute_scores(model, batch_vectorizer, doctopic_eps)
        score = scores["distinctness"]
        converged = abs(score - prev_score) / prev_score < converge_thresh
        if not quiet and not ((start_iter + num_iter) % 25):
            print_scores(logger, scores, "\tIteration %d" % (start_iter + num_iter))
        prev_score = score
    return model, score, start_iter + num_iter


def _save_model(
    model: ARTM,
    logger: logging.Logger,
    batch_vectorizer: BatchVectorizer,
    num_docs: int,
    batch_vectorizer_train: BatchVectorizer,
    num_docs_train: int,
    doctopic_output_path: str,
    wordtopic_output_path: str,
    consolidate: bool,
) -> None:
    if consolidate:
        check_doctopic(
            logger,
            model.transform_sparse(batch_vectorizer_train)[0].todense().T,
            num_docs_train,
        )
    doctopic = model.transform_sparse(batch_vectorizer)[0].todense().T
    if check_doctopic(logger, doctopic, num_docs):
        if os.path.exists(doctopic_output_path):
            logger.info("Removing previous document-topic matrix ...")
            os.remove(doctopic_output_path)
        if os.path.exists(wordtopic_output_path):
            logger.info("Removing previous document-topic matrix ...")
            os.remove(wordtopic_output_path)
        logger.info("Saving topics per document ...")
        np.save(doctopic_output_path, doctopic)
        logger.info("Saved topics per document in '%s'.", doctopic_output_path)
        logger.info("Saving word/topic distribution ...")
        np.save(wordtopic_output_path, model.get_phi_dense()[0].T)
        logger.info("Saved word/topic distribution in '%s'.\n", wordtopic_output_path)
    else:
        logger.info("Document-topic matrix is corrupted, no saving.\n")


@register_command(parser_definer=_define_parser)
def train_artm(
    bow_name: str,
    exp_name: str,
    force: bool,
    batch_size: int,
    max_topic: int,
    converge_thresh: float,
    max_iter: int,
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
    consolidate: bool,
    log_level: str,
) -> None:
    """Train ARTM model from the input BoW."""
    logger = create_logger(log_level, __name__)

    input_dir = os.path.join(BOW_DIR, bow_name)
    check_file_exists(os.path.join(input_dir, VOCAB_FILENAME))
    docword_input_path = os.path.join(input_dir, DOCWORD_FILENAME)
    check_file_exists(docword_input_path)
    if consolidate:
        check_file_exists(os.path.join(input_dir, VOCAB_CONCAT_FILENAME))
        docword_concat_input_path = os.path.join(input_dir, DOCWORD_CONCAT_FILENAME)
        check_file_exists(docword_concat_input_path)

    output_dir = os.path.join(TOPICS_DIR, bow_name, exp_name)
    doctopic_output_path = os.path.join(output_dir, DOCTOPIC_FILENAME)
    check_remove(doctopic_output_path, logger, force)
    wordtopic_output_path = os.path.join(output_dir, WORDTOPIC_FILENAME)
    check_remove(wordtopic_output_path, logger, force)
    create_directory(output_dir, logger)

    logger.info("Creating batch vectorizer from bags of words ...")
    batch_vectorizer, num_docs = create_artm_batch_vectorizer(
        "bow_tm", input_dir, batch_size, docword_input_path, logger
    )
    if consolidate:
        logger.info("Creating batch vectorizer from consolidated  bags of words ...")
        batch_vectorizer_train, num_docs_train = create_artm_batch_vectorizer(
            "bow_concat_tm", input_dir, batch_size, docword_concat_input_path, logger
        )
    else:
        batch_vectorizer_train, num_docs_train = batch_vectorizer, num_docs

    if min_docs_rel is None:
        min_docs = min_docs_abs
    else:
        check_range(min_docs_rel, "min-docs-rel")
        min_docs = int(num_docs_train * min_docs_rel)

    model_artm = ARTM(
        cache_theta=True,
        reuse_theta=True,
        theta_name="theta",
        dictionary=batch_vectorizer_train.dictionary,
        num_document_passes=1,
        num_topics=max_topic,
        scores=[SparsityPhiScore(name="topic sparsity", eps=wordtopic_eps)],
        regularizers=[
            SmoothSparsePhiRegularizer(name="Sparse Topic", tau=0),
            SmoothSparseThetaRegularizer(name="Sparse Doc", tau=0),
            DecorrelatorPhiRegularizer(name="Decorrelator", tau=decor_coeff),
            TopicSelectionThetaRegularizer(name="Selector", tau=0),
        ],
    )
    num_iter = 0

    def loop_until_convergence(model: ARTM, n_iter: int) -> Tuple[ARTM, float, int]:
        return _loop_until_convergence(
            model,
            n_iter,
            logger,
            batch_vectorizer_train,
            converge_thresh,
            max_iter,
            doctopic_eps,
            quiet,
        )

    def save_model(model: ARTM) -> None:
        _save_model(
            model,
            logger,
            batch_vectorizer,
            num_docs,
            batch_vectorizer_train,
            num_docs_train,
            doctopic_output_path,
            wordtopic_output_path,
            consolidate,
        )

    logger.info("Starting training ...")

    logger.info("Decorrelating topics ...")
    model_artm, _, num_iter = loop_until_convergence(model_artm, num_iter)
    check_doctopic(
        logger,
        model_artm.transform_sparse(batch_vectorizer_train)[0].todense().T,
        num_docs_train,
    )
    logger.info("Finished first phase at iteration %d", num_iter)
    print_scores(
        logger, compute_scores(model_artm, batch_vectorizer, doctopic_eps), "Scores:"
    )

    logger.info("Applying selection regularization on topics ...")
    model_artm.regularizers["Sparse Topic"].tau = 0
    model_artm.regularizers["Sparse Doc"].tau = 0
    model_artm.regularizers["Decorrelator"].tau = 0
    model_artm.regularizers["Selector"].tau = select_coeff
    model_artm, score_1, num_iter = loop_until_convergence(model_artm, num_iter)
    logger.info("Finished second phase at iteration %d", num_iter)
    print_scores(
        logger,
        compute_scores(model_artm, batch_vectorizer, doctopic_eps),
        "Scores before topic removal:",
    )

    logger.info(
        "Removing topics with less than %d documents with probability over %.2f.",
        min_docs,
        min_prob,
    )
    doctopic = model_artm.transform_sparse(batch_vectorizer_train)[0].todense().T
    valid_topics = np.sum(doctopic > min_prob, axis=0) > min_docs
    topic_names = [
        topic_name
        for topic_ind, topic_name in enumerate(model_artm.topic_names)
        if valid_topics[0, topic_ind]
    ]
    model_artm.reshape_topics(topic_names)
    if len(valid_topics):
        logger.info("New number of topics: %d", len(topic_names))
    else:
        raise RuntimeError(
            "Removed all topics, please soften your selection criteria (aborting)."
        )
    print_scores(
        logger,
        compute_scores(model_artm, batch_vectorizer, doctopic_eps),
        "Scores after topic removal:",
    )
    save_model(model_artm)

    logger.info("Inducing sparsity ...")
    model_artm.regularizers["Selector"].tau = 0
    model_artm.regularizers["Decorrelator"].tau = decor_coeff
    model_artm.regularizers["Sparse Topic"].tau = -sparse_word_coeff
    model_artm.regularizers["Sparse Doc"].tau = -sparse_doc_coeff
    model_artm, score_2, num_iter = loop_until_convergence(model_artm, num_iter)
    logger.info("Finished last phase of training at iteration %d", num_iter)
    print_scores(
        logger, compute_scores(model_artm, batch_vectorizer, doctopic_eps), "Scores:"
    )
    if score_1 < score_2:
        save_model(model_artm)
    else:
        logger.info("Sparsity worsened the model, no saving.")
