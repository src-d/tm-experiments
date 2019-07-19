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

from .io_constants import (
    BOW_DIR,
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
    logger = create_logger(log_level, __name__)

    input_dir = os.path.join(BOW_DIR, bow_name)
    check_file_exists(os.path.join(input_dir, VOCAB_FILENAME))
    docword_input_path = os.path.join(input_dir, DOCWORD_FILENAME)
    check_file_exists(docword_input_path)

    output_dir = os.path.join(TOPICS_DIR, bow_name, exp_name)
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
        "Removing topics with less then %d documents with probability over %.2f",
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
    doctopic, _, _ = model_artm.get_theta_sparse()
    doctopic = doctopic.todense()
    logger.info("Saving topics per document ...")
    np.save(doctopic_output_path, doctopic.T)
    logger.info("Saved topics per document in '%s'.", doctopic_output_path)

    wordtopic, _, _ = model_artm.get_phi_dense()
    logger.info("Saving word/topic distribution ...")
    np.save(wordtopic_output_path, wordtopic.T)
    logger.info("Saved word/topic distribution in '%s'.", wordtopic_output_path)
