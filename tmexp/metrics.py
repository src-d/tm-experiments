from logging import Logger
import os
import pickle

import numpy as np

from .io_constants import (
    BOW_DIR,
    MEMBERSHIP_FILENAME,
    METRICS_FILENAME,
    REF_FILENAME,
    TOPICS_DIR,
    WORDCOUNT_FILENAME,
    WORDTOPIC_FILENAME,
)
from .utils import check_file_exists, check_remove, create_logger


def metric_stats(metric: np.array, num_tabs: int, logger: Logger) -> None:
    for op, op_name in zip(
        [np.min, np.mean, np.median, np.max],
        ["Minimum", "Mean   ", "Median ", "Maximum"],
    ):
        logger.info("%s%s : %.2f", num_tabs * "\t", op_name, op(metric))


def compute_metrics(bow_name: str, exp_name: str, force: bool, log_level: str) -> None:

    logger = create_logger(log_level, __name__)

    input_dir_bow = os.path.join(BOW_DIR, bow_name)
    refs_input_path = os.path.join(input_dir_bow, REF_FILENAME)
    check_file_exists(refs_input_path)

    dir_exp = os.path.join(TOPICS_DIR, bow_name, exp_name)
    membership_input_path = os.path.join(dir_exp, MEMBERSHIP_FILENAME)
    check_file_exists(membership_input_path)
    wordcount_input_path = os.path.join(dir_exp, WORDCOUNT_FILENAME)
    check_file_exists(wordcount_input_path)
    wordtopic_input_path = os.path.join(dir_exp, WORDTOPIC_FILENAME)
    check_file_exists(wordtopic_input_path)

    metrics_output_path = os.path.join(dir_exp, METRICS_FILENAME)
    check_remove(metrics_output_path, logger, force)

    logger.info("Loading tagged refs ...")
    with open(refs_input_path, "r", encoding="utf-8") as fin:
        refs = fin.read().split("\n")
    logger.info("Loaded tagged refs, found %d." % len(refs))

    logger.info("Loading document membership ...")
    with open(membership_input_path, "rb") as fin_b:
        membership = pickle.load(fin_b)
    logger.info("Loaded memberships.")

    logger.info("Loading document total word count ...")
    with open(wordcount_input_path, "rb") as fin_b:
        wordcount = pickle.load(fin_b)
    logger.info("Loaded word counts.")

    logger.info("Loading word topic distributions ...")
    wordtopic = np.load(wordtopic_input_path)
    num_topics, num_words = wordtopic.shape
    logger.info("Loaded, found %d words and %d topics.", num_words, num_topics)

    similarity = np.zeros((num_topics, num_topics))
    logger.info("Computing similarity between topics ...")
    for i in range(num_topics):
        dist_i = wordtopic[i, :]
        for j in range(num_topics):
            dist_j = wordtopic[j, :]
            similarity[i, j] = np.sum(dist_i * np.log(dist_i / dist_j))
    similarity = (similarity + similarity.T) / 2
    logger.info("Computed similarity between topics.")

    logger.info("Mean similarity per topic:")
    metric_stats(np.sum(similarity, axis=1) / (num_topics - 1), 1, logger)

    assignment: np.array = np.empty((len(refs), num_topics))
    weight: np.array = np.empty((len(refs), num_topics))
    scatter: np.array = np.empty((len(refs), num_topics))
    focus: np.array = np.empty((len(refs), num_topics))
    logger.info("Computing topic assignment, weight, scatter and focus ...")
    for ind_ref, ref in enumerate(refs):
        ref_doc_count = len(membership[ref])
        ref_wc = sum(wordcount[ref].values())
        ref_docs = sorted(membership[ref])
        ref_membership = np.stack([membership[ref][doc] for doc in ref_docs], axis=0)
        ref_wc = np.array([wordcount[ref][doc] for doc in ref_docs]) / ref_wc
        assignment[ind_ref, :] = np.sum(ref_membership, axis=0) / ref_doc_count
        weight[ind_ref, :] = ref_wc @ ref_membership
        scatter[ind_ref, :] = (
            -np.sum(ref_membership * np.log(ref_membership), axis=0) / ref_doc_count
        )
        focus[ind_ref, :] = np.sum(ref_membership > 0.5, axis=0) / ref_doc_count
    logger.info("Computed metrics.")

    for metric, metric_name in zip(
        [assignment, weight, scatter, focus],
        ["Assignment", "Weight", "Scattering", "Focus"],
    ):
        logger.info("%s :" % metric_name)
        logger.info("\tAcross all tagged references:")
        metric_stats(metric, 2, logger)
        logger.info("\tAveraged over all tagged references:")
        metric_stats(np.mean(metric, axis=0), 2, logger)

    logger.info("Saving metrics ...")
    with open(metrics_output_path, "wb") as fout:
        pickle.dump(
            {
                "similarity": similarity,
                "assignment": assignment,
                "weight": weight,
                "scatter": scatter,
                "focus": focus,
            },
            fout,
        )
    logger.info("Saved metrics in '%s'." % metrics_output_path)
