from argparse import ArgumentParser
import os
import pickle
from typing import Optional

import numpy as np

from .cli import CLIBuilder, register_command
from .data import Metric, Metrics
from .io_constants import (
    BOW_DIR,
    MEMBERSHIP_FILENAME,
    METRICS_FILENAME,
    REF_FILENAME,
    TOPICS_DIR,
    WORDCOUNT_FILENAME,
    WORDTOPIC_FILENAME,
)
from .utils import check_file_exists, check_remove, create_logger, load_refs_dict


def _define_parser(parser: ArgumentParser) -> None:
    cli_builder = CLIBuilder(parser)
    cli_builder.add_bow_arg(required=True)
    cli_builder.add_experiment_arg(required=True)
    cli_builder.add_force_arg()


def compute_distinctness(
    wordtopic: np.ndarray, num_topics: int, num_words: int
) -> np.ndarray:
    distinctness = np.zeros((num_topics, num_topics))
    wordtopic += 1 / num_words
    wordtopic /= np.sum(wordtopic, axis=1)[:, None]
    for i, dist_i in enumerate(wordtopic):
        for j, dist_j in enumerate(wordtopic):
            distinctness[i, j] = np.sum(dist_i * np.log(dist_i / dist_j))
    return (distinctness + distinctness.T) / 2


def compute_scatter(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x)
    y[x != 0] = x[x != 0] * np.log(x[x != 0])
    return -np.mean(y, axis=0)


def metric_stats(metric: np.ndarray) -> str:
    return "min/med/mean/max : %.2f / %.2f / %.2f / %.2f\n" % (
        np.min(metric),
        np.median(metric),
        np.mean(metric),
        np.max(metric),
    )


def metrics_summary(metrics: Metrics, summary: str, repo: Optional[str] = None) -> str:
    for metric_ind, metric_name in enumerate(metrics._fields):
        if metric_name == "distinctness":
            continue
        if repo is None:
            metric = [np.mean(val, axis=0) for val in metrics[metric_ind].values()]
            summary += "\t%s : %s" % (
                metric_name,
                metric_stats(np.mean(metric, axis=0)),
            )
        else:
            metric = np.mean(metrics[metric_ind][repo], axis=0)
            summary += "\t\t%s : %s" % (metric_name, metric_stats(metric))
    return summary


@register_command(parser_definer=_define_parser)
def compute_metrics(bow_name: str, exp_name: str, force: bool, log_level: str) -> None:
    """Compute metrics given topic distributions over each version."""
    logger = create_logger(log_level, __name__)

    input_dir_bow = os.path.join(BOW_DIR, bow_name)
    refs_input_path = os.path.join(input_dir_bow, REF_FILENAME)
    check_file_exists(refs_input_path)
    wordcount_input_path = os.path.join(input_dir_bow, WORDCOUNT_FILENAME)
    check_file_exists(wordcount_input_path)

    dir_exp = os.path.join(TOPICS_DIR, bow_name, exp_name)
    membership_input_path = os.path.join(dir_exp, MEMBERSHIP_FILENAME)
    check_file_exists(membership_input_path)
    wordtopic_input_path = os.path.join(dir_exp, WORDTOPIC_FILENAME)
    check_file_exists(wordtopic_input_path)

    metrics_output_path = os.path.join(dir_exp, METRICS_FILENAME)
    check_remove(metrics_output_path, logger, force)

    refs_dict = load_refs_dict(logger, refs_input_path)

    logger.info("Loading document membership ...")
    with open(membership_input_path, "rb") as fin_b:
        memberships = pickle.load(fin_b)
    logger.info("Loaded memberships.")

    logger.info("Loading document total word counts ...")
    with open(wordcount_input_path, "rb") as fin_b:
        total_word_counts = pickle.load(fin_b)
    logger.info("Loaded word counts.")

    logger.info("Loading word topic distributions ...")
    wordtopic = np.load(wordtopic_input_path)
    num_topics, num_words = wordtopic.shape
    logger.info(
        "Loaded distributions, found %d words and %d topics.", num_words, num_topics
    )

    logger.info("Computing distinctness between topics ...")
    distinctness = compute_distinctness(wordtopic, num_topics, num_words)
    logger.info("Computed distinctness between topics.\n")

    summary = "Summary:\n\tDistinctness per topic: %s"
    summary = summary % metric_stats(np.sum(distinctness, axis=1) / (num_topics - 1))
    summary += "\tDistinctness between topic %s" % metric_stats(distinctness)
    logger.info(summary)
    assignment = Metric()
    weight = Metric()
    scatter = Metric()
    focus = Metric()
    logger.info("Computing topic assignment, weight, scatter and focus per repo ...")

    for repo, refs in refs_dict.items():
        logger.info("\tProcessing repository '%s'", repo)
        for metric in [assignment, weight, scatter, focus]:
            metric[repo] = np.empty((len(refs), num_topics))
        for ind_ref, ref in enumerate(refs):
            ref_memberships = memberships[repo][ref]
            ref_wc = total_word_counts[repo][ref]

            docs = sorted(ref_memberships)
            ref_total_wc = sum(ref_wc.values())
            ref_memberships = np.stack([ref_memberships[doc] for doc in docs], axis=0)
            ref_wc = np.array([ref_wc[doc] for doc in docs]) / ref_total_wc

            assignment[repo][ind_ref, :] = np.mean(ref_memberships, axis=0)
            weight[repo][ind_ref, :] = ref_wc @ ref_memberships
            scatter[repo][ind_ref, :] = compute_scatter(ref_memberships)
            focus[repo][ind_ref, :] = np.sum(ref_memberships > 0.5, axis=0) / np.sum(
                ref_memberships > 0, axis=0
            )
    logger.info("Computed metrics.")

    metrics = Metrics(
        distinctness=distinctness,
        assignment=assignment,
        weight=weight,
        scatter=scatter,
        focus=focus,
    )
    if len(refs_dict) != 1:
        summary = "Global summary (metrics averaged over refs and repos):\n"
        logger.info(metrics_summary(metrics, summary))
    summary = "Repository summary (metrics averaged over refs):\n"
    for repo in refs_dict:
        summary += "\t%s\n" % repo
        summary = metrics_summary(metrics, summary, repo=repo)
    logger.info(summary)

    logger.info("Saving metrics ...")
    with open(metrics_output_path, "wb") as fout:
        pickle.dump(metrics, fout)
    logger.info("Saved metrics in '%s'." % metrics_output_path)
