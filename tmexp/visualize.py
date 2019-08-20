from argparse import ArgumentParser
import os
import pickle
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from .cli import CLIBuilder, register_command
from .data import RefList
from .io_constants import (
    BOW_DIR,
    EVOLUTION_FILENAME,
    HEATMAP_FILENAME,
    LABELS_FILENAME,
    METRICS_FILENAME,
    REF_FILENAME,
    TOPICS_DIR,
    VIZ_DIR,
)
from .utils import (
    check_file_exists,
    check_remove,
    create_directory,
    create_logger,
    load_refs_dict,
)


def _define_parser(parser: ArgumentParser) -> None:
    cli_builder = CLIBuilder(parser)
    cli_builder.add_bow_arg(required=True)
    cli_builder.add_experiment_arg(required=True)
    cli_builder.add_force_arg()
    parser.add_argument(
        "--max-topics",
        help="Limit to this amount the number of topics displayed on visualizations "
        "simultaniously (will select most distinct), defaults to %(default)s.",
        default=10,
        type=int,
    )


def create_heatmap(
    output_path: str,
    data: np.ndarray,
    title: str,
    y_labels: List[str],
    x_label: Optional[str] = None,
    x_labels: Optional[Union[List[str], RefList]] = None,
) -> None:
    plt.figure(figsize=data.T.shape)
    heatmap = plt.pcolor(
        data,
        cmap=plt.cm.Blues,
        vmin=0.0,
        vmax=max(np.max(data), 1.0),
        edgecolor="white",
        linewidths=1,
    )
    plt.colorbar(heatmap)
    plt.title(title, fontsize=18)
    if x_label is None:
        x_label = "Topics"
        x_labels = y_labels
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel("Topics", fontsize=16)
    x_ticks = [i + 0.5 for i, _ in enumerate(x_labels)]
    plt.xticks(x_ticks, x_labels, rotation=90, fontsize=12)
    y_ticks = [i + 0.5 for i, _ in enumerate(y_labels)]
    plt.yticks(y_ticks, y_labels, fontsize=12)
    plt.tick_params(axis="both", which="both", bottom=False, left=False)
    plt.savefig(output_path, bbox_inches="tight")


@register_command(parser_definer=_define_parser)
def visualize(
    bow_name: str, exp_name: str, force: bool, max_topics: int, log_level: str
) -> None:
    """Create visualizations for precomputed metrics."""

    logger = create_logger(log_level, __name__)

    input_dir_bow = os.path.join(BOW_DIR, bow_name)
    refs_input_path = os.path.join(input_dir_bow, REF_FILENAME)
    check_file_exists(refs_input_path)
    input_dir_exp = os.path.join(TOPICS_DIR, bow_name, exp_name)
    metrics_input_path = os.path.join(input_dir_exp, METRICS_FILENAME)
    check_file_exists(metrics_input_path)
    labels_input_path = os.path.join(input_dir_exp, LABELS_FILENAME)
    check_file_exists(labels_input_path)

    output_dir = os.path.join(VIZ_DIR, bow_name, exp_name)
    check_remove(output_dir, logger, force, is_dir=True)
    create_directory(output_dir, logger)

    refs_dict = load_refs_dict(logger, refs_input_path)

    logger.info("Loading metrics ...")
    with open(metrics_input_path, "rb") as fin_b:
        metrics = pickle.load(fin_b)
    num_topics = metrics.distinctness.shape[0]
    logger.info("Loaded metrics, found %d topics." % num_topics)

    logger.info("Loading topic labels ...")
    with open(labels_input_path, "r", encoding="utf-8") as fin:
        topic_labels = [label for label in fin.read().split("\n")]
    logger.info("Loaded topic labels.")
    if num_topics > max_topics:
        logger.info("Selecting most distinct topics ...")
        topic_distinctness = np.sum(metrics.distinctness, axis=1) / (num_topics - 1)
        topics = np.argsort(topic_distinctness)[::-1][:max_topics]
        topic_labels = [topic_labels[ind_label] for ind_label in topics]
        logger.info("Selected the following topics:\n\t%s", "\n\t".join(topic_labels))

    logger.info("Creating and saving distinctness heatmap ...")
    output_path = os.path.join(output_dir, HEATMAP_FILENAME % "distinctness")
    data = metrics.distinctness[topics][:, topics]
    create_heatmap(output_path, data, "Topic distinctness", topic_labels)
    logger.info("Saved distinctness heatmap in '%s'", output_path)

    num_repos = len(refs_dict)
    if num_repos == 1:
        repo, x_labels = refs_dict.popitem()
        logger.info(
            "Found only one repository, creating evolution plots per topic ...."
        )
        x_label = "Tagged references"
        for ind_topic, topic_label in tqdm.tqdm(
            enumerate(topic_labels), total=max_topics
        ):
            plt.figure(figsize=(10, 5))
            x_ticks = [i + 0.5 for i in range(len(x_labels))]
            for metric_ind, metric_name in enumerate(metrics._fields):
                if metric_name == "distinctness":
                    continue
                metric = metrics[metric_ind][repo]
                plt.plot(x_ticks, metric[:, topics[ind_topic]], "-+", label=metric_name)
            plt.xlabel(x_label, fontsize=14)
            plt.xticks(x_ticks, x_labels, rotation=45)
            plt.ylabel("Metric value", fontsize=14)
            plt.ylim(0, 1)
            plt.title("Metrics for topic '%s'" % topic_label, fontsize=18)
            plt.legend(fontsize=16)
            plt.savefig(
                os.path.join(output_dir, EVOLUTION_FILENAME % (ind_topic + 1)),
                bbox_inches="tight",
            )
        logger.info("Saved evolution plots.")
    if num_repos > 1:
        logger.info("Found %d repositories, (assuming one ref per repo).", num_repos)
        x_labels = sorted(refs_dict)  # type: ignore
        x_label = "Repositories"

    logger.info("Creating heatmaps for each metric ...")
    for metric_ind, metric_name in enumerate(metrics._fields):
        if metric_name == "distinctness":
            continue
        metric = metrics[metric_ind]
        output_path = os.path.join(output_dir, HEATMAP_FILENAME % metric_name)
        if num_repos == 1:
            data = metric[repo][:, topics].T
            title = "Topics %s across versions" % metric_name
        else:
            data = np.empty((num_topics, num_repos))
            for ind_repo, repo in enumerate(x_labels):
                data[:, ind_repo] = metric[repo][0][topics]
            title = "Topics %s across repositories" % metric_name
        create_heatmap(output_path, data, title, topic_labels, x_label, x_labels)
    logger.info("Saved heatmaps.")
