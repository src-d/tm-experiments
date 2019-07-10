import os
import pickle
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from .io_constants import (
    BOW_DIR,
    EVOLUTION_FILENAME,
    HEATMAP_FILENAME,
    METRICS_FILENAME,
    REF_FILENAME,
    TOPICS_DIR,
    VIZ_DIR,
)
from .utils import check_file_exists, check_remove, create_directory, create_logger


def create_heatmap(
    output_path: str,
    data: np.array,
    title: str,
    vmin: float,
    vmax: float,
    x_label: str,
    y_label: str,
    x_ticks: Optional[List[float]] = None,
    y_ticks: Optional[List[float]] = None,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
) -> None:
    plt.figure(figsize=data.T.shape)
    heatmap = plt.pcolor(
        data, cmap=plt.cm.Blues, vmin=vmin, vmax=vmax, edgecolor="white", linewidths=1
    )
    plt.colorbar(heatmap)
    plt.title(title, fontsize=18)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    if x_ticks is not None and x_labels is not None:
        plt.xticks(x_ticks, x_labels, rotation=90, fontsize=12)
    if y_ticks is not None and y_labels is not None:
        plt.yticks(y_ticks, y_labels, fontsize=12)
    plt.tick_params(axis="both", which="both", bottom=False, left=False)
    plt.savefig(output_path, bbox_inches="tight")


def visualize(
    bow_name: str, exp_name: str, force: bool, max_topics: int, log_level: str
) -> None:

    # TODO: include topic names when inference is added

    logger = create_logger(log_level, __name__)

    input_dir_bow = os.path.join(BOW_DIR, bow_name)
    refs_input_path = os.path.join(input_dir_bow, REF_FILENAME)
    check_file_exists(refs_input_path)
    input_dir_exp = os.path.join(TOPICS_DIR, bow_name, exp_name)
    metrics_input_path = os.path.join(input_dir_exp, METRICS_FILENAME)
    check_file_exists(metrics_input_path)

    output_dir = os.path.join(VIZ_DIR, bow_name, exp_name)
    check_remove(output_dir, logger, force, is_dir=True)
    create_directory(output_dir, logger)
    evolution_output_dir = os.path.join(output_dir, "evolution_per_topic")
    create_directory(evolution_output_dir, logger)

    logger.info("Loading tagged refs ...")
    with open(refs_input_path, "r", encoding="utf-8") as fin:
        refs = [ref.replace("refs/tags/", "") for ref in fin.read().split("\n")]
    logger.info("Loaded tagged refs, found %d." % len(refs))

    logger.info("Loading metrics ...")
    with open(metrics_input_path, "rb") as fin_b:
        metrics = pickle.load(fin_b)
    num_topics = metrics["similarity"].shape[0]
    logger.info("Loaded metrics, found %d topics." % num_topics)

    logger.info(
        "Creating and saving evolution plots for assignment, weight, scatter and focus "
        "per topic ..."
    )
    ref_ticks = [i + 0.5 for i in range(len(refs))]
    for ind_topic in tqdm.tqdm(range(num_topics)):
        plt.figure(figsize=(10, 5))
        for metric_name, metric in metrics.items():
            if metric_name == "similarity":
                continue
            plt.plot(ref_ticks, metric[:, ind_topic], "-+", label=metric_name)
        plt.xlabel("Tagged reference", fontsize=14)
        plt.xticks(ref_ticks, refs, rotation=45)
        plt.ylabel("Metric value", fontsize=14)
        plt.ylim(0, 1)
        plt.title("Metrics for topic %d" % (ind_topic + 1), fontsize=18)
        plt.legend(fontsize=16)
        plt.savefig(
            os.path.join(evolution_output_dir, EVOLUTION_FILENAME % (ind_topic + 1)),
            bbox_inches="tight",
        )
    logger.info("Saved evolution plots in '%s'." % evolution_output_dir)

    logger.info("Creating and saving heatmaps  per metric ...")
    for metric_name, metric in metrics.items():
        output_path = os.path.join(output_dir, HEATMAP_FILENAME % metric_name)
        if metric_name == "similarity":
            label = "Topic index"
            title = "Topic similarity"
            create_heatmap(output_path, metric, title, 0, np.max(metric), label, label)
        else:
            metric = metric.T
            topics_ind = np.argsort(np.max(metric, axis=1))[::-1][:max_topics]
            metric = metric[topics_ind]
            title = (
                "Evolution of top %d topics for %s metrics per topic and version"
                % (max_topics, metric_name)
            )
            create_heatmap(
                output_path,
                metric,
                title,
                0,
                1,
                "Tagged reference",
                "Topic",
                ref_ticks,
                [i + 0.5 for i in range(max_topics)],
                refs,
                ["topic %d" % (ind + 1) for ind in topics_ind],
            )
        logger.info(
            "Created and saved heatmap for %s metric in '%s'", metric_name, output_path
        )
