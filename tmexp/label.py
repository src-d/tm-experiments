from argparse import ArgumentParser
from collections import defaultdict
from enum import Enum
from functools import partial
import itertools
import os
from typing import DefaultDict, Dict, List, Optional, Union

import numpy as np

from .cli import CLIBuilder, register_command
from .constants import ADD, DEL, DIFF_MODEL, DOC, HALL_MODEL, SEP
from .data import RefList, RepoMapping
from .io_constants import (
    BOW_DIR,
    DOC_FILENAME,
    DOCWORD_FILENAME,
    LABELS_FILENAME,
    REF_FILENAME,
    TOPICS_DIR,
    VOCAB_FILENAME,
    WORDTOPIC_FILENAME,
)
from .reduce import (
    concat_reducer,
    diff_to_hall_reducer,
    FileReducer,
    last_ref_reducer,
    max_reducer,
    mean_reducer,
    median_reducer,
    reduce_corpus,
)
from .utils import check_file_exists, check_range, check_remove, create_logger


def _define_parser(parser: ArgumentParser) -> None:
    cli_builder = CLIBuilder(parser)
    cli_builder.add_bow_arg(required=True)
    cli_builder.add_experiment_arg(required=True)
    cli_builder.add_force_arg()
    parser.add_argument(
        "--mu",
        help="Weights how discriminative we want the label to be relative to other"
        " topics , defaults to %(default)s.",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--label-size",
        help="Number of words in a label, defaults to %(default)s.",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--min-prob",
        help="Admissible words for a topic label must have a topic probability over "
        "this value, defaults to %(default)s.",
        default=0.001,
        type=float,
    )
    parser.add_argument(
        "--max-topics",
        help="Admissible words for a topic label must be admissible for less then this"
        " amount of topics, defaults to %(default)s.",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--no-smoothing",
        help="To ignore words that don't cooccur with a given label rather then use "
        "Laplace smoothing on the joint word/label probabilty.",
        dest="smoothing",
        action="store_false",
    )
    parser.add_argument(
        "--context",
        help="Context creation method.",
        choices=list(Context),
        type=Context.from_string,
        required=True,
    )


class Context(Enum):
    last = partial(last_ref_reducer)
    max = partial(max_reducer)
    mean = partial(mean_reducer)
    median = partial(median_reducer)
    concat = partial(concat_reducer)
    hall = None

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def from_string(s: str) -> "Context":
        try:
            return Context[s]
        except KeyError:
            raise ValueError()

    @property
    def reducer(self) -> Optional[FileReducer]:
        return self.value


@register_command(parser_definer=_define_parser)
def label(
    bow_name: str,
    exp_name: str,
    force: bool,
    log_level: str,
    mu: float,
    label_size: int,
    min_prob: float,
    max_topics: int,
    smoothing: bool,
    context: Context,
) -> None:
    """Infer a label for each topic automatically given a topic model."""
    logger = create_logger(log_level, __name__)
    input_dir_bow = os.path.join(BOW_DIR, bow_name)
    doc_input_path = os.path.join(input_dir_bow, DOC_FILENAME)
    check_file_exists(doc_input_path)
    docword_input_path = os.path.join(input_dir_bow, DOCWORD_FILENAME)
    check_file_exists(docword_input_path)
    refs_input_path = os.path.join(input_dir_bow, REF_FILENAME)
    check_file_exists(refs_input_path)
    vocab_input_path = os.path.join(input_dir_bow, VOCAB_FILENAME)
    check_file_exists(vocab_input_path)

    dir_exp = os.path.join(TOPICS_DIR, bow_name, exp_name)
    wordtopic_input_path = os.path.join(dir_exp, WORDTOPIC_FILENAME)
    check_file_exists(wordtopic_input_path)

    labels_output_path = os.path.join(dir_exp, LABELS_FILENAME)
    check_remove(labels_output_path, logger, force)

    check_range(min_prob, "min-prob")

    logger.info("Loading tagged refs ...")
    with open(refs_input_path, "r", encoding="utf-8") as fin:
        refs: DefaultDict[str, RefList] = defaultdict(RefList)
        for line in fin:
            repo, ref = line.split(SEP)
            refs[repo].append(ref.replace("\n", ""))
    logger.info("Loaded tagged refs:")
    for repo, repo_refs in refs.items():
        logger.info("\tRepository '%s': %d refs", repo, len(repo_refs))

    logger.info("Loading document index ...")
    with open(doc_input_path, "r", encoding="utf-8") as fin:
        line = fin.readline()
        if SEP + ADD in line or SEP + DEL in line:
            topic_model = DIFF_MODEL
        else:
            topic_model = HALL_MODEL
        fin.seek(0)
        repo_mapping = RepoMapping()
        for doc_ind, line in enumerate(fin):
            doc_info = line.split()
            if topic_model == HALL_MODEL:
                repo, file_path, _ = doc_info[0].split(SEP)
                delta_type = DOC
            else:
                repo, file_path, delta_type, _ = doc_info[0].split(SEP)
            for ref in doc_info[1:]:
                repo_mapping[repo][file_path][ref][delta_type] = doc_ind
    logger.info("Loaded document index, detected %s topic model.", topic_model)

    logger.info("Loading bags of words ...")
    with open(docword_input_path, "r", encoding="utf-8") as fin:
        num_docs = int(fin.readline())
        num_words = int(fin.readline())
        fin.readline()
        corpus = np.zeros((num_docs, num_words))
        for line in fin:
            doc_id, word_id, count = map(int, line.split())
            corpus[doc_id, word_id - 1] = count
    logger.info("Loaded %d bags of words.", num_docs)

    if topic_model == DIFF_MODEL:
        logger.info("Recreating hall model corpus (we can't use delta-documents) ...")
        corpus = reduce_corpus(corpus, logger, repo_mapping, refs, diff_to_hall_reducer)
        num_docs = corpus.shape[0]
        logger.info("Recreated hall model corpus, found %d documents ...", num_docs)

    if context.reducer is not None:
        logger.info("Creating %s context ...", str(context))
        corpus = reduce_corpus(corpus, logger, repo_mapping, refs, context.reducer)
        num_docs = corpus.shape[0]
        logger.info("Created context, found %d documents ...", num_docs)

    logger.info("Loading word index ...")
    with open(vocab_input_path, "r", encoding="utf-8") as fin:
        word_index: Dict[int, str] = {
            i: word.replace("\n", "") for i, word in enumerate(fin)
        }
    logger.info("Loaded word index, found %d words.", num_words)

    logger.info("Loading word topic distributions ...")
    topic_words = np.load(wordtopic_input_path)
    num_topics = topic_words.shape[0]
    logger.info("Loaded distributions, found %d topics.", num_topics)

    logger.info("Finding common words for each topic ...")
    common_words = np.argwhere(np.sum(topic_words > min_prob, axis=0) > max_topics)
    mask = np.ones(num_words, dtype=bool)
    mask[common_words] = False
    logger.info(
        "Found %d words with probability over %.4f for more then %d topics, "
        "they will not be considered for labels.",
        len(common_words),
        min_prob,
        max_topics,
    )
    if len(common_words) == num_words:
        logger.info("All words were excluded, cannot infer label.")
        return
    coeff = mu / (num_topics - 1)
    words_counts = np.sum(corpus, axis=0)
    logger.info("Inferring labels for each topic ...")
    best_labels_per_topic: Dict[int, Dict[str, float]] = {}
    best_scores: Dict[str, float] = defaultdict(lambda: -np.inf)
    for cur_topic in range(num_topics):
        logger.info("Topic %d:", cur_topic + 1)
        num_admissible = len(np.argwhere(topic_words[cur_topic] > min_prob).flatten())
        admissible_words = np.argwhere(
            topic_words[cur_topic, mask] > min_prob
        ).flatten()
        if not len(admissible_words):
            logger.info("No admissible words where found, cannot infer label.")
            return
        logger.info(
            "\tFound %d words with probability over %.4f, %d remained after removing "
            "common words.",
            num_admissible,
            min_prob,
            len(admissible_words),
        )
        candidates = []
        candidates_names = []
        candidates_counts: Union[List, np.array] = []
        candidates_sizes = []
        for candidate in itertools.combinations(admissible_words, label_size):
            if np.min(corpus[:, candidate], axis=1).any():
                candidates.append(candidate)
                candidates_names.append(" ".join(word_index[w] for w in candidate))
                candidates_counts.append(np.prod(corpus[:, list(candidate)], axis=1))
                candidates_sizes.append(len(candidate))
        num_cand = len(candidates_names)
        if not num_cand:
            logger.info("No candidates where found, cannot infer label.")
            return
        logger.info("\tFound %d candidate labels, computing their scores ...", num_cand)
        candidates_counts = np.array(candidates_counts)
        joint_counts = candidates_counts @ corpus
        candidates_counts = np.sum(candidates_counts, axis=1)
        if smoothing:
            joint_counts += 1
        else:
            inds = np.argwhere(joint_counts == 0)
            joint_counts[joint_counts == 0] = (
                candidates_counts[inds[:, 0]] * words_counts[inds[:, 1]]
            )
        for cand_ind, candidate in enumerate(candidates):
            joint_counts[cand_ind, list(candidate)] = candidates_counts[cand_ind]

        # denominator = constant term > so we use counts instead of probs to compute PMI

        pmi = np.log(
            joint_counts / (candidates_counts[:, None] @ words_counts[:, None].T)
        )
        topic_probs = np.copy(topic_words).T
        topic_probs[:, cur_topic] *= coeff + 1
        topic_probs[:, [t for t in range(num_topics) if t != cur_topic]] *= -coeff
        scores = {
            name: score
            for name, score in zip(candidates_names, np.sum(pmi @ topic_probs, axis=1))
        }
        logger.info("\tTop 5 candidates:")
        best_labels = sorted(scores, key=scores.get, reverse=True)[:num_topics]
        best_labels_per_topic[cur_topic] = {}
        for label in best_labels:
            if scores[label] > best_scores[label]:
                for topic in best_labels_per_topic:
                    if label in best_labels_per_topic[topic]:
                        best_labels_per_topic[topic].pop(label)
                best_labels_per_topic[cur_topic][label] = scores[label]
                best_scores[label] = scores[label]
        for i, label_name in enumerate(best_labels[:5]):
            logger.info("\t\t %d. %s : %.4f", i + 1, label_name, scores[label_name])

    topic_labels: List[str] = []
    for cur_topic in range(num_topics):
        scores = best_labels_per_topic[cur_topic]
        topic_labels.append(sorted(scores, key=scores.get, reverse=True)[0])

    logger.info("Selected the following labels:")
    for ind_label, label in enumerate(topic_labels):
        logger.info(
            "\tTopic %d : %s (score: %.4f)", ind_label + 1, label, best_scores[label]
        )

    logger.info("Saving topic labels ...")
    with open(labels_output_path, "w", encoding="utf-8") as fout:
        fout.write("\n".join(label for label in topic_labels))
    logger.info("Saved topic labels in '%s'.", labels_output_path)
