import os
from typing import Dict, List, Set, Tuple

import gensim
import numpy as np
import tqdm

from .create_bow import DOCWORD_FILE_NAME, VOCAB_FILE_NAME
from .utils import check_exists, check_remove_file, create_directory, create_logger


DOCTOPIC_FILE_NAME = "doc.topic.txt"
WORDTOPIC_FILENAME = "word.topic.npy"


def train_hdp(
    input_dir: str,
    output_dir: str,
    dataset_name: str,
    force: bool,
    chunk_size: int,
    kappa: float,
    tau: float,
    K: int,
    T: int,
    alpha: int,
    gamma: int,
    eta: float,
    scale: float,
    var_converge: float,
    min_proba: float,
    log_level: str,
) -> None:
    logger = create_logger(log_level, __name__)

    input_dir = os.path.join(input_dir, dataset_name)
    output_dir = os.path.join(output_dir, dataset_name)
    words_input_path = os.path.join(input_dir, VOCAB_FILE_NAME)
    check_exists(words_input_path)
    docword_input_path = os.path.join(input_dir, DOCWORD_FILE_NAME)
    check_exists(docword_input_path)
    doctopic_output_path = os.path.join(output_dir, DOCTOPIC_FILE_NAME)
    check_remove_file(doctopic_output_path, logger, force)
    wordtopic_output_path = os.path.join(output_dir, WORDTOPIC_FILENAME)
    check_remove_file(wordtopic_output_path, logger, force)
    create_directory(output_dir, logger)

    logger.info("Loading bags of words ...")
    with open(docword_input_path, "r", encoding="utf-8") as fin:
        corpus: List[List[Tuple[int, int]]] = [[] for _ in range(int(fin.readline()))]
        logger.info("Number of documents: %d", len(corpus))
        num_words = int(fin.readline())
        logger.info("Number of words: %d", num_words)
        num_rows = int(fin.readline())
        logger.info("Number of document/word pairs: %d", num_rows)
        for line in tqdm.tqdm(fin, total=num_rows):
            doc_id, word_id, count = map(int, line.split())
            corpus[doc_id].append((word_id, count))
    logger.info("Corpus created.")

    logger.info("Loading vocabulary ...")
    with open(words_input_path, "r", encoding="utf-8") as fin:
        word_index: Dict[int, str] = {
            i: word.replace("\n", "") for i, word in enumerate(fin)
        }
    id2word = gensim.corpora.Dictionary.from_corpus(corpus, word_index)
    logger.info("Word index created.")

    logger.info("Training HDP model ...")
    hdp = gensim.models.HdpModel(
        corpus,
        id2word,
        chunksize=chunk_size,
        kappa=kappa,
        tau=tau,
        K=K,
        T=T,
        alpha=alpha,
        gamma=gamma,
        eta=eta,
        scale=scale,
        var_converge=var_converge,
    )
    logger.info("Trained the model.")

    logger.info(
        "Inferring topics per document (pruning topics with probability"
        " inferior to %.2f)..." % min_proba
    )
    num_topics: List[int] = []
    document_topics: List[List[Tuple[int, float]]] = []
    used_topics: Set[int] = set()
    for bow in tqdm.tqdm(corpus):
        gammas = hdp.inference([bow])[0]
        topic_dist = gammas / sum(gammas)
        topics = [
            (topic_id, topic_proba)
            for topic_id, topic_proba in enumerate(topic_dist)
            if topic_proba >= min_proba
        ]
        used_topics.update(topic_id for topic_id, _ in topics)
        num_topics.append(len(topics))
        topics.sort(key=lambda pair: pair[1], reverse=True)
        document_topics.append(topics)
    logger.info("%d topics remain after pruning." % len(used_topics))
    topic_mapping = {old: new for new, old in enumerate(sorted(used_topics))}
    for op, info in zip(
        [np.min, np.median, np.mean, np.max], ["Minimum", "Median", "Mean", "Maximum"]
    ):
        logger.info(
            "%s amount of topics per document after pruning: %.2f", info, op(num_topics)
        )
    logger.info("Saving pruned topics per document ...")
    with open(doctopic_output_path, "w", encoding="utf-8") as fout:
        for topics in document_topics:
            fout.write(
                "%s\n"
                % " ".join(
                    "%d,%f" % (topic_mapping[topic_id], topic_proba)
                    for topic_id, topic_proba in topics
                )
            )
    logger.info("Saved pruned topics per document in '%s'." % doctopic_output_path)

    logger.info("Saving word/topic distribution ...")
    np.save(wordtopic_output_path, hdp.get_topics()[sorted(used_topics), :])
    logger.info("Saved word/topic distribution in '%s'." % wordtopic_output_path)
