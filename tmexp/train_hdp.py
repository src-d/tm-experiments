import os
from typing import Dict, List, Tuple

import gensim
import numpy as np
import tqdm

from .io_constants import (
    BOW_DIR,
    DOCTOPIC_FILENAME,
    DOCWORD_FILENAME,
    TOPICS_DIR,
    VOCAB_FILENAME,
    WORDTOPIC_FILENAME,
)
from .utils import check_file_exists, check_remove, create_directory, create_logger


def train_hdp(
    bow_name: str,
    exp_name: str,
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
    log_level: str,
) -> None:
    logger = create_logger(log_level, __name__)

    input_dir = os.path.join(BOW_DIR, bow_name)
    words_input_path = os.path.join(input_dir, VOCAB_FILENAME)
    check_file_exists(words_input_path)
    docword_input_path = os.path.join(input_dir, DOCWORD_FILENAME)
    check_file_exists(docword_input_path)

    output_dir = os.path.join(TOPICS_DIR, bow_name, exp_name)
    doctopic_output_path = os.path.join(output_dir, DOCTOPIC_FILENAME)
    check_remove(doctopic_output_path, logger, force)
    wordtopic_output_path = os.path.join(output_dir, WORDTOPIC_FILENAME)
    check_remove(wordtopic_output_path, logger, force)
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

    logger.info("Inferring topics per document ...")
    document_topics = np.empty((len(corpus), T))
    for ind_doc, bow in tqdm.tqdm(enumerate(corpus)):
        gammas = hdp.inference([bow])[0]
        document_topics[ind_doc, :] = gammas / sum(gammas)

    logger.info("Saving topics per document ...")
    np.save(doctopic_output_path, document_topics)
    logger.info("Saved topics per document in '%s'." % doctopic_output_path)

    logger.info("Saving word/topic distribution ...")
    np.save(wordtopic_output_path, hdp.get_topics())
    logger.info("Saved word/topic distribution in '%s'." % wordtopic_output_path)
