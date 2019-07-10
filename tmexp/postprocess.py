from collections import Counter, defaultdict
import os
import pickle
from typing import Dict

import numpy as np

from .create_bow import DIFF_MODEL, HALL_MODEL, SEP
from .io_constants import (
    BOW_DIR,
    DOC_FILENAME,
    DOCTOPIC_FILENAME,
    DOCWORD_FILENAME,
    MEMBERSHIP_FILENAME,
    REF_FILENAME,
    TOPICS_DIR,
    WORDCOUNT_FILENAME,
)
from .utils import check_file_exists, check_remove, create_logger


def postprocess(bow_name: str, exp_name: str, force: bool, log_level: str) -> None:

    logger = create_logger(log_level, __name__)

    input_dir_bow = os.path.join(BOW_DIR, bow_name)
    doc_input_path = os.path.join(input_dir_bow, DOC_FILENAME)
    check_file_exists(doc_input_path)
    docword_input_path = os.path.join(input_dir_bow, DOCWORD_FILENAME)
    check_file_exists(docword_input_path)
    refs_input_path = os.path.join(input_dir_bow, REF_FILENAME)
    check_file_exists(refs_input_path)

    dir_exp = os.path.join(TOPICS_DIR, bow_name, exp_name)
    doctopic_input_path = os.path.join(dir_exp, DOCTOPIC_FILENAME)
    check_file_exists(doctopic_input_path)

    membership_output_path = os.path.join(dir_exp, MEMBERSHIP_FILENAME)
    check_remove(membership_output_path, logger, force)
    wordcount_output_path = os.path.join(dir_exp, WORDCOUNT_FILENAME)
    check_remove(wordcount_output_path, logger, force)

    logger.info("Loading tagged refs ...")
    with open(refs_input_path, "r", encoding="utf-8") as fin:
        refs = fin.read().split("\n")
    logger.info("Loaded tagged refs, found %d." % len(refs))

    logger.info("Loading document topics matrix ...")
    doctopic = np.load(doctopic_input_path)
    num_docs, num_topics = doctopic.shape
    logger.info(
        "Loaded matrix, found %d documents and %d topics.", num_docs, num_topics
    )

    logger.info("Loading document word counts ...")
    docword: Dict[int, int] = Counter()
    with open(docword_input_path, "r", encoding="utf-8") as fin:
        for _ in range(3):
            fin.readline()
        for line in fin:
            ind_doc, _, count = map(int, line.split())
            docword[ind_doc] += count
    logger.info("Loaded word counts.")

    membership: Dict[str, Dict[str, np.array]] = defaultdict(dict)
    wordcount: Dict[str, Dict[str, int]] = defaultdict(dict)

    logger.info("Loading document index ...")
    with open(doc_input_path, "r", encoding="utf-8") as fin:
        line = fin.readline()
        if ":added" in line or ":removed" in line:
            topic_model = DIFF_MODEL
            diff_mapping: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
                lambda: defaultdict(dict)
            )
        else:
            topic_model = HALL_MODEL
        fin.seek(0)
        doc_index = fin.read().split("\n")
    logger.info("Loaded document index, detected %s topic model.", topic_model)

    logger.info("Computing topic membership and total word count per document ...")
    for ind_doc in range(num_docs):
        name_refs = doc_index[ind_doc].split()
        doc_name = name_refs[0].split(SEP)[0]
        doc_refs = name_refs[1:]
        if topic_model == DIFF_MODEL:
            doc_type = name_refs[0].split(SEP)[1]
            diff_mapping[doc_refs[0]][doc_name][doc_type] = ind_doc
            for ref in doc_refs[1:]:
                diff_mapping[ref][doc_name][doc_type] = None
        else:
            for ref in doc_refs:
                membership[ref][doc_name] = doctopic[ind_doc, :]
                wordcount[ref][doc_name] = docword[ind_doc]
    if topic_model == DIFF_MODEL:
        last_membership: Dict[str, np.array] = defaultdict(lambda: np.zeros(num_topics))
        last_wordcount: Dict[str, int] = defaultdict(int)
        for ref in refs:
            for doc_name, doc_mapping in diff_mapping[ref].items():
                last = last_membership[doc_name]
                last_wc = last_wordcount[doc_name]
                add_wc, rem_wc = 0, 0
                add, rem = np.zeros(num_topics), np.zeros(num_topics)
                if "added" in doc_mapping and doc_mapping["added"] is not None:
                    add_wc += docword[doc_mapping["added"]]
                    add += doctopic[doc_mapping["added"], :]
                if "removed" in doc_mapping and doc_mapping["removed"] is not None:
                    rem_wc += docword[doc_mapping["removed"]]
                    rem += doctopic[doc_mapping["removed"], :]

                cur = np.zeros(num_topics)
                cur_wc = last_wc + add_wc - rem_wc
                if cur_wc:
                    wordcount[ref][doc_name] = cur_wc
                    cur = (last * last_wc + add * add_wc - rem * rem_wc) / cur_wc
                    cur[cur > 1] = 1
                    cur[cur <= 0] = 1e-20
                    membership[ref][doc_name] = cur
                last_wordcount[doc_name] = cur_wc
                last_membership[doc_name] = cur
    logger.info("Computed topic membership per reference.")

    logger.info("Saving document memberships ...")
    with open(membership_output_path, "wb") as fout:
        pickle.dump(dict(membership), fout)
    logger.info("Saved memberships in '%s'." % membership_output_path)

    logger.info("Saving document total word counts ...")
    with open(wordcount_output_path, "wb") as fout:
        pickle.dump(dict(wordcount), fout)
    logger.info("Saved word counts in '%s'." % wordcount_output_path)
