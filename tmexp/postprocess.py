from collections import Counter, defaultdict
import os
import pickle
from typing import Counter as CounterType, DefaultDict, Dict, List

import numpy as np

from .create_bow import DIFF_MODEL, HALL_MODEL, SEP
from .io_constants import (
    BOW_DIR,
    DOC_ARTM_FILENAME,
    DOC_FILENAME,
    DOCTOPIC_FILENAME,
    DOCWORD_FILENAME,
    MEMBERSHIP_FILENAME,
    REF_FILENAME,
    TOPICS_DIR,
    WORDCOUNT_FILENAME,
)
from .utils import check_file_exists, check_remove, create_logger


def postprocess(
    bow_name: str,
    exp_name: str,
    force: bool,
    # TODO(https://github.com/src-d/tm-experiments/issues/21)
    original_document_index: bool,
    log_level: str,
) -> None:

    logger = create_logger(log_level, __name__)

    input_dir_bow = os.path.join(BOW_DIR, bow_name)
    dir_exp = os.path.join(TOPICS_DIR, bow_name, exp_name)
    # TODO(https://github.com/src-d/tm-experiments/issues/21)
    if original_document_index:
        doc_input_path = os.path.join(input_dir_bow, DOC_FILENAME)
    else:
        doc_input_path = os.path.join(dir_exp, DOC_ARTM_FILENAME)
    check_file_exists(doc_input_path)
    docword_input_path = os.path.join(input_dir_bow, DOCWORD_FILENAME)
    check_file_exists(docword_input_path)
    refs_input_path = os.path.join(input_dir_bow, REF_FILENAME)
    check_file_exists(refs_input_path)

    doctopic_input_path = os.path.join(dir_exp, DOCTOPIC_FILENAME)
    check_file_exists(doctopic_input_path)

    membership_output_path = os.path.join(dir_exp, MEMBERSHIP_FILENAME)
    check_remove(membership_output_path, logger, force)
    wordcount_output_path = os.path.join(dir_exp, WORDCOUNT_FILENAME)
    check_remove(wordcount_output_path, logger, force)

    logger.info("Loading tagged refs ...")
    with open(refs_input_path, "r", encoding="utf-8") as fin:
        refs: DefaultDict[str, List[str]] = defaultdict(list)
        for line in fin:
            repo, ref = line.split(SEP)
            refs[repo].append(ref)
    logger.info(
        "Loaded tagged refs, found %d."
        % sum(len(repo_refs) for repo_refs in refs.values())
    )

    logger.info("Loading document topics matrix ...")
    doctopic = np.load(doctopic_input_path)
    num_docs, num_topics = doctopic.shape
    logger.info(
        "Loaded matrix, found %d documents and %d topics.", num_docs, num_topics
    )

    logger.info("Loading document word counts ...")
    docword: CounterType[int] = Counter()
    with open(docword_input_path, "r", encoding="utf-8") as fin:
        for _ in range(3):
            fin.readline()
        for line in fin:
            ind_doc, _, count = map(int, line.split())
            docword[ind_doc] += count
    logger.info("Loaded word counts.")

    membership: DefaultDict[str, DefaultDict[str, Dict[str, np.array]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    wordcount: DefaultDict[str, DefaultDict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    logger.info("Loading document index ...")
    with open(doc_input_path, "r", encoding="utf-8") as fin:
        line = fin.readline()
        if ":added" in line or ":removed" in line:
            topic_model = DIFF_MODEL
            diff_mapping: DefaultDict[
                str, DefaultDict[str, DefaultDict[str, Dict[str, int]]]
            ] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        else:
            topic_model = HALL_MODEL
        fin.seek(0)
        doc_index = fin.read().split("\n")
    logger.info("Loaded document index, detected %s topic model.", topic_model)

    logger.info("Computing topic membership and total word count per document ...")
    for ind_doc in range(num_docs):
        name_refs = doc_index[ind_doc].split()
        doc_name = name_refs[0].split(SEP)
        doc_repo = doc_name[0]
        doc_path = doc_name[1]
        doc_refs = name_refs[1:]
        if topic_model == DIFF_MODEL:
            doc_type = doc_name[2]
            diff_mapping[doc_repo][doc_refs[0]][doc_path][doc_type] = ind_doc
            for ref in doc_refs[1:]:
                diff_mapping[ref][doc_path][doc_type] = None
        else:
            for ref in doc_refs:
                membership[doc_repo][ref][doc_path] = doctopic[ind_doc, :]
                wordcount[doc_repo][ref][doc_path] = docword[ind_doc]
    if topic_model == DIFF_MODEL:

        for repo, repo_refs in refs.items():
            last_membership: DefaultDict[str, np.array] = defaultdict(
                lambda: np.zeros(num_topics)
            )
            last_wordcount: CounterType[str] = Counter()
            for ref in repo_refs:
                for doc_path, doc_mapping in diff_mapping[repo][ref].items():
                    last = last_membership[doc_path]
                    last_wc = last_wordcount[doc_path]
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
                        wordcount[repo][ref][doc_path] = cur_wc
                        cur = (last * last_wc + add * add_wc - rem * rem_wc) / cur_wc
                        cur[cur > 1] = 1
                        cur[cur <= 0] = 1e-20
                        membership[repo][ref][doc_path] = cur
                    last_wordcount[doc_path] = cur_wc
                    last_membership[doc_path] = cur
    logger.info("Computed topic membership per reference.")

    logger.info("Saving document memberships ...")
    with open(membership_output_path, "wb") as fout:
        pickle.dump(dict(membership), fout)
    logger.info("Saved memberships in '%s'." % membership_output_path)

    logger.info("Saving document total word counts ...")
    with open(wordcount_output_path, "wb") as fout:
        pickle.dump(dict(wordcount), fout)
    logger.info("Saved word counts in '%s'." % wordcount_output_path)
