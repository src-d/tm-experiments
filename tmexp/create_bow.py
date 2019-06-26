from collections import Counter, defaultdict
import logging
import os
import pickle
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

import tqdm

from .gitbase_constants import SUPPORTED_LANGUAGES

DIFF_MODEL = "diff"
HALL_MODEL = "hall"
SEP = ":"


def create_bow(
    input_path: str,
    output_dir: str,
    dataset_name: str,
    langs: Optional[List[str]],
    exclude_langs: Optional[List[str]],
    features: List[str],
    topic_model: str,
    log_level: str,
) -> None:
    logger = logging.getLogger("create_bow")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(log_level)

    if not os.path.exists(input_path):
        raise RuntimeError("File '%s' does not exists, aborting." % input_path)
    if not (output_dir == "" or os.path.exists(output_dir)):
        logger.warn("Creating directory '%s'." % output_dir)
        os.makedirs(output_dir)
    if dataset_name == "":
        dataset_name = topic_model
    words_output_path = os.path.join(output_dir, "vocab." + dataset_name + ".txt")
    if os.path.exists(words_output_path):
        raise RuntimeError("File '%s' already exists, aborting." % words_output_path)
    docword_output_path = os.path.join(output_dir, "docword." + dataset_name + ".txt")
    if os.path.exists(docword_output_path):
        raise RuntimeError("File '%s' already exists, aborting." % docword_output_path)
    docs_output_path = os.path.join(output_dir, "docs." + dataset_name + ".txt")
    if os.path.exists(docs_output_path):
        raise RuntimeError("File '%s' already exists, aborting." % docs_output_path)

    logger.info("Reading pickled dict from '%s' ..." % input_path)
    with open(input_path, "rb") as _in:
        input_dict = pickle.load(_in)

    logger.info("Computing bag of words ...")
    if langs is None:
        langs = SUPPORTED_LANGUAGES
        if exclude_langs is not None:
            langs = [lang for lang in langs if lang not in exclude_langs]
    all_refs = sorted([ref for ref in input_dict["files_info"]])
    vocabulary = set()
    bow = {}
    num_docwords = 0
    for file_path, blob_dict in tqdm.tqdm(input_dict["files_content"].items()):
        refs = sorted(
            [
                ref
                for ref, file_dict in input_dict["files_info"].items()
                if file_path in file_dict
            ]
        )
        lang = input_dict["files_info"][refs[0]][file_path]["language"]
        if lang not in langs:
            continue
        merged_features = {}
        for blob_hash, feature_dict in blob_dict.items():
            word_dict: Counter = Counter()
            for feature in features:
                if feature not in feature_dict:
                    continue
                for word, count in feature_dict[feature].items():
                    word_dict[word] += count
                    vocabulary.add(word)
            if word_dict:
                merged_features[blob_hash] = word_dict
        if not merged_features:
            continue
        file_bows: Dict[int, Dict[str, Any]] = {}
        prev_ref = None
        prev_blob = None
        if topic_model == DIFF_MODEL:
            prev_features: Counter = Counter()
        for ref in all_refs:
            if ref in refs:
                cur_blob = input_dict["files_info"][ref][file_path]["blob_hash"]
                cur_features = merged_features[cur_blob]
            else:
                if prev_ref is None:
                    continue
                cur_blob = None
                cur_features = Counter()
            if not (cur_blob is None and topic_model == HALL_MODEL):
                prev_id = len(file_bows) - 1
                if prev_blob == cur_blob:
                    file_bows[prev_id]["refs"].append(ref)
                else:
                    file_bows[prev_id + 1] = {"refs": [ref]}
                    if topic_model == HALL_MODEL:
                        file_bows[prev_id + 1]["all"] = cur_features
                        num_docwords += len(cur_features)
                    elif topic_model == DIFF_MODEL:
                        added, removed = {}, {}
                        words = [word for word in prev_features]
                        words = list(set(words + [word for word in cur_features]))
                        for word in words:
                            diff = cur_features[word] - prev_features[word]
                            if diff > 0:
                                added[word] = diff
                            if diff < 0:
                                removed[word] = diff
                        prev_features = cur_features
                        file_bows[prev_id + 1]["added"] = added
                        file_bows[prev_id + 1]["removed"] = removed
                        num_docwords += len(added) + len(removed)
                    prev_blob = cur_blob
            prev_ref = ref
        bow[file_path] = file_bows

    logger.info("Creating word index ...")
    sorted_vocabulary = sorted(vocabulary)
    word_index = {word: i for i, word in enumerate(sorted_vocabulary)}
    num_words = len(word_index)
    logger.info("Number of distinct words: %d" % num_words)
    logger.info("Saving word index ...")
    with open(words_output_path, "w") as _out:
        for word in sorted_vocabulary:
            _out.write(word + "\n")
    logger.info("Saved word index in '%s'" % words_output_path)

    logger.info("Creating document index ...")
    if topic_model == HALL_MODEL:
        suffixes = ["all"]
    elif topic_model == DIFF_MODEL:
        suffixes = ["added", "removed"]
    sorted_docs = sorted(
        [
            SEP.join([file_path, str(ind), suffix])
            for file_path, ind_dict in bow.items()
            for ind, content in ind_dict.items()
            for suffix in suffixes
            if content[suffix]
        ]
    )
    document_index = {doc: i for i, doc in enumerate(sorted_docs)}
    num_docs = len(document_index)
    logger.info("Number of distinct documents : %d" % len(document_index))
    logger.info("Saving document index ...")
    with open(docs_output_path, "w") as _out:
        for doc in sorted_docs:
            file_path, ind, suffix = doc.split(SEP)
            _out.write(" ".join([doc] + bow[file_path][int(ind)]["refs"]) + "\n")
    logger.info("Saved document index in '%s'" % docs_output_path)

    logger.info(
        "Sparsity of the document word co-occurence matrix : %f"
        % (num_docwords / (num_docs * num_words))
    )
    logger.info("Saving bags of words...")
    with open(docword_output_path, "w") as _out:
        for count in [num_docs, num_words, num_docwords]:
            _out.write(str(count) + "\n")
        for doc in sorted_docs:
            file_path, ind, suffix = doc.split(SEP)
            for word, count in bow[file_path][int(ind)][suffix].items():
                _out.write(
                    " ".join(
                        [str(document_index[doc]), str(word_index[word]), str(count)]
                    )
                    + "\n"
                )
    logger.info("Saved bags of words in '%s'" % docword_output_path)
