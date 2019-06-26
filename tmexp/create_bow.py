from collections import Counter
import copy
import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import tqdm

from .utils import (
    check_exists,
    check_remove_file,
    create_directory,
    create_language_list,
)

DIFF_MODEL = "diff"
HALL_MODEL = "hall"
SEP = ":"


def check_fraction(fraction: float, arg_name: str) -> None:
    if not (0 <= fraction <= 1):
        raise RuntimeError(
            "Argument '%s' must be in the range [0, 1], aborting." % arg_name
        )


def create_bow(
    input_path: str,
    output_dir: str,
    dataset_name: str,
    langs: Optional[List[str]],
    exclude_langs: Optional[List[str]],
    features: List[str],
    force: bool,
    topic_model: str,
    min_word_frac: float,
    max_word_frac: float,
    log_level: str,
) -> None:
    logger = logging.getLogger("create_bow")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(log_level)

    check_exists(input_path)
    create_directory(output_dir, logger)
    if dataset_name == "":
        dataset_name = topic_model
    words_output_path = os.path.join(output_dir, "vocab." + dataset_name + ".txt")
    check_remove_file(words_output_path, logger, force)
    docword_output_path = os.path.join(output_dir, "docword." + dataset_name + ".txt")
    check_remove_file(docword_output_path, logger, force)
    docs_output_path = os.path.join(output_dir, "docs." + dataset_name + ".txt")
    check_remove_file(docs_output_path, logger, force)
    check_fraction(min_word_frac, "min-word-frac")
    check_fraction(max_word_frac, "max-word-frac")

    logger.info("Reading pickled dict from '%s' ..." % input_path)
    with open(input_path, "rb") as _in:
        input_dict = pickle.load(_in)

    logger.info("Computing bag of words ...")
    langs = create_language_list(langs, exclude_langs)
    all_refs = sorted([ref for ref in input_dict["files_info"]])
    doc_freq: Counter = Counter()
    bow: Dict[str, Dict[int, Dict[str, Any]]] = {}
    num_docwords = 0
    num_blobs = 0
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
        word_counts = {}
        for blob_hash, feature_dict in blob_dict.items():
            word_dict: Counter = Counter()
            for feature in features:
                if feature not in feature_dict:
                    continue
                word_dict.update(feature_dict[feature])
            if word_dict:
                num_blobs += 1
                doc_freq.update(word_dict.keys())
                word_counts[blob_hash] = word_dict
        if not word_counts:
            continue
        file_bows: Dict[int, Dict[str, Any]] = {}
        prev_ref = None
        prev_blob = None
        if topic_model == DIFF_MODEL:
            prev_word_counts: Counter = Counter()
        for ref in all_refs:
            if ref in refs:
                cur_blob = input_dict["files_info"][ref][file_path]["blob_hash"]
                cur_word_counts = copy.deepcopy(word_counts[cur_blob])
            else:
                if prev_ref is None:
                    continue
                cur_blob = None
                cur_word_counts = Counter()
            if not (cur_blob is None and topic_model == HALL_MODEL):
                prev_id = len(file_bows) - 1
                if prev_blob == cur_blob:
                    file_bows[prev_id]["refs"].append(ref)
                else:
                    file_bows[prev_id + 1] = {"refs": [ref]}
                    if topic_model == HALL_MODEL:
                        file_bows[prev_id + 1]["all"] = cur_word_counts
                    elif topic_model == DIFF_MODEL:
                        cur_word_counts.subtract(prev_word_counts)
                        file_bows[prev_id + 1]["added"] = {
                            word: count
                            for word, count in cur_word_counts.items()
                            if count > 0
                        }
                        file_bows[prev_id + 1]["removed"] = {
                            word: -count
                            for word, count in cur_word_counts.items()
                            if count < 0
                        }
                        if cur_blob is None:
                            prev_word_counts = Counter()
                        else:
                            prev_word_counts = word_counts[cur_blob]
                    num_docwords += len(
                        [count for count in cur_word_counts.values() if count != 0]
                    )
                    prev_blob = cur_blob
            prev_ref = ref
        bow[file_path] = file_bows

    if min_word_frac > 0 or max_word_frac < 1:
        logger.info("Used %d blobs to create documents." % num_blobs)
        min_word_blob = int(min_word_frac * num_blobs)
        max_word_blob = int(max_word_frac * num_blobs)
        logger.info(
            "Finding words that appear in less then %d blobs or more then %d blobs ..."
            % (min_word_blob, max_word_blob)
        )
        blacklisted_words = set(
            [
                word
                for word, count in doc_freq.items()
                if count < min_word_blob or count > max_word_blob
            ]
        )
        logger.info("Found %d words. Removing them ..." % len(blacklisted_words))
        for file_path, file_bows in bow.items():
            for index, ind_dict in file_bows.items():
                for suffix, word_dict in ind_dict.items():
                    if suffix == "refs":
                        continue
                    bow[file_path][index][suffix] = {
                        word: count
                        for word, count in word_dict.items()
                        if word not in blacklisted_words
                    }

    logger.info("Creating word index ...")
    sorted_vocabulary = sorted(doc_freq.keys())
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
