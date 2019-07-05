from collections import Counter, defaultdict
import os
import pickle
from typing import Any, Counter as CounterType, DefaultDict, Dict, List, Optional, Set

import tqdm

from .utils import (
    BOW_DIR,
    check_file_exists,
    check_remove_file,
    create_directory,
    create_language_list,
    create_logger,
    DATASET_DIR,
    DOC_FILE_NAME,
    DOCWORD_FILE_NAME,
    VOCAB_FILE_NAME,
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
    dataset_name: str,
    bow_name: str,
    langs: Optional[List[str]],
    exclude_langs: Optional[List[str]],
    features: List[str],
    force: bool,
    topic_model: str,
    min_word_frac: float,
    max_word_frac: float,
    log_level: str,
) -> None:
    logger = create_logger(log_level, __name__)

    input_path = os.path.join(DATASET_DIR, dataset_name + ".pkl")
    check_file_exists(input_path)

    output_dir = os.path.join(BOW_DIR, bow_name)
    create_directory(output_dir, logger)
    words_output_path = os.path.join(output_dir, VOCAB_FILE_NAME)
    check_remove_file(words_output_path, logger, force)
    docword_output_path = os.path.join(output_dir, DOCWORD_FILE_NAME)
    check_remove_file(docword_output_path, logger, force)
    doc_output_path = os.path.join(output_dir, DOC_FILE_NAME)
    check_remove_file(doc_output_path, logger, force)

    check_fraction(min_word_frac, "min-word-frac")
    check_fraction(max_word_frac, "max-word-frac")

    logger.info("Reading pickled dict from '%s' ..." % input_path)
    with open(input_path, "rb") as fin:
        input_dict = pickle.load(fin)

    logger.info("Computing bag of words ...")
    langs = create_language_list(langs, exclude_langs)
    bow: DefaultDict[str, List[Dict[Any, int]]] = defaultdict(list)
    num_bow, num_blobs = 0, 0
    doc_freq: Counter = Counter()
    docs: DefaultDict[str, List[List[str]]] = defaultdict(list)
    for file_path, blobs in tqdm.tqdm(input_dict["files_content"].items()):
        previous_blob_hash = None
        previous_docs: List[str] = []
        previous_blobs: Set[str] = set()
        if topic_model == DIFF_MODEL:
            previous_count: CounterType = Counter()
            doc_added = file_path + SEP + "added"
            doc_deleted = file_path + SEP + "removed"
        for ref in input_dict["refs"]:
            if file_path not in input_dict["files_info"][ref]:
                if topic_model == HALL_MODEL:
                    continue
                blob_hash = None
            else:
                file_info = input_dict["files_info"][ref][file_path]
                if file_info["language"] not in langs:
                    break
                blob_hash = file_info["blob_hash"]
            if blob_hash == previous_blob_hash:
                if blob_hash is not None:
                    for doc_name in previous_docs:
                        docs[doc_name][-1].append(ref)
                continue
            elif blob_hash is None:
                bow[doc_deleted].append(previous_count)
                docs[doc_deleted].append([ref])
                previous_count = Counter()
                previous_docs = [doc_deleted]
            else:
                if blob_hash not in previous_blobs:
                    previous_blobs.add(blob_hash)
                    num_blobs += 1
                blob = blobs[blob_hash]
                word_counts: CounterType = Counter()
                for feature in features:
                    if feature not in blob:
                        continue
                    word_counts.update(blob[feature])
                if not word_counts:
                    continue
                doc_freq.update(word_counts.keys())
                if topic_model == HALL_MODEL:
                    bow[file_path].append(word_counts)
                    docs[file_path].append([ref])
                    previous_docs = [file_path]
                else:
                    word_counts.subtract(previous_count)
                    bow[doc_added].append(+word_counts)
                    docs[doc_added].append([ref])
                    previous_docs = [doc_added]
                    if previous_blob_hash is not None:
                        num_bow += 1
                        bow[doc_deleted].append(-word_counts)
                        docs[doc_deleted].append([ref])
                        previous_docs.append(doc_deleted)
                    word_counts.update(previous_count)
                    previous_count = +word_counts
            previous_blob_hash = blob_hash
            num_bow += 1
    logger.info("Computed %d bags of words from %d blobs.", num_bow, num_blobs)
    if min_word_frac > 0 or max_word_frac < 1:
        min_word_blob = int(min_word_frac * num_blobs)
        max_word_blob = int(max_word_frac * num_blobs)
        logger.info(
            "Finding words that appear in less then %d blobs or more then %d blobs ...",
            min_word_blob,
            max_word_blob,
        )
        blacklisted_words = {
            word
            for word, count in doc_freq.items()
            if count < min_word_blob or count > max_word_blob
        }
        logger.info("Found %d words." % len(blacklisted_words))
        logger.info("Pruning BOW...")
        for doc_name, counts_list in bow.items():
            bow[doc_name] = [
                {
                    word: count
                    for word, count in word_counts.items()
                    if word not in blacklisted_words
                }
                for word_counts in counts_list
            ]
            docs[doc_name] = [
                ref_list
                for i, ref_list in enumerate(docs[doc_name])
                if bow[doc_name][i]
            ]
            bow[doc_name] = [
                word_counts for word_counts in bow[doc_name] if word_counts
            ]
    logger.info("Creating word index ...")
    sorted_vocabulary = sorted(
        word for word in doc_freq if word not in blacklisted_words
    )
    word_index = {word: i for i, word in enumerate(sorted_vocabulary)}
    num_words = len(word_index)
    logger.info("Number of distinct words: %d" % num_words)
    logger.info("Saving word index ...")
    with open(words_output_path, "w", encoding="utf-8") as fout:
        fout.write("%s\n" % "\n".join(sorted_vocabulary))
    logger.info("Saved word index in '%s'" % words_output_path)

    logger.info("Creating and saving document index ...")
    document_index = {}
    num_docs = 0
    with open(doc_output_path, "w", encoding="utf-8") as fout:
        for doc in sorted(docs):
            for i, refs in enumerate(docs[doc]):
                doc_name = doc + SEP + str(i)
                document_index[doc_name] = num_docs
                num_docs += 1
                fout.write(" ".join([doc_name] + refs) + "\n")
    logger.info("Number of distinct documents : %d" % num_docs)
    logger.info("Saved document index in '%s'" % doc_output_path)

    num_nnz = sum(len(wc) for word_counts in bow.values() for wc in word_counts)
    logger.info("Number of document-word pairs: %d" % num_nnz)
    logger.info(
        "Sparsity of the document-word co-occurence matrix : %f"
        % (num_nnz / (num_docs * num_words))
    )
    logger.info("Saving bags of words ...")
    with open(docword_output_path, "w", encoding="utf-8") as fout:
        for count in [num_docs, num_words, num_nnz]:
            fout.write("%d\n" % count)
        for doc in sorted(docs):
            for i, words in enumerate(bow[doc]):
                doc_name = doc + SEP + str(i)
                for word, count in words.items():
                    fout.write(
                        "%d %d %d\n"
                        % (document_index[doc_name], word_index[word], count)
                    )
    logger.info("Saved bags of words in '%s'" % docword_output_path)
