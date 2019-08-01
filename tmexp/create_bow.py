from argparse import ArgumentParser
from collections import Counter, defaultdict
import os
import pickle
from typing import Any, Counter as CounterType, DefaultDict, Dict, List, Optional, Set

import tqdm

from .cli import CLIBuilder, register_command
from .io_constants import (
    BOW_DIR,
    Dataset,
    DATASET_DIR,
    DOC_FILENAME,
    DOCWORD_FILENAME,
    REF_FILENAME,
    VOCAB_FILENAME,
    WordCount,
)
from .utils import (
    check_file_exists,
    check_range,
    check_remove,
    create_directory,
    create_language_list,
    create_logger,
)

DIFF_MODEL = "diff"
HALL_MODEL = "hall"
SEP = ":"


def _define_parser(parser: ArgumentParser) -> None:
    cli_builder = CLIBuilder(parser)
    cli_builder.add_bow_arg(required=False)
    cli_builder.add_dataset_arg(required=True)
    cli_builder.add_feature_arg()
    cli_builder.add_force_arg()
    cli_builder.add_lang_args()

    parser.add_argument(
        "--topic-model",
        help="Topic evolution model to use.",
        required=True,
        choices=[DIFF_MODEL, HALL_MODEL],
    )
    parser.add_argument(
        "--min-word-frac",
        help="Words occuring in less then this draction of all documents are removed,"
        " defaults to %(default)s.",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--max-word-frac",
        help="Words occuring in more then this fraction of all documents are removed,"
        " defaults to %(default)s.",
        type=float,
        default=0.8,
    )


@register_command(parser_definer=_define_parser)
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
    """Create the BoW dataset from a pickled dict, in UCI format."""
    logger = create_logger(log_level, __name__)

    input_path = os.path.join(DATASET_DIR, dataset_name + ".pkl")
    check_file_exists(input_path)

    output_dir = os.path.join(BOW_DIR, bow_name)
    create_directory(output_dir, logger)
    words_output_path = os.path.join(output_dir, VOCAB_FILENAME)
    check_remove(words_output_path, logger, force)
    docword_output_path = os.path.join(output_dir, DOCWORD_FILENAME)
    check_remove(docword_output_path, logger, force)
    doc_output_path = os.path.join(output_dir, DOC_FILENAME)
    check_remove(doc_output_path, logger, force)
    refs_output_path = os.path.join(output_dir, REF_FILENAME)
    check_remove(refs_output_path, logger, force)

    check_range(min_word_frac, "min-word-frac")
    check_range(max_word_frac, "max-word-frac")

    logger.info("Loading dataset ...")
    with open(input_path, "rb") as fin:
        input_dataset: Dataset = pickle.load(fin)

    logger.info("Computing bag of words ...")
    langs = create_language_list(langs, exclude_langs)
    bow: DefaultDict[str, DefaultDict[str, List[Dict[Any, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    num_bow, num_blobs = 0, 0
    doc_freq: CounterType[str] = Counter()
    docs: DefaultDict[str, DefaultDict[str, List[List[str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for repo, files_content in input_dataset.files_content.items():
        logger.info("Processing repository '%s'", repo)
        for file_path, blobs in tqdm.tqdm(files_content.items()):
            previous_blob_hash = None
            previous_docs: List[str] = []
            previous_blobs: Set[str] = set()
            if topic_model == DIFF_MODEL:
                previous_count: CounterType[str] = Counter()
                doc_added = file_path + SEP + "added"
                doc_deleted = file_path + SEP + "removed"
            for ref in input_dataset.refs[repo]:
                file_info = input_dataset.files_info[repo][ref].get(file_path)
                if file_info is None:
                    if topic_model == HALL_MODEL:
                        continue
                    blob_hash = None
                else:
                    if file_info.language not in langs:
                        break
                    blob_hash = file_info.blob_hash
                if blob_hash == previous_blob_hash:
                    if blob_hash is not None:
                        for doc_name in previous_docs:
                            docs[repo][doc_name][-1].append(ref)
                    continue
                elif blob_hash is None:
                    bow[repo][doc_deleted].append(previous_count)
                    docs[repo][doc_deleted].append([ref])
                    previous_count = Counter()
                    previous_docs = [doc_deleted]
                else:
                    if blob_hash not in previous_blobs:
                        previous_blobs.add(blob_hash)
                        num_blobs += 1
                    blob = blobs[blob_hash]
                    word_counts: WordCount = Counter()
                    for feature in features:
                        if feature not in blob:
                            continue
                        word_counts.update(blob[feature])
                    if not word_counts:
                        continue
                    doc_freq.update(word_counts.keys())
                    if topic_model == HALL_MODEL:
                        bow[repo][file_path].append(word_counts)
                        docs[repo][file_path].append([ref])
                        previous_docs = [file_path]
                    else:
                        word_counts.subtract(previous_count)
                        bow[repo][doc_added].append(+word_counts)
                        docs[repo][doc_added].append([ref])
                        previous_docs = [doc_added]
                        if previous_blob_hash is not None:
                            num_bow += 1
                            bow[repo][doc_deleted].append(-word_counts)
                            docs[repo][doc_deleted].append([ref])
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
        for repo, repo_bow in bow.items():
            for doc_name, counts_list in repo_bow.items():
                repo_bow[doc_name] = [
                    {
                        word: count
                        for word, count in word_counts.items()
                        if word not in blacklisted_words
                    }
                    for word_counts in counts_list
                ]
                docs[repo][doc_name] = [
                    ref_list
                    for i, ref_list in enumerate(docs[repo][doc_name])
                    if repo_bow[doc_name][i]
                ]
                repo_bow[doc_name] = [
                    word_counts for word_counts in repo_bow[doc_name] if word_counts
                ]
    logger.info("Creating word index ...")
    sorted_vocabulary = sorted(
        word for word in doc_freq if word not in blacklisted_words
    )
    word_index = {word: i for i, word in enumerate(sorted_vocabulary, start=1)}
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
        for repo in sorted(docs):
            for doc in sorted(docs[repo]):
                for i, refs in enumerate(sorted(docs[repo][doc])):
                    doc_name = SEP.join((repo, doc, str(i)))
                    document_index[doc_name] = num_docs
                    num_docs += 1
                    fout.write(" ".join([doc_name] + refs) + "\n")
    logger.info("Number of distinct documents : %d" % num_docs)
    logger.info("Saved document index in '%s'" % doc_output_path)

    logger.info("Saving tagged refs ...")
    with open(refs_output_path, "w", encoding="utf-8") as fout:
        for repo, repo_refs in input_dataset.refs.items():
            for ref in repo_refs:
                fout.write("%s%s%s\n" % (repo, SEP, ref))
    logger.info("Saved tagged refs in '%s'" % refs_output_path)

    num_nnz = sum(len(wc) for word_counts in bow.values() for wc in word_counts)
    logger.info("Number of document-word pairs: %d" % num_nnz)
    logger.info(
        "Sparsity of the document-word co-occurence matrix : %f",
        (num_docs * num_words - num_nnz) / num_docs / num_words,
    )
    logger.info("Saving bags of words ...")
    with open(docword_output_path, "w", encoding="utf-8") as fout:
        fout.write("%d\n" * 3 % (num_docs, num_words, num_nnz))
        for repo in sorted(docs):
            for doc in sorted(docs[repo]):
                for i, words in enumerate(bow[repo][doc]):
                    doc_name = SEP.join((repo, doc, str(i)))
                    for word, count in words.items():
                        fout.write(
                            "%d %d %d\n"
                            % (document_index[doc_name], word_index[word], count)
                        )

    logger.info("Saved bags of words in '%s'" % docword_output_path)
