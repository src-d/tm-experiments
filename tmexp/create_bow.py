from argparse import ArgumentParser
import os
import pickle
from typing import Dict, List, Optional, Set

import tqdm

from .cli import CLIBuilder, register_command
from .constants import ADD, DEL, DIFF_MODEL, HALL_MODEL, SEP
from .data import Dataset, DocumentEvolution, EvolutionModel, RefList, WordCount
from .io_constants import (
    BOW_DIR,
    DATASET_DIR,
    DOC_FILENAME,
    DOCWORD_CONCAT_FILENAME,
    DOCWORD_FILENAME,
    REF_FILENAME,
    VOCAB_CONCAT_FILENAME,
    VOCAB_FILENAME,
)
from .utils import (
    check_file_exists,
    check_range,
    check_remove,
    create_directory,
    create_language_list,
    create_logger,
)


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
    refs_output_path = os.path.join(output_dir, REF_FILENAME)
    check_remove(refs_output_path, logger, force)
    if topic_model == DIFF_MODEL:
        words_concat_output_path = os.path.join(output_dir, VOCAB_CONCAT_FILENAME)
        check_remove(words_concat_output_path, logger, force, is_symlink=True)
        docword_concat_output_path = os.path.join(output_dir, DOCWORD_CONCAT_FILENAME)
        check_remove(docword_concat_output_path, logger, force)
    words_output_path = os.path.join(output_dir, VOCAB_FILENAME)
    check_remove(words_output_path, logger, force)
    docword_output_path = os.path.join(output_dir, DOCWORD_FILENAME)
    check_remove(docword_output_path, logger, force)
    doc_output_path = os.path.join(output_dir, DOC_FILENAME)
    check_remove(doc_output_path, logger, force)

    check_range(min_word_frac, "min-word-frac")
    check_range(max_word_frac, "max-word-frac")

    logger.info("Loading dataset ...")
    with open(input_path, "rb") as fin:
        input_dataset: Dataset = pickle.load(fin)

    logger.info("Creating topic evolution model  ...")
    langs = create_language_list(langs, exclude_langs)
    evolution_model = EvolutionModel()
    doc_freq = WordCount()
    num_blobs = 0
    for repo, files_content in input_dataset.files_content.items():
        logger.info("Processing repository '%s'", repo)
        for file_path, blobs in tqdm.tqdm(files_content.items()):
            bows: Dict[str, WordCount] = {}
            for blob_hash, feature_dict in blobs.items():
                bow = WordCount()
                for feature in features:
                    if feature not in feature_dict:
                        continue
                    bow.update(feature_dict[feature])
                bows[blob_hash] = bow
                doc_freq.update(bow.keys())
            doc_evolution = DocumentEvolution(bows=[], refs=[])
            seen_blobs: Set[str] = set()
            refs = input_dataset.refs_dict[repo]
            prev_bow = None
            for ref in refs:
                file_info = input_dataset.files_info[repo][ref].get(file_path)
                if file_info is None:
                    if not seen_blobs:
                        continue
                    cur_blob_hash = None
                    cur_bow = WordCount()
                else:
                    if file_info.language not in langs:
                        break
                    cur_blob_hash = file_info.blob_hash
                    cur_bow = bows[cur_blob_hash]
                    seen_blobs.add(cur_blob_hash)
                if prev_bow == cur_bow:
                    doc_evolution.refs[-1].append(ref)
                else:
                    doc_evolution.bows.append(cur_bow)
                    doc_evolution.refs.append(RefList([ref]))
                    prev_bow = cur_bow
            if not doc_evolution.bows:
                continue
            num_blobs += len(seen_blobs)
            if topic_model == HALL_MODEL:
                evolution_model[SEP.join([repo, file_path])] = doc_evolution
            else:
                added_evolution = DocumentEvolution(bows=[], refs=[])
                removed_evolution = DocumentEvolution(bows=[], refs=[])
                prev_bow = WordCount()
                for bow, refs in zip(doc_evolution.bows, doc_evolution.refs):
                    added_evolution.bows.append(bow - prev_bow)
                    added_evolution.refs.append(RefList(refs))
                    removed_evolution.bows.append(prev_bow - bow)
                    removed_evolution.refs.append(RefList(refs))
                    prev_bow = bow
                evolution_model[SEP.join([repo, file_path, ADD])] = added_evolution
                evolution_model[SEP.join([repo, file_path, DEL])] = removed_evolution

    logger.info("Computed bags of words from %d blobs.", num_blobs)
    if min_word_frac > 0 or max_word_frac < 1:
        min_word_blob = int(min_word_frac * num_blobs)
        max_word_blob = int(max_word_frac * num_blobs)
        logger.info(
            "Finding words that appear in less then %d blobs or more then %d blobs ...",
            min_word_blob,
            max_word_blob,
        )
        blacklisted_words = frozenset(
            word
            for word, count in doc_freq.items()
            if count < min_word_blob or count > max_word_blob
        )
        logger.info("Found %d words." % len(blacklisted_words))
        logger.info("Pruning bags of words ...")
        for doc_name in sorted(evolution_model):
            doc_evolution = evolution_model.pop(doc_name)
            pruned_evolution = DocumentEvolution(bows=[], refs=[])
            prev_bow = None
            for cur_bow, refs in zip(doc_evolution.bows, doc_evolution.refs):
                for word in blacklisted_words:
                    if word in cur_bow:
                        del cur_bow[word]
                if cur_bow and (cur_bow != prev_bow or topic_model == DIFF_MODEL):
                    pruned_evolution.bows.append(cur_bow)
                    pruned_evolution.refs.append(refs)
                elif pruned_evolution.bows:
                    pruned_evolution.refs[-1] += refs
                prev_bow = cur_bow
            if pruned_evolution.bows:
                evolution_model[doc_name] = pruned_evolution

    logger.info("Saving tagged refs ...")
    with open(refs_output_path, "w", encoding="utf-8") as fout:
        for repo, refs in input_dataset.refs_dict.items():
            for ref in refs:
                fout.write("%s%s%s\n" % (repo, SEP, ref))
    logger.info("Saved tagged refs in '%s'" % refs_output_path)

    logger.info("Creating word index ...")
    sorted_vocabulary = sorted(
        word for word in doc_freq if word not in blacklisted_words
    )
    word_index = {word: i for i, word in enumerate(sorted_vocabulary, start=1)}
    num_words = len(word_index)
    logger.info("Number of words: %d" % num_words)
    logger.info("Saving word index ...")
    with open(words_output_path, "w", encoding="utf-8") as fout:
        fout.write("%s\n" % "\n".join(sorted_vocabulary))
    logger.info("Saved word index in '%s'" % words_output_path)

    logger.info("Creating and saving document index ...")
    document_index = {}
    num_docs = 0
    with open(doc_output_path, "w", encoding="utf-8") as fout:
        for doc_name in sorted(evolution_model):
            doc_evolution = evolution_model[doc_name]
            for i, refs in enumerate(doc_evolution.refs):
                doc_name_ind = SEP.join([doc_name, str(i)])
                document_index[doc_name_ind] = num_docs
                fout.write(" ".join([doc_name_ind] + refs) + "\n")
                num_docs += 1
    logger.info("Number of documents : %d" % num_docs)
    logger.info("Saved document index in '%s'" % doc_output_path)

    num_nnz = sum(
        len(bow)
        for doc_evolution in evolution_model.values()
        for bow in doc_evolution.bows
    )
    logger.info(
        "Found %d non-zero entries in the document-word co-occurence matrix "
        "(sparsity of %.4f)",
        num_nnz,
        (num_docs * num_words - num_nnz) / (num_docs * num_words),
    )
    logger.info("Saving bags of words ...")
    with open(docword_output_path, "w", encoding="utf-8") as fout:
        fout.write("%d\n" * 3 % (num_docs, num_words, num_nnz))
        for doc_name, doc_evolution in evolution_model.items():
            for i, bow in enumerate(doc_evolution.bows):
                doc_name_ind = SEP.join([doc_name, str(i)])
                for word, count in bow.items():
                    fout.write(
                        "%d %d %d\n"
                        % (document_index[doc_name_ind], word_index[word], count)
                    )
    logger.info("Saved bags of words in '%s'" % docword_output_path)

    if topic_model == DIFF_MODEL:
        logger.info("Creating consolidated corpus data ...")

        logger.info("Creating symlink to the word index ...")
        os.symlink(words_output_path, words_concat_output_path)
        logger.info("Created symlink pointing to '%s'" % words_concat_output_path)

        logger.info("Creating consolidated bags of words ...")
        consolidated_bows: List[WordCount] = []
        for doc_name, doc_evolution in evolution_model.items():
            if doc_name.split(SEP)[-1] != ADD:
                continue
            consolidated_bow = WordCount()
            for bow in doc_evolution.bows:
                consolidated_bow.update(bow)
            consolidated_bows.append(consolidated_bow)
        num_docs = len(consolidated_bows)
        logger.info("Number of consolidated documents : %d" % num_docs)
        num_nnz = sum(len(bow) for bow in consolidated_bows)
        logger.info(
            "Found %d non-zero entries in the consolidated document-word co-occurence "
            "matrix (sparsity of %.4f)",
            num_nnz,
            (num_docs * num_words - num_nnz) / (num_docs * num_words),
        )

        logger.info("Saving consolidated bags of words ...")
        with open(docword_concat_output_path, "w", encoding="utf-8") as fout:
            fout.write("%d\n" * 3 % (num_docs, num_words, num_nnz))
            for doc_ind, bow in enumerate(consolidated_bows):
                for word, count in bow.items():
                    fout.write("%d %d %d\n" % (doc_ind, word_index[word], count))
        logger.info(
            "Saved consolidated bags of words in '%s'" % docword_concat_output_path
        )
