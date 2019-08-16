from argparse import ArgumentParser
import os
import pickle

import numpy as np

from .cli import CLIBuilder, register_command
from .constants import ADD, DEL, DIFF_MODEL, DOC, HALL_MODEL
from .data import RepoMapping, RepoMembership, RepoWordCounts
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
from .reduce import diff_to_hall_reducer
from .utils import check_file_exists, check_remove, create_logger, load_refs_dict


def _define_parser(parser: ArgumentParser) -> None:
    cli_builder = CLIBuilder(parser)
    cli_builder.add_bow_arg(required=True)
    cli_builder.add_experiment_arg(required=True)
    cli_builder.add_force_arg()


@register_command(parser_definer=_define_parser)
def postprocess(bow_name: str, exp_name: str, force: bool, log_level: str) -> None:
    """Compute document word count and membership given a topic model."""
    logger = create_logger(log_level, __name__)

    dir_bow = os.path.join(BOW_DIR, bow_name)
    dir_exp = os.path.join(TOPICS_DIR, bow_name, exp_name)
    doc_input_path = os.path.join(dir_bow, DOC_FILENAME)
    check_file_exists(doc_input_path)
    docword_input_path = os.path.join(dir_bow, DOCWORD_FILENAME)
    check_file_exists(docword_input_path)
    refs_input_path = os.path.join(dir_bow, REF_FILENAME)
    check_file_exists(refs_input_path)

    doctopic_input_path = os.path.join(dir_exp, DOCTOPIC_FILENAME)
    check_file_exists(doctopic_input_path)

    wordcount_output_path = os.path.join(dir_bow, WORDCOUNT_FILENAME)
    check_remove(wordcount_output_path, logger, force)
    membership_output_path = os.path.join(dir_exp, MEMBERSHIP_FILENAME)
    check_remove(membership_output_path, logger, force)

    refs_dict = load_refs_dict(logger, refs_input_path)

    logger.info("Loading document topics matrix ...")
    doctopic = np.load(doctopic_input_path)
    num_topics = doctopic.shape[1]
    logger.info("Loaded matrix, found %d topics.", num_topics)

    memberships = RepoMembership()
    total_word_counts = RepoWordCounts()

    repo_mapping = RepoMapping()
    repo_mapping.build(logger, doc_input_path)

    corpus = repo_mapping.create_corpus(logger, docword_input_path)
    if repo_mapping.topic_model == DIFF_MODEL:
        logger.info("Creating hall model corpus ...")
        hall_corpus = repo_mapping.reduce_corpus(
            corpus, logger, refs_dict, diff_to_hall_reducer
        )
        num_docs = hall_corpus.shape[0]
        logger.info("Recreated hall model corpus, found %d documents ...", num_docs)

    logger.info("Computing topic membership and total word count per document ...")
    for repo, file_mapping in repo_mapping.items():
        logger.info("\tProcessing repository '%s'", repo)
        for doc_name, ref_mapping in file_mapping.items():
            if repo_mapping.topic_model == HALL_MODEL:
                for ref, doc_mapping in ref_mapping.items():
                    ind_doc = doc_mapping[DOC]
                    memberships[repo][ref][doc_name] = doctopic[ind_doc]
                    total_word_counts[repo][ref][doc_name] = np.sum(corpus[ind_doc])
            else:
                prev_membership = np.zeros(num_topics)
                prev_count = 0
                for ref in refs_dict[repo]:
                    if ref not in ref_mapping or DOC not in ref_mapping[ref]:
                        prev_membership = np.zeros(num_topics)
                        prev_count = 0
                        continue
                    ind_doc = ref_mapping[ref][DOC]
                    count = np.sum(hall_corpus[ind_doc])
                    membership = prev_membership * prev_count
                    for key, mult in zip([ADD, DEL], [1, -1]):
                        ind = ref_mapping[ref].get(key)
                        if ind is not None:
                            membership += mult * np.sum(corpus[ind]) * doctopic[ind]
                    membership = membership / count
                    membership[membership < 0] = 0
                    membership[membership > 1] = 1
                    membership = membership / np.sum(membership)
                    memberships[repo][ref][doc_name] = membership
                    total_word_counts[repo][ref][doc_name] = count
                    prev_membership = membership
                    prev_count = count
    logger.info("Computed topic membership per reference.")

    logger.info("Saving document memberships ...")
    with open(membership_output_path, "wb") as fout:
        pickle.dump(memberships, fout)
    logger.info("Saved memberships in '%s'." % membership_output_path)

    logger.info("Saving document total word counts ...")
    with open(wordcount_output_path, "wb") as fout:
        pickle.dump(total_word_counts, fout)
    logger.info("Saved word counts in '%s'." % wordcount_output_path)
