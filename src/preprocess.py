import os
import re
import pickle
import logging
import pymysql
import warnings
import numpy as np

from nltk import PorterStemmer
from collections import defaultdict

warnings.filterwarnings("ignore")

GITBASE_EXTRACT_VERSION = """
SELECT rf.ref_name
FROM repositories r
    NATURAL JOIN refs rf
WHERE r.repository_id = '{}' AND is_tag(rf.ref_name);
"""

GITBASE_EXTRACT_WORDS = """
SELECT f.file_path,
       language(f.file_path, f.blob_content) as lang,
       uast_extract(uast(f.blob_content, language(f.file_path, f.blob_content), '{}'), '{}') as words
FROM repositories r
    NATURAL JOIN refs rf
    NATURAL JOIN commits c
    NATURAL JOIN commit_files cf
    NATURAL JOIN files f
WHERE r.repository_id = '{}' AND rf.ref_name='{}' AND words IS NOT NULL;
"""

ID = "identifiers"
LIT = "literals"
COM = "comments"

IDENTIFIER_XPATH = "//uast:Identifier"
LITERAL_XPATH = "//uast:String"
COMMENT_XPATH = "//uast:Comment"

IDENTIFIER_KEY = "Name"
LITERAL_KEY = "Value"
COMMENT_KEY = "Text"

FEATURE_MAPPING = {ID: (IDENTIFIER_XPATH, IDENTIFIER_KEY),
                   LIT: (LITERAL_XPATH, LITERAL_KEY),
                   COM: (COMMENT_XPATH, COMMENT_KEY)}


def extract(host, port, user, password, sql):
    try:
        connection = pymysql.connect(host=host, port=port, user=user, password=password, db="",
                                     cursorclass=pymysql.cursors.SSDictCursor, use_unicode=False)
        with connection.cursor() as cursor:
            cursor.execute(sql)
            for row in cursor.fetchall_unbuffered():
                yield row
    finally:
        connection.close()


def process_row(row, tokenize, stem):
    document = row["file_path"].decode()
    lang = row["lang"].decode()
    words = row["words"].decode()[1:-1].split(",")
    words = [w for word in words for w in word.split()]
    if tokenize:
        words = [w for word in words for w in word.split("_")]
        words = [w for word in words for w in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", word)]
    words = [word.lower() for word in words]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    word_dict = defaultdict(int)
    for word in words:
        word_dict[word] += 1
    return document, lang, dict(word_dict)


def preprocess(args):
    logger = logging.getLogger("preprocess")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(args.log_level)
    if os.path.exists(args.output):
        if args.force:
            logger.warn("File {} already exists, removing it.".format(args.output))
            os.remove(args.output)
        else:
            logger.error("File {} already exists, aborting (use force to remove).".format(args.output))
            return 1
    repos = []
    if args.repo is None:
        if not os.path.exists(args.repo_list):
            logger.error("File {} does not exist, aborting.".format(args.repo_list))
            return 1
        logger.info("Reading {} ...")
        with open(args.repo_list) as _in:
            for line in _in.readlines():
                repos.append(line)
        logger.info("Found {} repository names".format(len(repos)))
    else:
        repos.append(args.repo)
    output_dict = {repo: {} for repo in repos}
    document_count = 0
    vocabulary = set()
    host, port, user, password = args.gitbase_host, args.gitbase_port, args.gitbase_user, args.gitbase_pass
    for repo in repos:
        logger.info("Processing repository '{}'".format(repo))
        logger.info("Extracting tagged references ...")
        sql = GITBASE_EXTRACT_VERSION.format(repo)
        refs = [row["ref_name"].decode() for row in extract(host, port, user, password, sql)]
        logger.info("Found {} tagged references.".format(len(refs)))
        for ref in refs:
            doc_word_dict = defaultdict(dict)
            logger.info("Processing '{}' ...".format(ref))
            for feature, include_feature in {ID: True, LIT: args.literals, COM: args.comments}.items():
                if not include_feature:
                    continue
                logger.info("Extracting {} ...".format(feature))
                uast_xpath, uast_key = FEATURE_MAPPING[feature]
                sql = GITBASE_EXTRACT_WORDS.format(uast_xpath, uast_key, repo, ref)
                lang_count = defaultdict(int)
                for row in extract(host, port, user, password, sql):
                    document, lang, word_dict = process_row(row, args.tokenize, args.stem)
                    document_count += 1
                    vocabulary = vocabulary.union(word_dict.keys())
                    doc_word_dict[document][feature] = {"words": word_dict}
                    doc_word_dict[document]["lang"] = lang
                    lang_count[lang] += 1
                logger.info("Extracted {} from:".format(feature))
                for lang in sorted(lang_count):
                    logger.info("   {} {} files.".format(lang_count[lang], lang))
            output_dict[repo][ref] = dict(doc_word_dict)
    logger.info("Extracted {} distinct words from {} documents.".format(len(vocabulary), document_count))
    output_dir = os.path.dirname(args.output)
    if not (output_dir == "" or os.path.exists(output_dir)):
        logger.warn("Creating directory {}.".format(output_dir))
        os.makedirs(output_dir)
    logger.info("Saving features in {} ...".format(args.output))
    with open(args.output, "wb") as _out:
        pickle.dump(output_dict, _out)
    logger.info("Saved features.")
