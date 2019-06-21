from collections import Counter, defaultdict
import logging
import os
import pickle
import re
from typing import Any, Counter as CounterType, Dict, Iterator, Optional, Set, Tuple
import warnings

import bblfsh
from nltk import PorterStemmer
import pymysql
import pymysql.cursors

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
       uast(f.blob_content,
            language(f.file_path, f.blob_content),
            ('{}')) as uast
FROM repositories r
    NATURAL JOIN refs rf
    NATURAL JOIN commits c
    NATURAL JOIN commit_files cf
    NATURAL JOIN files f
WHERE r.repository_id = '{}' AND rf.ref_name='{}' AND uast IS NOT NULL;
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

FEATURE_MAPPING = {
    ID: {"xpath": IDENTIFIER_XPATH, "key": IDENTIFIER_KEY},
    LIT: {"xpath": LITERAL_XPATH, "key": LITERAL_KEY},
    COM: {"xpath": COMMENT_XPATH, "key": COMMENT_KEY},
}


def extract(
    host: str, port: str, user: str, password: str, sql: str
) -> Iterator[Dict[str, Any]]:
    try:
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            db="",
            cursorclass=pymysql.cursors.SSDictCursor,
            use_unicode=False,
        )
        with connection.cursor() as cursor:
            cursor.execute(sql)
            for row in cursor.fetchall_unbuffered():
                yield row
    finally:
        connection.close()


def process_row(
    row: Dict[str, Any], tokenize: bool, stem: bool, feature_dict: Dict[str, bool]
) -> Tuple[str, str, Dict[str, Dict[str, int]]]:
    document = row["file_path"].decode()
    lang = row["lang"].decode()
    ctx = bblfsh.decode(row["uast"])
    word_dict: Dict[str, Counter] = {
        feature: Counter()
        for feature, include_feature in feature_dict.items()
        if include_feature
    }
    if stem:
        stemmer = PorterStemmer()
    for node in ctx.load():
        for feature, uast_dict in FEATURE_MAPPING.items():
            if (
                node["@type"] not in uast_dict["xpath"]
                or node[uast_dict["key"]] is None
            ):
                continue
            words = node[uast_dict["key"]].split()
            if tokenize:
                words = [w for word in words for w in word.split("_")]
                words = [
                    w
                    for word in words
                    for w in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", word)
                ]
            words = [word.lower() for word in words]
            if stem:
                words = [stemmer.stem(word) for word in words]
            word_dict[feature].update(words)
            break
    return (
        document,
        lang,
        {
            feature: dict(feature_word_dict)
            for feature, feature_word_dict in word_dict.items()
        },
    )


def preprocess(
    repo: Optional[str],
    repo_list: Optional[str],
    force: bool,
    output: str,
    literals: bool,
    comments: bool,
    tokenize: bool,
    stem: bool,
    log_level: str,
    gitbase_host: str,
    gitbase_port: str,
    gitbase_user: str,
    gitbase_pass: str,
) -> None:
    logger = logging.getLogger("preprocess")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(log_level)
    if os.path.exists(output):
        if force:
            logger.warn("File {} already exists, removing it.".format(output))
            os.remove(output)
        else:
            raise RuntimeError(
                "File {} already exists, aborting (use force to remove).".format(output)
            )
    repos = []
    if repo is None:
        if not os.path.exists(repo_list):
            raise RuntimeError("File {} does not exist, aborting.".format(repo_list))
        logger.info("Reading {} ...")
        with open(repo_list) as _in:
            for line in _in.readlines():
                repos.append(line)
        logger.info("Found {} repository names".format(len(repos)))
    else:
        repos.append(repo)
    output_dict: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
        r: {} for r in repos
    }
    document_count = 0
    vocabulary: Set[str] = set()
    host, port, user, password = (
        gitbase_host,
        gitbase_port,
        gitbase_user,
        gitbase_pass,
    )
    for repo in repos:
        logger.info("Processing repository '{}'".format(repo))
        logger.info("Extracting tagged references ...")
        sql = GITBASE_EXTRACT_VERSION.format(repo)
        refs = [
            row["ref_name"].decode() for row in extract(host, port, user, password, sql)
        ]
        logger.info("Found {} tagged references.".format(len(refs)))
        feature_dict = {ID: True, LIT: literals, COM: comments}
        uast_xpath = " | ".join(
            [
                FEATURE_MAPPING[feature]["xpath"]
                for feature, include_feature in feature_dict.items()
                if include_feature
            ]
        )
        for ref in refs:
            doc_word_dict: Dict[str, Dict[str, Any]] = defaultdict(dict)
            logger.info("Processing '{}' ...".format(ref))
            sql = GITBASE_EXTRACT_WORDS.format(uast_xpath, repo, ref)
            lang_count: CounterType[str] = Counter()
            for row in extract(host, port, user, password, sql):
                document, lang, word_dict = process_row(
                    row, tokenize, stem, feature_dict
                )
                for word_feature_dict in word_dict.values():
                    vocabulary = vocabulary.union(word_feature_dict.keys())
                document_count += 1
                doc_word_dict[document] = {"words": word_dict}
                lang_count[lang] += 1
            logger.info("Extracted words from:")
            for lang in sorted(lang_count):
                logger.info("   {} {} files.".format(lang_count[lang], lang))
            output_dict[repo][ref] = dict(doc_word_dict)
    logger.info(
        "Extracted {} distinct words from {} documents.".format(
            len(vocabulary), document_count
        )
    )
    output_dir = os.path.dirname(output)
    if not (output_dir == "" or os.path.exists(output_dir)):
        logger.warn("Creating directory {}.".format(output_dir))
        os.makedirs(output_dir)
    logger.info("Saving features in {} ...".format(output))
    with open(output, "wb") as _out:
        pickle.dump(output_dict, _out)
    logger.info("Saved features.")
