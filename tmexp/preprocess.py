from collections import Counter, defaultdict
import logging
import os
import pickle
import re
import shutil
from typing import (
    Any,
    Counter as CounterType,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)
import warnings

import bblfsh
from nltk import PorterStemmer
import pymysql
import pymysql.cursors
import tqdm

from .gitbase_constants import (
    FEATURE_MAPPING,
    FILE_CONTENT,
    FILE_INFO,
    SUPPORTED_LANGUAGES,
    TAGGED_VERSIONS,
)

warnings.filterwarnings("ignore")


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


def create_dir(dir_path: str, logger: logging.Logger) -> None:
    if not (dir_path == "" or os.path.exists(dir_path)):
        logger.warn("Creating directory {}.".format(dir_path))
        os.makedirs(dir_path)


def preprocess(
    repo: str,
    batch_size: int,
    output: str,
    batch_dir: str,
    langs: Optional[List[str]],
    exclude_langs: Optional[List[str]],
    features: List[str],
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
        raise RuntimeError(
            "File {} already exists, aborting (use force to remove).".format(output)
        )
    create_dir(os.path.dirname(output), logger)
    create_dir(
        batch_dir, logger
    )  # TODO: remove this and everything associated when gitbase works

    if langs is None:
        langs = SUPPORTED_LANGUAGES
        if exclude_langs is not None:
            langs = [lang for lang in langs if lang not in exclude_langs]
    languages = ",".join(["'" + lang + "'" for lang in langs])
    uast_xpath = " | ".join([FEATURE_MAPPING[feature]["xpath"] for feature in features])
    if stem:
        stemmer = PorterStemmer()

    vocabulary: Dict[str, Set[str]] = {feature: set() for feature in features}
    host, port, user, password = (
        gitbase_host,
        gitbase_port,
        gitbase_user,
        gitbase_pass,
    )
    logger.info("Processing repository '{}'".format(repo))
    logger.info("Extracting tagged references ...")
    sql = TAGGED_VERSIONS.format(repo)
    refs = sorted(
        [row["ref_name"].decode() for row in extract(host, port, user, password, sql)]
    )
    logger.info("Found {} tagged references.".format(len(refs)))
    logger.info("Extracting file information ...")
    sql = FILE_INFO.format(repo, languages)

    files_info: Dict[str, Dict[str, Dict[str, str]]] = {ref: {} for ref in refs}
    lang_count: CounterType[str] = Counter()
    seen_files: Set[Tuple[str, str]] = set()
    raw_count = 0
    for row in extract(host, port, user, password, sql):
        raw_count += 1
        ref = row["ref_name"].decode()
        file_path = row["file_path"].decode()
        blob_hash = row["blob_hash"].decode()
        lang = row["lang"].decode()
        if (file_path, blob_hash) not in seen_files:
            lang_count[lang] += 1
            seen_files.add((file_path, blob_hash))
        files_info[ref][file_path] = {"blob_hash": blob_hash, "language": lang}
    logger.info("Found {} parsable files:".format(raw_count))
    for ref in refs:
        logger.info("   '{}' : {} files.".format(ref, len(files_info[ref])))
    logger.info("Found {} distinct parsable files:".format(len(seen_files)))
    for lang in sorted(lang_count):
        logger.info("   {} : {} files.".format(lang, lang_count[lang]))
    if batch_size <= 0:
        batch_size = len(seen_files)
    batch_start = len(os.listdir(batch_dir))
    if batch_start:
        logger.info("Resuming parsing from batch {}.".format(batch_start + 1))
    num_batches = len(seen_files) // batch_size + int(
        bool(len(seen_files) % batch_size)
    )
    files_content: Dict[str, Dict[str, Any]] = {}

    for batch in range(batch_start, num_batches):
        sql = FILE_CONTENT.format(
            uast_xpath, repo, languages, batch * batch_size, (batch + 1) * batch_size
        )
        batch_content: Dict[str, Dict[str, Any]] = defaultdict(dict)
        logger.info(
            "Extracting words from batch {} / {} ...".format(batch + 1, num_batches)
        )
        for row in tqdm.tqdm(
            extract(host, port, user, password, sql), total=batch_size
        ):
            file_path = row["file_path"].decode()
            blob_hash = row["blob_hash"].decode()
            ctx = bblfsh.decode(row["uast"])
            word_dict: Dict[str, Counter] = {feature: Counter() for feature in features}
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
            batch_content[file_path][blob_hash] = {
                feature: dict(feature_word_dict)
                for feature, feature_word_dict in word_dict.items()
            }
            for feature, word_feature_dict in word_dict.items():
                vocabulary[feature].update(word_feature_dict.keys())
        if num_batches > 1 and batch_dir:
            batch_path = os.path.join(batch_dir, "batch_{}.pkl".format(batch + 1))
            logger.info("Saving batch in {} ...".format(batch_path))
            with open(batch_path, "wb") as _out:
                pickle.dump(dict(batch_content), _out)
        else:
            files_content.update(dict(batch_content))
    if num_batches > 1 and batch_dir:
        for batch in range(num_batches):
            with open(
                os.path.join(batch_dir, "batch_{}.pkl".format(batch + 1)), "rb"
            ) as _in:
                files_content.update(pickle.load(_in))
    parsed_files = set(
        [
            (file_path, blob_hash)
            for file_path in files_content
            for blob_hash in files_content[file_path]
        ]
    )
    lang_count = Counter()
    for file_path, blob_hash in parsed_files:
        for ref_files in files_info.values():
            if (
                file_path in ref_files
                and ref_files[file_path]["blob_hash"] == blob_hash
            ):
                lang_count.update(ref_files[file_path]["language"])
                break
    logger.info("Parsed {} distinct files:".format(len(parsed_files)))
    for ref, ref_files in files_info.items():
        for file_path in ref_files:
            if (file_path, files_info[ref][file_path]["blob_hash"]) not in parsed_files:
                files_info[ref].pop(file_path)
    output_dict = {"files_info": dict(files_info), "files_content": files_content}
    logger.info("Saving features in {} ...".format(output))
    with open(output, "wb") as _out:
        pickle.dump(output_dict, _out)
    logger.info("Saved features.")
    shutil.rmtree(batch_dir)
    logger.info("Removed temporary data.")
