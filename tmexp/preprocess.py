from collections import Counter, defaultdict
import logging
import os
import pickle
import re
from typing import (
    Any,
    Counter as CounterType,
    DefaultDict,
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

from .gitbase_constants import FEATURE_MAPPING, FILE_CONTENT, FILE_INFO, TAGGED_REFS
from .utils import check_remove_file, create_directory, create_language_list

warnings.filterwarnings("ignore")


def extract(
    host: str, port: int, user: str, password: str, sql: str
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


def remove_file_from_dict(
    file_path: str, blob_hash: str, files_info: Dict[str, Dict[str, Dict[str, str]]]
) -> None:
    for ref in files_info:
        if (
            file_path in files_info[ref]
            and files_info[ref][file_path]["blob_hash"] == blob_hash
        ):
            files_info[ref].pop(file_path)


def preprocess(
    repo: str,
    exclude_refs: List[str],
    only_by_date: bool,
    version_sep: str,
    output_path: str,
    langs: Optional[List[str]],
    exclude_langs: Optional[List[str]],
    features: List[str],
    force: bool,
    tokenize: bool,
    stem: bool,
    gitbase_host: str,
    gitbase_port: int,
    gitbase_user: str,
    gitbase_pass: str,
    bblfsh_host: str,
    bblfsh_port: int,
    log_level: str,
) -> None:
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(log_level)

    check_remove_file(output_path, logger, force)
    create_directory(os.path.dirname(output_path), logger)

    host, port, user, password = (
        gitbase_host,
        gitbase_port,
        gitbase_user,
        gitbase_pass,
    )
    logger.info("Processing repository '%s'" % repo)
    logger.info("Retrieving tagged references ...")
    sql = TAGGED_REFS % repo
    refs_dict: DefaultDict[int, DefaultDict[int, List[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    refs = [
        row["ref_name"].decode() for row in extract(host, port, user, password, sql)
    ]
    for keyword in exclude_refs:
        refs = [ref for ref in refs if keyword not in ref]
    if not only_by_date:
        for ref in refs:
            major, minor = [
                int(re.findall(r"[0-9]+", version)[0])
                for version in ref.split(version_sep)[:2]
            ]
            refs_dict[major][minor].append(ref)
        refs = [
            ref
            for major in sorted(refs_dict)
            for minor in sorted(refs_dict[major])
            for ref in refs_dict[major][minor]
        ]
    logger.info("Found %d tagged references." % len(refs))

    languages = ",".join(
        "'%s'" % lang for lang in create_language_list(langs, exclude_langs)
    )
    sql = FILE_INFO % (repo, ",".join("'%s'" % ref for ref in refs), languages)
    files_info: Dict[str, Dict[str, Dict[str, str]]] = {ref: {} for ref in refs}
    lang_count: CounterType[str] = Counter()
    seen_files: Set[Tuple[str, str]] = set()
    raw_count = 0
    logger.info("Retrieving file information ...")
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
    logger.info("Found %d parsable blobs:" % raw_count)
    for ref in refs:
        logger.info("   '%s' : %d blobs.", ref, len(files_info[ref]))
    logger.info("Found %d distinct parsable blobs:" % len(seen_files))
    for lang in sorted(lang_count):
        logger.info("   %s : %d files.", lang, lang_count[lang])

    files_content: Dict[str, Dict[str, Any]] = defaultdict(dict)
    sql = FILE_CONTENT % (repo, ",".join("'%s'" % ref for ref in refs), languages)
    uast_xpath = " | ".join([FEATURE_MAPPING[feature]["xpath"] for feature in features])
    if stem:
        stemmer = PorterStemmer()
    vocabulary: Dict[str, Set[str]] = {feature: set() for feature in features}
    client = bblfsh.BblfshClient(bblfsh_host + ":" + str(bblfsh_port))
    lang_count = Counter()
    logger.info("Retrieving file content ...")
    for row in tqdm.tqdm(
        extract(host, port, user, password, sql), total=len(seen_files)
    ):
        file_path = row["file_path"].decode()
        blob_hash = row["blob_hash"].decode()
        lang = row["lang"].decode()
        contents = row["blob_content"].decode()
        if contents == "":
            remove_file_from_dict(file_path, blob_hash, files_info)
            continue
        try:
            ctx = client.parse(
                filename="", language=lang, contents=contents, timeout=5.0
            )
        except Exception:
            remove_file_from_dict(file_path, blob_hash, files_info)
            continue
        word_dict: Dict[str, Counter] = {feature: Counter() for feature in features}
        num_nodes = 0
        for node in ctx.filter(uast_xpath):
            num_nodes += 1
            node = node.get()
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
        if num_nodes == 0:
            remove_file_from_dict(file_path, blob_hash, files_info)
            continue
        for feature, word_feature_dict in word_dict.items():
            vocabulary[feature].update(word_feature_dict.keys())
        files_content[file_path][blob_hash] = {
            feature: dict(feature_word_dict)
            for feature, feature_word_dict in word_dict.items()
        }
        lang_count[lang] += 1
    logger.info("Parsed %d distinct blobs:" % sum(lang_count.values()))
    for lang in sorted(lang_count):
        logger.info("   %s : %d blobs.", lang, lang_count[lang])
    output_dict = {
        "files_info": dict(files_info),
        "files_content": dict(files_content),
        "refs": refs,
    }
    logger.info("Saving features in '%s' ..." % output_path)
    with open(output_path, "wb") as fout:
        pickle.dump(output_dict, fout)
    logger.info("Saved features.")
