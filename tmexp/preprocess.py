from collections import Counter, defaultdict
import os
import pickle
import re
import subprocess
import time
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

from .gitbase_queries import FILE_CONTENT, FILE_INFO, TAGGED_REFS
from .io_constants import DATASET_DIR
from .utils import (
    check_env_exists,
    check_remove,
    create_directory,
    create_language_list,
    create_logger,
)

warnings.filterwarnings("ignore")

IDENTIFIERS = "identifiers"
LITERALS = "literals"
COMMENTS = "comments"

IDENTIFIER_XPATH = "uast:Identifier"
LITERAL_XPATH = "uast:String"
COMMENT_XPATH = "uast:Comment"

IDENTIFIER_KEY = "Name"
LITERAL_KEY = "Value"
COMMENT_KEY = "Text"

FEATURE_MAPPING = {
    IDENTIFIER_XPATH: (IDENTIFIER_KEY, IDENTIFIERS),
    LITERAL_XPATH: (LITERAL_KEY, LITERALS),
    COMMENT_XPATH: (COMMENT_KEY, COMMENT_XPATH),
}


def remove_file_from_dict(
    file_path: str,
    blob_hash: str,
    files_info: Dict[str, Dict[str, Dict[str, str]]],
    files_content: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    if files_content is not None and file_path in files_content:
        files_content.pop(file_path)
    for ref in files_info:
        if file_path in files_info[ref] and (
            files_content is not None
            or files_info[ref][file_path]["blob_hash"] == blob_hash
        ):
            files_info[ref].pop(file_path)


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


def preprocess(
    repo: str,
    dataset_name: str,
    exclude_refs: List[str],
    only_by_date: bool,
    version_sep: str,
    langs: Optional[List[str]],
    exclude_langs: Optional[List[str]],
    features: List[str],
    force: bool,
    tokenize: bool,
    stem: bool,
    bblfsh_timeout: float,
    log_level: str,
) -> None:
    def feature_extractor(uast_obj: Any) -> Iterator[Tuple[str, str]]:
        if type(uast_obj) == dict:
            if "@type" in uast_obj and uast_obj["@type"] in feature_mapping:
                key, feature = feature_mapping[uast_obj["@type"]]
                if uast_obj[key] is not None:
                    yield uast_obj[key], feature
            for key in uast_obj:
                if type(uast_obj[key]) in {dict, list}:
                    yield from feature_extractor(uast_obj[key])
        elif type(uast_obj) == list:
            for uast in uast_obj:
                yield from feature_extractor(uast)

    logger = create_logger(log_level, __name__)

    output_path = os.path.join(DATASET_DIR, dataset_name + ".pkl")
    check_remove(output_path, logger, force)
    create_directory(os.path.dirname(output_path), logger)

    bblfsh_host = check_env_exists("BBLFSH_HOSTNAME")
    bblfsh_port = int(check_env_exists("BBLFSH_PORT"))
    host = check_env_exists("GITBASE_HOSTNAME")
    port = int(check_env_exists("GITBASE_PORT"))
    user = check_env_exists("GITBASE_USERNAME")
    password = check_env_exists("GITBASE_PASSWORD")

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
    if stem:
        stemmer = PorterStemmer()
    blacklisted_files: Set[str] = set()
    client = bblfsh.BblfshClient("%s:%d" % (bblfsh_host, bblfsh_port))
    parsed_count: CounterType = Counter()
    feature_mapping = {
        xpath: feature_tuple
        for xpath, feature_tuple in FEATURE_MAPPING.items()
        if feature_tuple[1] in features
    }
    logger.info("Retrieving file content ...")
    # TODO: Remove docker restart logic when this
    #       https://github.com/bblfsh/bblfshd/issues/297 is done
    for row in tqdm.tqdm(
        extract(host, port, user, password, sql), total=len(seen_files)
    ):
        file_path = row["file_path"].decode()
        if file_path in blacklisted_files:
            continue
        blob_hash = row["blob_hash"].decode()
        lang = row["lang"].decode()
        contents = row["blob_content"].decode()
        if contents == "":
            remove_file_from_dict(file_path, blob_hash, files_info)
            continue
        for attempt in range(2):
            try:
                start = time.time()
                ctx = client.parse(
                    filename="",
                    language=lang,
                    contents=contents,
                    timeout=bblfsh_timeout,
                )
                uast = ctx.get_all()
            except Exception:
                if time.time() - start > bblfsh_timeout - 0.1 and attempt == 0:
                    logger.warn("Babelfish timed out, restarting the container ...")
                    subprocess.call(
                        ["docker", "restart", bblfsh_host], stdout=subprocess.DEVNULL
                    )
                    time.sleep(10)
                    logger.warn("Restarted the container.")
                uast = None
        if uast is None:
            logger.debug(
                "Failed to parse '%s' : %s (%s file), blacklisting it.",
                file_path,
                blob_hash,
                lang,
            )
            remove_file_from_dict(file_path, blob_hash, files_info, files_content)
            blacklisted_files.add(file_path)
            continue

        parsed_count[lang] += 1
        word_dict: Dict[str, Counter] = {feature: Counter() for feature in features}
        num_nodes = 0
        for word, feature in feature_extractor(uast):
            words = word.split()
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
            if words:
                num_nodes += 1
                word_dict[feature].update(words)
        if num_nodes == 0:
            remove_file_from_dict(file_path, blob_hash, files_info)
            continue
        files_content[file_path][blob_hash] = {
            feature: dict(feature_word_dict)
            for feature, feature_word_dict in word_dict.items()
        }
    total_parsed = sum(parsed_count.values())
    logger.info("Extracted features from %d distinct blobs.", total_parsed)
    logger.debug(
        "Parsed successfully %f %% blobs.", total_parsed * 100 / len(seen_files)
    )
    for lang in sorted(parsed_count):
        logger.info("   %s : %d blobs.", lang, parsed_count[lang])
        logger.debug(
            "   Parsed successfully %f %% blobs.",
            parsed_count[lang] * 100 / lang_count[lang],
        )
    output_dict = {
        "files_info": dict(files_info),
        "files_content": dict(files_content),
        "refs": refs,
    }
    logger.info("Saving features in '%s' ..." % output_path)
    with open(output_path, "wb") as fout:
        pickle.dump(output_dict, fout)
    logger.info("Saved features.")
