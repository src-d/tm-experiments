from argparse import ArgumentParser
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
from nltk.corpus import stopwords
import pymysql
import pymysql.cursors
import tqdm

from .cli import CLIBuilder, register_command
from .gitbase_queries import (
    get_file_content_sql,
    get_file_info_sql,
    get_tagged_refs_sql,
)
from .io_constants import (
    Dataset,
    DATASET_DIR,
    FeatureContent,
    FileInfo,
    FilesContent,
    FilesInfo,
    WordCount,
)
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


def _define_parser(parser: ArgumentParser) -> None:
    cli_builder = CLIBuilder(parser)
    cli_builder.add_dataset_arg(required=False)
    cli_builder.add_feature_arg()
    cli_builder.add_force_arg()
    cli_builder.add_lang_args()

    parser.add_argument(
        "-r", "--repo", help="Name of the repo to preprocess.", required=True
    )
    parser.add_argument(
        "--exclude-refs",
        help="All refs containing one of these keywords will be excluded "
        "(e.g. all refs with `alpha`).",
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--keep-vendors", help="Keep vendors in processed files.", action="store_true"
    )
    parser.add_argument(
        "--only-head", help="Preprocess only the head revision.", action="store_true"
    )
    parser.add_argument(
        "--only-by-date",
        help="To sort the references only by date (may cause errors).",
        action="store_true",
    )
    parser.add_argument(
        "--version-sep",
        help="If sorting by version, provide the seperator between major and minor.",
        default=".",
    )
    parser.add_argument(
        "--bblfsh-timeout",
        help="Timeout for parse requests made to Babelfish.",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--use-nn",
        help="To enable the use of the neural splitter.",
        action="store_true",
    )


def good_token(token: str) -> bool:
    if len(token) < 3 or len(set(token)) == 1:
        return False
    prev_char, count = None, 0
    for char in token:
        if prev_char == char:
            count += 1
        if count == 3:
            return False
    return True


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


@register_command(parser_definer=_define_parser)
def preprocess(
    repo: str,
    dataset_name: str,
    exclude_refs: List[str],
    only_head: bool,
    only_by_date: bool,
    version_sep: str,
    langs: Optional[List[str]],
    exclude_langs: Optional[List[str]],
    keep_vendors: bool,
    features: List[str],
    force: bool,
    bblfsh_timeout: float,
    use_nn: bool,
    log_level: str,
) -> None:
    """Extract features from a repository and store them as a pickled dict."""

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

    if use_nn:
        from sourced.ml.core.algorithms.token_parser import TokenParser

        token_parser = TokenParser(single_shot=True, use_nn=True)

    logger.info("Processing repository '%s'" % repo)
    logger.info("Retrieving tagged references ...")
    sql = get_tagged_refs_sql(repository_id=repo)
    refs_dict: DefaultDict[int, DefaultDict[int, List[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    if only_head:
        refs = ["HEAD"]
        logger.info("Only extracting HEAD revision.")
    else:
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

    used_langs = create_language_list(langs, exclude_langs)
    exclude_vendors = not keep_vendors
    sql = get_file_info_sql(
        repository_id=repo,
        ref_names=refs,
        exclude_vendors=exclude_vendors,
        langs=used_langs,
    )
    files_info = FilesInfo(refs)
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
        files_info[ref][file_path] = FileInfo(blob_hash=blob_hash, language=lang)
    if raw_count:
        logger.info("Found %d parsable blobs:" % raw_count)
    else:
        logger.info("Found no parsable blobs, stopping.")
        return
    for ref in refs:
        logger.info("   '%s' : %d blobs.", ref, len(files_info[ref]))
    logger.info("Found %d distinct parsable blobs:" % len(seen_files))
    for lang in sorted(lang_count):
        logger.info("   %s : %d files.", lang, lang_count[lang])

    files_content = FilesContent(files_info)
    sql = get_file_content_sql(
        repository_id=repo,
        ref_names=refs,
        exclude_vendors=exclude_vendors,
        langs=used_langs,
    )
    stop_words = frozenset(stopwords.words("english"))
    stemmer = PorterStemmer()
    stem_mapping: Dict[str, WordCount] = defaultdict(Counter)
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
            files_info.remove(file_path, blob_hash)
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
            files_info.remove(file_path, blob_hash)
            blacklisted_files.add(file_path)
            continue

        parsed_count[lang] += 1
        feature_dict: FeatureContent = {feature: Counter() for feature in features}
        num_nodes = 0
        for word, feature in feature_extractor(uast):
            if feature == COMMENTS:
                words = [w for w in word.split() if w.lower() not in stop_words]
            else:
                words = [word]
            if use_nn:
                words = [token_parser.split(w) for w in words]
            else:
                words = [w for word in words for w in word.split("_")]
                words = [
                    w
                    for word in words
                    for w in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", word)
                ]
            words = [w.lower() for w in words]
            stems_words: List[Tuple[str, str]] = [(stemmer.stem(w), w) for w in words]
            stems_words = [(s, w) for s, w in stems_words if good_token(s)]
            if stems_words:
                num_nodes += 1
                feature_dict[feature].update(s for s, _ in stems_words)
                for stem, word in stems_words:
                    stem_mapping[stem][word] += 1
        if num_nodes == 0:
            files_info.remove(file_path, blob_hash)
            continue
        files_content[file_path][blob_hash] = {
            feature: feature_word_dict
            for feature, feature_word_dict in feature_dict.items()
        }
    files_content.purge(blacklisted_files)
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
    logger.info("Creating reverse stem mapping ...")
    reverse_mapping: Dict[str, str] = {}
    for stem in stem_mapping:
        reverse_mapping[stem] = stem_mapping[stem].most_common(1)[0][0]
    logger.info("Reversing stemming ...")
    files_content.map_words(reverse_mapping)

    dataset = Dataset(
        files_info={repo: files_info},
        files_content={repo: files_content},
        refs={repo: refs},
    )
    logger.info("Saving features ...")
    with open(output_path, "wb") as fout:
        pickle.dump(dataset, fout)
    logger.info("Saved features in '%s'." % output_path)
