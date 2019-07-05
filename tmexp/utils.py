import logging
from logging import Handler, Logger, LogRecord, NOTSET
import os
import time
from typing import List, Optional

import tqdm

# TODO: stop hardcoding when https://github.com/bblfsh/client-python/issues/168 is done
SUPPORTED_LANGUAGES = [
    "C#",
    "C++",
    "C",
    "Cuda",
    "OpenCL",
    "Metal",
    "Bash",
    "Shell",
    "Go",
    "Java",
    "JavaScript",
    "JS",
    "JSX",
    "PHP",
    "Python",
    "Ruby",
    "TypeScript",
]

DATASET_DIR = "/data/datasets"

BOW_DIR = "/data/bows"
DOC_FILE_NAME = "doc.bow_tm.txt"
DOCWORD_FILE_NAME = "docword.bow_tm.txt"
VOCAB_FILE_NAME = "vocab.bow_tm.txt"

TOPICS_DIR = "/data/topics"
DOCTOPIC_FILE_NAME = "doc.topic.txt"
WORDTOPIC_FILENAME = "word.topic.npy"

VIZ_DIR = "/data/visualisations"

CUR_TIME = None


class TqdmLoggingHandler(Handler):
    def __init__(self, level: int = NOTSET) -> None:
        super().__init__(level)

    def emit(self, record: LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


def create_logger(log_level: str, name: str) -> Logger:
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.addHandler(TqdmLoggingHandler())
    return logger


def check_create_default(out_type: str) -> str:
    global CUR_TIME
    if CUR_TIME is None:
        CUR_TIME = time.strftime("%m-%d-%H:%M")
    return "%s-%s" % (CUR_TIME, out_type)


def check_file_exists(file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError("File '%s' does not exists, aborting." % file_path)


def check_env_exists(env_name: str) -> str:
    if env_name not in os.environ:
        raise RuntimeError("Variable '%s' does not exists, aborting." % env_name)
    return os.environ[env_name]


def check_remove_file(file_path: str, logger: Logger, force: bool) -> None:
    if os.path.exists(file_path):
        if not os.path.isfile(file_path):
            raise RuntimeError(
                "Path '%s' is a directory or a link, aborting." % file_path
            )
        if not force:
            raise RuntimeError(
                "File  %s already exists, aborting (use force to remove)." % file_path
            )

        logger.warn("File '%s' already exists, removing it." % file_path)
        os.remove(file_path)


def create_directory(dir_path: str, logger: Logger) -> None:
    if not (dir_path == "" or os.path.exists(dir_path)):
        logger.warn("Creating directory '%s'." % dir_path)
        os.makedirs(dir_path)


def create_language_list(
    langs: Optional[List[str]], exclude_langs: Optional[List[str]]
) -> List[str]:
    if langs is None:
        langs = SUPPORTED_LANGUAGES
        if exclude_langs is not None:
            langs = [lang for lang in langs if lang not in exclude_langs]
    return langs
