from logging import Logger
import os
from typing import Counter, List, Optional

from .gitbase_constants import SUPPORTED_LANGUAGES

WordCount = Counter[str]


def check_exists(file_path: str) -> None:
    if not os.path.exists(file_path):
        raise RuntimeError("File '%s' does not exists, aborting." % file_path)


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
