from collections.abc import Mapping, MutableMapping
import logging
from logging import Handler, Logger, LogRecord, NOTSET
import os
import shutil
import time
from typing import (
    List,
    Mapping as MappingType,
    MutableMapping as MutableMappingType,
    Optional,
)

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


def check_range(
    value: float, arg_name: str, min_value: float = 0.0, max_value: float = 1.0
) -> None:
    if not (min_value <= value <= max_value):
        raise RuntimeError(
            "Argument '%s' must be in the range [%.1f, %.1f], aborting.",
            arg_name,
            min_value,
            max_value,
        )


def check_remove(
    path: str,
    logger: Logger,
    force: bool,
    is_dir: bool = False,
    is_symlink: bool = False,
) -> None:
    if os.path.exists(path):
        if not force:
            raise RuntimeError(
                "'%s' already exists, aborting (use force to remove)." % path
            )
        if is_dir:
            if not os.path.isdir(path):
                raise RuntimeError("'%s' is a file or a link, aborting." % path)
            logger.warn("Directory '%s' already exists, removing it." % path)
            shutil.rmtree(path)
        if is_symlink:
            if not os.path.islink(path):
                raise RuntimeError("'%s' is a directory or a file, aborting." % path)
            logger.warn("Symbolic link '%s' already exists, removing it." % path)
            os.unlink(path)
        else:
            if not os.path.isfile(path):
                raise RuntimeError("'%s' is a directory or a link, aborting." % path)
            logger.warn("File '%s' already exists, removing it." % path)
            os.remove(path)


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


def recursive_update(d: MutableMappingType, u: MappingType) -> None:
    def recursive_worker(d: MutableMappingType, u: MappingType) -> MutableMappingType:
        for k, v in u.items():
            dv = d.get(k, {})
            if not isinstance(dv, MutableMapping):
                d[k] = v
            elif isinstance(v, Mapping):
                d[k] = recursive_worker(dv, v)
            else:
                d[k] = v
        return d

    recursive_worker(d, u)
