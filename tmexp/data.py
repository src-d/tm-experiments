import logging
from typing import Any, Callable, Counter, DefaultDict, Dict, List, NamedTuple, Set

import numpy as np

from .constants import ADD, DEL, DIFF_MODEL, DOC, HALL_MODEL, SEP


class RefList(List[str]):
    pass


class RefsDict(DefaultDict[str, RefList]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.default_factory = RefList  # type: ignore


class FileInfo(NamedTuple):
    blob_hash: str
    language: str


class RefInfo(Dict[str, FileInfo]):
    pass


class FilesInfo(Dict[str, RefInfo]):
    def __init__(self, refs: RefList):
        super().__init__()
        for ref in refs:
            self[ref] = RefInfo()

    def remove(self, file_path: str, blob_hash: str) -> None:
        for ref_dict in self.values():
            if file_path in ref_dict and blob_hash == ref_dict[file_path].blob_hash:
                ref_dict.pop(file_path)


class WordCount(Counter[str]):
    def __sub__(self, other):  # type: ignore
        if not isinstance(other, WordCount):
            return NotImplemented
        result = WordCount()
        for elem in set(self) | set(other):
            newcount = self[elem] - other[elem]
            if newcount > 0:
                result[elem] = newcount
        return result


class FeatureContent(Dict[str, WordCount]):
    def __init__(self, features: List[str]):
        for feature in features:
            self[feature] = WordCount()


class BlobContent(Dict[str, FeatureContent]):
    pass


class FilesContent(Dict[str, BlobContent]):
    def __init__(self, files_info: FilesInfo):
        super().__init__()
        for ref_dict in files_info.values():
            for file_path in ref_dict:
                self[file_path] = BlobContent()

    def purge(self, blacklist: Set[str]) -> None:
        for file_path in blacklist:
            self.pop(file_path)

    def map_words(self, mapping: Dict[str, str]) -> None:
        for file_path, blob_dict in self.items():
            for blob_hash, feature_dict in blob_dict.items():
                for feature, word_dict in feature_dict.items():
                    new_wc = WordCount()
                    for word, count in word_dict.items():
                        new_wc[mapping[word]] = count
                    self[file_path][blob_hash][feature] = new_wc


class Dataset(NamedTuple):
    files_info: Dict[str, FilesInfo] = {}
    files_content: Dict[str, FilesContent] = {}
    refs_dict: RefsDict = RefsDict()


# ------------------------------------------------------------------


class DocumentEvolution(NamedTuple):
    bows: List[WordCount]
    refs: List[RefList]


class EvolutionModel(Dict[str, DocumentEvolution]):
    pass


# ------------------------------------------------------------------


class DocMapping(Dict[str, int]):
    pass


class RefMapping(DefaultDict[str, DocMapping]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.default_factory = DocMapping  # type: ignore


class FileMapping(DefaultDict[str, RefMapping]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.default_factory = RefMapping  # type: ignore


FileReducer = Callable[[np.ndarray, np.ndarray, RefList, RefMapping, int], int]


class RepoMapping(DefaultDict[str, FileMapping]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.default_factory = FileMapping  # type: ignore

    def build(self, logger: logging.Logger, input_path: str) -> None:
        logger.info("Loading document index ...")
        with open(input_path, "r", encoding="utf-8") as fin:
            line = fin.readline()
            if SEP + ADD in line or SEP + DEL in line:
                self.topic_model = DIFF_MODEL
            else:
                self.topic_model = HALL_MODEL
            fin.seek(0)
            for doc_ind, line in enumerate(fin):
                doc_info = line.split()
                if self.topic_model == HALL_MODEL:
                    repo, file_path, _ = doc_info[0].split(SEP)
                    delta_type = DOC
                else:
                    repo, file_path, delta_type, _ = doc_info[0].split(SEP)
                for ref in doc_info[1:]:
                    self[repo][file_path][ref][delta_type] = doc_ind
        logger.info("Loaded document index, detected %s topic model.", self.topic_model)

    def create_corpus(self, logger: logging.Logger, input_path: str) -> np.ndarray:
        logger.info("Loading bags of words ...")
        with open(input_path, "r", encoding="utf-8") as fin:
            num_docs = int(fin.readline())
            num_words = int(fin.readline())
            fin.readline()
            corpus = np.zeros((num_docs, num_words))
            for line in fin:
                doc_id, word_id, count = map(int, line.split())
                corpus[doc_id, word_id - 1] = count
        logger.info("Loaded %d bags of words.", num_docs)
        return corpus

    def reduce_corpus(
        self,
        corpus: np.ndarray,
        logger: logging.Logger,
        refs_dict: RefsDict,
        file_reducer: FileReducer,
    ) -> np.ndarray:
        new_corpus = np.zeros_like(corpus)
        cur_doc_ind = -1
        for repo, file_mapping in self.items():
            logger.info("\tProcessing repository '%s'", repo)
            new_num_docs = 0
            for ref_mapping in file_mapping.values():
                num_added_docs = file_reducer(
                    corpus, new_corpus, refs_dict[repo], ref_mapping, cur_doc_ind
                )
                new_num_docs += num_added_docs
                cur_doc_ind += num_added_docs
            logger.info("\tExtracted %d documents.", new_num_docs)
        num_docs = cur_doc_ind + 1
        return new_corpus[:num_docs]


# ------------------------------------------------------------------


class DocWordCounts(Dict[str, int]):
    pass


class RefWordCounts(DefaultDict[str, DocWordCounts]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.default_factory = DocWordCounts  # type: ignore


class RepoWordCounts(DefaultDict[str, RefWordCounts]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.default_factory = RefWordCounts  # type: ignore


class DocMembership(Dict[str, np.ndarray]):
    pass


class RefMembership(DefaultDict[str, DocMembership]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.default_factory = DocMembership  # type: ignore


class RepoMembership(DefaultDict[str, RefMembership]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.default_factory = RefMembership  # type: ignore


# ------------------------------------------------------------------


class Metric(Dict[str, np.ndarray]):
    pass


class Metrics(NamedTuple):
    distinctness: np.ndarray
    assignment: Metric
    weight: Metric
    scatter: Metric
    focus: Metric
