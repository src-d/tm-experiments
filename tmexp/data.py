from typing import Any, Counter, DefaultDict, Dict, List, NamedTuple, Set


class FileInfo(NamedTuple):
    blob_hash: str
    language: str


class RefInfo(Dict[str, FileInfo]):
    pass


class FilesInfo(Dict[str, RefInfo]):
    def __init__(self, refs: List[str]):
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
    refs: Dict[str, List[str]] = {}


# ------------------------------------------------------------------


class RefList(List[str]):
    pass


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


class RepoMapping(DefaultDict[str, FileMapping]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.default_factory = FileMapping  # type: ignore
