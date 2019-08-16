from typing import Callable

import numpy as np

from .constants import ADD, DEL, DOC
from .data import RefList, RefMapping


def diff_to_hall_reducer(
    corpus: np.ndarray,
    new_corpus: np.ndarray,
    refs: RefList,
    ref_mapping: RefMapping,
    cur_doc_ind: int,
) -> int:
    cur_count: np.ndarray = np.zeros_like(corpus[0])
    prev_add_doc, prev_del_doc = None, None
    new_num_docs = 0
    for ref in refs:
        if ref not in ref_mapping:
            continue
        cur_add_doc = ref_mapping[ref].get(ADD)
        cur_del_doc = ref_mapping[ref].get(DEL)

        new_doc = cur_add_doc != prev_add_doc or cur_del_doc != prev_del_doc
        if new_doc:
            if cur_add_doc != prev_add_doc:
                cur_count += corpus[cur_add_doc]
                prev_add_doc = cur_add_doc
            if cur_del_doc != prev_del_doc:
                cur_count -= corpus[cur_del_doc]
                prev_del_doc = cur_del_doc
        if cur_count.any():
            new_num_docs += int(new_doc)
            new_corpus[cur_doc_ind + new_num_docs] = cur_count
            ref_mapping[ref][DOC] = cur_doc_ind + new_num_docs
    return new_num_docs


def last_ref_reducer(
    corpus: np.ndarray,
    new_corpus: np.ndarray,
    refs: RefList,
    ref_mapping: RefMapping,
    cur_doc_ind: int,
) -> int:
    if refs[-1] in ref_mapping and DOC in ref_mapping[refs[-1]]:
        new_corpus[cur_doc_ind + 1] = corpus[ref_mapping[refs[-1]][DOC]]
    return 1


def numpy_op_reducer(
    corpus: np.ndarray,
    new_corpus: np.ndarray,
    _: RefList,
    ref_mapping: RefMapping,
    cur_doc_ind: int,
    numpy_op: Callable[..., np.ndarray],
) -> int:
    doc_inds = [
        doc_mapping[DOC] for doc_mapping in ref_mapping.values() if DOC in doc_mapping
    ]
    new_corpus[cur_doc_ind + 1] = numpy_op(corpus[doc_inds], axis=0)
    return 1


def max_reducer(
    corpus: np.ndarray,
    new_corpus: np.ndarray,
    refs: RefList,
    ref_mapping: RefMapping,
    cur_doc_ind: int,
) -> int:
    return numpy_op_reducer(corpus, new_corpus, refs, ref_mapping, cur_doc_ind, np.max)


def mean_reducer(
    corpus: np.ndarray,
    new_corpus: np.ndarray,
    refs: RefList,
    ref_mapping: RefMapping,
    cur_doc_ind: int,
) -> int:
    return numpy_op_reducer(corpus, new_corpus, refs, ref_mapping, cur_doc_ind, np.mean)


def median_reducer(
    corpus: np.ndarray,
    new_corpus: np.ndarray,
    refs: RefList,
    ref_mapping: RefMapping,
    cur_doc_ind: int,
) -> int:
    return numpy_op_reducer(
        corpus, new_corpus, refs, ref_mapping, cur_doc_ind, np.median
    )


def concat_reducer(
    corpus: np.ndarray,
    new_corpus: np.ndarray,
    refs: RefList,
    ref_mapping: RefMapping,
    cur_doc_ind: int,
) -> int:
    prev_doc = np.zeros_like(corpus[0])
    empty_doc = np.zeros_like(corpus[0])
    for ref in refs:
        if ref not in ref_mapping or DOC not in ref_mapping[ref]:
            continue
        cur_doc = corpus[ref_mapping[ref][DOC]]
        new_corpus[cur_doc_ind] += np.maximum(empty_doc, cur_doc - prev_doc)
        prev_doc = cur_doc
    return 1
