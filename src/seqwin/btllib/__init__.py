"""
btllib
======

Generate minimizer sketches, with code adopted from `btllib<https://github.com/bcgsc/btllib>`__. 

Usage:
------
```python
>>> from btllib import indexlr
>>> kmers, record_ids, record_offsets, assembly_offsets = indexlr(
>>>     assembly_paths=[Path('example1.fa'), Path('example2.fa.gz')], 
>>>     kmerlen=21, 
>>>     windowsize=200, 
>>>     assembly_idx=[0, 1], 
>>>     is_target=[True, False], 
>>> )
```

Dependencies:
-------------
- numpy
- zlib

Functions:
----------
- indexlr

Attributes:
-----------
- KMER_DTYPE (np.dtype)
"""

__license__ = 'GPL 3.0'

from pathlib import Path
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

# _core.so will be installed under this dir (see cpp/CMakeLists.txt)
from ._core import indexlr_native

KMER_DTYPE = np.dtype([
    ('hash', np.uint64), 
    ('pos', np.uint32), 
    ('record_idx', np.uint16), 
    ('assembly_idx', np.uint16), 
    ('is_target', np.bool_), 
])


def indexlr(
    assembly_paths: Iterable[Path],  
    kmerlen: int, 
    windowsize: int, 
    assembly_idx: Iterable[int], 
    is_target: Iterable[bool]
) -> tuple[
    NDArray[np.void], 
    list[tuple[str, ...]], 
    NDArray[np.uint64], 
    NDArray[np.uint64]
]:
    """Compute minimizers for all FASTA records in each assembly in order. 
    `assembly_paths`, `assembly_idx`, and `is_target` are parallel lists. 

    Args:
        assembly_paths (Iterable[Path]): Path to each assembly file in FASTA format (gzip supported). 
        kmerlen (int): k-mer length. 
        windowsize (int): Window size for minimizer sketch. 
        assembly_idx (Iterable[int]): Index of each assembly. 
        is_target (Iterable[bool]): True for target assemblies. 

    Returns:
        tuple: A tuple containing
            1. NDArray[np.void]: A 1-D Numpy structured array of k-mers from all assemblies, with dtype `KMER_DTYPE`. 
                Each element represents a minimizer, with fields, 
                - 'hash' (uint64): Hash value of the minimizer. 
                - 'pos' (uint32): Position of the first base of the minimizer. 
                - 'record_idx' (uint16): 0-based index of the sequence records, in the same order as they appear in the FASTA file. 
                - 'assembly_idx' (uint16): Assembly index. 
                - 'is_target' (bool): True for target assemblies. 
            2. list[tuple[str, ...]]: FASTA record IDs of each assembly. 
            3. NDArray[np.uint64]: Global minimizer indices where a new FASTA record starts contributing minimizers. 
            4. NDArray[np.uint64]: Global minimizer indices where a new assembly starts contributing minimizers. 
    """
    kmers, idx_to_id, record_offsets, assembly_offsets = indexlr_native(
        list(str(p) for p in assembly_paths), 
        int(kmerlen), 
        int(windowsize), 
        list(int(idx) for idx in assembly_idx), 
        list(bool(target) for target in is_target)
    )
    return kmers.view(KMER_DTYPE), list(tuple(ids) for ids in idx_to_id), record_offsets, assembly_offsets


__all__ = ['indexlr']
