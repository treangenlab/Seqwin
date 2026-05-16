"""
Minimizer Graph
===============

Core functions and dtypes for Seqwin minimizer graphs. 

Usage:
------
```python
>>> from pathlib import Path
>>> from seqwin.graph import build
>>> kmers, idx, nodes, edges, record_ids = build(
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

Functions:
----------
- build

Attributes:
-----------
- KMER_DTYPE (np.dtype)
"""

__license__ = 'GPL 3.0'
__author__ = 'Michael X. Wang'

from pathlib import Path
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from ._core import _build_native

from .utils import OrderedKmers


KMER_DTYPE = np.dtype([
    ('hash', np.uint64), 
    ('pos', np.uint32), 
    ('record_idx', np.uint16), 
    ('assembly_idx', np.uint16), 
    ('is_target', np.bool_), 
])

NODE_DTYPE = np.dtype([
    ('hash', np.uint64), 
    ('n_tar', np.uint32), 
    ('n_neg', np.uint32), 
    ('penalty', np.float64), 
    ('start', np.uint64), 
    ('stop', np.uint64), 
])


def build(
    assembly_paths: Iterable[Path],  
    kmerlen: int, 
    windowsize: int, 
    assembly_idx: Iterable[int], 
    is_target: Iterable[bool], 
    n_cpu: int = 1
) -> tuple[
    NDArray[np.void], 
    NDArray[np.uint64], 
    NDArray[np.void], 
    NDArray[np.uint64], 
    list[tuple[str, ...]]
]:
    """Build a Seqwin minimizer graph. 
    - `assembly_paths`, `assembly_idx`, and `is_target` are parallel lists. 
    - In the returned arrays, `kmers` is grouped by node/hash. 
    - For every node:
    ```python
    >>> kmer_group = kmers[node["start"]:node["stop"]]
    >>> assert np.all(kmer_group["hash"] == node["hash"])
    >>> original_indices = idx[node["start"]:node["stop"]] # strictly increasing
    ```

    Args:
        assembly_paths (Iterable[Path]): Path to each assembly file in FASTA format (gzip supported). 
        kmerlen (int): k-mer length. 
        windowsize (int): Window size for minimizer sketch. 
        assembly_idx (Iterable[int]): Index of each assembly. 
        is_target (Iterable[bool]): True for target assemblies. 
        n_cpu (int, optional): Number of threads. [1]

    Returns:
        tuple: A tuple containing
            1. NDArray[np.void]: A 1-D Numpy structured array of k-mers from all assemblies, with dtype `KMER_DTYPE`. 
                Each element represents a minimizer, with fields, 
                - 'hash' (uint64): Hash value of the minimizer. 
                - 'pos' (uint32): Position of the first base of the minimizer. 
                - 'record_idx' (uint16): 0-based index of the sequence records, in the same order as they appear in the FASTA file. 
                - 'assembly_idx' (uint16): Assembly index. 
                - 'is_target' (bool): True for target assemblies. 
            2. NDArray[np.uint64]: The original indices assigned when k-mers are generated (k-mers with consecutive indices are adjacent in the genome). 
            3. NDArray[np.void]: A 1-D Numpy structured array of k-mer nodes, with dtype `NODE_DTYPE`. 
            4. NDArray[np.uint64]: A 3-column Numpy array of weighted, undirected edges (u, v, w). 
            5. list[tuple[str, ...]]: FASTA record IDs of each assembly. 
    """
    kmers, idx, nodes, edges, idx_to_id = _build_native(
        list(str(p) for p in assembly_paths), 
        int(kmerlen), 
        int(windowsize), 
        list(int(idx) for idx in assembly_idx), 
        list(bool(target) for target in is_target), 
        int(n_cpu)
    )
    return kmers.view(KMER_DTYPE), idx, nodes, edges, idx_to_id
