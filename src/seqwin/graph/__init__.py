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

from ._core import _build_native, _filter_kmers_native

from .utils import OrderedKmers

KMER_DTYPE = np.dtype([
    ('pos', np.uint32),
    ('record_idx', np.uint16),
    ('assembly_idx', np.uint16)
])

NODE_DTYPE = np.dtype([
    ('hash', np.uint64),
    ('start', np.uint64),
    ('stop', np.uint64),
    ('n_tar', np.uint32),
    ('n_neg', np.uint32),
    ('penalty', np.float64)
])

EDGE_DTYPE = np.dtype([
    ("first", np.uint64),
    ("second", np.uint64),
    ("weight", np.uint64),
])


def build(
    assembly_paths: Iterable[Path],
    kmerlen: int,
    windowsize: int,
    assembly_indices: Iterable[int],
    is_targets: Iterable[bool],
    n_cpu: int = 1
) -> tuple[
    NDArray[np.void],
    NDArray[np.uint64],
    NDArray[np.void],
    NDArray[np.void],
    list[tuple[str, ...]]
]:
    """Build a Seqwin minimizer graph.

    Example usage:
    ```python
    >>> from seqwin.graph import build
    >>> kmers, idx, nodes, edges, record_ids = build(
    >>>     assembly_paths = ...,
    >>>     kmerlen = 21,
    >>>     windowsize = 200,
    >>>     assembly_indices = ...,
    >>>     is_targets = ...,
    >>>     n_cpu = 4,
    >>> )
    ```
    - `assembly_paths`, `assembly_indices`, and `is_targets` are parallel lists.
    - `kmers` stores minimizer occurrences in all assemblies, grouped and sorted by hash.
    - `idx` is parallel to `kmers` and stores each minimizer's original generation index, ordered by genomic position.
    - `nodes` and `edges` are sorted by hash.

    The `[start, stop)` range in each node identifies minimizers with this hash.
    ```python
    >>> kmer_group = kmers[node['start']:node['stop']]
    >>> group_hash = node['hash']
    >>> original_indices = idx[node['start']:node['stop']] # strictly increasing
    ```

    Args:
        assembly_paths (Iterable[Path]): Paths to input assemblies in FASTA format (plain or gzipped).
        kmerlen (int): K-mer length for minimizer sketch.
        windowsize (int): Window size for minimizer sketch.
        assembly_indices (Iterable[int]): Assembly indices.
        is_targets (Iterable[bool]): Whether each assembly is a target assembly.
        n_cpu (int, optional): Number of worker threads to use. [1]

    Returns:
        tuple: A tuple containing
            1. NDArray[np.void]: A 1-D NumPy structured array of minimizers from all assemblies.
                Dtype: `KMER_DTYPE`
                - 'pos' (uint32): 0-based position of the minimizer within its FASTA record.
                - 'record_idx' (uint16): 0-based index of the FASTA record within the assembly.
                - 'assembly_idx' (uint16): Assembly index assigned to the source assembly.
            2. NDArray[np.uint64]: The original indices assigned when minimizers are generated (ordered by genomic positions).
            3. NDArray[np.void]: A 1-D NumPy structured array of minimizer nodes.
                Dtype: `NODE_DTYPE`
                - 'hash' (uint64): Hash value of the minimizers represented by this node.
                - 'start' (uint64): Start of the half-open range for this node's minimizer entries.
                - 'stop' (uint64): End of the half-open range for this node's minimizer entries.
                - 'n_tar' (uint32): Number of target assemblies containing this minimizer hash.
                - 'n_neg' (uint32): Number of non-target assemblies containing this minimizer hash.
                - 'penalty' (float64): Node penalty score used for downstream graph filtering.
            4. NDArray[np.void]: A 1-D NumPy structured array of weighted, undirected edges.
                Dtype: `EDGE_DTYPE`
                - 'first' (uint64): Smaller endpoint hash of the undirected edge.
                - 'second' (uint64): Larger endpoint hash of the undirected edge.
                - 'weight' (uint64): Number of assemblies where the endpoints are adjacent.
            5. list[tuple[str, ...]]: FASTA record IDs of each assembly.
    """
    return _build_native(
        list(str(p) for p in assembly_paths),
        int(kmerlen),
        int(windowsize),
        list(int(i) for i in assembly_indices),
        list(bool(t) for t in is_targets),
        int(n_cpu)
    )


def _filter_kmers(
    kmers: NDArray[np.void],
    idx: NDArray[np.uint64],
    nodes: NDArray[np.void],
    used_hashes: frozenset[np.uint64]
) -> tuple[
    NDArray[np.void],
    NDArray[np.uint64],
    NDArray[np.void]
]:
    """
    1. Remove k-mers (`kmers`, `idx` and `nodes`) not included in `used_hashes`.
    2. Update 'start' and 'stop' in nodes.

    Args:
        kmers (NDArray): See `KmerGraph.kmers`.
        idx (NDArray): See `KmerGraph.idx`.
        nodes (NDArray): See `KmerGraph.nodes`.
        used_hashes (frozenset[np.uint64]): K-mers and nodes with these hash values are kept.

    Returns:
        tuple: A tuple containing
            1. NDArray: See `KmerGraph.kmers`.
            2. NDArray: See `KmerGraph.idx`.
            3. NDArray: See `KmerGraph.nodes`.
    """
    return _filter_kmers_native(kmers, idx, nodes, used_hashes)
