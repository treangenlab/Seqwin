"""
Minimizer Graph
===============

Core classes and dtypes for Seqwin minimizer graphs.

Usage:
------
```python
>>> from seqwin.graph import KmerGraph
>>> help(KmerGraph)
```

Dependencies:
-------------
- numpy

Classes:
----------
- KmerGraph

Attributes:
-----------
- KMER_DTYPE (np.dtype)
- NODE_DTYPE (np.dtype)
- EDGE_DTYPE (np.dtype)
"""

__license__ = 'GPL 3.0'
__author__ = 'Michael X. Wang'

from pathlib import Path
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from ._core import _build_native, _get_penalty_native, _filter_kmers_native, _get_subgraphs_native

from .utils import OrderedKmers

KMER_DTYPE = np.dtype([
    ('pos', np.uint32),
    ('record_idx', np.uint32),
])

NODE_DTYPE = np.dtype([
    ('hash', np.uint64),
    ('start', np.uintp),
    ('stop', np.uintp),
    ('n_tar', np.uint32),
    ('n_neg', np.uint32),
    ('penalty', np.float64)
])

EDGE_DTYPE = np.dtype([
    ("first", np.uint64),
    ("second", np.uint64),
    ("weight", np.uintp),
])


class KmerGraph:
    r"""The minimizer graph class.

    Example usage:
    ```python
    >>> from seqwin.graph import KmerGraph
    >>> graph = KmerGraph(
    >>>     assembly_paths = ...,
    >>>     kmerlen = 21,
    >>>     windowsize = 200,
    >>>     n_cpu = 4,
    >>>     low_memory = False
    >>> )
    ```
    - `kmers` stores minimizer occurrences in all assemblies, grouped and sorted by hash.
    - `nodes` and `edges` are sorted by hash.

    The `[start, stop)` range in each node identifies minimizers with this hash.
    ```python
    >>> kmer_group = kmers[node['start']:node['stop']]
    >>> group_hash = node['hash']
    ```

    Use `record_offsets` to recover the original assembly and record index of each minimizer.
    ```python
    >>> import numpy as np
    >>> assembly_idx = np.searchsorted(
    >>>     record_offsets,
    >>>     kmers['record_idx'],
    >>>     side='right',
    >>> ) - 1
    >>> record_idx = kmers['record_idx'] - record_offsets[assembly_idx]
    ```

    Attributes:
        kmers (NDArray[np.void]): A 1-D NumPy structured array of minimizers from all assemblies.
            Dtype: `KMER_DTYPE`
            - 'pos' (uint32): 0-based position of the minimizer within its FASTA record.
            - 'record_idx' (uint32): 0-based global index of the FASTA record.
        nodes (NDArray[np.void]): A 1-D NumPy structured array of minimizer nodes.
            Dtype: `NODE_DTYPE`
            - 'hash' (uint64): Hash value of the minimizers represented by this node.
            - 'start' (uintp): Start of the half-open range for this node's minimizer entries.
            - 'stop' (uintp): End of the half-open range for this node's minimizer entries.
            - 'n_tar' (uint32): Node scoring placeholder initialized to 0.
            - 'n_neg' (uint32): Node scoring placeholder initialized to 0.
            - 'penalty' (float64): Node scoring placeholder initialized to 0.0.
        edges (NDArray[np.void]): A 1-D NumPy structured array of weighted, undirected edges.
            Dtype: `EDGE_DTYPE`
            - 'first' (uint64): Smaller endpoint hash of the undirected edge.
            - 'second' (uint64): Larger endpoint hash of the undirected edge.
            - 'weight' (uintp): Number of assemblies where the endpoints are adjacent.
        record_offsets (NDArray[np.uint32]): Cumulative global FASTA record offsets by assembly.
        record_ids (NDArray[np.str\_]): FASTA record IDs in global record order.
    """
    __slots__ = ('kmers', 'nodes', 'edges', 'record_offsets', 'record_ids')
    kmers: NDArray[np.void]
    nodes: NDArray[np.void]
    edges: NDArray[np.void]
    record_offsets: NDArray[np.uint32]
    record_ids: NDArray[np.str_]

    def __init__(
        self,
        assembly_paths: Iterable[Path],
        kmerlen: int,
        windowsize: int,
        low_memory: bool = False,
        n_cpu: int = 1
    ) -> None:
        """Build a minimizer graph.

        Args:
            assembly_paths (Iterable[Path]): Paths to input assemblies in FASTA format (plain or gzipped).
            kmerlen (int): K-mer length for minimizer sketch.
            windowsize (int): Window size for minimizer sketch.
            n_cpu (int, optional): Number of worker threads to use. [1]
            low_memory (bool, optional): Recompute minimizers in a second pass to reduce peak memory. [False]
        """
        self.kmers, self.nodes, self.edges, self.record_offsets, record_ids = _build_native(
            list(str(p) for p in assembly_paths),
            int(kmerlen),
            int(windowsize),
            int(n_cpu),
            bool(low_memory)
        )
        self.record_ids = np.asarray(record_ids, dtype='U')

    def save(self, path: str | Path) -> None:
        """Save the minimizer graph as a directory of NumPy arrays. Existing files are overwritten.

        Args:
            path (str | Path): Path to the graph directory.
        """
        path = Path(path)
        for name in self.__slots__:
            np.save(path / f'{name}.npy', getattr(self, name), allow_pickle=False)

    @classmethod
    def load(cls, path: str | Path) -> 'KmerGraph':
        """Load a memory-mapped minimizer graph.

        Args:
            path (str | Path): Path to the graph directory.

        Returns:
            KmerGraph: A graph backed by the saved NumPy array files.
        """
        path = Path(path)
        if not path.is_dir():
            raise NotADirectoryError(f'Not a graph directory: {path}')

        modes = {
            'kmers': 'r',
            'nodes': 'c', # copy-on-write: changes affect data in memory, but are not saved to disk
            'edges': 'r',
            'record_offsets': 'r',
            'record_ids': 'r',
        }
        paths = {name: path / f'{name}.npy' for name in modes}
        missing = [array_path.name for array_path in paths.values() if not array_path.is_file()]
        if missing:
            raise FileNotFoundError(f'Missing graph array file(s): {", ".join(missing)}')

        arrays = {
            name: np.load(array_path, mmap_mode=modes[name], allow_pickle=False)
            for name, array_path in paths.items()
        }
        expected_dtypes = {
            'kmers': KMER_DTYPE,
            'nodes': NODE_DTYPE,
            'edges': EDGE_DTYPE,
            'record_offsets': np.dtype(np.uint32),
        }
        for name, array in arrays.items():
            if array.ndim != 1:
                raise ValueError(f'Graph array {name!r} must be one-dimensional, got shape {array.shape}')
            if not array.flags.c_contiguous:
                raise ValueError(f'Graph array {name!r} must be C-contiguous')
        for name, dtype in expected_dtypes.items():
            if arrays[name].dtype != dtype:
                raise ValueError(f'Graph array {name!r} has dtype {arrays[name].dtype}, expected {dtype}')

        if arrays['record_offsets'].size == 0:
            raise ValueError("Graph array 'record_offsets' must not be empty")
        if int(arrays['record_offsets'][0]) != 0:
            raise ValueError("Graph array 'record_offsets' must start at zero")

        if arrays['record_ids'].dtype.kind != 'U':
            raise ValueError(f"Graph array 'record_ids' has dtype {arrays['record_ids'].dtype}, expected 'U'")
        if len(arrays['record_ids']) != int(arrays['record_offsets'][-1]):
            raise ValueError("Graph array 'record_ids' length must equal the final record offset")

        graph = cls.__new__(cls)
        for name, array in arrays.items():
            setattr(graph, name, array)
        return graph


def _get_penalty(
    kmers: NDArray[np.void],
    nodes: NDArray[np.void],
    record_offsets: NDArray[np.uint32],
    is_targets: Iterable[bool],
    n_cpu: int = 1
) -> None:
    """Populate node target counts and penalty scores in place.

    Args:
        kmers (NDArray): See `KmerGraph.kmers`.
        nodes (NDArray): See `KmerGraph.nodes`.
        record_offsets (NDArray[np.uint32]): See `KmerGraph.record_offsets`.
        is_targets (Iterable[bool]): Whether each assembly is a target assembly.
        n_cpu (int, optional): Number of worker threads to use. [1]
    """
    _get_penalty_native(
        kmers,
        nodes,
        record_offsets,
        np.asarray(is_targets, dtype=np.bool_, order='C'),
        int(n_cpu)
    )


def _filter_kmers(
    kmers: NDArray[np.void],
    nodes: NDArray[np.void],
    used_hashes: frozenset[np.uint64]
) -> tuple[
    NDArray[np.void],
    NDArray[np.void]
]:
    """
    1. Remove k-mers (`kmers` and `nodes`) not included in `used_hashes`.
    2. Update 'start' and 'stop' in nodes.

    Args:
        kmers (NDArray): See `KmerGraph.kmers`.
        nodes (NDArray): See `KmerGraph.nodes`.
        used_hashes (frozenset[np.uint64]): K-mers and nodes with these hash values are kept.

    Returns:
        tuple: A tuple containing
            1. NDArray: See `KmerGraph.kmers`.
            2. NDArray: See `KmerGraph.nodes`.
    """
    return _filter_kmers_native(kmers, nodes, used_hashes)
