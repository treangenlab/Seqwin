"""
Graph
=====

Classes and dtypes for Seqwin minimizer graphs.

Classes:
----------
- Graph

Attributes:
-----------
- KMER_DTYPE (np.dtype)
- EDGE_DTYPE (np.dtype)
"""

__license__ = 'GPL 3.0'
__author__ = 'Michael X. Wang'

from pathlib import Path
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ._native import _build_native

KMER_DTYPE = np.dtype([
    ('pos', np.uint32),
    ('record_idx', np.uint32),
])

EDGE_DTYPE = np.dtype([
    ("first", np.uintp),
    ("second", np.uintp),
    ("weight", np.uintp),
])


class Graph:
    r"""The Seqwin minimizer graph class.

    Example usage:
    ```python
    >>> from seqwin.core import Graph
    >>> graph = Graph(
    ...     assembly_paths = ['asm1.fasta', 'asm2.fasta'],
    ...     kmerlen = 21,
    ...     windowsize = 200,
    ...     n_cpu = 4,
    ...     low_memory = False
    ... )
    ```
    - `kmers` stores minimizer occurrences in all assemblies, grouped and sorted by hash.
    - `hashes` stores unique minimizer hashes in strictly increasing order.
    - `prevalence` stores the number of assemblies containing each node.
    - `edges` endpoints are node indices; sorted by descending weight, then by ascending endpoints.

    Node i's k-mers occupy `kmers[offsets[i]:offsets[i + 1]]`.
    `offsets` has one more element than `hashes`, starts at zero, ends at
    `len(kmers)`, and is strictly increasing. Empty graphs have `offsets == [0]`.
    ```python
    >>> kmers = graph.kmers
    >>> node_idx = 0
    >>> kmer_group = kmers[graph.offsets[node_idx]:graph.offsets[node_idx + 1]]
    >>> group_hash = graph.hashes[node_idx]
    ```

    Use `record_offsets` to recover the original assembly and record index of each minimizer.
    ```python
    >>> import numpy as np
    >>> record_offsets = graph.record_offsets
    >>> assembly_idx = np.searchsorted(
    ...     record_offsets,
    ...     kmers['record_idx'],
    ...     side='right',
    ... ) - 1
    >>> record_idx = kmers['record_idx'] - record_offsets[assembly_idx]
    ```

    `assembly_nodes` stores the sorted, unique node indices present in each assembly.
    For assembly `i`, `assembly_nodes[node_offsets[i]:node_offsets[i + 1]]`
    contains unique nodes present in that assembly, in ascending node-index order.

    Attributes:
        kmers (NDArray[np.void]): A 1-D NumPy structured array of minimizers from all assemblies.
            - 'pos' (uint32): 0-based position of the minimizer within its FASTA record.
            - 'record_idx' (uint32): 0-based global index of the FASTA record.
        offsets (NDArray[np.uintp]): Cumulative offsets into `kmers` by node, including the final sentinel.
        prevalence (NDArray[np.uint32]): Number of assemblies containing each node.
        hashes (NDArray[np.uint64]): Unique minimizer hashes in ascending order.
        edges (NDArray[np.void]): A 1-D NumPy structured array of weighted, undirected edges.
            - 'first' (uintp): Index of the smaller endpoint in the graph.
            - 'second' (uintp): Index of the larger endpoint in the graph.
            - 'weight' (uintp): Number of assemblies where the endpoints are adjacent.
        record_offsets (NDArray[np.uint32]): Cumulative global FASTA record offsets by assembly.
        record_ids (NDArray[np.str\_]): FASTA record IDs in global record order.
        assembly_nodes (NDArray[np.uintp]): Node indices grouped by assembly.
        node_offsets (NDArray[np.uintp]): Cumulative offsets into `assembly_nodes` by assembly.
    """
    __module__ = 'seqwin.core'

    __slots__ = (
        'kmers',
        'offsets',
        'prevalence',
        'hashes',
        'edges',
        'record_offsets',
        'record_ids',
        'assembly_nodes',
        'node_offsets',
    )
    kmers: NDArray[np.void]
    offsets: NDArray[np.uintp]
    prevalence: NDArray[np.uint32]
    hashes: NDArray[np.uint64]
    edges: NDArray[np.void]
    record_offsets: NDArray[np.uint32]
    record_ids: NDArray[np.str_]
    assembly_nodes: NDArray[np.uintp]
    node_offsets: NDArray[np.uintp]

    def __init__(
        self,
        assembly_paths: Sequence[str | Path],
        kmerlen: int,
        windowsize: int,
        n_cpu: int = 4,
        low_memory: bool = False,
    ) -> None:
        """Build a minimizer graph.

        Args:
            assembly_paths (Sequence[str | Path]): Paths to input assemblies in FASTA format (plain or gzipped).
            kmerlen (int): K-mer length for minimizer sketch.
            windowsize (int): Window size for minimizer sketch.
            n_cpu (int, optional): Number of worker threads to use. [4]
            low_memory (bool, optional): Recompute minimizers in a second pass to reduce peak memory. [False]
        """
        (
            self.kmers,
            self.offsets,
            self.prevalence,
            self.hashes,
            self.edges,
            self.record_offsets,
            record_ids,
            self.assembly_nodes,
            self.node_offsets,
        ) = _build_native(
            list(map(str, assembly_paths)),
            int(kmerlen),
            int(windowsize),
            int(n_cpu),
            bool(low_memory),
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
    def load(cls, path: str | Path) -> 'Graph':
        """Load a memory-mapped minimizer graph.

        Args:
            path (str | Path): Path to the graph directory.

        Returns:
            Graph: A graph backed by the saved NumPy array files.
        """
        path = Path(path)
        if not path.is_dir():
            raise NotADirectoryError(f'Not a graph directory: {path}')

        modes = {
            'kmers': 'r',
            'offsets': 'r',
            'prevalence': 'r',
            'hashes': 'r',
            'edges': 'r',
            'record_offsets': 'r',
            'record_ids': 'r',
            'assembly_nodes': 'r',
            'node_offsets': 'r',
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
            'offsets': np.dtype(np.uintp),
            'prevalence': np.dtype(np.uint32),
            'hashes': np.dtype(np.uint64),
            'edges': EDGE_DTYPE,
            'record_offsets': np.dtype(np.uint32),
            'assembly_nodes': np.dtype(np.uintp),
            'node_offsets': np.dtype(np.uintp),
        }
        for name, array in arrays.items():
            if array.ndim != 1:
                raise ValueError(f'Graph array {name!r} must be one-dimensional, got shape {array.shape}')
            if not array.flags.c_contiguous:
                raise ValueError(f'Graph array {name!r} must be C-contiguous')
        for name, dtype in expected_dtypes.items():
            if arrays[name].dtype != dtype:
                raise ValueError(f'Graph array {name!r} has dtype {arrays[name].dtype}, expected {dtype}')

        n_nodes = len(arrays['hashes'])
        if len(arrays['prevalence']) != n_nodes:
            raise ValueError("Graph arrays 'prevalence' and 'hashes' must have equal lengths")
        offsets = arrays['offsets']
        if len(offsets) != n_nodes + 1:
            raise ValueError("Graph array 'offsets' length must equal n_nodes + 1")
        if offsets[0] != 0 or offsets[-1] != len(arrays['kmers']):
            raise ValueError("Graph array 'offsets' must start at zero and end at len(kmers)")

        if arrays['record_offsets'].size == 0:
            raise ValueError("Graph array 'record_offsets' must not be empty")
        if int(arrays['record_offsets'][0]) != 0:
            raise ValueError("Graph array 'record_offsets' must start at zero")

        if arrays['record_ids'].dtype.kind != 'U':
            raise ValueError(f"Graph array 'record_ids' has dtype {arrays['record_ids'].dtype}, expected 'U'")
        if len(arrays['record_ids']) != int(arrays['record_offsets'][-1]):
            raise ValueError("Graph array 'record_ids' length must equal the final record offset")

        if arrays['node_offsets'].size == 0:
            raise ValueError("Graph array 'node_offsets' must not be empty")
        if int(arrays['node_offsets'][0]) != 0:
            raise ValueError("Graph array 'node_offsets' must start at zero")
        if int(arrays['node_offsets'][-1]) != len(arrays['assembly_nodes']):
            raise ValueError("Graph array assembly_nodes length must equal 'node_offsets' final value")
        if len(arrays['node_offsets']) != len(arrays['record_offsets']):
            raise ValueError("Graph arrays 'node_offsets' and 'record_offsets' must have equal lengths")

        graph = cls.__new__(cls)
        for name, array in arrays.items():
            setattr(graph, name, array)
        return graph
