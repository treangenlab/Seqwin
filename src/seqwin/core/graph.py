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
- NODE_DTYPE (np.dtype)
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

NODE_DTYPE = np.dtype([
    ('hash', np.uint64),
    ('start', np.uintp),
    ('prevalence', np.uintp),
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
    >>>     assembly_paths = ...,
    >>>     kmerlen = 21,
    >>>     windowsize = 200,
    >>>     n_cpu = 4,
    >>>     low_memory = False
    >>> )
    ```
    - `kmers` stores minimizer occurrences in all assemblies, grouped and sorted by hash.
    - `nodes` are sorted by hash.
    - `edges` endpoints are indices into `nodes`; sorted by descending weight, then by ascending endpoints.

    A node's `start` is the beginning of its minimizers in `kmers`. Its end is
    the next node's `start`, or `len(kmers)` for the final node.
    ```python
    >>> start = nodes[node_idx]['start']
    >>> end = nodes[node_idx + 1]['start'] if node_idx + 1 < len(nodes) else len(kmers)
    >>> kmer_group = kmers[start:end]
    >>> group_hash = nodes[node_idx]['hash']
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

    `assembly_nodes` stores the sorted, unique node indices present in each assembly.
    For assembly `i`, `assembly_nodes[node_offsets[i]:node_offsets[i + 1]]`
    contains unique nodes present in that assembly, in ascending node-index order.

    Attributes:
        kmers (NDArray[np.void]): A 1-D NumPy structured array of minimizers from all assemblies.
            - 'pos' (uint32): 0-based position of the minimizer within its FASTA record.
            - 'record_idx' (uint32): 0-based global index of the FASTA record.
        nodes (NDArray[np.void]): A 1-D NumPy structured array of minimizer nodes.
            - 'hash' (uint64): Hash value of the minimizers represented by this node.
            - 'start' (uintp): Start of this node's minimizer entries in `kmers`.
            - 'prevalence' (uintp): Number of assemblies containing this node's minimizer.
        edges (NDArray[np.void]): A 1-D NumPy structured array of weighted, undirected edges.
            - 'first' (uintp): Index of the smaller endpoint in `nodes`.
            - 'second' (uintp): Index of the larger endpoint in `nodes`.
            - 'weight' (uintp): Number of assemblies where the endpoints are adjacent.
        record_offsets (NDArray[np.uint32]): Cumulative global FASTA record offsets by assembly.
        record_ids (NDArray[np.str\_]): FASTA record IDs in global record order.
        assembly_nodes (NDArray[np.uintp]): Node indices grouped by assembly.
        node_offsets (NDArray[np.uintp]): Cumulative offsets into `assembly_nodes` by assembly.
    """
    __module__ = 'seqwin.core'

    __slots__ = (
        'kmers',
        'nodes',
        'edges',
        'record_offsets',
        'record_ids',
        'assembly_nodes',
        'node_offsets',
    )
    kmers: NDArray[np.void]
    nodes: NDArray[np.void]
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
        low_memory: bool = False,
        n_cpu: int = 1,
    ) -> None:
        """Build a minimizer graph.

        Args:
            assembly_paths (Sequence[str | Path]): Paths to input assemblies in FASTA format (plain or gzipped).
            kmerlen (int): K-mer length for minimizer sketch.
            windowsize (int): Window size for minimizer sketch.
            low_memory (bool, optional): Recompute minimizers in a second pass to reduce peak memory. [False]
            n_cpu (int, optional): Number of worker threads to use. [1]
        """
        (
            self.kmers,
            self.nodes,
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
            'nodes': 'r',
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
            'nodes': NODE_DTYPE,
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
