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
    """The Seqwin minimizer graph class.

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

    Nodes:
        Each node represents a unique k-mer hash. Nodes are ordered by ascending hash.

        The node arrays share the same node indices. For node `i`,
        `kmers[offsets[i]:offsets[i + 1]]` contains its k-mer occurrences,
        `prevalences[i]` is its assembly prevalence, and `hashes[i]` is its hash.
        ```python
        >>> kmers, offsets = graph.kmers, graph.offsets
        >>> prevalences, hashes = graph.prevalences, graph.hashes
        >>> node_idx = 0
        >>> node_kmers = kmers[offsets[node_idx]:offsets[node_idx + 1]]
        >>> node_hash = hashes[node_idx]
        >>> node_prevalence = prevalences[node_idx]
        ```

    Edges:
        Each edge connects two nodes by their node indices.
        Edges are undirected and sorted by descending weight.

    FASTA record indices:
        Each k-mer's `record_idx` is a global FASTA record index.
        Use `record_offsets` to recover the assembly index and the original record index of each k-mer.
        ```python
        >>> import numpy as np
        >>> record_offsets = graph.record_offsets
        >>> assembly_indices = np.searchsorted(
        ...     record_offsets,
        ...     kmers['record_idx'],
        ...     side='right',
        ... ) - 1
        >>> record_indices = kmers['record_idx'] - record_offsets[assembly_indices]
        ```

    Assembly-to-node index:
        `assembly_nodes` stores the unique node indices present in each assembly, in ascending node-index order.
        For assembly `i`, `assembly_nodes[node_offsets[i]:node_offsets[i + 1]]` contains its node indices.

    Attributes:
        kmers (NDArray[np.void]): K-mer occurrences grouped by node, ordered by record index and position.
            - 'pos' (uint32): 0-based position of the k-mer within its FASTA record.
            - 'record_idx' (uint32): 0-based global index of the FASTA record.
        offsets (NDArray[np.uintp]): Cumulative offsets into `kmers` by node, including the final sentinel.
        prevalences (NDArray[np.uint32]): Number of assemblies containing each node's k-mer.
        hashes (NDArray[np.uint64]): Unique k-mer hashes in strictly ascending order.
        edges (NDArray[np.void]): Endpoints are node indices; sorted by descending weight, then ascending endpoints.
            - 'first' (uintp): Index of the smaller endpoint.
            - 'second' (uintp): Index of the larger endpoint.
            - 'weight' (uintp): Number of assemblies where the endpoints are adjacent.
        record_offsets (NDArray[np.uint32]): Cumulative global FASTA record offsets by assembly, including the final sentinel.
        assembly_nodes (NDArray[np.uintp]): Unique node indices grouped by assembly, sorted in ascending order within each assembly.
        node_offsets (NDArray[np.uintp]): Cumulative offsets into `assembly_nodes` by assembly, including the final sentinel.
    """
    __module__ = 'seqwin.core'

    __slots__ = (
        'kmers',
        'offsets',
        'prevalences',
        'hashes',
        'edges',
        'record_offsets',
        'assembly_nodes',
        'node_offsets',
    )
    kmers: NDArray[np.void]
    offsets: NDArray[np.uintp]
    prevalences: NDArray[np.uint32]
    hashes: NDArray[np.uint64]
    edges: NDArray[np.void]
    record_offsets: NDArray[np.uint32]
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
            self.prevalences,
            self.hashes,
            self.edges,
            self.record_offsets,
            self.assembly_nodes,
            self.node_offsets,
        ) = _build_native(
            list(map(str, assembly_paths)),
            int(kmerlen),
            int(windowsize),
            int(n_cpu),
            bool(low_memory),
        )

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
            'prevalences': 'r',
            'hashes': 'r',
            'edges': 'r',
            'record_offsets': 'r',
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
            'prevalences': np.dtype(np.uint32),
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
        if len(arrays['prevalences']) != n_nodes:
            raise ValueError("Graph arrays 'prevalences' and 'hashes' must have equal lengths")
        offsets = arrays['offsets']
        if len(offsets) != n_nodes + 1:
            raise ValueError("Graph array 'offsets' length must equal n_nodes + 1")
        if offsets[0] != 0 or offsets[-1] != len(arrays['kmers']):
            raise ValueError("Graph array 'offsets' must start at zero and end at len(kmers)")

        if arrays['record_offsets'].size == 0:
            raise ValueError("Graph array 'record_offsets' must not be empty")
        if int(arrays['record_offsets'][0]) != 0:
            raise ValueError("Graph array 'record_offsets' must start at zero")

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
