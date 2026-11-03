"""
Type declarations for the Seqwin C++ extension

Classes:
--------
- Graph
- Nodes
- AssemblyPaths
- FilteredGraph
- SubgraphLoc
- Signature
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Columns in the signature metadata CSV.
_METADATA_NAMES: tuple[str, ...]

class Nodes:
    """Parallel arrays representing nodes in the final k-mer graph.
    """
    def __len__(self) -> int:
        """Number of graph nodes."""
        ...
    @property
    def kmers(self) -> NDArray[np.void]:
        """K-mer occurrences grouped by node, ordered by record index and position.
            - 'pos' (uint32): 0-based position of the k-mer within its FASTA record.
            - 'record_idx' (uint32): 0-based global index of the FASTA record.
        """
        ...
    @property
    def offsets(self) -> NDArray[np.uintp]:
        """Cumulative offsets into `kmers` by node, including the final sentinel."""
        ...
    @property
    def prevalences(self) -> NDArray[np.uint32]:
        """Number of assemblies containing each node's k-mer."""
        ...
    @property
    def hashes(self) -> NDArray[np.uint64]:
        """Unique k-mer hashes in strictly ascending order."""
        ...

class Graph:
    """The Seqwin minimizer graph class.

    Example usage:
    ```python
    >>> from seqwin.core import Graph
    >>> graph = Graph(
    ...     assembly_paths = ['asm1.fasta', 'asm2.fasta'],
    ...     kmerlen = 21,
    ...     windowsize = 200
    ... )
    >>> graph_dir = Path('graph')
    >>> graph_dir.mkdir(exist_ok=True)
    >>> graph.save(graph_dir)
    >>> loaded = Graph.load(graph_dir)
    >>> len(loaded.nodes) == len(graph.nodes)
    True
    ```
    `Graph.save()` writes the native graph to NumPy `.npy` files in an existing directory.
    `Graph.load()` loads these files using read-only memory mapping.

    Nodes:
        Each node represents a unique k-mer hash. Nodes are ordered by ascending hash.

        The node arrays share the same node indices. For node `i`,
        `kmers[offsets[i]:offsets[i + 1]]` contains its k-mer occurrences,
        `prevalences[i]` is its assembly prevalence, and `hashes[i]` is its hash.
    ```python
    >>> kmers, offsets = graph.nodes.kmers, graph.nodes.offsets
    >>> prevalences, hashes = graph.nodes.prevalences, graph.nodes.hashes
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
    """
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
        ...
    @property
    def nodes(self) -> Nodes:
        """Parallel node arrays; the returned object keeps graph storage alive."""
        ...
    @property
    def edges(self) -> NDArray[np.void]:
        """Endpoints are node indices; sorted by descending weight, then ascending endpoints.
            - 'first' (uintp): Index of the smaller endpoint.
            - 'second' (uintp): Index of the larger endpoint.
            - 'weight' (uintp): Number of assemblies where the endpoints are adjacent.
        """
        ...
    @property
    def record_offsets(self) -> NDArray[np.uint32]:
        """Cumulative global FASTA record offsets by assembly, including the final sentinel."""
        ...
    @property
    def assembly_nodes(self) -> NDArray[np.uintp]:
        """Unique node indices grouped by assembly, sorted in ascending order within each assembly."""
        ...
    @property
    def node_offsets(self) -> NDArray[np.uintp]:
        """Cumulative offsets into `assembly_nodes` by assembly, including the final sentinel."""
        ...
    def save(self, path: str | Path) -> None:
        """Save a native graph as a directory of NumPy arrays (`.npy` files).
        Existing files are overwritten.

        Args:
            path (str | Path): Existing directory for the graph files.
        """
        ...
    @staticmethod
    def _from_numpy(
        *,
        kmers: NDArray[np.void],
        offsets: NDArray[np.uintp],
        prevalences: NDArray[np.uint32],
        hashes: NDArray[np.uint64],
        edges: NDArray[np.void],
        record_offsets: NDArray[np.uint32],
        assembly_nodes: NDArray[np.uintp],
        node_offsets: NDArray[np.uintp],
    ) -> Graph:
        """Construct a graph backed by NumPy arrays without copying.

        Input arrays may be writable, but they are treated as read-only.
        Callers must not modify their contents while the graph is in use.
        """
        ...
    @staticmethod
    def load(path: str | Path) -> Graph:
        """Load a graph from NumPy files using read-only memory mapping.

        Args:
            path (str | Path): Directory containing the graph's NumPy arrays (`.npy` files).

        Returns:
            Graph: A graph backed by read-only NumPy memory maps.
        """
        ...

class AssemblyPaths:
    """A reusable collection of assembly paths for native C++ calls.

    Paths are normalized using `os.fspath()` and stored as C++ strings, avoiding repeated conversions.
    """
    def __init__(self, assembly_paths: Sequence[str | Path]) -> None:
        """Normalize and store assembly paths for native C++ calls."""
        ...
    def __len__(self) -> int:
        """Number of assembly paths."""
        ...

class FilteredGraph:
    """Includes filtered graph arrays, low-penalty subgraphs, and calculated values.

    Filtered nodes and edges follow their original order.
    """
    @property
    def nodes(self) -> NDArray[np.void]:
        """Nodes retained by edge filtering.
            - 'idx' (uintp): Node index in the original graph.
            - 'n_tar' (uint32): Number of target assemblies containing this node's k-mer.
            - 'n_neg' (uint32): Number of non-target assemblies containing this node's k-mer.
            - 'penalty' (float64): Node penalty score.
        """
        ...
    @property
    def edges(self) -> NDArray[np.void]:
        """Edges passing weight and target-support thresholds."""
        ...
    @property
    def subgraphs(self) -> list[list[int]]:
        """Low-penalty subgraphs represented by node indices of the original graph."""
        ...
    @property
    def total_tar(self) -> int:
        """Number of target assemblies."""
        ...
    @property
    def total_neg(self) -> int:
        """Number of non-target assemblies."""
        ...
    @property
    def e_absence_tar(self) -> float | None:
        """Expected k-mer absence in target assemblies."""
        ...
    @property
    def e_presence_neg(self) -> float | None:
        """Expected k-mer presence in non-target assemblies."""
        ...
    @property
    def penalty_th(self) -> float:
        """Node penalty threshold (user input or auto-computed)."""
        ...
    @property
    def edge_weight_th(self) -> float:
        """Graph edge weight threshold."""
        ...
    @property
    def min_nodes(self) -> int:
        """Minimum number of nodes for a low-penalty subgraph."""
        ...
    @property
    def max_nodes(self) -> int | None:
        """Maximum number of nodes for a low-penalty subgraph."""
        ...

class SubgraphLoc:
    """Location and k-mer metadata for a subgraph found in an assembly.

    This is determined by the longest consecutive k-mer run in the assembly,
    when considering only the k-mers included in the subgraph.

    Note that a subgraph may appear more than once in an assembly.
    """
    @property
    def assembly_idx(self) -> int:
        """Index of the assembly containing the subgraph."""
        ...
    @property
    def record_idx(self) -> int:
        """Index of the FASTA record within the assembly."""
        ...
    @property
    def start(self) -> int:
        """0-based start position in the FASTA record."""
        ...
    @property
    def stop(self) -> int:
        """Exclusive stop position in the FASTA record."""
        ...
    @property
    def n_kmers(self) -> int:
        """Size of the longest consecutive k-mer run in the assembly."""
        ...
    @property
    def n_repeats(self) -> int:
        """Number of consecutive k-mer runs found in the assembly."""
        ...

class Signature:
    """A signature is extracted from a low-penalty subgraph, represented by a
    consecutive k-mer run (a.k.a. the representative) found in target assemblies.

    Forward and reversed k-mer orders are treated as the same canonical ordering.

    The nucleotide sequence of the signature is fetched from the first target assembly
    containing the representative.
    """
    @property
    def subgraph_idx(self) -> int:
        """Index of the low-penalty subgraph that produced the signature."""
        ...
    @property
    def location(self) -> SubgraphLoc:
        """Location and metadata of the representative."""
        ...
    @property
    def n_tar(self) -> int:
        """Number of target assemblies containing the representative."""
        ...

def _filter_native(
    graph: Graph,
    assembly_paths: Sequence[str | Path] | AssemblyPaths,
    is_targets: NDArray[np.bool_],
    jaccard: NDArray[np.float64] | None,
    kmerlen: int,
    windowsize: int,
    penalty_th: float | None,
    stringency: float,
    min_len: int,
    max_len: int | None,
    penalty_th_cap: float,
    edge_w_th_mul: float,
    min_nodes_floor: int,
    max_nodes_cap: int | None,
    consec_kmer_mul: float,
    n_cpu: int = 1,
    fasta_path: str | None = None,
    csv_path: str | None = None,
) -> tuple[
    FilteredGraph,
    list[Signature],
]: ...
