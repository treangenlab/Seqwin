"""
Type declarations for the Seqwin C++ extension

Classes:
--------
- FilteredGraph
- SubgraphLoc
- Signature
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

class FilteredGraph:
    """Includes filtered graph arrays, low-penalty subgraphs, and calculated values.

    Filtered nodes and edges follow their original order.

    Attributes:
        nodes (NDArray[np.void]): Nodes retained by edge filtering.
        edges (NDArray[np.void]): Low-weight edges are filtered.
        subgraphs (list[list[int]]): Low-penalty subgraphs represented by indices of retained nodes.
        total_tar (int): Number of target assemblies.
        total_neg (int): Number of non-target assemblies.
        e_absence_tar (float): Expected k-mer absence in target assemblies.
        e_presence_neg (float): Expected k-mer presence in non-target assemblies.
        penalty_th (float): Node penalty threshold (user input or auto-computed).
        edge_weight_th (float): Graph edge weight threshold.
        min_nodes (int): Minimum number of nodes for a low-penalty subgraph.
        max_nodes (int | None): Maximum number of nodes for a low-penalty subgraph.
    """
    @property
    def nodes(self) -> NDArray[np.void]: ...
    @property
    def edges(self) -> NDArray[np.void]: ...
    @property
    def subgraphs(self) -> list[list[int]]: ...
    @property
    def total_tar(self) -> int: ...
    @property
    def total_neg(self) -> int: ...
    @property
    def e_absence_tar(self) -> float: ...
    @property
    def e_presence_neg(self) -> float: ...
    @property
    def penalty_th(self) -> float: ...
    @property
    def edge_weight_th(self) -> float: ...
    @property
    def min_nodes(self) -> int: ...
    @property
    def max_nodes(self) -> int | None: ...

class SubgraphLoc:
    """Location and minimizer metadata for a subgraph found in an assembly.

    This is determined by the longest consecutive minimizer run in the assembly,
    when considering only the minimizers included in the subgraph.

    Note that a subgraph may appear more than once in an assembly.

    Attributes:
        assembly_idx (int): Index of the assembly containing the subgraph.
        record_idx (int): Index of the FASTA record within the assembly.
        start (int): 0-based start position in the FASTA record.
        stop (int): Exclusive stop position in the FASTA record.
        n_kmers (int): Size of the longest consecutive minimizer run in the assembly.
        n_repeats (int): Number of consecutive minimizer runs found in the assembly.
    """
    @property
    def assembly_idx(self) -> int: ...
    @property
    def record_idx(self) -> int: ...
    @property
    def start(self) -> int: ...
    @property
    def stop(self) -> int: ...
    @property
    def n_kmers(self) -> int: ...
    @property
    def n_repeats(self) -> int: ...

class Signature:
    """A signature is extracted from a low-penalty subgraph, represented by a
    consecutive minimizer run (a.k.a. the representative) found in target assemblies.

    Forward and reversed k-mer orders are treated as the same canonical ordering.

    The nucleotide sequence of the signature is fetched from the first target assembly
    containing the representative.

    Attributes:
        subgraph_idx (int): Index of the low-penalty subgraph that produced the signature.
        location (SubgraphLoc): Location and metadata of the representative.
        sequence (str): Nucleotide sequence of the signature.
        length (int): Length of the nucleotide sequence.
        n_rep (int): Number of target assemblies containing the representative.
        rep_ratio (float): Fraction of target assemblies containing the representative.
    """
    @property
    def subgraph_idx(self) -> int: ...
    @property
    def location(self) -> SubgraphLoc: ...
    @property
    def sequence(self) -> str: ...
    @property
    def length(self) -> int: ...
    @property
    def n_rep(self) -> int: ...
    @property
    def rep_ratio(self) -> float: ...

def _build_native(
    assembly_paths: Sequence[str],
    kmerlen: int,
    windowsize: int,
    n_cpu: int = ...,
    low_memory: bool = ...,
) -> tuple[
    NDArray[np.void],
    NDArray[np.void],
    NDArray[np.void],
    NDArray[np.uint32],
    list[str],
]: ...

def _filter_native(
    kmers: NDArray[np.void],
    nodes: NDArray[np.void],
    edges: NDArray[np.void],
    record_offsets: NDArray[np.uint32],
    assembly_paths: Sequence[str],
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
    n_cpu: int,
) -> tuple[
    FilteredGraph,
    list[Signature],
]: ...
