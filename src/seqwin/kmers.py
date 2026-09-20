"""
K-mer Graph
===========

A core module of Seqwin. Build a k-mer graph from all input assemblies and extract low-penalty subgraphs.

Dependencies:
-------------
- numpy
- .graph
- .assemblies
- .utils
- .config

Classes:
--------
- FilterResults
- Signature

Functions:
----------
- build_graph
- filter_graph
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import logging
from time import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .graph import KmerGraph, _filter_native
from .assemblies import Assemblies
from .markers import SignatureMetrics
from .utils import print_time_delta
from .config import Config, RunState, HAS_MASH, WORKINGDIR


@dataclass(slots=True, frozen=True)
class FilterResults:
    """Includes filtered graph arrays, low-penalty subgraphs,
    Jaccard indices and calculated values.

    Filtered nodes and edges follow their original order.

    Attributes:
        nodes (NDArray[np.void]): Nodes retained by edge filtering.
        edges (NDArray[np.void]): Low-weight edges are filtered.
        subgraphs (list[list[int]]): Low-penalty subgraphs represented by indices of retained nodes.
        jaccard (NDArray[np.float64] | None): Pairwise assembly Jaccard matrix.
        total_tar (int): Number of target assemblies.
        total_neg (int): Number of non-target assemblies.
        e_absence_tar (float): Expected k-mer absence in target assemblies.
        e_presence_neg (float): Expected k-mer presence in non-target assemblies.
        penalty_th (float): Node penalty threshold (user input or auto-computed).
        edge_weight_th (float): Graph edge weight threshold.
        min_nodes (int): Minimum number of nodes for a low-penalty subgraph.
        max_nodes (int | None): Maximum number of nodes for a low-penalty subgraph.
    """
    nodes: NDArray[np.void]
    edges: NDArray[np.void]
    subgraphs: list[list[int]]
    jaccard: NDArray[np.float64] | None
    total_tar: int
    total_neg: int
    e_absence_tar: float
    e_presence_neg: float
    penalty_th: float
    edge_weight_th: float
    min_nodes: int
    max_nodes: int | None


@dataclass(slots=True)
class Signature:
    """A signature extracted from one low-penalty subgraph."""
    subgraph_idx: int
    location: object
    sequence: str
    length: int
    n_rep: int
    rep_ratio: float
    blast: pd.DataFrame | None = None
    metrics: SignatureMetrics = SignatureMetrics()


def build_graph(assemblies: Assemblies, config: Config) -> KmerGraph:
    """
    Args:
        assemblies (Assemblies): See `Assemblies` in `assemblies.py`.
        config (Config): See `Config` in `config.py`.

    Returns:
        KmerGraph: The unscored minimizer graph.
    """
    logger.info(f'Building minimizer graph from {len(assemblies)} assemblies...')
    if config.low_memory:
        logger.warning(' - Low-memory mode is enabled; graph construction may take longer.')
    tik = time()

    graph = KmerGraph(
        assembly_paths=assemblies.paths,
        kmerlen=config.kmerlen,
        windowsize=config.windowsize,
        n_cpu=config.n_cpu,
        low_memory=config.low_memory
    )

    logger.info(f' - Found {len(graph.kmers)} minimizers')
    logger.info(f' - Found {len(graph.nodes)} nodes (unique minimizers)')
    logger.info(f' - Found {len(graph.edges)} weighted edges')

    print_time_delta(time()-tik)
    return graph


def filter_graph(
    graph: KmerGraph, assemblies: Assemblies, config: Config, state: RunState
) -> tuple[FilterResults, list[Signature]]:
    """Filter a minimizer graph and find low-penalty subgraphs.
    """
    logger.info('Filtering minimizer graph...')
    tik = time()
    jaccard = None
    if config.penalty_th is None and config.run_mash:
        if HAS_MASH:
            jaccard = assemblies.mash(
                kmerlen=config.kmerlen,
                sketchsize=config.sketchsize,
                out_path=state.working_dir / WORKINGDIR.mash,
                overwrite=config.overwrite,
                n_cpu=config.n_cpu
            )
        else:
            logger.error('Mash is not installed. Falling back to minimizer sketches.')

    (
        nodes,
        edges,
        subgraphs,
        native_signatures,
        total_tar,
        total_neg,
        e_absence_tar,
        e_presence_neg,
        penalty_th,
        edge_weight_th,
        min_nodes,
        max_nodes
     ) = _filter_native(
        kmers=graph.kmers,
        nodes=graph.nodes,
        edges=graph.edges,
        record_offsets=graph.record_offsets,
        assembly_paths=[str(path) for path in assemblies.paths],
        is_targets=assemblies.is_targets,
        jaccard=jaccard,
        kmerlen=config.kmerlen,
        windowsize=config.windowsize,
        penalty_th=config.penalty_th,
        stringency=config.stringency,
        min_len=config.min_len,
        max_len=config.max_len,
        penalty_th_cap=config.penalty_th_cap,
        edge_w_th_mul=config.edge_w_th_mul,
        min_nodes_floor=config.min_nodes_floor,
        max_nodes_cap=config.max_nodes_cap,
        consec_kmer_mul=config.consec_kmer_mul,
        n_cpu=config.n_cpu
    )

    signatures = list(
        Signature(
            subgraph_idx=s.subgraph_idx,
            location=s.location,
            sequence=s.sequence,
            length=s.length,
            n_rep=s.n_rep,
            rep_ratio=s.rep_ratio
        )
        for s in native_signatures
    )
    filtered = FilterResults(
        nodes=nodes,
        edges=edges,
        subgraphs=subgraphs,
        jaccard=jaccard,
        total_tar=total_tar,
        total_neg=total_neg,
        e_absence_tar=e_absence_tar,
        e_presence_neg=e_presence_neg,
        penalty_th=penalty_th,
        edge_weight_th=edge_weight_th,
        min_nodes=min_nodes,
        max_nodes=max_nodes
    )

    print_time_delta(time() - tik)
    return filtered, signatures
