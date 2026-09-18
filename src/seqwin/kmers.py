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
- Signature
- FilterResult

Functions:
----------
- build_graph
- filter_graph
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import logging
from time import time

logger = logging.getLogger(__name__)

import numpy as np
from numpy.typing import NDArray

from .graph import KmerGraph, _filter_native
from .assemblies import Assemblies
from .utils import print_time_delta
from .config import Config, RunState, HAS_MASH, WORKINGDIR, CONSEC_KMER_MUL


class Signature:
    """A signature extracted from one low-penalty subgraph."""
    __slots__ = (
        'subgraph_idx', 'location', 'sequence', 'length', 'n_rep', 'rep_ratio',
        'blast', 'metrics'
    )

    def __init__(
        self, subgraph_idx: int, location, sequence: str, length: int,
        n_rep: int, rep_ratio: float
    ) -> None:
        self.subgraph_idx = subgraph_idx
        self.location = location
        self.sequence = sequence
        self.length = length
        self.n_rep = n_rep
        self.rep_ratio = rep_ratio
        self.blast = None
        self.metrics = None


class FilterResult:
    """Filtered graph data and signatures produced by the native pipeline."""
    __slots__ = ('nodes', 'edges', 'subgraphs', 'signatures', 'jaccard')

    def __init__(
        self, nodes: NDArray[np.void], edges: NDArray[np.void],
        subgraphs: list[list[int]], signatures: list[Signature],
        jaccard: NDArray[np.float64] | None
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.subgraphs = subgraphs
        self.signatures = signatures
        self.jaccard = jaccard


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
) -> FilterResult:
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

    (nodes, edges, subgraphs, native_signatures, total_tar, total_neg,
     penalty_th, edge_weight_th, min_nodes, max_nodes) =  _filter_native(
        graph.kmers,
        graph.nodes,
        graph.edges,
        graph.record_offsets,
        [str(path) for path in assemblies.paths],
        assemblies.is_targets,
        jaccard,
        config.kmerlen,
        config.windowsize,
        config.penalty_th,
        config.stringency,
        config.min_len,
        config.max_len,
        config.penalty_th_cap,
        config.edge_w_th_mul,
        config.min_nodes_floor,
        config.max_nodes_cap,
        CONSEC_KMER_MUL,
        config.n_cpu
    )

    signatures = [
        Signature(
            signature.subgraph_idx, signature.location, signature.sequence,
            signature.length, signature.n_rep, signature.rep_ratio
        )
        for signature in native_signatures
    ]
    filtered = FilterResult(nodes, edges, subgraphs, signatures, jaccard)
    state.total_tar = total_tar
    state.total_neg = total_neg
    state.penalty_th = penalty_th
    state.edge_weight_th = edge_weight_th
    state.min_nodes = min_nodes
    state.max_nodes = max_nodes

    print_time_delta(time() - tik)
    return filtered
