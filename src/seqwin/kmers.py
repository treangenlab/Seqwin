"""
K-mer Graph
===========

A core module of Seqwin. Build a k-mer graph from all input assemblies and extract low-penalty subgraphs.

Dependencies:
-------------
- numpy
- networkx
- .graph
- .assemblies
- .utils
- .config

Classes:
--------
- FilteredGraph

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
import networkx as nx
from numpy.typing import NDArray

from .graph import KmerGraph, _filter_native
from .assemblies import Assemblies
from .utils import print_time_delta
from .config import Config, RunState, HAS_MASH, WORKINGDIR, EDGE_W, NODE_P


class FilteredGraph(KmerGraph):
    r"""The filtered minimizer graph class.

    Attributes:
        kmers (NDArray[np.void]): Only includes k-mers with hashes in `subgraphs`.
        nodes (NDArray[np.void]): Only includes nodes with hashes in `subgraphs`.
        edges (NDArray[np.void]): Low-weight edges are filtered.
        record_offsets (NDArray[np.uint32]): Inherited from `KmerGraph.record_offsets`.
        record_ids (NDArray[np.str\_]): Inherited from `KmerGraph.record_ids`.
        nx_graph (nx.Graph): The NetworkX graph instance built from filtered nodes and edges.
        subgraphs (tuple[frozenset[np.uint64], ...]): Low-penalty subgraphs. Each subgraph is a set of k-mer hash values.
    """
    __slots__ = ('nx_graph', 'subgraphs')
    nx_graph: nx.Graph
    subgraphs: tuple[frozenset[np.uint64], ...]

    def __init__(
        self,
        kmers: NDArray[np.void],
        nodes: NDArray[np.void],
        edges: NDArray[np.void],
        record_offsets: NDArray[np.uint32],
        record_ids: NDArray[np.str_],
        subgraphs: tuple[frozenset[np.uint64], ...]
    ) -> None:
        """Initialized a filtered minimizer graph from computed graph data.
        """
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(nodes['hash'])
        nx_graph.add_weighted_edges_from(
            edges.view(np.uint64).reshape(-1, 3),
            weight=EDGE_W
        )
        nx.set_node_attributes(
            nx_graph,
            values=dict(zip(nodes['hash'], nodes['penalty'])),
            name=NODE_P
        )
        self.kmers = kmers
        self.nodes = nodes
        self.edges = edges
        self.record_offsets = record_offsets
        self.record_ids = record_ids
        self.nx_graph = nx_graph
        self.subgraphs = subgraphs


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
) -> tuple[FilteredGraph, NDArray[np.float64] | None]:
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

    (kmers, nodes, edges, subgraphs,
     total_tar, total_neg, penalty_th, edge_weight_th, min_nodes, max_nodes) =  _filter_native(
        graph.kmers,
        graph.nodes,
        graph.edges,
        graph.record_offsets,
        assemblies.is_targets,
        jaccard,
        config.penalty_th,
        config.stringency,
        config.penalty_th_cap,
        config.edge_w_th_mul,
        config.windowsize,
        config.min_len,
        config.max_len,
        config.min_nodes_floor,
        config.max_nodes_cap,
        config.n_cpu
    )

    filtered = FilteredGraph(
        kmers=kmers,
        nodes=nodes,
        edges=edges,
        record_offsets=graph.record_offsets,
        record_ids=graph.record_ids,
        subgraphs=tuple(frozenset(map(np.uint64, subgraph)) for subgraph in subgraphs)
    )
    state.total_tar = total_tar
    state.total_neg = total_neg
    state.penalty_th = penalty_th
    state.edge_weight_th = edge_weight_th
    state.min_nodes = min_nodes
    state.max_nodes = max_nodes

    print_time_delta(time() - tik)
    return filtered, jaccard
