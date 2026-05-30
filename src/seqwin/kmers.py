"""
K-mer Graph
===========

A core module of Seqwin. Create a k-mer graph from k-mers of all input genome assemblies.

Dependencies:
-------------
- numpy
- networkx
- .graph
- .assemblies
- .helpers
- .utils
- .config

Classes:
--------
- KmerGraph

Functions:
----------
- get_kmers
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import logging
from random import Random
from time import time

logger = logging.getLogger(__name__)

import numpy as np
import networkx as nx
from numpy.typing import NDArray

from .graph import build, _filter_kmers
from .assemblies import Assemblies
from .helpers import get_subgraphs
from .utils import print_time_delta, log_and_raise
from .config import Config, RunState, HAS_MASH, WORKINGDIR, EDGE_W, NODE_P


class KmerGraph(object):
    """
    1. Create a weighted, undirected k-mer graph, and calculate node penalty scores.
    2. Extract low-penalty subgraphs from the k-mer graph with `self.filter()`.

    Attributes:
        kmers (NDArray[np.void]): A 1-D NumPy structured array of k-mers from all assemblies, grouped and sorted by k-mer hashes.
        idx (NDArray[np.uint64] | None): The original indices assigned when k-mers are generated (ordered by genomic positions).
            Parallel to `kmers`.
        nodes (NDArray[np.void]): A 1-D NumPy structured array of k-mer nodes.
            For each node, `kmers[node['start']:node['stop']]` is the k-mer group with `node['hash']`.
        edges (NDArray[np.void]): A 1-D NumPy structured array of weighted, undirected edges.
            Edge weight is the number of assemblies where the two k-mers are adjacent.
        record_offsets (NDArray[np.uint64]): Cumulative global FASTA record offsets by assembly.
        graph (nx.Graph): The graph instance built from filtered nodes and edges.
        subgraphs (tuple[frozenset[np.uint64], ...] | None): Low-penalty subgraphs. Each subgraph is a set of k-mer hash values.
            Generated with `self.filter()`.
    """
    __slots__ = (
        'kmers', 'idx', 'nodes', 'edges', 'record_offsets', 'graph', 'subgraphs', '_filtered_flag'
    )
    kmers: NDArray[np.void]
    idx: NDArray[np.uint64]
    nodes: NDArray[np.void]
    edges: NDArray[np.void]
    record_offsets: NDArray[np.uint64]
    graph: nx.Graph
    subgraphs: tuple[frozenset[np.uint64], ...] | None
    _filtered_flag: bool # True if `self.filter()` is called

    def __init__(self, assemblies: Assemblies, kmerlen: int, windowsize: int, n_cpu: int) -> None:
        """
        1. Generate minimizer sketches and weighted edges.
        2. Generate k-mer nodes and calculate node penalty scores.

        Args:
            assemblies (Assemblies): See `Assemblies` in `assemblies.py`.
            kmerlen (int): See `Config` in `config.py`.
            windowsize (int): See `Config` in `config.py`.
            n_cpu (int): See `Config` in `config.py`.
        """
        n_assemblies = len(assemblies)
        logger.info(f'Building minimizer graph from {n_assemblies} assemblies...')
        tik = time()

        kmers, idx, nodes, edges, record_offsets, record_ids = build(
            assemblies.path,
            kmerlen,
            windowsize,
            assemblies.is_target,
            n_cpu=n_cpu,
        )
        # calculate penalty for each node
        n_tar = sum(assemblies.is_target)
        n_neg = n_assemblies - n_tar
        nodes['penalty'] = _frac_to_penalty(
            nodes['n_tar'] / n_tar,
            nodes['n_neg'] / n_neg
        )
        assemblies.record_ids = record_ids

        logger.info(f' - Found {len(kmers)} minimizers')
        logger.info(f' - Found {len(nodes)} nodes (unique minimizers)')
        logger.info(f' - Found {len(edges)} weighted edges')

        print_time_delta(time()-tik)

        self.kmers = kmers
        self.idx = idx
        self.nodes = nodes
        self.edges = edges
        self.record_offsets = record_offsets
        self.graph = None # create graph after filtering nodes and edges
        self.subgraphs = None
        self._filtered_flag = False

    def filter(
        self, penalty_th: float, edge_weight_th: float, min_nodes: int, max_nodes: int | None, rng: Random
    ) -> None:
        """Filter graph edges and nodes.
        1. Remove low-weight edges and isolated nodes.
        2. Create the graph instance and extract low-penalty subgraphs.
        3. Remove k-mers not included in any of the subgraphs.

        Args:
            penalty_th (float): See `RunState` in `config.py`.
            edge_weight_th (float): See `RunState` in `config.py`.
            min_nodes (int): See `RunState` in `config.py`.
            max_nodes (int | None): See `RunState` in `config.py`.
            rng (random.Random): See `RunState` in `config.py`.
        """
        kmers = self.kmers
        idx = self.idx
        nodes = self.nodes
        edges = self.edges
        filtered_flag = self._filtered_flag

        if filtered_flag:
            logger.error(f'K-mers are already filtered, cannot filter again.')
            return None

        logger.info('Extracting low-penalty subgraphs from the k-mer graph...')
        tik = time()

        if max_nodes is None:
            logger.warning(f' - Upper limit of subgraph size is not set. Lower limit is set to {min_nodes}')
        else:
            logger.info(f' - Subgraph size limit is set to [{min_nodes}, {max_nodes}]')

        # remove low-weight edges and isolated nodes
        nodes, edges, graph = KmerGraph.__filter_graph(nodes, edges, edge_weight_th)

        # get low-penalty subgraphs
        subgraphs, used_hashes = get_subgraphs(graph, penalty_th, min_nodes, max_nodes, rng)

        # remove unused k-mers
        logger.info(' - Removing k-mers not included in any of the subgraphs...')
        kmers, idx, nodes = _filter_kmers(kmers, idx, nodes, used_hashes)
        logger.info(f' - {len(kmers)} k-mers left')

        print_time_delta(time()-tik)
        self.kmers = kmers
        self.idx = idx
        self.nodes = nodes
        self.edges = edges
        self.graph = graph
        self.subgraphs = subgraphs
        self._filtered_flag = True

    @staticmethod
    def __filter_graph(nodes: NDArray, edges: NDArray, edge_weight_th: float) -> tuple[NDArray, NDArray, nx.Graph]:
        """Remove low-weight edges and isolated nodes, and create the graph instance.

        Args:
            nodes (NDArray): See `KmerGraph.nodes`.
            edges (NDArray): See `KmerGraph.edges`.
            edge_weight_th (float): See `RunState` in `config.py`.

        Returns:
            tuple: A tuple containing
                1. NDArray: Filtered nodes.
                2. NDArray: Filtered edges.
                3. nx.Graph: See `KmerGraph.graph`.
        """
        logger.info(' - Filtering graph edges and nodes...')
        n_nodes, n_edges = len(nodes), len(edges)

        # remove low-weight edges
        th = np.uint64(edge_weight_th) # for faster comparison
        edges = edges[edges['weight'] > th]
        edge_values = edges.view(np.uint64).reshape(-1, 3)
        logger.info(f' - Removed {n_edges - len(edges)} edges with weight<{edge_weight_th:.3f}, {len(edges)} edges left')

        # remove isolated nodes
        nodes_to_keep = np.unique(edge_values[:, :2])
        nodes = nodes[
            np.searchsorted(nodes['hash'], nodes_to_keep)
        ]
        logger.info(f' - Removed {n_nodes - len(nodes)} isolated nodes, {len(nodes)} nodes left')

        logger.info(' - Building graph...')
        graph = nx.Graph()
        graph.add_weighted_edges_from(edge_values, weight=EDGE_W)
        nx.set_node_attributes(
            graph,
            values=dict(zip(nodes['hash'], nodes['penalty'])),
            name=NODE_P
        )

        return nodes, edges, graph


def _expected_frac(jaccard_mtx: NDArray) -> np.floating:
    """Calculate the expected fraction from a matrix of pairwise Jaccard indices.
    Here fraction means: for a k-mer `h` in a group of genomes (k-mer sets), the fraction of sets in
    a second group that also contain `h`. `jaccard_mtx` is the pairwise Jaccard indices between sets
    in the two groups. Note that this could be a self comparison (two groups are the same).
    - `E(frac) = mean(2J / (1+J))`, where `J` is the Jaccard matrix.
    - This expectation ≥ mean of the Jaccard matrix, (`2J / (1+J) ≥ J, 0 ≤ J ≤ 1`).
    """
    return np.mean(2 * jaccard_mtx / (1 + jaccard_mtx))


def _frac_to_penalty(frac_tar: NDArray | float, frac_neg: NDArray | float) -> NDArray | float:
    """The penalty formula (Euclidean / L2 norm of the two fractions).
    - When `frac_tar` and `frac_neg` are expectations, this formula doesn't give the expected penalty, since it's non-linear.
    - However, since it is convex, the returned value ≤ the true expectation (Jensen’s inequality).
    """
    return ((1 - frac_tar)**2 + frac_neg**2)**0.5


def get_kmers(
    assemblies: Assemblies, config: Config, state: RunState
) -> tuple[KmerGraph, NDArray | None]:
    """Create a `KmerGraph` instance, calculate filtering thresholds and run the `filter()` method to generate low-penalty subgraphs.

    Args:
        assemblies (Assemblies): See `Assemblies` in `assemblies.py`.
        config (Config): See `Config` in `config.py`.
        state (RunState): See `RunState` in `config.py`.

    Returns:
        tuple: A tuple containing
            1. KmerGraph: The KmerGraph instance.
            2. NDArray | None: A matrix of Jaccard indices of all assembly pairs.
    """
    overwrite = config.overwrite
    kmerlen = config.kmerlen
    windowsize = config.windowsize
    penalty_th = config.penalty_th
    run_mash = config.run_mash
    stringency = config.stringency
    min_len = config.min_len
    max_len = config.max_len
    no_filter = config.no_filter
    penalty_th_cap = config.penalty_th_cap
    edge_w_th_mul = config.edge_w_th_mul
    min_nodes_floor = config.min_nodes_floor
    max_nodes_cap = config.max_nodes_cap
    sketchsize = config.sketchsize
    n_cpu = config.n_cpu

    working_dir = state.working_dir
    rng = state.rng
    n_tar = state.n_tar
    n_neg = state.n_neg

    kmers = KmerGraph(assemblies, kmerlen, windowsize, n_cpu)

    if no_filter:
        return kmers, None

    # calculate filter params
    # 1. calculate penalty threshold
    if penalty_th is None:
        logger.info(f'Calculating penalty threshold...')
        tik = time()

        # we only care about the presence & absence of k-mers in target assemblies
        if run_mash and HAS_MASH:
            jaccard = assemblies.mash(
                kmerlen=kmerlen,
                sketchsize=sketchsize,
                out_path=working_dir / WORKINGDIR.mash,
                overwrite=overwrite,
                n_cpu=n_cpu
            )
            e_absence_tar = 1 - _expected_frac(jaccard[:n_tar, :n_tar])
            e_presence_neg = _expected_frac(jaccard[n_tar:, :n_tar])
        else:
            if run_mash:
                logger.error('Mash is not installed. Falling back to minimizer sketches.')
            # calculate expected fractions directly from minimizer sketches
            # for tar absence or neg presence, k-mer weights should always be nodes['n_tar'] (how many target assemblies have this k-mer)
            nodes = kmers.nodes
            frac_tar = nodes['n_tar'] / n_tar
            e_absence_tar = 1 - np.sum(frac_tar * nodes['n_tar']) / np.sum(nodes['n_tar'])
            frac_neg = nodes['n_neg'] / n_neg
            e_presence_neg = np.sum(frac_neg * nodes['n_tar']) / np.sum(nodes['n_tar'])
            jaccard = None

        logger.info(f' - expected k-mer absence in targets: {e_absence_tar:.5f}')
        logger.info(f' - expected k-mer presence in non-targets: {e_presence_neg:.5f}')

        penalty_th_mul = 1 - stringency / 10
        penalty_th = penalty_th_mul * (e_absence_tar * e_presence_neg)**0.5 # geometric mean
        logger.info(f' - calculated penalty threshold: {penalty_th:.5f}')

        if penalty_th > penalty_th_cap:
            penalty_th = penalty_th_cap
            logger.warning(f' - calculated penalty threshold is too large (capped at {penalty_th})')

        print_time_delta(time()-tik)
    else:
        logger.warning(f'Penalty threshold is provided (--penalty-th), skip auto estimation')
        jaccard = None

    # 2. calculate edge weight threshold
    # consider N as the number of assemblies that include a certain k-mer
    # since we want k-mers with penalty lower than penalty_th
    # based on the definition of penalty, N ≥ (1 - penalty_th) * n_tar
    # so edge weight threshold is calculated based on the lower bound of N, times a multiplier
    edge_weight_th = edge_w_th_mul * (1 - penalty_th) * n_tar

    # 3. calculate size range of subgraphs
    gap_len = (windowsize + 1) // 2 # average length of gap between minimizers
    min_nodes = max(min_nodes_floor, min_len // gap_len + 1)
    if max_len is None:
        max_nodes = max_nodes_cap
    else:
        max_nodes = max_len // gap_len + 1

    # extract low-penalty subgraphs and remove unused k-mers
    kmers.filter(penalty_th, edge_weight_th, min_nodes, max_nodes, rng)

    state.penalty_th = penalty_th
    state.edge_weight_th = edge_weight_th
    state.min_nodes = min_nodes
    state.max_nodes = max_nodes
    return kmers, jaccard
