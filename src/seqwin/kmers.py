"""
K-mer Graph
===========

A core module of Seqwin. Create an instance for k-mers of all input genome assemblies, 
including the weighted k-mer graph and low-penalty subgraphs. 

Dependencies:
-------------
- numpy
- pandas
- networkx
- scipy (optional)
- .assemblies
- .minimizer
- .graph
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
from itertools import repeat, chain
from time import time
from heapq import heappush, heappop

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import networkx as nx
try:
    from scipy.sparse import coo_matrix
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from .assemblies import Assemblies
from .minimizer import indexlr
from .graph import WeightedGraph, compose_weighted_graphs, EDGE_W
from .utils import print_time_delta, log_and_raise, mp_wrapper, get_chunks
from .config import Config, RunState, WORKINGDIR, NODE_P

_IDX_TYPE = np.uint16 # dtype for assembly_idx


class KmerGraph(object):
    """
    1. Create a weighted k-mer graph, and calculate node penalty scores. 
    2. (Optional) Calculate `Mash distances<https://mash.readthedocs.io/en/latest/distances.html>`__ for each assembly pair. 
    3. Extract low-penalty subgraphs from the k-mer graph with `self.filter()`. 
    
    Attributes:
        kmers (pd.DataFrame): Each row represents a k-mer, with columns, 
            1. 'hash' (uint64): Hash value of the k-mer. 
            2. 'pos' (uint32): Position of the first base of the k-mer. 
            3. 'record_idx' (uint16): Sequence record index. 
            4. 'assembly_idx' (uint16): Assembly index. 
            5. 'is_target' (bool): True for target assemblies. 
        graph (nx.Graph): A weighted, undirected graph of k-mers. 
            Edge weight is the number of assemblies where the two k-mers are adjacent. 
        clusters (pd.DataFrame): Each row represents a graph node, with its k-mer hash value as index, 
            and columns ['n_tar', 'n_neg', 'true_pos', 'false_pos', 'penalty']. 
        cnt_mtx (np.ndarray | None): A matrix of the number of shared k-mers between each assembly pair. 
            Calculated when`get_dist=True`. 
        dist_mtx (np.ndarray | None): A matrix of the Mash distance between each assembly pair. 
            Calculated when `get_dist=True`. 
        subgraphs (list[set[int]] | None): A list of low-penalty subgraphs. Each subgraph is a set of k-mer hash values. 
            Generated with `self.filter()`. 
        _filtered_flag (bool): True if `self.filter()` is called. 
    """
    __slots__ = (
        'kmers', 'graph', 'clusters', 'cnt_mtx', 'dist_mtx', 'subgraphs', '_filtered_flag'
    )
    kmers: pd.DataFrame
    graph: nx.Graph
    clusters: pd.DataFrame
    cnt_mtx: np.ndarray | None
    dist_mtx: np.ndarray | None
    subgraphs: list[set[int]] | None
    _filtered_flag: bool

    def __init__(self, assemblies: Assemblies, kmerlen: int, windowsize: int, get_dist: bool, n_cpu: int) -> None:
        """
        1. Create a weighted, undirected graph of k-mers. 
        2. Calculate node penalty scores. 
        3. Calculate assembly distances if `get_dist=True`. 

        Args:
            assemblies (Assemblies): See `Assemblies` in `assemblies.py`. 
            kmerlen (bool): See `Config` in `config.py`. 
            windowsize (bool): See `Config` in `config.py`. 
            get_dist (bool): See `Config` in `config.py`. 
            n_cpu (int): See `Config` in `config.py`. 
        """
        # merge k-mers from all assemblies and create the graph
        kmers, graph = KmerGraph.__get_graph(assemblies, kmerlen, windowsize, n_cpu)

        # calculate penalty scores by clustering k-mers
        clusters, cnt_mtx = KmerGraph.__get_penalty(kmers, assemblies, get_dist)

        # sanity check
        if set(graph.nodes) != set(clusters.index):
            log_and_raise(ValueError, 'Inconsistent nodes from pandas GroupBy output.')

        # add penalties to networkx graph nodes
        nx.set_node_attributes(graph, clusters['penalty'].to_dict(), name=NODE_P)

        # calculate assembly distances
        if cnt_mtx is not None:
            dist_mtx = KmerGraph.__get_dist(kmers, cnt_mtx, kmerlen)
        else:
            dist_mtx = None

        self.kmers = kmers
        self.graph = graph
        self.clusters = clusters
        self.cnt_mtx = cnt_mtx
        self.dist_mtx = dist_mtx
        self.subgraphs = None
        self._filtered_flag = False

    @staticmethod
    def __get_graph(assemblies: Assemblies, kmerlen: int, windowsize: int, n_cpu: int) -> tuple[pd.DataFrame, nx.Graph]:
        """
        1. Merge k-mers from all assemblies into a single DataFrame. 
        2. Create a weighted, undirected graph of k-mers; edge weight is the number of assemblies where the two k-mers are adjacent. 
        3. Add record IDs as a column in assemblies. 

        Args:
            assemblies (Assemblies): See `Assemblies` in `assemblies.py`. 
            kmerlen (bool): See `Config` in `config.py`. 
            windowsize (bool): See `Config` in `config.py`. 
            get_dist (bool): See `Config` in `config.py`. 
            n_cpu (int): See `Config` in `config.py`. 
        
        Returns:
            tuple: A tuple containing
                1. kmers (pd.DataFrame): See `KmerGraph.kmers`. 
                2. graph (nx.Graph): See `KmerGraph.graph`. 
        """
        n_assemblies = len(assemblies)
        logger.info(f'Creating a weighted, undirected minimizer graph from {n_assemblies} assemblies...')
        tik = time()

        # merge k-mers from all assembies
        if n_cpu <= 1:
            # create the graph with a single thread
            kmers, graph, isolates, all_record_ids = _get_graph(assemblies, kmerlen, windowsize)
        else:
            # to make mp work, the method/function must be static and should not start with double underscores (single is fine)
            # difference between single & double underscores: https://docs.python.org/3/tutorial/classes.html#private-variables
            logger.info(f' - Parallelizing across {n_cpu} threads (~{n_assemblies//n_cpu} assemblies per thread)...')
            graph_args = zip(
                get_chunks(assemblies, n_cpu), 
                repeat(kmerlen, n_cpu), 
                repeat(windowsize, n_cpu)
            )
            kmers, graph, isolates, all_record_ids = mp_wrapper(
                _get_graph, graph_args, n_cpu, unpack_output=True
            )
            # merge outputs from multiple processes
            logger.info(' - Merging from all threads...')
            kmers = pd.concat(kmers, ignore_index=True)
            graph = compose_weighted_graphs(graph)
            isolates = set.union(*isolates)
            all_record_ids = chain.from_iterable(all_record_ids)
        assemblies.record_ids = list(all_record_ids) # save record IDs

        # convert to a networkx graph
        graph = graph.to_nxGraph()
        # add isolated nodes to graph, so that the number of nodes is the same as the number of k-mer clusters
        # we are doing this because when the graphs are merged, only the edges are merged and isolated nodes are not merged
        graph.add_nodes_from(isolates)

        logger.info(f' - {len(graph)} nodes and {len(graph.edges)} edges from {len(kmers)} k-mers')
        logger.info(f' - {len(isolates)} nodes are isolates')
        print_time_delta(time()-tik)
        return kmers, graph

    @staticmethod
    def __get_penalty(
        kmers: pd.DataFrame, assemblies: Assemblies, get_dist: bool
    ) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Calculate node penalties by clustering k-mers with the same hash value. 
        Ideally, this can be done during graph creation. But in python, it is faster to do this with pandas groupby. 
        Assembly distance can also be calculated in this step (`get_dist=True`), using a slower clustering function `_cluster_kmers_dist()`. 

        Args:
            kmers (pd.DataFrame): See `KmerGraph.kmers`. 
            assemblies (Assemblies): See `Assemblies` in `assemblies.py`. 
            get_dist (bool): See `Config` in `config.py`. 
        
        Returns:
            tuple: A tuple containing
                1. clusters (pd.DataFrame): See `KmerGraph.clusters`. 
                2. cnt_mtx (np.ndarray | None): See `KmerGraph.cnt_mtx`. 
        """
        logger.info(f'Calculating penalty score for each k-mer node...')
        tik = time()

        tar_idx = set(assemblies[assemblies.is_target == True].index)
        neg_idx = set(assemblies[assemblies.is_target == False].index)

        if get_dist and (not _HAS_SCIPY):
            logger.error(' - SciPy is not installed, skip assembly distance calculation')
            get_dist = False

        # choose which clustering function to use
        if get_dist:
            logger.warning(' - Assembly distance calculation is turned on, extra time and memory needed')
            clusters, cnt_mtx = KmerGraph.__cluster_dist(kmers, tar_idx, neg_idx)
        else:
            clusters = KmerGraph.__cluster(kmers, tar_idx, neg_idx)
            cnt_mtx = None

        clusters.sort_values(
            ['penalty', 'hash'], # 'hash' is the index
            ascending=True, ignore_index=False, inplace=True
        )
        print_time_delta(time()-tik)
        return clusters, cnt_mtx

    @staticmethod
    def __cluster_dist(
        kmers: pd.DataFrame, tar_idx: set[int], neg_idx: set[int]
    ) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Cluster k-mers by their hash values with `df.groupby()`, and calculate the penalty for each cluster. 
        `KmerGraph.cnt_mtx` is also calculated. Only used when distance calculation is turned on. 

        Args:
            kmers (pd.DataFrame): See `KmerGraph.kmers`. 
            tar_idx (set[int]): See `KmerGraph.__cluster()`. 
            neg_idx (set[int]): See `KmerGraph.__cluster()`. 

        Returns:
            tuple: A tuple containing
                1. clusters (pd.DataFrame): See `KmerGraph.clusters`. 
                2. cnt_mtx (np.ndarray | None): See `KmerGraph.cnt_mtx`. Return None if scipy is not installed. 
        """
        # drop duplicate hash values for each assembly
        clusters = kmers[
            ['hash', 'assembly_idx', 'is_target']
        ].drop_duplicates(keep='first', inplace=False)

        # cluster by hash and count the number of target / total assemblies in each cluster
        clusters = clusters.groupby(
            # as_index=True to use hash values as index
            by='hash', as_index=True, sort=False
        ).agg(
            assembly_idx=pd.NamedAgg(column='assembly_idx', aggfunc=frozenset), 
            n_tar=pd.NamedAgg(column='is_target', aggfunc='sum'), 
            total=pd.NamedAgg(column='is_target', aggfunc='size'), 
        )
        assembly_idx = clusters.pop('assembly_idx')

        # formatting
        clusters['n_neg'] = clusters.pop('total') - clusters['n_tar']

        # true positive rate & false positive rate
        clusters['true_pos'] = clusters['n_tar']/len(tar_idx)
        clusters['false_pos'] = clusters['n_neg']/len(neg_idx)

        # calculate cluster penalty based on distance to true_pos=1 and false_pos=0
        clusters['penalty'] = ((1 - clusters['true_pos'])**2 + clusters['false_pos']**2)**0.5

        # calculate cnt_mtx
        # 1. build a membership matrix (use a sparse matrix to save memory)
        # each row represents a k-mer cluster, each column represents an assembly
        # if the i'th k-mer exists in the j'th assembly, then M[i, j] = 1; else M[i, j] = 0
        kmer_idx = list(chain.from_iterable( # row indexes
            repeat(i, len(s)) for i, s in enumerate(assembly_idx)
        ))
        assembly_idx = list(chain.from_iterable(assembly_idx)) # col indexes
        M = coo_matrix( # the membership matrix
            ( # construct from three arrays: data entries, row indexes, col indexes
                np.ones(len(kmer_idx)), (kmer_idx, assembly_idx)
            ), 
            shape=( # N_kmers, N_assemblies
                len(clusters), len(tar_idx)+len(neg_idx)
            )
        ).tocsr() # convert to sparse

        # 2. calculate the number of shared k-mers by matrix multiplication
        cnt_mtx: coo_matrix = M.T @ M

        return clusters, cnt_mtx.toarray()

    @staticmethod
    def __cluster(kmers: pd.DataFrame, tar_idx: set[int], neg_idx: set[int]) -> pd.DataFrame:
        """Cluster k-mers by their hash values with `df.groupby()`, and calculate the penalty for each cluster. 

        Args:
            kmers (pd.DataFrame): See `KmerGraph.kmers`. 
            tar_idx (set[int]): See `KmerGraph.__cluster()`. 
            neg_idx (set[int]): See `KmerGraph.__cluster()`. 

        Returns:
            pd.DataFrame: See `KmerGraph.clusters`. 
        """
        # cluster k-mers by hash value, and count the number of unique target / non-target assemblies in each cluster
        clusters = kmers.groupby(
            # as_index=True to return a series
            by=['hash', 'is_target'], as_index=True, sort=False
        )['assembly_idx'].nunique()
        clusters.rename('n', inplace=True)

        # into two groups (target / non-target), and drop index level 'is_target'
        df_tar = clusters.xs(True, level='is_target', drop_level=True)
        df_neg = clusters.xs(False, level='is_target', drop_level=True)

        # merge into one df (this will replace column 'n' with 'n_tar' and 'n_neg')
        # must set on='hash', otherwise the resulted df will use index of 0, 1, 2, ...
        clusters = pd.merge(
            df_tar, df_neg, how='outer', on='hash', sort=False, suffixes=('_tar', '_neg')
        )
        clusters.fillna(0, inplace=True)
        # when NaNs are inserted by pd.merge(), int64 is converted to float64
        # here the index (k-mer hash values) is not converted, and remains uint64
        clusters = clusters.astype('int64', copy=False, errors='raise')

        # true positive rate & false positive rate
        clusters['true_pos'] = clusters['n_tar'] / len(tar_idx)
        clusters['false_pos'] = clusters['n_neg'] / len(neg_idx)

        # calculate cluster penalty based on distance to true_pos=1 and false_pos=0
        clusters['penalty'] = ((1 - clusters['true_pos'])**2 + clusters['false_pos']**2)**0.5

        return clusters

    @staticmethod
    def __get_dist(kmers: pd.DataFrame, cnt_mtx: np.ndarray, kmerlen: int) -> np.ndarray:
        """Calculate distances between each assembly pair based on the number of shared k-mers, 
        using the distance formula used by `Mash<https://mash.readthedocs.io/en/latest/distances.html>`__. 

        Args:
            kmers (pd.DataFrame): See `KmerGraph.kmers`. 
            cnt_mtx (pd.DataFrame): See `KmerGraph.cnt_mtx`. 
            kmerlen (int): See `Config` in `config.py`. 
        
        Returns:
            np.ndarray: See `KmerGraph.dist_mtx`. 
        """
        logger.info(f'Calculating distances between each assembly pair...')
        tik = time()

        # count the number of unique k-mers in each assembly
        nunique_kmer = kmers.groupby(
            # use 'assembly_idx' as index by as_index=True (returns a pandas series)
            # cannot rely on iloc since some some assemblies might be missing after filtering
            # sort=True since cnt_mtx is upper triangular
            by='assembly_idx', as_index=True, sort=True
        )['hash'].nunique()

        # calculate jaccard and distance
        dist_mtx = np.ones(cnt_mtx.shape)
        for i, idx1 in enumerate(nunique_kmer.index):
            n1 = nunique_kmer.loc[idx1] # number of k-mers in assembly 1
            for idx2 in nunique_kmer.index[i+1:]:
                # assembly_idx is sorted in groupby, so idx1 < idx2
                n2 = nunique_kmer.loc[idx2] # number of k-mers in assembly 2
                c = cnt_mtx[idx1, idx2] # number of shared k-mers
                jaccard = c / (n1 + n2 - c)
                if jaccard > 0:
                    dist_mtx[idx1, idx2] = (-1/kmerlen) * np.log(2*jaccard/(1+jaccard))

        dist_mtx = dist_mtx - np.tril(dist_mtx) # remove all the ones in the lower triangle
        dist_mtx = dist_mtx + dist_mtx.T # make it symmetric

        print_time_delta(time()-tik)
        return dist_mtx

    def filter(
        self, penalty_th: float, edge_weight_th: float, min_nodes: int, max_nodes: int | None, rng: Random
    ) -> None:
        """Filter graph edges and nodes. 
        1. Remove low-weight edges and isolated nodes. 
        2. Extract low-penalty subgraphs. 
        3. Remove k-mers not included in any of the subgraphs. 

        Args:
            penalty_th (float): See `RunState` in `config.py`. 
            edge_weight_th (float): See `RunState` in `config.py`. 
            min_nodes (int): See `RunState` in `config.py`. 
            max_nodes (int | None): See `RunState` in `config.py`. 
            rng (random.Random): See `RunState` in `config.py`. 
        """
        kmers = self.kmers
        graph = self.graph
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
        graph = KmerGraph.__filter_graph(graph, edge_weight_th)

        # get low-penalty subgraphs
        subgraphs, used = KmerGraph.__get_subgraphs(graph, penalty_th, min_nodes, max_nodes, rng)

        logger.info(' - Removing k-mers not included in any of the subgraphs...')
        kmers = kmers[kmers['hash'].isin(used)]
        # kmers.drop(
        #     index=kmers.index[~kmers['hash'].isin(used)], 
        #     inplace=True
        # )
        logger.info(f' - {len(kmers)} k-mers left')

        print_time_delta(time()-tik)
        self.kmers = kmers
        self.graph = graph
        self.subgraphs = subgraphs
        self._filtered_flag = True

    @staticmethod
    def __filter_graph(graph: nx.Graph, edge_weight_th: float) -> nx.Graph:
        """Remove low-weight edges and isolated nodes from the graph. 
        
        Args:
            graph (nx.Graph): See `KmerGraph.graph`. 
            edge_weight_th (float): See `RunState` in `config.py`. 
        
        Returns:
            nx.Graph: The filtered graph. 
        """
        logger.info(' - Filtering graph edges and nodes...')

        # remove low-weight edges
        low_weight_edges = list((u, v) for u, v, w in graph.edges(data=EDGE_W) if w < edge_weight_th)
        graph.remove_edges_from(low_weight_edges)
        logger.info(f' - Removed {len(low_weight_edges)} edges with weight<{edge_weight_th:.3f}, {len(graph.edges)} edges left')

        # remove isolated nodes
        isolates = set(nx.isolates(graph)) # NOTE: might still be useful
        graph.remove_nodes_from(isolates)
        logger.info(f' - Removed {len(isolates)} isolated nodes from graph, {len(graph)} nodes left')

        return graph

    @staticmethod
    def __get_subgraphs(
        graph: nx.Graph, penalty_th: float, min_nodes: int, max_nodes: int | None, rng: Random
    ) -> tuple[list[set[int]], set[int]]:
        """Find disjoint (no shared node) subgraphs whose average node-penalty ≤ `penalty_th` and size within `size_th`. 
        1. Remove low-weight edges and isolated nodes from `graph`. 
        2. Find nodes with penalty ≤ `penalty_th` as seeds of subgraphs. 
        3. Greedy seed-expansion with breadth first search (BFS), where the neighboring node with the lowest penalty is 
            selected in each iteration. 

        A heap frontier (nodes to be visited in BFS) is used to accelerate the expansion process. 
        The heap is implemented with the built-in Python `heapq` module, which is a min-heap. 
        E.g., when tuples of `(penalty, node)` are pushed to the heap, it will always pop the tuple with the smallest `penalty` first. 
        This is faster than calling `min()` everytime to fetch the node with the lowest penalty. 
        When tested on the Salmonella dataset (576 genomes, no edge filtering), this implementation is more than 3x faster than the naive one. 
        However, the performance gain becomes less significant when more low-weight edges are removed. 

        Args:
            graph (nx.Graph): See `KmerGraph.graph`. 
            penalty_th (float): See `Config` in `config.py`. 
            min_nodes (int): See `Config` in `config.py`. 
            max_nodes (int | None): See `Config` in `config.py`. 
            rng (random.Random): See `RunState` in `config.py`. 

        Returns:
            tuple: A tuple containing
                1. subgraphs (list[set[int]]): See `KmerGraph.subgraphs`. 
                2. used (set[int]): Union of k-mer hash values in all subgraphs. 
        """
        # a dict mapping node to penalty for faster lookup
        node_penalty: dict[int, float] = dict(
            # sort nodes for reproducibility (nodes order decides seeds order)
            sorted(graph.nodes(data=NODE_P))
        )

        # collect all seed nodes and shuffle
        # use <=, otherwise there will be no seed when penalty_th = 0
        seeds = list(n for n, p in node_penalty.items() if p <= penalty_th)
        rng.shuffle(seeds)
        logger.info(f' - Expanding subgraphs from {len(seeds)} seed nodes (penalty≤{penalty_th:.5f})...')

        used: set[int] = set() # nodes already assigned to a subgraph
        subgraphs: list[set[int]] = list() # list of subgraphs to return

        for s in seeds:
            if s in used:
                continue

            # initialize the subgraph (sg)
            sg = {s}
            sum_penalty = node_penalty[s]

            #---------- subgraph expansion (the naive way) ----------#
            # # add initial neighbors to frontier
            # frontier = set(graph.neighbors(s)) - used - sg

            # # expand the subgraph by adding the node in the frontier with the lowest penalty
            # while frontier and len(sg) < max_size:
            #     node = min(frontier, key=lambda n: (node_penalty[n], n))

            #     # whether to accept this node
            #     new_sum_penalty = sum_penalty + node_penalty[node]
            #     if new_sum_penalty / (len(sg)+1) <= penalty_th:
            #         # accept
            #         sg.add(node)
            #         sum_penalty = new_sum_penalty

            #         # bring in its neighbors
            #         frontier |= (set(graph.neighbors(node)) - used - sg)

            #     # whether accepted or not, never reconsider this node
            #     frontier.remove(node)
            #---------- subgraph expansion (the naive way) ----------#

            #---------- subgraph expansion (heap frontier) ----------#
            # min‐heap of (penalty, node)
            frontier_heap: list[tuple[float, int]] = list()
            # a set synced with frontier_heap
            # for faster membership checking, also guarantees every node in the heap is unique
            frontier_set: set[int] = set()

            # add initial neighbors to frontier
            for nbr in graph.neighbors(s):
                if (nbr not in used) and (nbr not in sg):
                    heappush(frontier_heap, (node_penalty[nbr], nbr))
                    frontier_set.add(nbr)

            # expand the subgraph by adding the node in the frontier with the lowest penalty
            while frontier_heap and ((max_nodes is None) or (len(sg) < max_nodes)):
                penalty, node = heappop(frontier_heap)
                # keep frontier_heap and frontier_set consistent (might not be necessary but to be safe)
                if node not in frontier_set:
                    continue

                # whether to accept this node
                new_sum_penalty = sum_penalty + penalty
                # use <=, otherwise there will be no new node when penalty_th = 0
                if new_sum_penalty / (len(sg)+1) <= penalty_th:
                    # accept
                    sg.add(node)
                    sum_penalty = new_sum_penalty

                    # bring in its neighbors
                    for nbr in graph.neighbors(node):
                        if (nbr not in used) and (nbr not in sg) and (nbr not in frontier_set):
                            heappush(frontier_heap, (node_penalty[nbr], nbr))
                            frontier_set.add(nbr)

                # whether accepted or not, never reconsider this node
                frontier_set.remove(node)
            #---------- subgraph expansion (heap frontier) ----------#

            # keep or discard the subgraph
            if len(sg) >= min_nodes:
                subgraphs.append(sg)
                used |= sg
        logger.info(f' - Found {len(subgraphs)} low-penalty subgraphs')

        # due to the greedy nature of node expansion, subgraphs created first are usually larger
        # by shuffling the subgraphs, we can get a more balanced distribution of sizes in downstream multiprocessing
        rng.shuffle(subgraphs)

        return subgraphs, used

    def filter_strict(self, penalty_th: float, edge_weight_th: float) -> None:
        """Filter graph edges and nodes. 
        1. Remove low-weight edges and isolated nodes. 
        2. Extract low-penalty subgraphs (only include nodes with penalty ≤ `penalty_th`). 
        3. Remove k-mers not included in any of the subgraphs. 

        Args:
            penalty_th (float): See `Config` in `config.py`. 
            edge_weight_th (float): See `RunState` in `config.py`. 
        """
        kmers = self.kmers
        graph = self.graph
        filtered_flag = self._filtered_flag

        if filtered_flag:
            logger.error(f'K-mers are already filtered, cannot filter again.')
            return None

        logger.info('Extracting low-penalty subgraphs from the k-mer graph...')
        tik = time()

        # remove high penalty nodes
        # use greater than (>), otherwise everything wiil be bad when penalty_th = 0
        bad_nodes: set[int] = set(n for n, p in graph.nodes(data=NODE_P) if p > penalty_th)
        # remove from graph
        graph.remove_nodes_from(bad_nodes)
        logger.info(f' - Removed {len(bad_nodes)} nodes with penalty>{penalty_th}')

        # remove low-weight edges and isolated nodes
        graph = KmerGraph.__filter_graph(graph, edge_weight_th)

        # get connected components (subgraphs)
        logger.info(f' - Finding connected components...')
        subgraphs: list[set[int]] = list(nx.connected_components(graph))
        used = set.union(*subgraphs)
        logger.info(f' - Found {len(subgraphs)} components (low-penalty subgraphs)')

        logger.info(' - Removing k-mers not included in any of the subgraphs...')
        kmers = kmers[kmers['hash'].isin(used)]
        logger.info(f' - {len(kmers)} k-mers left.')

        print_time_delta(time()-tik)
        self.kmers = kmers
        self.graph = graph
        self.subgraphs = subgraphs
        self._filtered_flag = True


def _get_graph(
    assemblies: Assemblies, kmerlen: int, windowsize: int
) -> tuple[pd.DataFrame, WeightedGraph, set, list[tuple[str]]]:
    """Merge k-mers of multiple assemblies into a single pandas DataFrame, and create a weighted, undirected graph. 
    - For multiprocessing, this function cannot be a method of `KmerGraph`. 

    Args:
        assemblies (Assemblies): See `Assemblies` in `assemblies.py` (could be a slice of the DataFrame). 
    
    Returns:
        tuple: A tuple containing
            1. kmers (pd.DataFrame): See `KmerGraph.kmers`. 
            2. graph (WeightedGraph): See `WeightedGraph` in `graph.py`. 
            3. isolates (set): Isolated k-mers. Some short sequence records might only have one k-mer, leaving isolated graph nodes. 
            4. all_record_ids (list[tuple[str]]): Record IDs of each assembly. 
    """
    kmers = list() # to concat
    graph = list() # edges to be added to graph
    isolates: set[int] = set() # isolated k-mers found in short sequence records
    all_record_ids: list[tuple[str]] = list()

    for idx, assembly in assemblies.iterrows():
        # get the minimizer sketch of the current assembly
        kmers_assembly, record_ids = indexlr(assembly.path, kmerlen, windowsize)

        graph_assembly: set[tuple[int, int]] = set() # unique edges of the current assembly
        for kmers_record in kmers_assembly:
            if len(kmers_record) == 1:
                # isolate nodes cannot be added to graph since they don't have any edge
                # they might be included in another path of another record, but that doesn't matter
                isolates.add(kmers_record['hash'].iloc[0])
            elif len(kmers_record) > 1:
                # data type of an edge should be hashable
                # cannot use frozenset() since an edge could be a self loop
                # so we can only use tuple, but note tuple is ordered and we want a undirected graph
                graph_assembly.update(
                    tuple((u, v)) if u < v else tuple((v, u))
                    for u, v in zip(kmers_record['hash'].iloc[:-1], kmers_record['hash'].iloc[1:])
                )
            # add assembly columns to the DataFrame
            kmers_record['assembly_idx'] = _IDX_TYPE(idx)
            kmers_record['is_target'] = assembly.is_target

        kmers.extend(kmers_assembly)
        graph.extend(graph_assembly)
        all_record_ids.append(record_ids)

    kmers = pd.concat(kmers, ignore_index=True)
    graph = WeightedGraph(graph) # get edge weight; a dict of {edge: weight}
    return kmers, graph, isolates, all_record_ids


def get_kmers(
    assemblies: Assemblies, config: Config, state: RunState
) -> tuple[KmerGraph, pd.DataFrame | None]:
    """Get clustered k-mers from all assemblies, create a k-mer graph and find low-penalty subgraphs. 

    Args:
        assemblies (Assemblies): See `Assemblies` in `assemblies.py`. 
        config (Config): See `Config` in `config.py`. 
        state (RunState): See `RunState` in `config.py`. 

    Returns:
        Returns:
        tuple: A tuple containing
            1. kmers (KmerGraph): The KmerGraph instance. 
            2. mash_dist (pd.DataFrame | None): Tabular output of `mash dist`. 
    """
    overwrite = config.overwrite
    kmerlen = config.kmerlen
    windowsize = config.windowsize
    penalty_th = config.penalty_th
    min_len = config.min_len
    max_len = config.max_len
    no_filter = config.no_filter
    penalty_th_cap = config.penalty_th_cap
    edge_w_th_mul = config.edge_w_th_mul
    min_nodes_floor = config.min_nodes_floor
    max_nodes_cap = config.max_nodes_cap
    sketchsize = config.sketchsize
    get_dist = config.get_dist
    n_cpu = config.n_cpu

    working_dir = state.working_dir
    rng = state.rng
    n_tar = state.n_tar
    n_neg = state.n_neg

    kmers = KmerGraph(assemblies, kmerlen, windowsize, get_dist, n_cpu)

    if no_filter:
        # skip kmers.filter(), debug only
        return kmers, None

    # calculate filter params
    # 1. calculate penalty threshold
    if penalty_th is None:
        logger.info(f'Calculating penalty threshold...')
        tik = time()

        mash_dist = assemblies.mash(
            kmerlen=kmerlen, 
            sketchsize=sketchsize, 
            out_path=working_dir / WORKINGDIR.mash, 
            overwrite=overwrite, 
            n_cpu=n_cpu
        )

        # calculate penalty_th with jaccard index
        jaccard = np.array(mash_dist['jaccard']).reshape(len(assemblies), -1)
        # average jaccard index of all target-target pairs
        j_tar = (np.sum(jaccard[:n_tar, :n_tar]) - n_tar) / (n_tar * (n_tar - 1))
        logger.info(f' - average Jaccard within targets: {j_tar:.5f}')
        # average jaccard index of all target-(non-target) pairs
        j_neg = np.sum(jaccard[n_tar:, :n_tar]) / (n_tar * n_neg)
        logger.info(f' - average Jaccard between targets and non-targets: {j_neg:.5f}')
        # choose the more stringent threshold
        penalty_th = min(1 - j_tar, j_neg)
        if penalty_th < penalty_th_cap:
            logger.info(f' - penalty threshold set as {penalty_th:.5f} (calculated with Jaccard)')
        else:
            penalty_th = penalty_th_cap
            logger.warning(f' - penalty threshold is capped at {penalty_th} (check Jaccard)')

        print_time_delta(time()-tik)
    else:
        logger.warning(f'Penalty threshold is provided (--penalty-th), skip running Mash')
        mash_dist = None

    # 2. calculate edge weight threshold
    # consider N as the number of assemblies that include a certain k-mer
    # since we want k-mers with penalty lower than penalty_th
    # based on the definition of penalty, N ≥ (1 - penalty_th) * n_tar
    # so edge weight threshold is calculated based on the lower bound of N, times a multiplier
    edge_weight_th = edge_w_th_mul * (1 - penalty_th) * n_tar

    # 3. calculate size range of subgraphs
    gap_len = windowsize // 2 # average length of gap between minimizers
    min_nodes = max(min_nodes_floor, min_len // gap_len + 1)
    if max_len is None:
        max_nodes = max_nodes_cap
    else:
        max_nodes = max_len // gap_len + 1

    # extract low-penalty subgraphs and remove unused k-mers
    kmers.filter(penalty_th, edge_weight_th, min_nodes, max_nodes, rng)
    #kmers.filter_strict(penalty_th, edge_weight_th)

    state.penalty_th = penalty_th
    state.edge_weight_th = edge_weight_th
    state.min_nodes = min_nodes
    state.max_nodes = max_nodes
    return kmers, mash_dist
