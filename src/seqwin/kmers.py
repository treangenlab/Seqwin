"""
K-mer Graph
===========

A core module of Seqwin. Create an instance for k-mers of all input genome assemblies, 
including the weighted k-mer graph and low-penalty subgraphs. 

Dependencies:
-------------
- numpy
- numba
- networkx
- pandas (optional)
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
from multiprocessing.shared_memory import SharedMemory

logger = logging.getLogger(__name__)

import numpy as np
import networkx as nx
from numba import njit, types, from_dtype
from numba.typed import List, Dict
try:
    # try import dependencies for distance calculation
    import pandas as pd
    from scipy.sparse import coo_matrix
    _HAS_DIST_DEPS = True
except ImportError:
    _HAS_DIST_DEPS = False

from .assemblies import Assemblies
from .minimizer import indexlr_py, KMER_DTYPE
from .utils import StartMethod, SharedArr, print_time_delta, log_and_raise, mp_wrapper, \
    get_chunks, concat_to_shm, concat_from_shm
from .config import Config, RunState, WORKINGDIR, EDGE_W, NODE_P

# numpy dtype of k-mer hash values
_HASH_NP_DT = KMER_DTYPE['hash']

# dtypes for numba
_HASH_NB_DT = from_dtype(_HASH_NP_DT)
_HASH_ARR_NB_DT = types.Array(_HASH_NB_DT, 1, 'A') # 'A': accepts arbitrary/strided arrays
_EDGE_NB_DT = types.UniTuple(_HASH_NB_DT, 2)

_IDX_DTYPE = np.uint32 # dtype for k-mer indices
_CLUSTER_DTYPE = np.dtype([ # dtype for each k-mer cluster
    ('hash', _HASH_NP_DT), 
    ('n_tar', KMER_DTYPE['assembly_idx']), 
    ('n_neg', KMER_DTYPE['assembly_idx']), 
    ('penalty', np.float64)
])


class KmerGraph(object):
    """
    1. Create a weighted k-mer graph, and calculate node penalty scores. 
    2. (Optional) Calculate `Mash distances<https://mash.readthedocs.io/en/latest/distances.html>`__ for each assembly pair. 
    3. Extract low-penalty subgraphs from the k-mer graph with `self.filter()`. 
    
    Attributes:
        kmers (np.ndarray): A 1-D Numpy array of k-mers from all assemblies, with dtype `KMER_DTYPE` defined in `minimizer.py`. 
        idx (np.ndarray | None): The original indices when k-mers are generated (k-mers with consecutive indices are adjacent in the genome). 
        graph (nx.Graph): A weighted, undirected graph of k-mers. 
            Edge weight is the number of assemblies where the two k-mers are adjacent. 
        clusters (np.ndarray): A 1-D Numpy array of k-mer clusters, with fields ['hash', 'n_tar', 'n_neg', 'penalty']. 
        cnt_mtx (np.ndarray | None): A matrix of the number of shared k-mers between each assembly pair. 
            Calculated when`get_dist=True`. 
        dist_mtx (np.ndarray | None): A matrix of the Mash distance between each assembly pair. 
            Calculated when `get_dist=True`. 
        subgraphs (tuple[tuple[frozenset[int], ...] | None): Low-penalty subgraphs. Each subgraph is a set of k-mer hash values. 
            Generated with `self.filter()`. 
    """
    __slots__ = (
        'kmers', 'idx', 'graph', 'clusters', 'cnt_mtx', 'dist_mtx', 'subgraphs', '_filtered_flag'
    )
    kmers: np.ndarray
    idx: np.ndarray
    graph: nx.Graph
    clusters: np.ndarray
    cnt_mtx: np.ndarray | None
    dist_mtx: np.ndarray | None
    subgraphs: tuple[frozenset[int], ...] | None
    _filtered_flag: bool # True if `self.filter()` is called. 

    def __init__(self, assemblies: Assemblies, kmerlen: int, windowsize: int, get_dist: bool, n_cpu: int) -> None:
        """
        1. Create a weighted, undirected graph of k-mers. 
        2. Calculate node penalty scores. 
        3. Calculate assembly distances if `get_dist=True`. 

        Args:
            assemblies (Assemblies): See `Assemblies` in `assemblies.py`. 
            kmerlen (int): See `Config` in `config.py`. 
            windowsize (int): See `Config` in `config.py`. 
            get_dist (bool): See `Config` in `config.py`. 
            n_cpu (int): See `Config` in `config.py`. 
        """
        # merge k-mers from all assemblies and create the graph
        # k-mers in the array have the same order as they appear in the genomes, 
        # and their indices are just their positions in the array (0, 1, 2, ...)
        kmers, graph = KmerGraph.__get_graph(assemblies, kmerlen, windowsize, n_cpu)

        # calculate penalty scores by clustering k-mers
        # k-mers are now sorted by hash values, idx is the original indices
        kmers, idx, clusters, cnt_mtx = KmerGraph.__get_penalty(kmers, assemblies, get_dist)

        # sanity check
        if set(graph.nodes) != set(clusters['hash']):
            log_and_raise(ValueError, 'Inconsistent graph nodes and k-mer clusters.')

        # add penalties to networkx graph nodes
        nx.set_node_attributes(
            graph, 
            values=dict(zip(clusters['hash'], clusters['penalty'])), 
            name=NODE_P
        )

        # calculate assembly distances
        if cnt_mtx is not None:
            dist_mtx = KmerGraph.__get_dist(cnt_mtx, kmerlen)
        else:
            dist_mtx = None

        self.kmers = kmers
        self.idx = idx
        self.graph = graph
        self.clusters = clusters
        self.cnt_mtx = cnt_mtx
        self.dist_mtx = dist_mtx
        self.subgraphs = None
        self._filtered_flag = False

    @staticmethod
    def __get_graph(assemblies: Assemblies, kmerlen: int, windowsize: int, n_cpu: int) -> tuple[np.ndarray, nx.Graph]:
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
                1. np.ndarray: See `KmerGraph.kmers`. 
                2. nx.Graph: See `KmerGraph.graph`. 
        """
        n_assemblies = len(assemblies)
        logger.info(f'Creating a weighted, undirected minimizer graph from {n_assemblies} assemblies...')
        tik = time()

        # merge k-mers from all assembies
        if n_cpu <= 1:
            # create the graph with a single thread
            kmers, edges, isolates, record_ids = _get_graph(assemblies, kmerlen, windowsize, return_shm=False)
        else:
            # to make mp work, the method/function must be static and should not start with double underscores (single is fine)
            # difference between single & double underscores: https://docs.python.org/3/tutorial/classes.html#private-variables
            logger.info(f' - Parallelizing across {n_cpu} processes (~{n_assemblies//n_cpu} assemblies per process)...')
            graph_args = zip(
                get_chunks(assemblies, n_cpu), 
                repeat(kmerlen, n_cpu), 
                repeat(windowsize, n_cpu)
            )
            kmers, edges, isolates, record_ids = mp_wrapper(
                _get_graph, graph_args, n_cpu, unpack_output=True, 
                start_method=StartMethod.forkserver # must use spawn/forkserver for the indexlr python wrapper
            )
            # merge outputs from multiple processes
            logger.info(' - Merging from all processes...')
            kmers = concat_from_shm(kmers)
            edges = concat_from_shm(edges)
            edges = _merge_weighted_edges(edges)
            isolates = np.unique(np.concatenate(isolates))
            record_ids = list(chain.from_iterable(record_ids))

        # convert to a networkx graph
        graph = nx.Graph()
        graph.add_weighted_edges_from(edges, weight=EDGE_W)
        # add isolated nodes to graph, so that the number of nodes is the same as the number of k-mer clusters
        graph.add_nodes_from(isolates)

        logger.info(f' - {len(graph)} nodes and {len(graph.edges)} edges from {len(kmers)} k-mers')
        logger.info(f' - {len(isolates)} nodes are isolates')
        print_time_delta(time()-tik)

        assemblies.record_ids = record_ids
        return kmers, graph

    @staticmethod
    def __get_penalty(
        kmers: np.ndarray, assemblies: Assemblies, get_dist: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Calculate node penalty scores by 
        1. sorting k-mers by hash values, 
        2. aggregating k-mers with the same hash value to calculate penalty. 
        - Ideally, this can be done during graph creation. But in python, it is faster to do this in bulk after all k-mers are merged. 
        Plus, sorting k-mers here can also make removing unused k-mers faster in `KmerGraph.__filter_kmers()`. 
        - Assembly distance can also be calculated in this step (`get_dist=True`), using a slower clustering function `KmerGraph.__cluster_dist()`. 

        Args:
            kmers (np.ndarray): See `KmerGraph.kmers`. 
            assemblies (Assemblies): See `Assemblies` in `assemblies.py`. 
            get_dist (bool): See `Config` in `config.py`. 
        
        Returns:
            tuple: A tuple containing
                1. np.ndarray: See `KmerGraph.kmers`. 
                2. np.ndarray: See `KmerGraph.idx`. 
                3. np.ndarray: See `KmerGraph.clusters`. 
                4. np.ndarray | None: See `KmerGraph.cnt_mtx`. 
        """
        logger.info(f'Calculating penalty score for each k-mer node...')
        tik = time()

        n_assemblies = len(assemblies)
        n_tar = sum(assemblies.is_target)
        n_neg = n_assemblies - n_tar

        logger.info(' - Sorting k-mers by hash values...')
        # stable sort in-place (preserve assembly order), and return the sorted indices
        # here we don't sort with the built-in numpy functions, because argsort (or lexsort) 
        # creates a C contiguous copy of the strided kmers['hash'], and it uses int64 for indices; 
        # reordering a struct array with kmers[:] = kmer[idx] also needs a large buffer; 
        # all these would increase peak RAM by a lot. 
        # to solve this, we could 
        # 1) structure kmers as a tuple of contiguous arrays (instead of a structured array), and use numpy sorting; 
        # 2) write a custom sorting function that works on a strided array. 
        # option 1) involves rewriting a lot of functions, and numpy still uses in64 for indexing
        # option 2) is easier to implement and equally fast
        idx = _sort_by_hash(kmers)

        if get_dist and (not _HAS_DIST_DEPS):
            logger.error(' - Pandas and SciPy are not installed, skip assembly distance calculation')
            get_dist = False

        # choose which clustering function to use
        if get_dist:
            logger.warning(' - Assembly distance calculation is turned on, extra time and memory needed')
            clusters, cnt_mtx = KmerGraph.__cluster_dist(kmers, n_assemblies)
        else:
            logger.info(' - Aggregating k-mer clusters...')
            # better pass individual fields to numba
            clusters = KmerGraph.__cluster(
                kmers['hash'], 
                kmers['assembly_idx'], 
                kmers['is_target']
            )
            cnt_mtx = None

        # calculate penalty for each cluster
        clusters['penalty'] = _penalty_func(
            clusters['n_tar'] / n_tar, 
            clusters['n_neg'] / n_neg
        )

        print_time_delta(time()-tik)
        return kmers, idx, clusters, cnt_mtx

    @staticmethod
    def __cluster_dist(kmers: np.ndarray, n_assemblies: int) -> tuple[np.ndarray, np.ndarray]:
        """
        1. Cluster k-mers by their hash values. 
        2. Count the number of target/non-target assemblies each cluster. 
        3. Count the number of shared k-mers between all assembly pairs. 

        Args:
            kmers (np.ndarray): See `KmerGraph.kmers`. 
            n_assemblies (int): Number of assemblies. 

        Returns:
            tuple: A tuple containing
                1. np.ndarray: See `KmerGraph.clusters`. 
                2. np.ndarray: See `KmerGraph.cnt_mtx`. 
        """
        logger.info(' - Clustering k-mers across all assemblies...')
        # drop duplicate hash values for each assembly
        clusters = pd.DataFrame(kmers, copy=False)[
            ['hash', 'assembly_idx', 'is_target']
        ].drop_duplicates(keep='first', inplace=False)

        # cluster by hash and count the number of target / total assemblies in each cluster
        clusters = clusters.groupby(
            by='hash', as_index=False, sort=False
        ).agg(
            assembly_idx=pd.NamedAgg(column='assembly_idx', aggfunc=frozenset), 
            n_tar=pd.NamedAgg(column='is_target', aggfunc='sum'), 
            total=pd.NamedAgg(column='is_target', aggfunc='size'), 
        )
        assembly_idx = clusters.pop('assembly_idx')

        # formatting
        clusters['n_neg'] = clusters.pop('total') - clusters['n_tar']
        clusters['penalty'] = .0 # placeholder for penalty
        # convert to structured array
        clusters = clusters.to_records(
            index=False, 
            # column_dtypes must be a dict
            column_dtypes={name: _CLUSTER_DTYPE[name] for name in _CLUSTER_DTYPE.names}
        )
        clusters = clusters.view(np.ndarray)

        logger.info(' - Counting shared k-mers between all assembly pairs...')
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
                len(clusters), n_assemblies
            )
        ).tocsr() # convert to sparse

        # 2. calculate the number of shared k-mers by matrix multiplication
        cnt_mtx: coo_matrix = M.T @ M

        return clusters, cnt_mtx.toarray()

    @staticmethod
    @njit(nogil=True)
    def __cluster(hashes: np.ndarray, assembly_idx: np.ndarray, is_target: np.ndarray) -> np.ndarray:
        """Count the number of target/non-target assemblies for each unique hash value. 

        Args:
            hashes (np.ndarray): Field of `KmerGraph.kmers`. 
            assembly_idx (np.ndarray): Field of `KmerGraph.kmers`. 
            is_target (np.ndarray): Field of `KmerGraph.kmers`. 

        Returns:
            np.ndarray: See `KmerGraph.clusters`. 
        """
        n = hashes.size
        # pre-allocate output array (never larger than n)
        clusters = np.empty(n, dtype=_CLUSTER_DTYPE) # define dtype outside the numba function

        cluster_i = 0
        i = 0
        while i < n:
            curr_hash = hashes[i]

            n_tar = n_neg = 0
            # walk all rows having the same hash
            while i < n and hashes[i] == curr_hash:
                curr_a = assembly_idx[i]
                curr_t = is_target[i]

                # skip any further duplicates in the same assembly
                j = i + 1
                while (
                    j < n 
                    and hashes[j] == curr_hash
                    and assembly_idx[j] == curr_a
                ):
                    j += 1

                # count the current assembly
                if curr_t:
                    n_tar += 1
                else:
                    n_neg += 1
                i = j # advance by the duplicates

            # set individual values in numba, no fancy indexing
            clusters[cluster_i]['hash'] = curr_hash
            clusters[cluster_i]['n_tar'] = n_tar
            clusters[cluster_i]['n_neg'] = n_neg
            clusters[cluster_i]['penalty'] = .0 # placeholder for penalty
            cluster_i += 1

        # trim the over-allocated buffers
        return clusters[:cluster_i]

    @staticmethod
    def __get_dist(cnt_mtx: np.ndarray, kmerlen: int) -> np.ndarray:
        """Calculate distances between each assembly pair based on the number of shared k-mers, 
        using the distance formula used by `Mash<https://mash.readthedocs.io/en/latest/distances.html>`__. 

        Args:
            cnt_mtx (np.ndarray): See `KmerGraph.cnt_mtx`. 
            kmerlen (int): See `Config` in `config.py`. 
        
        Returns:
            np.ndarray: See `KmerGraph.dist_mtx`. 
        """
        logger.info(f'Calculating distances between each assembly pair...')
        tik = time()

        # get the number of unique k-mers in each assembly
        nunique_kmer = np.diagonal(cnt_mtx)

        # calculate jaccard and distance
        dist_mtx = np.ones(cnt_mtx.shape)
        for i, n1 in enumerate(nunique_kmer):
            for j, n2 in enumerate(nunique_kmer[i+1:], start=i+1):
                c = cnt_mtx[i, j] # number of shared k-mers
                jaccard = c / (n1 + n2 - c)
                if jaccard > 0:
                    dist_mtx[i, j] = (-1/kmerlen) * np.log(2*jaccard/(1+jaccard))

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
        idx = self.idx
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

        # remove unused k-mers
        kmers, idx = KmerGraph.__filter_kmers(kmers, idx, used)

        print_time_delta(time()-tik)
        self.kmers = kmers
        self.idx = idx
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
    ) -> tuple[tuple[frozenset[int], ...], frozenset[int]]:
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
                1. tuple[tuple[frozenset[int], ...]: See `KmerGraph.subgraphs`. 
                2. frozenset[int]: Union of k-mer hash values in all subgraphs. 
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

        return tuple(frozenset(sg) for sg in subgraphs), frozenset(used)

    @staticmethod
    def __filter_kmers(kmers: np.ndarray, idx: np.ndarray, used: frozenset[int]) -> tuple[np.ndarray, np.ndarray]:
        """Remove k-mers not included in `used`. `kmers` is already sorted by 'hash' (see `__get_penalty()`). 

        Args:
            kmers (np.ndarray): See `KmerGraph.kmers`. 
            idx (np.ndarray): See `KmerGraph.idx`. 
            used (frozenset[int]): Output of `KmerGraph.__get_subgraphs()`. 
        
        Returns:
            tuple: A tuple containing
                1. np.ndarray: See `KmerGraph.kmers`. 
                2. np.ndarray: See `KmerGraph.idx`. 
        """
        logger.info(' - Removing k-mers not included in any of the subgraphs...')
        hashes = kmers['hash']

        # convert used to a sorted array (sorting is not necessary, but good practice)
        used_arr = np.fromiter(used, dtype=hashes.dtype, count=len(used))
        used_arr.sort()

        # create a mask of k-mers to be kept
        # np.isin is memory-hungry, plus kmers is already sorted
        # 1. find the index of used k-mers in hashes
        left  = np.searchsorted(hashes, used_arr, side='left')
        right = np.searchsorted(hashes, used_arr, side='right')
        # 2. mark start/stop with 1/-1
        flag = np.zeros(hashes.size + 1, dtype=np.int8)
        np.add.at(flag, left, 1)
        np.add.at(flag, right, -1)
        # 3. create a boolean mask (specify int8, otherwise intp will be used)
        np.cumsum(flag[:-1], dtype=np.int8, out=flag[:-1]) # in-place cumsum
        mask = flag[:-1] > 0

        # apply mask
        kmers = kmers[mask]
        idx = idx[mask]
        logger.info(f' - {len(kmers)} k-mers left')
        return kmers, idx

    def filter_strict(self, penalty_th: float, edge_weight_th: float) -> None:
        """Filter graph edges and nodes. 
        1. Remove nodes with penalty > `penalty_th`. 
        2. Remove low-weight edges and isolated nodes. 
        3. Extract connected components (subgraphs). 
        4. Remove k-mers not included in any of the subgraphs. 

        Args:
            penalty_th (float): See `Config` in `config.py`. 
            edge_weight_th (float): See `RunState` in `config.py`. 
        """
        kmers = self.kmers
        idx = self.idx
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
        logger.info(f' - Removed {len(bad_nodes)} nodes with penalty>{penalty_th:.5f}')

        # remove low-weight edges and isolated nodes
        graph = KmerGraph.__filter_graph(graph, edge_weight_th)

        # get connected components (subgraphs)
        logger.info(f' - Finding connected components...')
        subgraphs = tuple(frozenset(sg) for sg in nx.connected_components(graph))
        used = frozenset.union(*subgraphs)
        logger.info(f' - Found {len(subgraphs)} components (low-penalty subgraphs)')

        # remove unused k-mers
        kmers, idx = KmerGraph.__filter_kmers(kmers, idx, used)

        print_time_delta(time()-tik)
        self.kmers = kmers
        self.idx = idx
        self.graph = graph
        self.subgraphs = subgraphs
        self._filtered_flag = True


@njit(nogil=True)
def _get_edges(hashes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Called inside `_get_graph()` to get weighted edges and isolated nodes. 

    Args:
        hashes (List[List[Array]]): K-mer hash values from all assemblies. 
    
    Returns:
        tuple: A tuple containing
            1. np.ndarray: Unique edges. 
            2. np.ndarray: Edge weights. 
            3. np.ndarray: Isolated nodes. 
    """
    # a counter for edges and weight
    edge_w = Dict.empty(
        # all type variables must be defined outside this function
        key_type=_EDGE_NB_DT, 
        value_type=types.intp
    )
    # isolated k-mers found in short sequence records
    # there is no numba typed set, just use set()
    isolates = set()

    for hashes_assembly in hashes:
        # get unique edges of the current assembly
        edges_assembly = set()

        for hashes_record in hashes_assembly:
            n = len(hashes_record)
            if n == 1:
                # isolate nodes cannot be added to graph since they don't have any edge
                # they might be included in another path of another record, but that doesn't matter
                isolates.add(hashes_record[0])
            else:
                # get edges of the current sequence record
                for i in range(n - 1):
                    h1 = hashes_record[i]
                    h2 = hashes_record[i + 1]
                    # canonical order of edge nodes (undirected graph)
                    if h1 < h2:
                        edges_assembly.add((h1, h2))
                    else:
                        edges_assembly.add((h2, h1))

        # add to the counter dict
        for e in edges_assembly:
            if e in edge_w:
                edge_w[e] += 1
            else:
                edge_w[e] = 1

    # prepare output edges and weights as np arrays
    n_edges = len(edge_w)
    edges = np.empty((n_edges, 2), dtype=_HASH_NP_DT)
    weights = np.empty(n_edges, dtype=np.intp)
    i = 0
    for e, w in edge_w.items():
        edges[i, 0] = e[0]
        edges[i, 1] = e[1]
        weights[i] = w
        i += 1

    # convert isolates to a np array
    n_isolates = len(isolates)
    isolates_arr = np.empty(n_isolates, dtype=_HASH_NP_DT)
    i = 0
    for h in isolates:
        isolates_arr[i] = h
        i += 1
    return edges, weights, isolates_arr


def _stack_to_shm(edges: np.ndarray, weights: np.ndarray) -> SharedArr:
    """Merge edges and weights into a single 3-column array. 

    Args:
        edges (np.ndarray): A 2-column array of edges. 
        weights (np.ndarray): Edge weights with the same length of `edges`. 
    
    Returns:
        SharedArr: The merged array attached to a SharedMemory instance. 
    """
    dtype = edges.dtype
    weights = weights.astype(dtype, copy=False)

    # create shared memory
    n_edges = edges.shape[0]
    shm = SharedMemory(create=True, size=n_edges * 3 * dtype.itemsize)
    arr_view = np.ndarray((n_edges, 3), dtype=dtype, buffer=shm.buf)

    # copy data into shared memory
    arr_view[:, :2] = edges
    arr_view[:, 2] = weights

    edges = SharedArr(shm.name, arr_view.shape, dtype)
    shm.close()
    return edges


def _get_graph(
    assemblies: Assemblies, kmerlen: int, windowsize: int, return_shm: bool=True
) -> tuple[SharedArr | np.ndarray, SharedArr | np.ndarray, np.ndarray, list[tuple[str]]]:
    """
    1. Merge k-mers from multiple assemblies into a 1-D structured Numpy array. 
    2. Generate weighted, undirected edges for the k-mer graph. 
    
    Numpy arrays are returned as shared memory blocks for multiprocessing. 

    Args:
        assemblies (Assemblies): See `Assemblies` in `assemblies.py` (could be a slice of the DataFrame). 
        kmerlen (int): See `Config` in `config.py`. 
        windowsize (int): See `Config` in `config.py`. 
        return_shm (bool, optional): If True, return the kmers and edges arrays using shared memory blocks; else return regular Numpy arrays. 
    
    Returns:
        tuple: A tuple containing
            1. SharedArr | np.ndarray: Merged k-mers. Return a `SharedArr` instance if `return_shm` is True; else return a Numpy array. 
                dtype is defined in `minimizer.py` (`KMER_DTYPE`). 
            2. SharedArr | np.ndarray: The weighted edges in a 3-column Numpy array: the first two columns are edges and the third column is weights. 
                Return a `SharedArr` instance if `return_shm` is True; else return a Numpy array. 
            3. np.ndarray: Isolated k-mer nodes. Some short sequence records might only have one k-mer, leaving isolated graph nodes. 
            4. list[tuple[str]]: Record IDs of each assembly. 
    """
    #---------- generate k-mers ----------#
    kmers: list[list[np.ndarray]] = list()
    record_ids: list[tuple[str]] = list() # record ids of each assembly

    for idx, assembly in assemblies.iterrows():
        # get the minimizer sketch of the current assembly
        kmers_assembly, ids = indexlr_py(
            assembly.path, kmerlen, windowsize, idx, assembly.is_target
        )
        kmers.append(kmers_assembly)
        record_ids.append(ids)
    #---------- generate k-mers ----------#

    #---------- generate graph edges ----------#
    # define input types of the numba function
    # this is a List[List[Array]]
    hashes = List.empty_list(types.ListType(_HASH_ARR_NB_DT))

    for kmers_assembly in kmers:
        # this is a List[Array]
        hashes_assembly = List.empty_list(_HASH_ARR_NB_DT)

        for kmers_record in kmers_assembly:
            # this does not create a copy, kmers_record['hash'] returns a strided view
            # must be specified in the numba dtype ('A')
            hashes_assembly.append(kmers_record['hash'])

        hashes.append(hashes_assembly)

    # get weighted edges and isolated nodes
    edges, weights, isolates = _get_edges(hashes)
    #---------- generate graph edges ----------#

    # concat arrays and return
    kmers = list(chain.from_iterable(kmers)) # flatten kmers before concat
    if return_shm:
        kmers = concat_to_shm(kmers)
        edges = _stack_to_shm(edges, weights)
    else:
        kmers = np.concatenate(kmers)
        # must convert to the same dtype, otherwise both converted to float64
        edges = np.column_stack((
            edges, weights.astype(edges.dtype, copy=False)
        ))

    return kmers, edges, isolates, record_ids


def _merge_weighted_edges(edges: np.ndarray):
    """Add weights of the same edges. 

    Args:
        edges (np.ndarray): Concatenated outputs of `_get_graph()`. 
    
    Returns:
        np.ndarray: Unique edges with total weights. 
    """
    # idx maps edges to unique_edges
    unique_edges, idx = np.unique(edges[:, :2], axis=0, return_inverse=True)
    # calculate the weight of each unique edge; weights are sorted by idx
    weights = np.bincount(idx, weights=edges[:, 2])
    edges = np.column_stack((
        unique_edges, weights.astype(edges.dtype, copy=False)
    ))
    return edges


@njit(nogil=True)
def _sort_by_hash(kmers: np.ndarray) -> np.ndarray:
    """Sort `kmers` by 'hash' in-place in a stable manner, and return the sorted indices. 
    Use LSD (Least Significant Digit) radix sort. 
    - A buffer of `4 * len(kmers)` bytes (uint32 indices) is needed during this process, so memory usage should be around `kmers + idx + buffer`. 
    - Note that this function is hard coded for hash dtype of uint64. 

    Args:
        kmers (np.ndarray): See `KmerGraph.kmers`. 

    Returns:
        np.ndarray: See `KmerGraph.idx`. 
    """
    n = kmers.size
    # create k-mer indices
    idx = np.arange(n, dtype=_IDX_DTYPE)

    # 1. sort idx by 'hash' using a stable algorithm (LSD radix sort on 64-bit keys)
    idx_out = np.empty(n, dtype=_IDX_DTYPE) # buffer for idx
    # process 16 bits at a time (4 passes for 64-bit keys: 0-15, 16-31, 32-47, 48-63)
    for shift in range(0, 64, 16):
        # counting sort on the current 16-bit segment of the hash (stable)
        count = np.zeros(65536, dtype=np.int64) # frequency array for 16-bit values (0-65535)
        # count occurrences of each 16-bit key value
        for i in range(n):
            key = (kmers[idx[i]]['hash'] >> shift) & 0xFFFF
            count[key] += 1
        # compute prefix sums in count to get starting index for each key value in sorted order
        total = 0
        for value in range(65536):
            c = count[value]
            count[value] = total
            total += c
        # distribute indices into idx_out according to the current 16-bit key (stable order)
        for i in range(n):
            key = (kmers[idx[i]]['hash'] >> shift) & 0xFFFF
            idx_out[count[key]] = idx[i]
            count[key] += 1
        # swap idx and idx_out for the next pass (partial sort is now in idx)
        idx, idx_out = idx_out, idx

    # 2. reorder kmers in-place using the sorted indices.
    # build inverse mapping: dest[old_index] = new_position of that element.
    dest = idx_out # reuse buffer
    for new_pos in range(n):
        dest[idx[new_pos]] = new_pos
    # iterate through each element, and swap it into correct position if it's not already there.
    for i in range(n):
        while dest[i] != i:
            j = dest[i]
            # swap the record at i with the record at j, field by field (no fancy index for numba)
            kmers[i]['hash'],         kmers[j]['hash']         = kmers[j]['hash'],         kmers[i]['hash']
            kmers[i]['pos'],          kmers[j]['pos']          = kmers[j]['pos'],          kmers[i]['pos']
            kmers[i]['record_idx'],   kmers[j]['record_idx']   = kmers[j]['record_idx'],   kmers[i]['record_idx']
            kmers[i]['assembly_idx'], kmers[j]['assembly_idx'] = kmers[j]['assembly_idx'], kmers[i]['assembly_idx']
            kmers[i]['is_target'],    kmers[j]['is_target']    = kmers[j]['is_target'],    kmers[i]['is_target']
            # update the index mapping after the swap
            dest[i], dest[j] = dest[j], dest[i]
    return idx


def _penalty_func(frac_tar: np.ndarray | float, frac_neg: np.ndarray | float) -> np.ndarray | float:
    """The penalty function. 
    """
    return ((1 - frac_tar)**2 + frac_neg**2)**0.5


def get_kmers(
    assemblies: Assemblies, config: Config, state: RunState
) -> tuple[KmerGraph, np.ndarray | None]:
    """Get clustered k-mers from all assemblies, create a k-mer graph and find low-penalty subgraphs. 

    Args:
        assemblies (Assemblies): See `Assemblies` in `assemblies.py`. 
        config (Config): See `Config` in `config.py`. 
        state (RunState): See `RunState` in `config.py`. 

    Returns:
        tuple: A tuple containing
            1. KmerGraph: The KmerGraph instance. 
            2. np.ndarray | None: A matrix of Jaccard indexes of all assembly pairs. 
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

        jaccard = assemblies.mash(
            kmerlen=kmerlen, 
            sketchsize=sketchsize, 
            out_path=working_dir / WORKINGDIR.mash, 
            overwrite=overwrite, 
            n_cpu=n_cpu
        )

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
        jaccard = None

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
    return kmers, jaccard
