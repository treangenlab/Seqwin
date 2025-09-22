"""
Markers
=======

A core module of Seqwin. Extract candidate markers from subgraphs of a k-mer graph 
(`kmers.KmerGraph.subgraphs`). 

Dependencies
------------
- numpy
- pandas
- networkx
- .assemblies
- .kmers
- .ncbi
- .graph
- .config
- .utils

Classes:
--------
- ConnectedKmers

Functions:
----------
- eval_markers
- get_markers
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import logging
from pathlib import Path
from time import time
from itertools import repeat
from collections import Counter
from collections.abc import Generator

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import networkx as nx

from .assemblies import Assemblies
from .kmers import KmerGraph
from .ncbi import blast
from .graph import OrderedKmers
from .utils import print_time_delta, log_and_raise, file_to_write, mp_wrapper, most_common, most_common_weighted
from .config import Config, RunState, WORKINGDIR, BLASTCONFIG, CONSEC_KMER_TH, LEN_TH_MUL, NO_BLAST_DIV

# Translation table for complement k-mer strands, deprecated. ['+' -> '-', '-' -> '+']
_KMER_STRAND_COMP: dict[str, str] = str.maketrans('+-', '-+')


class ConnectedKmers(object):
    """The candidate marker Class, created from a low-penalty subgraph of the k-mer graph `KmerGraph.graph`. 
    
    Attributes:
        graph (nx.Graph): A low-penalty subgraph of the k-mer graph `KmerGraph.graph`. 
        kmers (pd.DataFrame): K-mers of each node in the subgraph, from all assemblies. 
            It's a subset of `KmerGraph.kmers`, with index inherited. 
            K-mers with adjacent indices are also adjacent in the assembly sequence. 
        loc (pd.DataFrame): Location of the subgraph in each assembly. 
            Columns: ['assembly_idx', 'record_idx', 'start', 'stop', 'n_kmers', 
            'kmers', 'is_target', 'n_repeats', 'len']. 
        path (OrderedKmers | None): K-mer ordering in the graph. None if the graph is not linear. 
        rep (pd.Series): Representative sequence of the subgraph (a certain row in `loc`). 
        len (int): Length of the representative sequence. 
        n_rep (int): Number of assemblies having the same k-mer order as the representative. 
        blast (pd.DataFrame | None): The best BLAST hit of the representative sequence in each assembly. 
        conservation (float | None): Average fraction of identical bases between the representative and target assemblies. 
        divergence (float | None): Average fraction of mismatches and gaps between the representative and non-target assemblies. 
        rep_ratio (float | None): Fraction of target assemblies that have the same k-mer ordering as the representative. 
        warning (set): Warning messages for debugging. 
        is_bad (bool): If True, this instance is not considered in downstream processing. 
    """
    __slots__ = (
        'graph', 'kmers', 'loc', 'path', 'rep', 'len', 'n_rep', 'blast', 
        'conservation', 'divergence', 'rep_ratio', 'warning', 'is_bad'
    )
    graph: nx.Graph
    kmers: pd.DataFrame
    loc: pd.DataFrame
    path: OrderedKmers | None
    rep: pd.Series
    len: int
    n_rep: int
    blast: pd.DataFrame | None
    conservation: float | None
    divergence: float | None
    rep_ratio: float | None
    warning: set
    is_bad: bool

    def __init__(self, graph: nx.Graph, kmers: pd.DataFrame, kmerlen: int) -> None:
        """Given a subgraph of the k-mer graph, 
        1. Determine the boundary of the subgraph in each assembly. 
        2. Determine the representative k-mer order. 
        3. (deprecated) Determine the orientation (strand, +/-) of the subgraph in each assembly. 

        Args:
            graph (nx.Graph): A connected low-penalty subgraph of the k-mer graph `KmerGraph.graph`. 
            kmers (pd.DataFrame): K-mers of each node in the subgraph, from all assemblies. 
                It's a subset of `KmerGraph.kmers`, with index inherited. 
                K-mers with adjacent indices are also adjacent in the assembly sequence. 
            kmerlen (int): See `Config` in `config.py`. 
        """
        self.warning = set() # warning messages for debugging
        # if is_bad is True, this instance is not considered in downstream processing. Conditions:
        # 1. duplicated k-mers in graph
        # 2. representative only has one k-mer
        self.is_bad = False

        # convert categorical columns back to string
        # kmers['strand'] = kmers['strand'].astype(str)

        # determine the boundary of the subgraph in each assembly
        loc = ConnectedKmers.__get_loc(kmers, kmerlen)

        # determine the representative k-mer order and the number of targets having this order
        rep_order, n_rep = self.__get_rep_order(loc)
        # get the representative assembly for BLAST check
        # among the assemblies with rep_order, choose the one with the smallest index
        # in this way, there will be fewer assemblies to be loaded when fetching the actual sequences
        rep = loc[loc['kmers'] == rep_order].iloc[0]

        # determine the representative path in the graph
        graph_order = self.__get_graph_order(graph, rep_order)

        # determine the orientation (+/-) of the subgraph in each assembly, based on k-mer ordering
        # if graph_order is not None:
        #     loc = self.__get_strand(loc, graph_order)
        # else:
        #     loc = self.__get_strand(loc, rep_order)

        # saving kmers and loc might take a lot of memory
        # self.graph = graph
        # self.kmers = kmers
        # self.loc = loc
        self.graph = None
        self.kmers = None
        self.loc = None

        self.path = graph_order
        self.rep = rep
        self.len = rep['len']
        self.n_rep = n_rep
        self.blast = None
        self.conservation = None
        self.divergence = None
        self.rep_ratio = None

    @staticmethod
    def __get_loc(kmers: pd.DataFrame, kmerlen: int) -> pd.DataFrame:
        """Determine the location / boundary of the subgraph in each assembly. 
        1. Find consecutive k-mers for each assembly. 
        2. Determine the boundaries (start & stop) of each group of consecutive k-mers. 
        3. Select the largest consecutive group for each assembly. 

        Args:
            kmers (pd.DataFrame): See `ConnectedKmers.__init__()`. 
            kmerlen (int): See `Config` in `config.py`. 
        
        Returns:
            pd.DataFrame: See `ConnectedKmers.loc`. 
        """
        # sort by k-mer index
        # essentially the same as sorting by k-mer position ['assembly_idx', 'record_idx', 'pos']
        kmers.sort_index(inplace=True)

        # since a subgraph might be repetitive in an assembly, we cannot simply groupby ['assembly_idx', 'record_idx']
        # we need to find runs of consecutive k-mers, by their indices (index is sorted)
        # if two k-mers have consecutive index, then they are also adjacent on the assembly (if on the same sequence record)
        # check KmerGraph.kmers for more info
        # consec_gp: consecutive group
        # the definition of "consecutive" can be adjusted by changing CONSEC_KMER_TH
        # we are doing this because the subgraph might occur more than one time in the same assembly
        kmers['consec_gp'] = (kmers.index.diff() > CONSEC_KMER_TH).cumsum()

        # find the start and stop of each consecutive group
        # df.groupby preserves the order of rows within each group
        # since position / index is sorted, start is the first row, and stop is the last row
        # strand can be dertermined by the order of k-mers, using k-mer strand as supplement
        loc = kmers.groupby(
            # group by record_idx is also needed, otherwise consec_gp might span across more than one records
            # as_index=False to keep 'assembly_idx' and 'record_idx' as columns in loc
            by=['assembly_idx', 'record_idx', 'consec_gp'], as_index=False, sort=False, observed=True
        ).agg(
            # if we only need 'pos', "groupby()['pos'].agg(['first', 'last', 'size'])" is faster
            start=pd.NamedAgg(column='pos', aggfunc='first'), 
            stop=pd.NamedAgg(column='pos', aggfunc='last'), 
            n_kmers=pd.NamedAgg(column='pos', aggfunc='size'), 
            kmers=pd.NamedAgg(column='hash', aggfunc=tuple), 
            # kmer_strand=pd.NamedAgg(column='strand', aggfunc='sum'), # much faster than "lambda x: ''.join(x)"
            is_target=pd.NamedAgg(column='is_target', aggfunc='first'), # should be the same within each group
        )
        loc.drop(columns='consec_gp', inplace=True)

        # select the largest consecutive group for each assembly (max number of consecutive k-mers)
        # also count the number of consecutive groups for each assembly (number of repeats)
        loc_max = loc.groupby(
            by='assembly_idx', sort=False
        )['n_kmers'].agg(['idxmax', 'size'])
        loc = loc.loc[loc_max['idxmax'].tolist()] # here .loc is a df method
        loc['n_repeats'] = loc_max['size'].tolist()

        # calculate sequence length
        loc['stop'] += kmerlen
        loc['len'] = loc['stop'] - loc['start']

        # add a placeholder for sequences
        loc['seq'] = None

        loc.reset_index(drop=True, inplace=True)
        return loc

    def __get_rep_order(self, loc: pd.DataFrame) -> tuple[OrderedKmers, int]:
        """Determine the representative k-mer order and the number of target assemblies having it. 
        1. Find the most common canonical k-mer ordering in target assemblies, weighted by the number of k-mers. 
        2. Sanity check. 

        Args:
            loc (pd.DataFrame): See `ConnectedKmers.loc`. 
        
        Returns:
            tuple: A tuple containing
                1. rep_order (OrderedKmers): The representative k-mer order. 
                2. n_rep (int): See `ConnectedKmers.n_rep`. 
        """
        # count the number of each unique k-mer ordering in target assemblies
        tar_kmers = loc[loc['is_target'] == True]['kmers']
        c: dict[tuple, int] = Counter(tar_kmers)

        # count the number of each unique canonical k-mer ordering (regardless of orientation)
        c_canonical: dict[tuple, int] = Counter()
        for kmers, n in c.items():
            c_canonical[
                sorted((kmers, kmers[::-1]))[0]
            ] += n

        # get the most common canonical ordering, weighted by the number of k-mers
        rep_canonical = max(
            c_canonical, 
            key=lambda k: len(k)*c_canonical[k]
        )
        # get the most common orientation
        rep_order = OrderedKmers(max(
            (rep_canonical, rep_canonical[::-1]), 
            key=lambda k: c[k] # if k does not exist in tar_kmers, c will return 0 (e.g., only one orientation exists)
        ))

        # sanity check
        if len(rep_order) == 1:
            # has only one k-mer
            self.warning.add('single')
            self.is_bad = True
        if rep_order.is_dup:
            # has duplicate k-mers
            self.warning.add('dup')
            self.is_bad = True

        return rep_order, c_canonical[rep_canonical]

    def __get_graph_order(self, graph: nx.Graph, rep_order: OrderedKmers) -> OrderedKmers | None:
        """Determine k-mer order in the subgraph. 
        1. Check if the subgraph is linear. Return None if not linear. 
        2. If linear, determine its k-mer ordering and check if it has the same orientation with `rep_order`. 
        3. Sanity check. 

        Args:
            graph (nx.Graph): See `ConnectedKmers.__init__()`. 
            rep_order (OrderedKmers): See `ConnectedKmers.__get_rep_order()`. 
        
        Returns:
            OrderedKmers | None: If the graph is linear, return the k-mer order in the graph; else return None. 
        """
        # check linearity
        leaf_nodes = tuple(node for node in graph if graph.degree[node] == 1)
        if len(leaf_nodes) != 2:
            # non-linear graph, cannot determine k-mer ordering
            self.warning.add('non-linear')
            return None

        # get k-mer ordering in the graph
        all_paths: list[list] = list(nx.all_simple_paths(graph, *leaf_nodes))
        if len(all_paths) == 1:
            graph_order = all_paths[0]
        else:
            # multiple paths, choose the best one
            self.warning.add('multi-paths')
            graph_order = None
            # choose the one that is the same as rep_order, if possible
            for path in all_paths:
                # convert to tuple before comparing to rep_order
                path = tuple(path)
                if path == rep_order:
                    graph_order = path
                    break
                elif path == rep_order.rev:
                    graph_order = path[::-1]
                    break
            # failed to find the same one, choose the longest one
            if graph_order is None:
                graph_order = max(all_paths, key=len)

        # make sure rep_order and graph_order have the same orientation
        if rep_order.which_strand(graph_order) == '-':
            graph_order = graph_order[::-1]

        # check if graph_order the same as rep_order
        graph_order = OrderedKmers(graph_order)
        if graph_order != rep_order:
            self.warning.add('inconsistent')

        # check if there is any duplicated k-mers
        if graph_order.is_dup:
            self.warning.add('dup')
            self.is_bad = True

        return graph_order

    def __get_strand(self, loc: pd.DataFrame, ref_order: OrderedKmers) -> pd.DataFrame:
        """Determine the orientation of the subgraph in each assembly, based on k-mer ordering 
        ('+': forward, '-': reverse, '?': undetermined, 'u': single k-mer). 

        Args:
            loc (pd.DataFrame): See `ConnectedKmers.loc`. 
            rep_order (OrderedKmers): See `ConnectedKmers.__get_rep_order()`. 
        
        Returns:
            pd.DataFrame: See `ConnectedKmers.loc` (with updated 'strand' column). 
        """
        # initialize
        loc['strand'] = '?'

        if ref_order != ref_order.rev:
            # forward and reverse ordering is not the same
            loc['strand'] = loc['kmers'].apply(ref_order.which_strand)
            self.warning.update(ref_order.warning)

            # handle sequences with only one shared k-mer with ref_order (strand as 'u'), deprecated
            # not needed if using graph_order as ref, since these are sequences with only one k-mer
            # loc = self.__get_strand_single(loc, ref_order)
        else:
            # k-mer ordering is reversible
            self.warning.add('rev')
            self.is_bad = True
            # determine strand based on 'kmer_strand', deprecated
            # loc = self.__get_strand_ks(loc)
        
        return loc
    
    def __get_strand_single(self, loc: pd.DataFrame, ref_order: OrderedKmers) -> pd.DataFrame:
        """Handle rows in `ConnectedKmers.loc` that have only one shared k-mer with `ref_order`. 

        Args:
            loc (pd.DataFrame): See `ConnectedKmers.loc`. 
            rep_order (OrderedKmers): See `ConnectedKmers.__get_rep_order()`. 
        
        Returns:
            pd.DataFrame: See `ConnectedKmers.loc` (with updated 'strand' column). 
        """
        is_single = (loc['strand'] == 'u')
        if not is_single.any():
            return loc
        
        # get kmer_strand of ref_order
        # it's likely that they all have the same kmer_strand, but just in case we find the most common one
        loc_ref = loc[loc['kmers'] == ref_order]
        if len(loc_ref) > 0:
            ref_strand: str = most_common(loc_ref['kmer_strand'])
        else:
            # ref_order is not found in loc
            # might happen when the most common order and graph_order is not the same
            return loc
        strand_map = {kmer: strand for kmer, strand in zip(ref_order, ref_strand)}
        
        # determine strand by k-mer strand
        def which_strand(row: pd.Series) -> str:
            """To be applied to all rows of loc[is_single]. 
            """
            # here row['kmers'] can have one k-mer, or multiple k-mers but only one of them is found in ref_order
            for kmer, strand in zip(row['kmers'], row['kmer_strand']):
                try:
                    if strand == strand_map[kmer]:
                        return '+'
                    else:
                        return '-'
                except KeyError:
                    # the current k-mer is not included in the most common ordering
                    continue
            # no shared k-mer with mc_order, which should not happen since this is handled in OrderedKmers.which_strand()
            self.warning.add('single_?')
            return '?'
        loc.loc[is_single, 'strand'] = loc[is_single].apply(which_strand, axis=1)
        return loc
    
    def __get_strand_ks(self, loc: pd.DataFrame) -> pd.DataFrame:
        """Determine sequence strand based on 'kmer_strand'. Only used when strand cannot be dertermined by k-mer ordering
        (e.g., forward and reverse ordering are the same (ABA), or mc_order has only one k-mer). 
        NOTE: forward / reverse strand might have the same k-mer strand ordering (e.g., '++--'). 

        Args:
            loc (pd.DataFrame): See `ConnectedKmers.loc`. 
        
        Returns:
            pd.DataFrame: See `ConnectedKmers.loc` (with updated 'strand' column). 
        """
        # get the most common 'kmer_strand', weighted by the number of k-mers
        mc_strand: str = most_common_weighted(loc['kmer_strand'])
        # reverse complement of mc_strand
        mc_strand_rc = mc_strand.translate(_KMER_STRAND_COMP)[::-1]
        
        def which_strand(kmer_strand: str) -> str:
            if kmer_strand == mc_strand:
                return '+'
            elif kmer_strand == mc_strand_rc:
                return '-'
            elif (kmer_strand in mc_strand) and (kmer_strand not in mc_strand_rc):
                return '+'
            elif (kmer_strand in mc_strand_rc) and (kmer_strand not in mc_strand):
                return '-'
            else: # undetermined strand
                self.warning.add('rev_?')
                return '?'
        loc['strand'] = loc['kmer_strand'].apply(which_strand)
        return loc
    
    @staticmethod
    def __filter(loc: pd.DataFrame) -> pd.DataFrame:
        """Remove abnormal rows in loc. 

        Args:
            loc (pd.DataFrame): See `ConnectedKmers.loc`. 
        
        Returns:
            pd.DataFrame: See `ConnectedKmers.loc` (with rows removed). 
        """
        # remove sequences with
        # 1. only one k-mer
        loc = loc[loc['n_kmers'] > 1]
        # 2. undetermined strand
        loc = loc[(loc['strand'] == '+') | (loc['strand'] == '-')]
        # 3. abnormal length (should be done last)
        # a possible scenario resulting in a super long sequence is a long run of 'N's in the original sequence
        # Indexlr will skip those 'N's and there will be a huge gap between the coordinates of neighboring k-mers
        med_len = np.median(loc['len'])
        loc = loc[loc['len'] < LEN_TH_MUL*med_len]
        return loc


def _create_ck(graph: nx.Graph, kmers: pd.DataFrame, kmerlen: int) -> ConnectedKmers:
    """Creates a ConnectedKmers instance (for multiprocessing). 
    """
    return ConnectedKmers(graph, kmers, kmerlen)


def _get_create_ck_args(
    kmers: KmerGraph, kmerlen: int
) -> Generator[tuple[nx.Graph, pd.DataFrame, int], None, None]:
    """Generates input arguments for `_create_ck()`. 

    Args:
        kmers (KmerGraph): See `KmerGraph` in `kmers.py`. 
        kmerlen (int): See `Config` in `config.py`. 

    Yields:
        tuple: Input arguments of `_create_ck()`. 
    """
    # convert the k-mers df to a dict of k-mer clusters, this will make fetching k-mers with the same hash much faster
    # although this can be done before calculating cluster penalty, but converting groupby to dict is much slower than groupby.agg()
    # so unless we are not filtering any k-mer clusters, it is faster to 
    # 1. calculate cluster penalty with groupby.agg()
    # 2. filter out high penalty k-mer clusters
    # 3. convert the smaller df to a dict of k-mer clusters
    kmer_clusters = pd.DataFrame(kmers.kmers, index=kmers.idx, copy=False)
    kmer_clusters = dict(tuple(kmer_clusters.groupby('hash', sort=False)))

    # create a nx instance for each subgraph, and extract underlying k-mers from the df
    graph = kmers.graph
    for subgraph in kmers.subgraphs:
        # each subgraph is a set of nodes, so it's not ordered
        arg_graph = graph.subgraph(subgraph).copy()

        # fetch k-mers for each node (k-mer cluster) and keep k-mer index
        arg_kmers = pd.concat(
            (kmer_clusters.pop(node) for node in subgraph), # remove node from dict to save memory
            ignore_index=False
        )

        yield arg_graph, arg_kmers, kmerlen


def _fetch_cks_seq(
    all_cks: list[ConnectedKmers], assemblies: Assemblies, rep_only: bool, n_cpu: int
) -> list[str] | None:
    """Fetch the actual sequences for a list of ConnectedKmers instances. 

    Args:
        all_cks (list[ConnectedKmers]): See `_get_cks()`. 
        assemblies (Assemblies): See `Assemblies` in `assemblies.py`. 
        rep_only (bool): If True, only fetch the representative of each ConnectedKmers instance (fewer assemblies to be loaded); 
            else fetch all sequences. 
        n_cpu (int): See `Config` in `config.py`. 
    
    Returns:
        list[str] | None: If `rep_only=True`, return a list of representative sequences; else return None. 
    """
    if rep_only:
        # concat all ConnectedKmers.rep and transpose
        df_loc = pd.concat(
            (ck.rep for ck in all_cks), 
            axis=1, ignore_index=True
        ).transpose()
    else:
        # concat all ConnectedKmers.loc, and add another level of index to label each instance
        ck_idx = range(len(all_cks))
        # ck.loc.index is already sorted
        df_loc = pd.concat(
            (ck.loc for ck in all_cks), 
            ignore_index=False, keys=ck_idx
        )

    # fetch sequences (make sure df_loc.index is sorted with ascending=True)
    all_seq = assemblies.fetch_seq(df_loc, n_cpu)

    if rep_only:
        # update all ConnectedKmers.rep['seq']
        for ck, seq in zip(all_cks, all_seq):
            ck.rep['seq'] = seq
        return all_seq.to_list()
    else:
        # update all ConnectedKmers.loc['seq']
        for ck, i in zip(all_cks, ck_idx):
            #if len(ck.loc) > 0:
            # use to_list() since all_seq is already sorted
            ck.loc['seq'] = all_seq.loc[i].to_list()


def _get_cks(
    kmers: KmerGraph, assemblies: Assemblies, kmerlen: int, min_len: int, n_tar: int, n_cpu: int
) -> tuple[list[ConnectedKmers], list[str]]:
    """
    1. Create a ConnectedKmers instance for each low-penalty subgraph of the k-mer graph (`KmerGraph.subgraphs`). 
    2. Remove instances that are shorter than min_len or have defects (`ConnectedKmers.is_bad`). 
    3. Fetch the representative sequence for each remaining instances. 
    
    Args:
        kmers (KmerGraph): See `KmerGraph` in `kmers.py`. 
        assemblies (Assemblies): See `Assemblies` in `assemblies.py`. 
        kmerlen (int): See `Config` in `config.py`. 
        min_len (int): See `Config` in `config.py`. 
        n_tar (int): See `RunState` in `config.py`. 
        n_cpu (int): See `Config` in `config.py`. 

    Returns:
        tuple: A tuple containing
            1. all_cks (list[ConnectedKmers]): Candidate markers as ConnectedKmers instances. 
            2. all_reps (list[str]): Sequence of each marker. 
    """
    logger.info('Finding a representative for each low-penalty subgraph...')
    tik = time()

    # create a ConnectedKmers instance for each subgraph
    logger.info(' - Processing each subgraph...')
    all_cks: list[ConnectedKmers] = mp_wrapper(
        _create_ck, 
        _get_create_ck_args(kmers, kmerlen), 
        n_cpu=n_cpu, n_jobs=len(kmers.subgraphs)
    )

    # get candidate ConnectedKmers instances
    all_cks = list(
        ck for ck in all_cks 
        if (ck.len >= min_len) and (not ck.is_bad)
    )
    logger.info(f' - Found {len(all_cks)} candidate markers')

    logger.info(' - Fetching the representative sequence for each candidate...')
    all_reps = _fetch_cks_seq(all_cks, assemblies, rep_only=True, n_cpu=n_cpu)

    # update rep_ratio of each ck
    for ck in all_cks:
        ck.rep_ratio = ck.n_rep / n_tar

    print_time_delta(time()-tik)
    return all_cks, all_reps


def _get_avg_ident(blast_out: pd.DataFrame, query_len: int, n: int) -> float:
    """Given a list of BLAST hits, calculate the average sequence identity between the query and all subjects. 
    The denominator (`n`) is the number of subject sequences that are expected to include the query sequence. 
    Note that `n` might not be the same as `len(blast_out)`, since some subjects may have no hit. 
    
    Args:
        blast_out (pd.DataFrame): Each row should be a BLAST hit of the query, with column
            'nident' (number of identical matches). 
        query_len (int): Length of the query sequence. 
        n (int): The number of subjects that are expected to include the query sequence. 
    
    Returns:
        float:
    """
    return sum(blast_out['nident']) / query_len / n


def _get_avg_dist(blast_out: pd.DataFrame, query_len: int, n: int) -> float:
    """Given a list of BLAST hits, calculate the average distance between the query and all subjects. 
    The denominator (`n`) is the number of subject sequences that are expected to include the query sequence. 
    Note that `n` might not be the same as `len(blast_out)`, since some subjects may have no hit. 
    
    Args:
        blast_out (pd.DataFrame): Each row should be a BLAST hit of the query, with columns, 
            1. 'mismatch': number of mismatches. 
            2. 'gaps': total number of gaps in BOTH query and subject (might cause inaccuracy). 
        query_len (int): Length of the query sequence. 
        n (int): The number of subjects that are expected to include the query sequence. 
    
    Returns:
        float:
    """
    return sum(blast_out['mismatch'] + blast_out['gaps']) / query_len / n


def _get_scores(
    blast_out: pd.DataFrame, marker_len: int, n_tar: int | None=None, n_neg: int | None=None
) -> tuple[float | None, float | None]:
    """Calculate the conservation and divergence of a marker based on its BLAST hits in all assemblies. 
    - Conservation is calculated with `_get_avg_ident()` on target assemblies. 
    - Divergence is calculated with `_get_avg_dist()` on non-target assemblies. 
        For assemblies with no hit, divergence is assumed to be `NO_BLAST_DIST` in `config.py`. 
    
    Args:
        blast_out (pd.DataFrame): Each row is the best BLAST hit of the marker in a assembly. 
            Required columns: ['is_target', 'nident', 'mismatch', 'gaps']
        marker_len (int): Marker length. 
        n_tar (int | None): Number of target assemblies. If None, conservation is not calculated. 
        n_neg (int | None): Number of non-target assemblies. If None, divergence is not calculated. 
    
    Returns:
        tuple: (conservation, divergence)
    """
    if n_tar is None: # do not calculate conservation
        conservation = None
    else:
        conservation = .0 # conservation value when no blast hit in target assemblies

    if n_neg is None: # do not calculate divergence
        divergence = None
    else:
        divergence = NO_BLAST_DIV # divergence value when no blast hit in non-target assemblies

    if blast_out is None: # no blast hit in any assembly
        return conservation, divergence

    # calculate conservation
    if n_tar is not None:
        df_tar = blast_out[blast_out['is_target'] == True]
        conservation = _get_avg_ident(df_tar, marker_len, n_tar)

    # calculate divergence
    if n_neg is not None:
        df_neg = blast_out[blast_out['is_target'] == False]
        divergence = _get_avg_dist(df_neg, marker_len, n_neg)
        divergence += NO_BLAST_DIV * (n_neg - len(df_neg)) / n_neg

    return conservation, divergence


def eval_markers(
    all_seqs: list[str], blastdb: Path, n_tar: int, n_neg: int, n_cpu: int=1
) -> tuple[list[pd.DataFrame], list[float], list[float]]:
    """BLAST check each marker sequence against all / non-target assemblies, and calcuate the conservation and divergence of each marker. 
    
    Args:
        all_seqs (list[str]): A list of marker sequences. 
        blastdb (Path): Path to the BLAST database. 
        n_tar (int): Number of target assemblies. 
        n_neg (int): Number of non-target assemblies. 
        n_cpu (int, optional): Number of threads to use. [1]
    
    Returns:
        tuple: A tuple containing
            1. all_blast (list[pd.DataFrame]): BLAST hits of each marker. 
            2. all_conservation (list[float]): Conservation of each marker. 
            2. all_divergence (list[float]): Divergence of each marker. 
    """
    if blastdb.name == BLASTCONFIG.title_neg_only:
        neg_only = True
        logger.info('BLAST checking markers against non-target assemblies (less sensitive but faster)...')
    elif blastdb.name == BLASTCONFIG.title_all:
        neg_only = False
        logger.info('BLAST checking markers against all assemblies (more sensitive but slower)...')
    else:
        log_and_raise(ValueError, f'Invalid BLAST database title. Must be "{BLASTCONFIG.title_all}" or "{BLASTCONFIG.title_neg_only}"')
    tik = time()
    n_seqs = len(all_seqs)

    # blast check all markers against all / non-target assemblies
    blast_out = blast(all_seqs, db=blastdb, task=BLASTCONFIG.task, columns=BLASTCONFIG.columns, n_cpu=n_cpu, batch_size=BLASTCONFIG.batch_size)
    if len(blast_out) == 0:
        log_and_raise(RuntimeError, 'No BLAST hit found.')
    #blast_out.to_csv('blast_out.tsv', sep='\t')

    #---------- extract BLAST hits of each marker ----------#
    logger.info(' - Formatting BLAST output...')
    # get assembly id and record id, see Assemblies.makeblastdb()
    blast_out[['assembly_idx', 'is_target', 'record_id']] = blast_out['sseqid'].str.split(
        BLASTCONFIG.header_sep, expand=True
    )
    blast_out.drop(columns='sseqid', inplace=True)
    # unlike pd.read_csv() in blast(), df.str.split() does not do auto type conversion
    blast_out['assembly_idx'] = blast_out['assembly_idx'].astype(int)
    blast_out['is_target'] = blast_out['is_target'].map(BLASTCONFIG.str2bool)

    # keep the best alignment (highest bitscore) for each assembly
    blast_out.sort_values(
        by=['qseqid', 'assembly_idx', 'bitscore'], 
        ascending=[True, True, False], inplace=True
    )
    blast_out = blast_out.groupby(
        # as_index=True so that .agg() could output a series
        by=['qseqid', 'assembly_idx'], as_index=True, sort=False
    )
    # also keep bitscores of other less optimal alignments
    bitscore = blast_out['bitscore'].agg(tuple)
    blast_out = blast_out.head(1)
    blast_out['bitscore_other'] = bitscore.tolist()
    blast_out.reset_index(drop=True, inplace=True)

    # output a df for each query sequence (some markers might have no BLAST hit)
    all_blast = [None] * n_seqs
    for i, g in blast_out.groupby('qseqid', sort=False):
        g.drop(columns='qseqid', inplace=True)
        g.reset_index(drop=True, inplace=True)
        all_blast[i] = g
    #---------- extract BLAST hits of each marker ----------#
    
    if not neg_only: # check for markers with no BLAST hit
        for i, b in enumerate(all_blast):
            if b is None:
                logger.warning(f'Marker at index {i} (0-based) has no BLAST hit in any assembly ({all_seqs[i][:10]}...)')

    # calculate conservation and divergence for each marker based on its blast output
    logger.info(' - Evaluating each marker...')
    if neg_only:
        # do not calculate conservation since target assemblies are not included in the BLAST database
        n_tar = None
    scores_args = zip(
        all_blast, 
        map(len, all_seqs), 
        repeat(n_tar, n_seqs), 
        repeat(n_neg, n_seqs)
    )
    all_conservation, all_divergence = mp_wrapper(
        _get_scores, scores_args, n_cpu, unpack_output=True, n_jobs=n_seqs
    )

    print_time_delta(time()-tik)
    return all_blast, all_conservation, all_divergence


def _eval_cks(
    all_cks: list[ConnectedKmers], all_reps: list[str], blastdb: Path, n_tar: int, n_neg: int, n_cpu: int
) -> None:
    """
    1. BLAST check the representative sequence of each ConnectedKmers instance (ck), against all / non-target assemblies. 
    2. Calculate the conservation and divergence for each ck. 
    3. Update the attributes of each ck. 
    4. Sort all_cks by conservation + divergence. 
    
    Args:
        all_cks (list[ConnectedKmers]): ConnectedKmers instances. 
        all_reps (list[str]): Representative sequences, in the same order as all_cks. 
        blastdb (Path): See `RunState` in `config.py`. 
        n_tar (int): See `RunState` in `config.py`. 
        n_neg (int): See `RunState` in `config.py`. 
        n_cpu (int): See `Config` in `config.py`. 
    """
    # run evaluation
    results = eval_markers(all_reps, blastdb, n_tar, n_neg, n_cpu)

    # update attributes of each ck
    for ck, blast, conservation, divergence in zip(all_cks, *results):
        ck.blast, ck.conservation, ck.divergence = blast, conservation, divergence

    # sort in-place
    all_cks.sort(key=lambda ck: ck.conservation+ck.divergence, reverse=True)


def get_markers(
    kmers: KmerGraph, assemblies: Assemblies, config: Config, state: RunState
) -> list[ConnectedKmers]:
    """Extract candidate markers from a k-mer graph, and save them to files. 
    
    Args:
        kmers (KmerGraph): See `KmerGraph` in `kmers.py`. 
        assemblies (Assemblies): See `Assemblies` in `assemblies.py`. 
        config (Config): See `Config` in `config.py`. 
        state (RunState): See `RunState` in `config.py`. 

    Returns:
        list[ConnectedKmers]: Candidate markers. 
    """
    overwrite = config.overwrite
    kmerlen = config.kmerlen
    min_len = config.min_len
    run_blast = config.run_blast
    blast_neg_only = config.blast_neg_only
    n_cpu = config.n_cpu

    working_dir = state.working_dir
    n_tar = state.n_tar
    n_neg = state.n_neg

    # extract marker from each low-penalty subgraph
    # all_cks: ConnectedKmers (ck) instances; all_reps: representative sequences
    all_cks, all_reps = _get_cks(kmers, assemblies, kmerlen, min_len, n_tar, n_cpu)

    # evaluate each marker with BLAST
    if run_blast:
        logger.info('Evaluating candidate markers with BLAST...')
        blastdb = assemblies.makeblastdb(
            prefix=working_dir / WORKINGDIR.blast_dir, 
            neg_only=blast_neg_only, 
            overwrite=overwrite, 
            n_cpu=n_cpu
        )
        _eval_cks(all_cks, all_reps, blastdb, n_tar, n_neg, n_cpu)
    else:
        logger.warning(f'Marker evaluation is turned off (--no-blast), skip running BLAST')
        blastdb = None

    # save to fasta
    markers_fasta = working_dir / WORKINGDIR.markers_fasta
    file_to_write(markers_fasta, overwrite)
    fasta = ''
    csv = list()
    all_record_ids = assemblies.record_ids
    for ck in all_cks:
        rep = ck.rep
        assembly_idx = rep.assembly_idx
        record_id = all_record_ids[assembly_idx][rep.record_idx]
        header = f'{assembly_idx}-{record_id}-{rep.start}:{rep.stop}'
        fasta += f'>{header}\n{rep.seq}\n'
        csv.append(
            (header, ck.len, ck.conservation, ck.divergence, ck.rep_ratio, rep.n_kmers)
        )
    markers_fasta.write_text(fasta)
    logger.info(f'Candidate markers saved as {markers_fasta}')

    # save to csv
    markers_csv = working_dir / WORKINGDIR.markers_csv
    file_to_write(markers_csv, overwrite)
    pd.DataFrame(
        csv, 
        columns=('fasta_header', 'length', 'conservation', 'divergence', 'rep_ratio', 'n_nodes')
    ).to_csv(markers_csv, index=False)
    logger.info(f'Metrics of candidate markers saved as {markers_csv}')

    state.blastdb = blastdb
    return all_cks
