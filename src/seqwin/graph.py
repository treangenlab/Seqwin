"""
Graph
=====

Graph utilities. 

Dependencies:
-------------
- networkx
- matplotlib (optional)

Classes:
--------
- WeightedGraph
- OrderedKmers

Functions:
----------
- compose_weighted_graphs
- add_path_weighted
- draw_weighted_graph

Attributes:
-----------
- EDGE_W (str)
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import logging
from math import sqrt
from itertools import chain, tee
from collections import Counter
from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

import networkx as nx
try:
    from matplotlib import pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

EDGE_W: str = 'w' # Key for edge weight, used in networkx graphs. ['w']


class WeightedGraph(Counter):
    """A weighted graph object inherited from `collections.Counter`, with data structure of {edge: weight}. 
    An edge should contain two hashable elements and ordering matters, with edge direction of (first -> second). 
    """
    def __init__(self, edges: Iterable[tuple]=()) -> None:
        """Create a weighted graph with an Iterable of edges. 
        Each edge should be a tuple containing two hashable elements as nodes. 
        Note that the ordering of the two nodes matters, with edge direction of (first node -> second node). 

        Args:
            edges (Iterable[tuple], optional): Each edge should be a tuple containing two hashable elements as nodes. 
            Note that the ordering of the two nodes matters, with edge direction of (first node -> second node). 
        """
        super().__init__(edges)

    def add_path(self, nodes: Iterable, cyclic=False) -> None:
        """Add a path to the weighted graph. 

        Args:
            nodes (Iterable): A path will be constructed from the nodes (in order) and added to the graph.
        """
        nodes = iter(nodes)
        start_nodes, stop_nodes = tee(nodes, n=2) # edge direction: start node -> stop node
        try:
            first_node = next(stop_nodes)
        except StopIteration:
            # no node is provided
            return
        if cyclic:
            stop_nodes = chain(stop_nodes, (first_node,))
        
        # add edges to graph
        self.update(
            tuple((u, v)) for u, v in zip(start_nodes, stop_nodes)
        )

    def to_nxGraph(self) -> nx.Graph:
        """Convert to networkx.Graph (a undirected graph), with edge weights set as edge attribute EDGE_W. 
        """
        return nx.Graph((*edge, {EDGE_W: weight}) for edge, weight in self.items())


class OrderedKmers(tuple):
    """Ordered k-mers created from an Iterable of integers. 
    The `which_strand()` method can take another Iterable of k-mers and determine its strand ('+'/'-'/'?'/'u'), 
    by comparing its ordering to self. 
    
    Attributes:
        rev (tuple): K-mers in reversed order. 
        is_dup (bool): True if there are duplicates in the k-mers. 
        warning (set): For debugging only. 
    
    Examples:
        ```
        l = [
            (1,2,3,3,4,5),
            (5,4,3,3,2,1),
            (1,2,3,4,5), 
            (5,4,3,2,1), 
            (2,), 
            (0,), 
            (6,5), 
            (9,10),
            (1,3,5),
            (2,3,4),
            (1,0,2,4),
            (5,3,1),
            (4,3,2),
            (4,2,0,1),
            (3,2,4,6)
        ]
        for t in l:
            k = OrderedKmers((1,2,3,3,4,5))
            print(t)
            print(k.which_strand(t))
            print(k._warning)
            print()
        ```
    """
    def __new__(cls, kmers: Iterable[int]):
        # tuple is immutable, so the content of the object must be defined during object creation
        return super().__new__(cls, kmers)

    def __init__(self, kmers: Iterable[int]) -> None:
        """Ordered k-mers created from an Iterable of integers. 
        The `which_strand()` method can take another Iterable of k-mers and determine its strand ('+'/'-'/'?'/'u'), 
        by comparing its ordering to self. 

        Args:
            kmers (Iterable[int]): K-mers as an Iterable of integers. 
        """
        # here self is already created as a tuple
        # kmers is not used here, but have to keep it or it will raise a TypeError (for docstring as well)
        self.rev = self[::-1]
        self._idx_map = {kmer: idx for idx, kmer in enumerate(self)}
        self.is_dup = len(self._idx_map) < self.__len__() # True if there are duplicated k-mers
        self.warning = set()
    
    def which_strand(self, kmers: Iterable[int]) -> str:
        """Given an Iterable of k-mers, compare its ordering to self and determine its strand ('+'/'-'/'?'/'u'). 

        Args:
            kmers (Iterable[int]): K-mers as an Iterable of integers. 

        Returns:
            str: strand type
            - '+': forward strand, 
            - '-': reverse strand, 
            - '?': unknown strand, 
            - 'u': only one shared k-mer with self, so the strand has to be determined by other methods. 
        """
        # keep in mind that there might be k-mers not found in self
        idx_map = self._idx_map
        if kmers == self:
            return '+'
        elif kmers == self.rev:
            return '-'
        elif len(kmers) == 1:
            if kmers[0] in idx_map:
                return 'u'
            else:
                self.warning.add(1)
                return '?'
        # determine if k-mers appear in the same order as self
        elif not self.is_dup:
            # no duplicates in self, use idx_map to check k-mer order
            all_idx = list()
            for k in kmers:
                try:
                    all_idx.append(idx_map[k])
                except KeyError:
                    # the current k-mer is not included in self
                    continue
            # check if indexes are non-decreasing or non-increasing
            if len(all_idx) == 1:
                self.warning.add(2)
                return 'u'
            elif len(all_idx) == 0:
                self.warning.add(3)
                return '?'
            elif all_idx == sorted(all_idx, reverse=False):
                return '+'
            elif all_idx == sorted(all_idx, reverse=True):
                return '-'
            else:
                self.warning.add(4)
                return '?'
        else:
            # duplicates in self (use a less effecient method to check k-mer order)
            # only check k-mers shared with self
            kmers_shared = tuple(k for k in kmers if k in idx_map)
            n_kmers_shared = len(kmers_shared)
            if n_kmers_shared == 1:
                self.warning.add(5)
                return 'u'
            elif n_kmers_shared == 0:
                self.warning.add(6)
                return '?'
            def check_order(orderedKmers) -> bool:
                i = 0
                for kmer in orderedKmers:
                    if kmer == kmers_shared[i]:
                        i += 1
                        if i == n_kmers_shared:
                            return True
                return False
            if check_order(self):
                return '+'
            elif check_order(self.rev):
                return '-'
            else:
                self.warning.add(7)
                return '?'


def compose_weighted_graphs(graphs: Iterable[WeightedGraph]):
    """Compose multiple WeightedGraph objects by adding the weights of same edges together. 

    Args:
        graphs (Iterable[WeightedGraph]): Graphs to be composed. 

    Returns:
        WeightedGraph: The composed graph. 
    """
    graphs = iter(graphs)
    try:
        merged_graph = next(graphs)
    except StopIteration:
        raise ValueError('No graph is given to compose. ')
    
    merged_graph = merged_graph.copy() # a shallow copy
    for g in graphs:
        merged_graph.update(g)
    return merged_graph


def add_path_weighted(graph: nx.Graph, path: Sequence) -> None:
    """Add a path to a weighted, undirected graph. Increment edge weight by 1 if the edge already exists. 

    Args:
        graph (nx.Graph): A weighted, undirected graph. 
        path (Sequence): A path (sequence of nodes) to be added to graph. 
    """
    # loop through each consecutive pair of nodes in the path
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]

        # if edge already exists, increment the weight
        try:
            graph[u][v]['weight'] += 1
        except KeyError:
            # otherwise, add the edge with an initial weight of 1
            graph.add_edge(u, v, weight=1)


if _HAS_MPL:
    def draw_weighted_graph(
        graph: nx.Graph, 
        save_path: str | None=None, 
        figsize: tuple | None=None, 
        node_size: int=200, 
        edge_width: int=2, 
        font_size: int=8, 
        seed: int=0
    ) -> None:
        """Draw a networkx graph with edge attribute 'weight'. 
        Code adopted from `networkx doc<https://networkx.org/documentation/stable/auto_examples/drawing/plot_weighted_graph.html>`__.

        Args:
            graph (nx.Graph): A weighted, undirected graph. 
            save_path (str | None, optional): Path to save the figure in SVG format. None for showing the figure without saving. [None]
        """
        # positions for all nodes - seed for reproducibility
        pos = nx.spring_layout(graph, k=2/sqrt(len(graph)), iterations=5000, weight=None, seed=seed)

        if figsize is not None:
            plt.figure(figsize=figsize)
        
        # nodes
        nx.draw_networkx_nodes(graph, pos, node_size=node_size)

        # edges
        nx.draw_networkx_edges(graph, pos, edgelist=graph.edges, width=edge_width)

        # node labels
        #nx.draw_networkx_labels(graph, pos, font_size=font_size)
        # edge weight labels
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=font_size)

        ax = plt.gca()
        ax.margins(0.1)
        plt.axis('off')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, transparent=True, format='svg')
        plt.show()
else:
    def draw_weighted_graph(
        graph, save_path=None, figsize=None, node_size=None, edge_width=None, font_size=None, seed=None
    ) -> None:
        raise ImportError('Matplotlib is needed for drawing a graph') from None
