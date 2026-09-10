"""
Graph
=====

Graph utilities.

Dependencies:
-------------
- networkx (optional)
- matplotlib (optional)

Classes:
--------
- OrderedKmers

Functions:
----------
- draw_weighted_graph

Attributes:
-----------
- EDGE_W (str)
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import logging
from math import sqrt
from collections.abc import Iterable

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    nx = None
    _HAS_NX = False
try:
    from matplotlib import pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

EDGE_W: str = 'w' # Key for edge weight, used in networkx graphs. ['w']


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
            print(k.warning)
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
            # check if indices are non-decreasing or non-increasing
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


if _HAS_MPL and _HAS_NX:
    def draw_weighted_graph(
        graph: nx.Graph,
        save_path: str | None=None,
        figsize: tuple | None=None,
        node_size: int=200,
        edge_width: int=2,
        font_size: int=8,
        seed: int=0
    ) -> None:
        """Draw a NetworkX graph with edge attribute 'weight'.
        Code adapted from `networkx doc<https://networkx.org/documentation/stable/auto_examples/drawing/plot_weighted_graph.html>`__.

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
        missing = ' and '.join(
            name for name, available in (('NetworkX', _HAS_NX), ('Matplotlib', _HAS_MPL))
            if not available
        )
        raise ImportError(f'{missing} is needed for drawing a graph') from None
