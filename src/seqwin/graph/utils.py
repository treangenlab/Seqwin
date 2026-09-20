"""
Graph
=====

Graph utilities.

Dependencies:
-------------
- networkx (optional)
- matplotlib (optional)

Functions:
----------
- draw_weighted_graph
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import logging
from math import sqrt

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

_EDGE_W: str = 'w' # Key for edge weight, used in networkx graphs. ['w']

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
        """Draw a NetworkX graph with edge attribute 'w'.
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
        edge_labels = nx.get_edge_attributes(graph, _EDGE_W)
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
