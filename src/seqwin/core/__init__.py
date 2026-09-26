"""
Seqwin core
===========

Core classes and dtypes of Seqwin.

Usage:
------
```python
>>> from seqwin.core import Graph
>>> help(Graph)
```

Classes:
--------
- Graph
- FilteredGraph
- SubgraphLoc
- Signature

Attributes:
-----------
- KMER_DTYPE (np.dtype)
- NODE_DTYPE (np.dtype)
- EDGE_DTYPE (np.dtype)
"""

from .graph import KMER_DTYPE, NODE_DTYPE, EDGE_DTYPE, Graph
from ._native import FilteredGraph, SubgraphLoc, Signature
