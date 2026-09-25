"""
Seqwin core
===========

Core classes and dtypes of Seqwin.

Usage:
------
```python
>>> from seqwin.core import KmerGraph
>>> help(KmerGraph)
```

Classes:
----------
- KmerGraph
- FilteredGraph
- SubgraphLoc
- Signature

Attributes:
-----------
- KMER_DTYPE (np.dtype)
- NODE_DTYPE (np.dtype)
- EDGE_DTYPE (np.dtype)
"""

from .graph import KMER_DTYPE, NODE_DTYPE, EDGE_DTYPE, KmerGraph
from ._native import FilteredGraph, SubgraphLoc, Signature
