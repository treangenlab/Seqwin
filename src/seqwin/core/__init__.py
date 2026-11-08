"""
Seqwin core
===========

Core classes and functions of Seqwin.

Classes:
--------
- AssemblyPaths
- FastaRecord
- Nodes
- Graph
- FilteredGraph
- SubgraphLoc
- Signature

Functions:
----------
- read_fasta
"""

from ._native import (
    AssemblyPaths, FastaRecord, Nodes, Graph, FilteredGraph, SubgraphLoc, Signature,
    read_fasta,
)
