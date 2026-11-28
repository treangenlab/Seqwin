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
- RetainedGraph
- SubgraphLoc
- Signature

Functions:
----------
- read_fasta
"""

from ._native import (
    AssemblyPaths, FastaRecord, Nodes, Graph, RetainedGraph, SubgraphLoc, Signature,
    read_fasta,
)
