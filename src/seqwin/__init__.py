"""
Seqwin
======

Rapid and sensitive search of clade-specific microbial marker sequences via minimizer graphs
- Output sensitive and specific marker sequences. 
- High input scalability. 
- Robust to low-quality genomes. 
- Do not rely on annotations. 

Usage:
------
```python
>>> from seqwin import Config, run
>>> config = Config(tar_paths=..., neg_paths=..., ...)
>>> results = run(config)
```

Dependencies:
-------------
- python >=3.10
- numpy
- numba
- pandas
- networkx
- pydantic
- typer
- btllib
- mash
- blast
- ncbi-datasets-cli
- scipy (optional)

Modules:
--------
- core
- assemblies
- kmers
- markers
- helpers
- minimizer
- ncbi
- mash
- graph
- utils
- config
"""

__author__ = 'Michael X. Wang'
from ._version import __version__
__license__ = 'GPL 3.0'

from .config import Config # import first to init logger
from .core import Seqwin, run, load
