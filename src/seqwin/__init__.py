"""
Seqwin
======

Ultrafast identification of signature sequences in microbial genomes
- Output sensitive and specific genomic signatures. 
- High input scalability. 
- Robust to sequence variations and low-quality assemblies. 
- Use raw sequences as input. 

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
- numpy >=2
- pandas >=2
- networkx
- pydantic
- typer
- mash (optional)
- blast (optional)
- ncbi-datasets-cli (optional)

Packages:
--------
- graph

Modules:
--------
- core
- assemblies
- kmers
- markers
- helpers
- ncbi
- mash
- utils
- config
"""

__author__ = 'Michael X. Wang'
from ._version import __version__
__license__ = 'GPL 3.0'

from .config import Config # import first to init logger
from .core import Seqwin, run, load
